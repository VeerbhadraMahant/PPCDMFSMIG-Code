"""DP-SGD (Opacus) training of the pixel DDPM, and its non-DP twin (see PROTOCOL_V2.md, E5)."""
import argparse
import copy
import csv
import time
import warnings

import numpy as np
import torch

from common import (atomic_save, build_unet, ema_update, get_device, load_data, make_ddpm,
                    member_indices, resolve, set_seed, to_tensor, write_json, T_TRAIN)

# Opacus "hooks" mode returns an all-zero grad_sample for the conv in Downsample2D/Upsample2D
# (found with dp_gate.py's correctness check, absent in "functorch" mode), so "functorch" is the default.
GRAD_SAMPLE_MODE = "functorch"
AMP = "bf16"  # set from the measurements recorded in the commit message / dp_gate.json; see --amp

warnings.filterwarnings("ignore", message="Full backward hook is firing")
warnings.filterwarnings("ignore", message="Secure RNG turned off")


def per_example_loss(model, x, sched, g, amp):
    """One uniform timestep and one Gaussian noise per example; MSE over pixels, shape (b,)."""
    t = torch.randint(0, T_TRAIN, (len(x),), device=x.device, generator=g)
    noise = torch.randn(x.shape, device=x.device, generator=g)
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=amp and x.device.type == "cuda"):
        pred = model(sched.add_noise(x, noise, t), t).sample
    return ((pred.float() - noise) ** 2).flatten(1).mean(1)


def strip_prefix(sd):
    return {k.removeprefix("_module."): v for k, v in sd.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="training set size")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--eps", type=float, default=10.0, help="target epsilon (DP mode)")
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--arch", choices=["base", "small"], default="base")
    ap.add_argument("--batch", type=int, default=512, help="expected (Poisson) batch size, or exact batch for --no-dp")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", type=float, default=1.0, help="per-sample clipping norm")
    ap.add_argument("--physical-bs", type=int, default=64, help="micro-batch for memory only; the logical batch is --batch")
    ap.add_argument("--no-dp", action="store_true", help="non-DP twin: same model, lr, EMA and steps; no clipping or noise")
    ap.add_argument("--amp", choices=["bf16", "fp32"], default=AMP)
    ap.add_argument("--grad-sample-mode", choices=["hooks", "functorch", "ew"], default=GRAD_SAMPLE_MODE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop early (smoke test); resumable")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--tiny", action="store_true", help="self-test: fake data, tiny UNet")
    args = ap.parse_args()

    assert args.n % args.batch == 0, "n must be a multiple of --batch (one epoch = n / batch logical steps)"
    dp = not args.no_dp
    amp = args.amp == "bf16"
    steps = args.epochs * (args.n // args.batch)
    assert steps >= 8, "need at least 8 logical steps for 8 checkpoints"
    ckpt_steps = [round(steps * k / 8) for k in range(1, 9)]
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    final = out / f"ema_{steps:07d}.pt"
    if final.exists():
        print(f"{out} already finished ({final.name}); nothing to do")
        return

    train_u8, _ = load_data(args.tiny)
    idx = member_indices(len(train_u8), args.n)
    np.save(out / "members.npy", idx)
    x_cpu = to_tensor(train_u8[idx], torch.device("cpu"))

    set_seed(args.seed)
    model = build_unet(args.tiny, arch=args.arch).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)  # plain module, so its keys load into build_unet
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)  # default weight decay 0.01
    sched = make_ddpm()
    g = torch.Generator(device=device)  # timesteps, noise (and batch indices for the twin)
    g.manual_seed(args.seed)
    cfg_dp = {"mode": "dp" if dp else "none", "target_eps": args.eps if dp else None,
              "delta": args.delta if dp else None, "epochs": args.epochs, "sample_rate": args.batch / args.n,
              "noise_multiplier": None, "max_grad_norm": args.clip if dp else None,
              "accountant": "rdp" if dp else None, "achieved_eps": None,
              "grad_sample_mode": args.grad_sample_mode if dp else None, "amp": args.amp}
    cfg = {"n": args.n, "steps": steps, "ckpt_steps": ckpt_steps, "tiny": args.tiny, "seed": args.seed,
           "cond": False, "arch": args.arch, "batch": args.batch, "lr": args.lr, "dp": cfg_dp}

    engine = loader_gen = noise_gen = under = None
    if dp:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from torch.utils.data import DataLoader, TensorDataset
        loader_gen = torch.Generator().manual_seed(args.seed)  # Poisson sampler stream
        noise_gen = torch.Generator(device=device).manual_seed(args.seed + 1)  # DP noise stream
        loader = DataLoader(TensorDataset(x_cpu), batch_size=args.batch, shuffle=True, generator=loader_gen)
        engine = PrivacyEngine(accountant="rdp")
        under = model
        model, opt, loader = engine.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=loader, target_epsilon=args.eps, target_delta=args.delta,
            epochs=args.epochs, max_grad_norm=args.clip, poisson_sampling=True,
            grad_sample_mode=args.grad_sample_mode, noise_generator=noise_gen)
        cfg_dp["noise_multiplier"] = float(opt.noise_multiplier)
        cfg_dp["sample_rate"] = float(loader.sample_rate)
    else:
        under = model
        x_dev = x_cpu.to(device)

    eps_now = lambda: float(engine.accountant.get_epsilon(args.delta)) if dp else None
    step = 0
    rp = out / "resume.pt"
    if rp.exists():
        st = torch.load(rp, map_location=device, weights_only=False)
        assert (st["n"], st["steps"], st["tiny"], st["dp"], st["arch"]) == (args.n, steps, args.tiny, dp, args.arch), \
            "resume.pt is from a different run"
        under.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        (opt.original_optimizer if dp else opt).load_state_dict(st["opt"])
        g.set_state(st["g"].cpu())
        if dp:  # restoring the history keeps the privacy accounting correct across the restart
            engine.accountant.load_state_dict(st["accountant"])
            loader_gen.set_state(st["loader_gen"].cpu())
            noise_gen.set_state(st["noise_gen"].cpu())
        step = st["step"]
        print(f"resumed from step {step}", flush=True)

    def save_resume():
        st = {"model": under.state_dict(), "ema": ema.state_dict(),
              "opt": (opt.original_optimizer if dp else opt).state_dict(), "g": g.get_state(), "step": step,
              "n": args.n, "steps": steps, "tiny": args.tiny, "dp": dp, "arch": args.arch}
        if dp:
            st.update(accountant=engine.accountant.state_dict(), loader_gen=loader_gen.get_state(),
                      noise_gen=noise_gen.get_state())
        atomic_save(st, rp)

    def save_ema():
        sd = strip_prefix(ema.state_dict())
        build_unet(args.tiny, arch=args.arch).load_state_dict(sd)  # keys must load into a plain UNet
        atomic_save(sd, out / f"ema_{step:07d}.pt")

    cfg_dp["achieved_eps"] = eps_now()
    write_json(cfg, out / "config.json")

    log = out / "log.csv"
    new_log = not log.exists()
    f = open(log, "a", newline="")
    w = csv.writer(f)
    if new_log:
        w.writerow(["step", "loss", "examples_per_s", "elapsed_s", "eps"])

    t_start = t_mark = time.time()
    ex_mark = 0
    ex_total, loss_sum, ex_since_log = 0, torch.zeros((), device=device), 0
    deadline = t_start + args.max_minutes * 60 if args.max_minutes else None
    model.train()

    def end_of_step():
        """Bookkeeping after one logical optimizer step. Returns True to stop early."""
        nonlocal step, t_mark, ex_mark, loss_sum, ex_since_log
        ema_update(ema, under, 0.999)  # once per logical step, not per micro-batch
        step += 1
        if step % args.log_every == 0 or step == steps:
            now = time.time()
            eps = eps_now()
            eps_s = "" if eps is None else f"{eps:.4f}"
            lm = (loss_sum / max(ex_since_log, 1)).item()
            w.writerow([step, f"{lm:.6f}", f"{(ex_total - ex_mark) / (now - t_mark):.1f}", f"{now - t_start:.1f}", eps_s])
            f.flush()
            print(f"step {step}/{steps} loss {lm:.4f} {(ex_total - ex_mark) / (now - t_mark):.1f} ex/s eps {eps_s}", flush=True)
            loss_sum.zero_(); ex_since_log = 0; t_mark = now; ex_mark = ex_total
        stop = deadline is not None and time.time() > deadline
        if step in ckpt_steps:
            save_ema()
            if dp:
                cfg_dp["achieved_eps"] = eps_now()
                write_json(cfg, out / "config.json")
        if (step in ckpt_steps or stop) and step < steps:
            save_resume()
        return stop and step < steps

    stopped = False
    if dp:
        while step < steps and not stopped:
            with BatchMemoryManager(data_loader=loader, max_physical_batch_size=args.physical_bs, optimizer=opt) as mm:
                for (xb,) in mm:  # Poisson sampled, variable size
                    xb = xb.to(device)
                    loss = per_example_loss(model, xb, sched, g, amp)
                    loss.mean().backward()
                    opt.step()
                    opt.zero_grad()
                    ex_total += len(xb); ex_since_log += len(xb); loss_sum += loss.detach().sum()
                    if not opt._is_last_step_skipped:  # last micro-batch of the logical batch
                        stopped = end_of_step()
                        if step >= steps or stopped:
                            break
    else:
        pb = args.physical_bs
        while step < steps and not stopped:
            b = torch.randint(0, args.n, (args.batch,), device=device, generator=g)  # with replacement, as train.py
            opt.zero_grad(set_to_none=True)
            for i in range(0, args.batch, pb):
                xb = x_dev[b[i:i + pb]]
                loss = per_example_loss(model, xb, sched, g, amp)
                (loss.sum() / args.batch).backward()  # mean over the logical batch
                loss_sum += loss.detach().sum()
            opt.step()
            ex_total += args.batch; ex_since_log += args.batch
            stopped = end_of_step()
    f.close()
    if stopped:
        print(f"--max-minutes reached at step {step}; rerun the same command to resume", flush=True)
    elif step >= steps and rp.exists():
        rp.unlink()


if __name__ == "__main__":
    main()
