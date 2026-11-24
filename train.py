"""Train one pixel-space DDPM on a nested member subset (see PROTOCOL.md; --cond / --seed: PROTOCOL_V2.md)."""
import argparse
import copy
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (NUM_CLASSES, T_TRAIN, atomic_save, build_unet, ema_update, get_device, load_data, load_labels,
                    make_ddpm, member_indices, predict_eps, resolve, set_seed, to_tensor, write_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True, help="training set size (256, 1024 or 4096)")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop early (smoke test); resumable")
    ap.add_argument("--resume-every", type=int, default=1000, help="steps between resume.pt saves")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="training seed (init, batch order, t, noise); members never change")
    ap.add_argument("--cond", action="store_true", help="class-conditional, label dropout p=0.1 (E3)")
    ap.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16", help="training precision (fp16 uses a loss scaler; PROTOCOL_V2.md deviation 6)")
    ap.add_argument("--tiny", action="store_true", help="self-test: fake data, tiny UNet")
    args = ap.parse_args()

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    cuda = device.type == "cuda"
    ckpt_steps = [round(args.steps * k / 8) for k in range(1, 9)]
    final = out / f"ema_{args.steps:07d}.pt"
    if final.exists():
        print(f"{out} already finished ({final.name}); nothing to do")
        return

    train_u8, _ = load_data(args.tiny)
    idx = member_indices(len(train_u8), args.n)
    np.save(out / "members.npy", idx)
    cfg = {"n": args.n, "steps": args.steps, "tiny": args.tiny, "seed": args.seed, "cond": args.cond, "amp": args.amp,
           "arch": "base", "ckpt_steps": ckpt_steps}
    if args.cond:
        cfg["p_uncond"] = 0.1
    write_json(cfg, out / "config.json")
    x = to_tensor(train_u8[idx], device)
    if args.cond:
        y_all = torch.from_numpy(load_labels(args.tiny)[0][idx]).to(device)

    set_seed(args.seed)
    model = build_unet(args.tiny, NUM_CLASSES if args.cond else 0).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = make_ddpm()
    g = torch.Generator(device=device)  # one stream for batch indices, timesteps and noise
    g.manual_seed(args.seed)
    use_amp = cuda and args.amp != "fp32"  # on CPU autocast stays off
    amp_dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if cuda and args.amp == "fp16" else None

    step = 0
    rp = out / "resume.pt"
    if rp.exists():
        st = torch.load(rp, map_location=device, weights_only=False)
        assert (st["n"], st["steps"], st["tiny"], st.get("seed", 0), st.get("cond", False), st.get("amp", "bf16")) ==             (args.n, args.steps, args.tiny, args.seed, args.cond, args.amp), "resume.pt is from a different run"
        model.load_state_dict(st["model"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        g.set_state(st["g"].cpu())
        if scaler is not None and "scaler" in st:
            scaler.load_state_dict(st["scaler"])
        step = st["step"]
        print(f"resumed from step {step}", flush=True)

    def save_resume():
        atomic_save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                     "g": g.get_state(), "step": step, "n": args.n, "steps": args.steps,
                     "tiny": args.tiny, "seed": args.seed, "cond": args.cond, "amp": args.amp,
                     **({"scaler": scaler.state_dict()} if scaler is not None else {})}, rp)

    log = out / "log.csv"
    new_log = not log.exists()
    f = open(log, "a", newline="")
    w = csv.writer(f)
    if new_log:
        w.writerow(["step", "loss", "it_per_s", "elapsed_s"])

    t_start = t_mark = time.time()
    step_mark = step
    acc, nacc = torch.zeros((), device=device), 0
    deadline = t_start + args.max_minutes * 60 if args.max_minutes else None
    model.train()
    while step < args.steps:
        b = torch.randint(0, args.n, (64,), device=device, generator=g)
        t = torch.randint(0, T_TRAIN, (64,), device=device, generator=g)
        noise = torch.randn(x[b].shape, device=device, generator=g)
        xt = sched.add_noise(x[b], noise, t)
        y = None
        if args.cond:  # drawn last and only here, so the unconditional RNG stream is unchanged
            drop = torch.rand(64, device=device, generator=g) < 0.1
            y = y_all[b].masked_fill(drop, NUM_CLASSES)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            pred = predict_eps(model, xt, t, y)
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        else:  # fp16: scale, unscale before clipping; the step is skipped on inf/nan grads
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        ema_update(ema, model, 0.999)
        step += 1
        acc += loss.detach()
        nacc += 1

        if step % args.log_every == 0 or step == args.steps:
            now = time.time()
            ips = (step - step_mark) / (now - t_mark)
            w.writerow([step, f"{(acc / nacc).item():.6f}", f"{ips:.2f}", f"{now - t_start:.1f}"])
            f.flush()
            print(f"step {step}/{args.steps} loss {(acc / nacc).item():.4f} {ips:.2f} it/s", flush=True)
            acc.zero_(); nacc = 0; t_mark = now; step_mark = step

        stop = deadline is not None and time.time() > deadline
        if step in ckpt_steps:
            atomic_save(ema.state_dict(), out / f"ema_{step:07d}.pt")
        if step % args.resume_every == 0 or step in ckpt_steps or stop:
            if step < args.steps:
                save_resume()
        if stop and step < args.steps:
            print(f"--max-minutes reached at step {step}; rerun the same command to resume", flush=True)
            break
    f.close()
    if step >= args.steps and rp.exists():
        rp.unlink()


if __name__ == "__main__":
    main()
