"""E5 feasibility gate: per-sample-gradient check and DP throughput on random tensors (see PROTOCOL_V2.md).

Decides run_e5 / arch / epochs / physical_bs. Projections are estimates from a short timing run.
"""
import argparse
import copy
import gc
import json
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from common import build_unet, get_device, make_ddpm, resolve, set_seed, write_json, T_TRAIN
from train_dp import AMP, GRAD_SAMPLE_MODE, warnings  # noqa: F401  (importing train_dp silences Opacus warnings)

EPOCHS = (50, 100, 200, 400)


def mem_clear():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except RuntimeError:  # GPU full (shared machine); the caller treats that as OOM
            pass


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------- 1. correctness of per-sample gradients ----------

def _loss(model, xt, t, noise, amp):
    with torch.autocast(xt.device.type, dtype=torch.bfloat16, enabled=amp and xt.device.type == "cuda"):
        pred = model(xt, t).sample
    return ((pred.float() - noise) ** 2).flatten(1).mean(1)


def _flat(gs):
    return torch.cat([g.flatten() for g in gs])


def correctness(arch, tiny, mode, device, bs=4):
    """Opacus grad_sample vs bs separate manual backward passes, fp32 (TF32 off) and bf16 autocast."""
    from opacus.grad_sample import wrap_model
    tf32 = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False  # else conv noise ~1e-3 alone
    try:
        set_seed(0)
        base = build_unet(tiny, arch=arch).to(device)
        names = [n for n, _ in base.named_parameters()]
        x = torch.randn(bs, 1, 64, 64, device=device).clamp(-1, 1)
        t = torch.randint(0, T_TRAIN, (bs,), device=device)
        noise = torch.randn_like(x)
        xt = make_ddpm().add_noise(x, noise, t)

        def manual(amp):  # list over examples of list over params
            m = copy.deepcopy(base)
            out = []
            for i in range(bs):
                m.zero_grad()
                _loss(m, xt[i:i + 1], t[i:i + 1], noise[i:i + 1], amp).mean().backward()
                out.append([p.grad.detach().clone() for p in m.parameters()])
            return out

        def opacus(amp):
            m = wrap_model(copy.deepcopy(base), grad_sample_mode=mode, batch_first=True, loss_reduction="mean")
            _loss(m, xt, t, noise, amp).mean().backward()
            gs = {n.removeprefix("_module."): p.grad_sample.detach() for n, p in m.named_parameters()}
            return [[gs[n][i] for n in names] for i in range(bs)]

        def compare(a, b):
            """(max per-parameter relative error, global relative error) over examples.

            Parameters whose reference gradient is below 1e-6 of the example's total norm are skipped
            (attention to_k.bias has a mathematically zero gradient, so a relative error is noise).
            """
            per_param, glob = 0.0, 0.0
            for ga, gb in zip(a, b):  # per example
                tot = _flat(gb).norm()
                glob = max(glob, float((_flat(ga) - _flat(gb)).norm() / tot))
                for pa, pb in zip(ga, gb):
                    if pb.norm() > 1e-6 * tot:
                        per_param = max(per_param, float((pa - pb).norm() / pb.norm()))
            return per_param, glob

        ref32, ref_amp = manual(False), manual(True)
        o32, oamp = opacus(False), opacus(True)
        r = {"arch": arch, "grad_sample_mode": mode, "batch": bs}
        r["fp32_max_rel_err_per_param"], r["fp32_global_rel_err"] = compare(o32, ref32)
        r["amp_vs_amp_manual_max_rel_err_per_param"], r["amp_vs_amp_manual_global_rel_err"] = compare(oamp, ref_amp)
        r["amp_vs_fp32_manual_global_rel_err"] = compare(oamp, ref32)[1]
        r["zero_grad_sample_params"] = sum(1 for ga, gb in zip(o32[0], ref32[0]) if ga.norm() == 0 and gb.norm() > 0)
        r["fp32_pass"] = r["fp32_max_rel_err_per_param"] < 1e-3
        r["train_dp_amp_setting"] = AMP
        r["amp_ok"] = r["amp_vs_fp32_manual_global_rel_err"] < 5e-2  # bf16 rounding alone is ~1e-2
        return r
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = tf32


# ---------- 2. throughput ----------

def time_dp(arch, tiny, pbs, n, batch, logical, amp, device):
    """Examples/s of the real DP loop (Poisson batches, clipping, noise, AdamW) after one warmup logical step."""
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    set_seed(0)
    model = build_unet(tiny, arch=arch).to(device)
    loader = DataLoader(TensorDataset(torch.randn(n, 1, 64, 64).clamp(-1, 1)), batch_size=batch)
    model, opt, loader = PrivacyEngine(accountant="rdp").make_private(
        module=model, optimizer=torch.optim.AdamW(model.parameters(), lr=3e-4), data_loader=loader,
        noise_multiplier=1.0, max_grad_norm=1.0, poisson_sampling=True, grad_sample_mode=GRAD_SAMPLE_MODE)
    sched, g = make_ddpm(), torch.Generator(device=device).manual_seed(0)
    done, ex, t0 = 0, 0, None
    while done < logical + 1:
        with BatchMemoryManager(data_loader=loader, max_physical_batch_size=pbs, optimizer=opt) as mm:
            for (xb,) in mm:
                xb = xb.to(device)
                t = torch.randint(0, T_TRAIN, (len(xb),), device=device, generator=g)
                noise = torch.randn(xb.shape, device=device, generator=g)
                _loss(model, sched.add_noise(xb, noise, t), t, noise, amp).mean().backward()
                opt.step(); opt.zero_grad()
                if t0 is not None:
                    ex += len(xb)
                if not opt._is_last_step_skipped:
                    done += 1
                    if done == 1:  # warmup over
                        sync(); t0 = time.time()
                    if done == logical + 1:
                        break
    sync()
    return ex / (time.time() - t0), torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0


def time_twin(arch, tiny, pbs, batch, logical, amp, device):
    """Examples/s of one non-DP logical step (batch examples in micro-batches of pbs, AdamW, no clipping)."""
    set_seed(0)
    model = build_unet(tiny, arch=arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    sched, g = make_ddpm(), torch.Generator(device=device).manual_seed(0)
    x = torch.randn(pbs, 1, 64, 64, device=device).clamp(-1, 1)

    def step():
        opt.zero_grad(set_to_none=True)
        for _ in range(0, batch, pbs):
            t = torch.randint(0, T_TRAIN, (pbs,), device=device, generator=g)
            noise = torch.randn(x.shape, device=device, generator=g)
            (_loss(model, sched.add_noise(x, noise, t), t, noise, amp).sum() / batch).backward()
        opt.step()

    step(); sync(); t0 = time.time()
    for _ in range(logical):
        step()
    sync()
    return logical * batch / (time.time() - t0)


def measure(arch, args, device, amp):
    """Largest physical batch that does not run out of memory, then DP and twin throughput at it."""
    for pbs in args.pbs_candidates:
        mem_clear()
        try:
            dp_ex_s, peak = time_dp(arch, args.quick, pbs, args.n, args.batch, args.logical_batches, amp, device)
            mem_clear()
            twin_ex_s = time_twin(arch, args.quick, pbs, args.batch, max(1, args.logical_batches // 4), amp, device)
            return {"physical_bs": pbs, "dp_examples_per_s": dp_ex_s, "twin_examples_per_s": twin_ex_s, "peak_gb": peak}
        except (torch.OutOfMemoryError, torch.AcceleratorError) as ex:
            if "out of memory" not in str(ex).lower():
                raise
            print(f"  {arch}: physical batch {pbs} out of memory", flush=True)
        finally:
            mem_clear()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hours", type=float, default=2.5, help="budget per DP run")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--logical-batches", type=int, default=20, help="timed DP logical steps per measurement")
    ap.add_argument("--quick", action="store_true", help="self-test: tiny UNet, tiny sizes (numbers meaningless)")
    args = ap.parse_args()
    args.pbs_candidates = (32, 64, 128, 256)  # tried largest first
    if args.quick:
        args.n, args.batch, args.logical_batches, args.pbs_candidates = 64, 16, 1, (8, 4)
    args.pbs_candidates = sorted(args.pbs_candidates, reverse=True)
    device = get_device()
    amp = AMP == "bf16"

    res = {"note": "projections are estimates from a short timing run, not measurements of a full run",
           "quick": args.quick, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
           "n": args.n, "batch": args.batch, "hours_budget": args.hours, "grad_sample_mode": GRAD_SAMPLE_MODE,
           "amp": AMP, "logical_batches_timed": args.logical_batches, "correctness": [], "throughput": {}}
    for arch in ("base", "small"):
        c = correctness(arch, args.quick, GRAD_SAMPLE_MODE, device)
        res["correctness"].append(c)
        print(f"correctness {arch}: fp32 max per-param rel err {c['fp32_max_rel_err_per_param']:.2e} "
              f"(global {c['fp32_global_rel_err']:.2e}, pass={c['fp32_pass']}); amp {AMP}: vs amp-manual "
              f"{c['amp_vs_amp_manual_global_rel_err']:.2e}, vs fp32-manual {c['amp_vs_fp32_manual_global_rel_err']:.2e}", flush=True)
    for arch in ("base", "small"):
        res["throughput"][arch] = measure(arch, args, device, amp)
        print(f"throughput {arch}: {res['throughput'][arch]}", flush=True)

    table, choice = {}, None
    for arch in ("base", "small"):
        th = res["throughput"][arch]
        table[arch] = {str(E): (E * args.n / th["dp_examples_per_s"] / 3600 if th else None) for E in EPOCHS}
    for arch in ("base", "small"):
        fits = [E for E in EPOCHS if table[arch][str(E)] is not None and table[arch][str(E)] <= args.hours]
        if fits:  # base: largest E that fits; small only if E=50 fits (then also the largest E that fits)
            choice = (arch, max(fits)); break
    res["projection_table"] = table
    res["run_e5"] = choice is not None
    res["arch"], res["epochs"] = choice if choice else (None, None)
    res["physical_bs"] = res["throughput"][res["arch"]]["physical_bs"] if choice else None
    res["projected_hours_per_run"] = table[choice[0]][str(choice[1])] if choice else None  # read by bench.py
    resolve(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(res, resolve(args.out))

    print(f"\nProjected hours per DP run (n={args.n}, batch={args.batch}, budget {args.hours} h; ESTIMATES)")
    print("epochs " + "".join(f"{E:>10}" for E in EPOCHS))
    for arch in ("base", "small"):
        print(f"{arch:<7}" + "".join(f"{'n/a':>10}" if table[arch][str(E)] is None else f"{table[arch][str(E)]:>10.2f}" for E in EPOCHS))
    print(f"decision: run_e5={res['run_e5']} arch={res['arch']} epochs={res['epochs']} physical_bs={res['physical_bs']}")


if __name__ == "__main__":
    main()
