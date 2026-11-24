"""Speed benchmark for the college PC (about 10 minutes; see COLLEGE_README.md).

Measures training and sampling speed, concurrency, optionally runs dp_gate.py, then prints a
projection of the time each experiment block would take. Every projected number is an ESTIMATE.
Writes runs/bench.json (runs/bench_quick.json with --quick).
"""
import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_H = Path(__file__).resolve().parent
for _k, _v in (("TORCH_HOME", "torch"), ("HF_HOME", "hf"), ("TMP", "tmp"), ("TEMP", "tmp")):
    os.environ.setdefault(_k, str(_H / ".cache" / _v))  # never download to the C: drive
(_H / ".cache" / "tmp").mkdir(parents=True, exist_ok=True)

import torch
import torch.nn.functional as F

from common import (HERE, NUM_CLASSES, T_TRAIN, build_unet, ema_update, get_device, make_ddim, make_ddpm,
                    predict_eps, resolve, write_json)

STEPS = 30000          # training steps per run (PROTOCOL_V2.md)
N_SAMPLES, DDIM_STEPS = 1000, 100
MIA_FWD = 6 * 4        # loss MIA: 6 timesteps x 4 noise draws per image (evaluate.py)
# Assumptions used only for the projection; all are guesses, not measurements.
ASSUME = {
    "attack_forward_passes_per_image": 30,   # PIA + SecMI-stat together, deliberately high
    "cpu_seconds_per_eval": 30,              # bootstrap CIs, nearest-neighbour search, file writes
    "fid_seconds_per_image_default": 0.02,   # used only if FID speed could not be measured
    "eval_images_per_fid": N_SAMPLES + 1140, # fake samples + non-members
    "val_pool": 520, "test_pool": 620,       # approx. non-member pool sizes for the secondary MIA
}


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gb(x):
    return round(x / 1e9, 2)


# ---------- measurements ----------

def train_speed(cond, steps, warmup, bs, hook=None):
    """The train.py inner step (bf16 autocast, AdamW 1e-4, clip 1.0, EMA 0.999) on random data."""
    device = get_device()
    cuda = device.type == "cuda"
    torch.manual_seed(0)
    model = build_unet(num_classes=NUM_CLASSES if cond else 0).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = make_ddpm()
    x = torch.rand(1024, 1, 64, 64, device=device) * 2 - 1
    y_all = torch.randint(0, NUM_CLASSES, (1024,), device=device)
    if cuda:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model.train()

    def one_step():
        b = torch.randint(0, len(x), (bs,), device=device)
        t = torch.randint(0, T_TRAIN, (bs,), device=device)
        noise = torch.randn(x[b].shape, device=device)
        xt = sched.add_noise(x[b], noise, t)
        y = None
        if cond:  # label dropout 0.1 -> null label, as in the conditional training run
            y = torch.where(torch.rand(bs, device=device) < 0.1, torch.full_like(b, NUM_CLASSES), y_all[b])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
            pred = predict_eps(model, xt, t, y)
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema_update(ema, model, 0.999)

    for _ in range(warmup):
        one_step()
    sync()
    if hook:
        hook()
    t0 = time.time()
    for _ in range(steps):
        one_step()
    sync()
    el = time.time() - t0
    r = {"it_per_s": steps / el, "steps": steps, "elapsed_s": el, "batch": bs}
    if cuda:
        r["peak_allocated_gb"] = gb(torch.cuda.max_memory_allocated())
        r["peak_reserved_gb"] = gb(torch.cuda.max_memory_reserved())
    return r


@torch.no_grad()
def sample_speed(cfg, bs, steps, warmup):
    """fp32 DDIM step throughput in images*steps/s. cfg=True: conditional UNet, doubled batch."""
    device = get_device()
    unet = build_unet(num_classes=NUM_CLASSES if cfg else 0).to(device).eval()
    sched = make_ddim()
    sched.set_timesteps(DDIM_STEPS)
    x = torch.randn(bs, 1, 64, 64, device=device)
    y2 = torch.cat([torch.zeros(bs), torch.full((bs,), NUM_CLASSES)]).long().to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    def one_step(t):
        nonlocal x
        if cfg:
            ec, eu = predict_eps(unet, torch.cat([x, x]), t, y2).chunk(2)
            eps = eu + 3.0 * (ec - eu)
        else:
            eps = predict_eps(unet, x, t)
        x = sched.step(eps, t, x, eta=0.0).prev_sample

    ts = sched.timesteps
    for i in range(warmup):
        one_step(ts[i])
    sync()
    t0 = time.time()
    for i in range(steps):
        one_step(ts[warmup + i])
    sync()
    el = time.time() - t0
    r = {"img_steps_per_s": bs * steps / el, "batch": bs, "steps": steps}
    if device.type == "cuda":
        r["peak_allocated_gb"] = gb(torch.cuda.max_memory_allocated())
    return r


@torch.no_grad()
def fid_speed(n):
    """Seconds per image through the FID Inception net, or None if it cannot run (weights missing)."""
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        device = get_device()
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        imgs = torch.rand(n, 3, 64, 64, device=device)
        fid.update(imgs[:20], real=True)
        sync()
        t0 = time.time()
        for i in range(0, n, 100):
            fid.update(imgs[i:i + 100], real=False)
        sync()
        return (time.time() - t0) / n
    except Exception as e:  # noqa: BLE001
        print(f"FID speed not measured: {type(e).__name__}: {e}", flush=True)
        return None


def run_concurrent(k, steps, warmup, bs):
    """k simultaneous train-speed subprocesses, started together after all have warmed up."""
    with tempfile.TemporaryDirectory(dir=os.environ["TMP"]) as td:
        td = Path(td)
        procs = [subprocess.Popen([sys.executable, "-u", str(HERE / "bench.py"), "--worker", "--idx", str(i),
                                   "--barrier", str(td), "--steps", str(steps), "--warmup", str(warmup),
                                   "--batch", str(bs)], cwd=HERE, stdout=subprocess.DEVNULL) for i in range(k)]
        t0 = time.time()
        while len(list(td.glob("ready_*"))) < k:
            if any(p.poll() is not None for p in procs) or time.time() - t0 > 900:
                for p in procs:
                    p.kill()
                return {"error": "a worker exited before it was ready (out of memory?)"}
            time.sleep(0.1)
        (td / "go").write_text("1")
        for p in procs:
            p.wait()
        res = [json.loads(p.read_text()) for p in sorted(td.glob("r*.json"))]
        if len(res) != k or any(p.returncode for p in procs):
            return {"error": "a worker failed during timing"}
    el = max(r["elapsed_s"] for r in res)
    return {"aggregate_it_per_s": sum(r["steps"] for r in res) / el,
            "per_process_it_per_s": [round(r["it_per_s"], 3) for r in res],
            "peak_allocated_gb_per_process": [r.get("peak_allocated_gb") for r in res],
            "peak_reserved_gb_per_process": [r.get("peak_reserved_gb") for r in res]}


def worker(a):
    bdir = Path(a.barrier)

    def hook():
        (bdir / f"ready_{a.idx}").write_text("1")
        t0 = time.time()
        while not (bdir / "go").exists():
            if time.time() - t0 > 900:
                raise SystemExit("barrier timeout")
            time.sleep(0.05)

    r = train_speed(False, a.steps, a.warmup, a.batch, hook)
    (bdir / f"r{a.idx}.json").write_text(json.dumps(r))


def run_dp_gate(force):
    gate, out = HERE / "dp_gate.py", resolve("runs/dp_gate.json")
    if not gate.exists():
        return {"note": "dp_gate.py is missing, so E5 cannot be projected"}
    if out.exists() and not force:
        d = json.loads(out.read_text())
        d["note"] = "reused existing runs/dp_gate.json (use --force-dp-gate to rerun)"
        return d
    t0 = time.time()
    r = subprocess.run([sys.executable, "-u", str(gate), "--out", "runs/dp_gate.json"], cwd=HERE)
    if r.returncode or not out.exists():
        return {"note": f"dp_gate.py failed (exit {r.returncode})"}
    d = json.loads(out.read_text())
    d["gate_seconds"] = round(time.time() - t0, 1)
    return d


# ---------- projection (all numbers are estimates) ----------

def makespan(runs, steps, agg, k):
    """Seconds to train `runs` equal jobs with at most k at once; agg[m] = aggregate it/s with m running."""
    left, total = runs, 0.0
    while left:
        m = min(k, left)
        total += steps / (agg[m] / m)
        left -= m
    return total


def project(b):
    r1 = b["sampling"]["uncond"]["img_steps_per_s"]
    r2 = b["sampling"]["cfg"]["img_steps_per_s"]
    fid_s = b.get("fid_seconds_per_image") or ASSUME["fid_seconds_per_image_default"]
    cpu = ASSUME["cpu_seconds_per_eval"]
    it_u, it_c = b["train"]["uncond"]["it_per_s"], b["train"]["cond"]["it_per_s"]
    total_gb = b["env"].get("gpu_total_gb") or 1e9
    peak = b["train"]["uncond"].get("peak_reserved_gb") or 0
    agg = {1: it_u}
    for k, v in b["concurrency"].items():
        if "aggregate_it_per_s" in v:
            agg[int(k)] = v["aggregate_it_per_s"]
    ks = [k for k in sorted(agg) if all(m in agg for m in range(1, k + 1)) and (k == 1 or k * peak <= 0.92 * total_gb)]

    def eval_std(n):  # evaluate.py --ckpt last
        m = min(n, 1000)
        fwd = MIA_FWD * (2 * m + min(n, 1000, ASSUME["val_pool"]) + min(n, 1000, ASSUME["test_pool"]))
        return N_SAMPLES * DDIM_STEPS / r1 + fwd / r1 + ASSUME["eval_images_per_fid"] * fid_s + cpu

    def eval_cfg(n):  # sweep over 6 w (2 single-batch, 4 doubled), 6 FIDs, MIA with true and null label
        m = min(n, 1000)
        return (N_SAMPLES * DDIM_STEPS * (2 / r1 + 4 / r2) + 6 * ASSUME["eval_images_per_fid"] * fid_s
                + 2 * MIA_FWD * 2 * m / r1 + 2 * cpu)

    def attacks(n, labels=1):
        return labels * (ASSUME["attack_forward_passes_per_image"] * 2 * min(n, 1000) / r1 + cpu)

    def block(train_runs, train_it, other_s):
        scaled = {m: train_it * agg[m] / agg[1] for m in agg}  # cond runs: reuse the measured concurrency gain
        best = min((makespan(train_runs, STEPS, scaled, k), k) for k in ks)
        return {"sequential_h": (train_runs * STEPS / train_it + other_s) / 3600,
                "concurrent_h": (best[0] + other_s) / 3600, "best_k": best[1]}

    out = {}
    out["E3"] = block(2, it_c, sum(eval_cfg(n) + attacks(n, 2) for n in (1024, 4096)))
    out["E4b"] = block(4, it_u, 2 * sum(eval_std(n) + attacks(n) for n in (1024, 4096)))
    pilot = sum(attacks(n) for n in (256, 1024, 4096))
    out["E4a"] = {"sequential_h": pilot / 3600, "concurrent_h": pilot / 3600, "best_k": 1}
    dp = b.get("dp_gate", {})
    hrs = next((dp[k] for k in ("projected_hours_per_run", "projected_hours", "hours_per_run", "dp_run_hours",
                                "projected_run_hours") if isinstance(dp.get(k), (int, float))), None)
    if dp.get("run_e5") is False:
        out["E5"] = {"skipped": "dp_gate says run_e5 = false"}
    elif hrs is None:
        out["E5"] = {"skipped": "no dp_gate result with a projected hours field (" + dp.get("note", "see runs/dp_gate.json") + ")"}
    else:
        rest = 3 * (eval_std(4096) + attacks(4096))
        t = (3 * hrs * 3600 + rest) / 3600  # twin assumed as slow as a DP run: an upper bound
        out["E5"] = {"sequential_h": t, "concurrent_h": t, "best_k": 1}
    out["_attack_seconds_per_model"] = {str(n): attacks(n) for n in (256, 1024, 4096)}
    out["_attacks_all_models_h"] = (sum(attacks(n) for n in (256, 1024, 4096)) + attacks(1024, 2) + attacks(4096, 2)
                                    + 2 * (attacks(1024) + attacks(4096)) + 3 * attacks(4096)) / 3600
    return out


def show(b, proj):
    p = print
    e = b["env"]
    p("\n==== BENCHMARK RESULT (copy everything from here to the end line and send it to Claude) ====")
    if b["quick"]:
        p("QUICK MODE: tiny batches, numbers below are NOT representative")
    p(f"GPU: {e.get('gpu')} (compute capability {e.get('capability')}, {e.get('gpu_total_gb')} GB), torch {e['torch']}")
    for k in ("uncond", "cond"):
        t = b["train"][k]
        p(f"train {k:6s} batch {t['batch']}: {t['it_per_s']:.2f} it/s, peak GPU memory {t.get('peak_reserved_gb')} GB reserved")
    for k in ("uncond", "cfg"):
        s = b["sampling"][k]
        p(f"DDIM fp32 {k:6s} batch {s['batch']}: {s['img_steps_per_s']:.0f} image-steps/s")
    p(f"FID speed: {b['fid_seconds_per_image']} s/image" if b.get("fid_seconds_per_image") else "FID speed: not measured (default assumed)")
    for k, v in b["concurrency"].items():
        p(f"{k} runs at once: " + (v["error"] if "error" in v else
          f"aggregate {v['aggregate_it_per_s']:.2f} it/s, per process {v['per_process_it_per_s']}, "
          f"peak reserved GB per process {v['peak_reserved_gb_per_process']}"))
    dp = b.get("dp_gate", {})
    p("DP gate: " + json.dumps({k: v for k, v in dp.items()}) if dp else "DP gate: not run")
    p("\nPROJECTION (ESTIMATES, hours; evaluation and attack steps are added in full to both columns)")
    p(f"{'block':6s} {'sequential h':>14s} {'best concurrency h':>20s}  best K")
    ts = tc = 0.0
    for name in ("E4a", "E3", "E5", "E4b"):
        v = proj[name]
        if "skipped" in v:
            p(f"{name:6s} not projected: {v['skipped']}")
            continue
        ts += v["sequential_h"]; tc += v["concurrent_h"]
        p(f"{name:6s} {v['sequential_h']:14.1f} {v['concurrent_h']:20.1f}  {v['best_k']}")
    p(f"{'total':6s} {ts:14.1f} {tc:20.1f}")
    p("Attack cost estimate per model (seconds, " + f"{ASSUME['attack_forward_passes_per_image']} forward passes per image, conservative): "
      + ", ".join(f"n={n}: {s:.0f}" for n, s in proj["_attack_seconds_per_model"].items())
      + f"; all models in all blocks: {proj['_attacks_all_models_h']:.2f} h")
    p("E5 note: concurrency not projected for DP runs; the non-DP twin is assumed as slow as a DP run.")
    p("==== END OF BENCHMARK RESULT ====")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few steps, small batches; proves the script runs")
    ap.add_argument("--no-dp-gate", action="store_true", help="do not run dp_gate.py (it takes about 30 minutes)")
    ap.add_argument("--force-dp-gate", action="store_true", help="rerun dp_gate.py even if runs/dp_gate.json exists")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    for n in ("idx", "steps", "warmup", "batch"):
        ap.add_argument("--" + n, type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--barrier", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    resolve("runs").mkdir(exist_ok=True)
    q = a.quick
    bs, steps, warm = (8, 10, 3) if q else (64, 200, 30)
    cs, cw = (8, 2) if q else (150, 20)
    sbs, ss, sw = (16, 3, 1) if q else (250, 10, 2)
    dev = get_device()
    env = {"torch": torch.__version__, "device": str(dev)}
    if dev.type == "cuda":
        pr = torch.cuda.get_device_properties(0)
        env.update(gpu=pr.name, capability=f"{pr.major}.{pr.minor}", gpu_total_gb=gb(pr.total_memory))
    b = {"quick": q, "env": env, "train": {}, "sampling": {}, "concurrency": {}}
    print("train speed ...", flush=True)
    for k, cond in (("uncond", False), ("cond", True)):
        b["train"][k] = train_speed(cond, steps, warm, bs)
        print(f"  {k}: {b['train'][k]['it_per_s']:.2f} it/s", flush=True)
    print("sampling speed ...", flush=True)
    b["sampling"]["uncond"] = sample_speed(False, sbs, ss, sw)
    b["sampling"]["cfg"] = sample_speed(True, sbs, ss, sw)
    b["fid_seconds_per_image"] = None if q else fid_speed(300)
    for k in (2, 3):
        print(f"{k} concurrent runs ...", flush=True)
        b["concurrency"][str(k)] = run_concurrent(k, cs, cw, bs)
    if q or a.no_dp_gate:
        b["dp_gate"] = {"note": "not run (--quick or --no-dp-gate)"}
    else:
        print("dp_gate.py ...", flush=True)
        b["dp_gate"] = run_dp_gate(a.force_dp_gate)
    b["assumptions"] = ASSUME
    b["projection"] = project(b)
    write_json(b, resolve("runs/bench_quick.json" if q else "runs/bench.json"))
    show(b, b["projection"])


if __name__ == "__main__":
    main()
