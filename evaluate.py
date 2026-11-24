"""Evaluate EMA checkpoints of one run: copy detection, loss-based MIA, FID (see PROTOCOL.md)."""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import auc as sk_auc, roc_curve

from common import (get_device, load_data_with_origin, make_ddim, make_ddpm, unet_from_config, resolve, to_tensor,
                    write_json)

MIA_TS = (25, 50, 100, 150, 200, 300)
MIA_DRAWS = 4


# ---------- copy detection ----------

def unit_rows(x):
    """Flatten, zero-mean, unit-norm each image (float64), so a dot product is Pearson r."""
    v = x.flatten(1).double()
    v = v - v.mean(1, keepdim=True)
    return v / v.norm(dim=1, keepdim=True).clamp_min(1e-12)


def nn_sim(q, ref, chunk=256):
    """Max Pearson similarity of each query image to the reference set, plus argmax index."""
    qn, rn = unit_rows(q), unit_rows(ref)
    sims, args = [], []
    for i in range(0, len(qn), chunk):
        m, a = (qn[i:i + chunk] @ rn.T).max(1)
        sims.append(m.cpu()); args.append(a.cpu())
    return torch.cat(sims).numpy(), torch.cat(args).numpy()


def ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------- membership inference ----------

def roc_metrics(y, scores):
    """ROC AUC and TPR at 1% FPR (linear interpolation on the ROC curve)."""
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    return float(sk_auc(fpr, tpr)), float(np.interp(0.01, fpr, tpr))


@torch.no_grad()
def mia_scores(unet, x, noise, sched, device, bs=125):
    """score = -mean over (t, draw) of ||eps - eps_theta(x_t, t)||^2; same noise for every image."""
    total = torch.zeros(len(x), device=device)
    for ti, t in enumerate(MIA_TS):
        for k in range(MIA_DRAWS):
            for i in range(0, len(x), bs):
                xb = x[i:i + bs]
                e = noise[ti, k][None].expand_as(xb)
                tt = torch.full((len(xb),), t, device=device, dtype=torch.long)
                pred = unet(sched.add_noise(xb, e, tt), tt).sample
                total[i:i + bs] += ((pred - e) ** 2).flatten(1).sum(1)
    return (-total / (len(MIA_TS) * MIA_DRAWS)).cpu().numpy()


# ---------- sampling and FID ----------

def mem_reset():
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def mem_peak():
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


@torch.no_grad()
def sample(unet, n, seed, steps, bs, device, amp=False):
    sched = make_ddim()
    sched.set_timesteps(steps)
    # initial noise drawn up front so samples do not depend on bs
    x_all = torch.randn(n, 1, 64, 64, generator=torch.Generator().manual_seed(seed))
    out = []
    for i in range(0, n, bs):
        x = x_all[i:i + bs].to(device)
        for t in sched.timesteps:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                eps = unet(x, t).sample
            x = sched.step(eps.float(), t, x, eta=0.0).prev_sample
        out.append(x.clamp(-1, 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def compute_fid(fake_pm1, real_pm1, device, bs=100):
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    prep = lambda x: ((x.to(device) + 1) / 2).clamp(0, 1).repeat(1, 3, 1, 1)
    for i in range(0, len(real_pm1), bs):
        fid.update(prep(real_pm1[i:i + bs]), real=True)
    for i in range(0, len(fake_pm1), bs):
        fid.update(prep(fake_pm1[i:i + bs]), real=False)
    return float(fid.compute())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", choices=["all", "last"], default="all")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=250, help="sampling batch size")
    ap.add_argument("--bf16-sampling", action="store_true", help="bf16 autocast for DDIM sampling only (about 4x faster)")
    ap.add_argument("--fp32-all", action="store_true", help="fp32 sampling at every checkpoint (default: bf16 at intermediate ones, fp32 at the last)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fid", action="store_true", help="self-test only")
    ap.add_argument("--force", action="store_true", help="recompute checkpoints already in metrics.json")
    args = ap.parse_args()

    run = resolve(args.run)
    cfg = json.loads((run / "config.json").read_text())
    device = get_device()
    train_u8, non_u8, is_val = load_data_with_origin(cfg["tiny"])
    mem_idx = np.load(run / "members.npy")
    members = to_tensor(train_u8[mem_idx], device)
    nonmem = to_tensor(non_u8, device)
    n = cfg["n"]
    if cfg.get("cond"):
        raise SystemExit("conditional runs are evaluated with eval_cfg.py")

    # fixed per-run quantities
    non_nn, _ = nn_sim(nonmem, members)
    threshold = float(np.percentile(non_nn, 99))
    m = min(n, 1000)
    rs = np.random.RandomState(0)
    sel_mem = rs.choice(n, m, replace=False) if n > m else np.arange(n)
    sel_non = rs.choice(len(nonmem), m, replace=False)
    mia_mem, mia_non = members[sel_mem], nonmem[sel_non]
    # secondary analysis: non-member pools by origin (val vs test split); primary metric is unaffected
    pools = {"val": np.flatnonzero(is_val), "test": np.flatnonzero(~is_val)}
    sec = {}
    for name, pool in pools.items():
        k = min(n, 1000, len(pool))
        sec[name] = (k, rs.choice(len(sel_mem), k, replace=False), rs.choice(pool, k, replace=False))
    noise = torch.randn(len(MIA_TS), MIA_DRAWS, 1, 64, 64, generator=torch.Generator().manual_seed(0)).to(device)
    ddpm = make_ddpm()

    unet = unet_from_config(cfg).to(device).eval()
    ckpts = sorted(run.glob("ema_*.pt"))
    found = {int(p.stem.split("_")[1]) for p in ckpts}
    if found != set(cfg["ckpt_steps"]):
        raise SystemExit(f"{run}: checkpoint steps {sorted(found)} != config ckpt_steps {cfg['ckpt_steps']}")
    last_step = int(ckpts[-1].stem.split("_")[1])
    if args.ckpt == "last":
        ckpts = ckpts[-1:]

    mpath = run / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else {"checkpoints": {}}
    metrics.update(n=n, steps=cfg["steps"], tiny=cfg["tiny"], copy_threshold=threshold,
                   nonmember_nn_median=float(np.median(non_nn)),
                   nonmember_nn_mean_val=float(non_nn[is_val].mean()), nonmember_nn_mean_test=float(non_nn[~is_val].mean()),
                   n_nonmembers=len(non_nn), n_nonmembers_val=int(is_val.sum()))

    for p in ckpts:
        step = int(p.stem.split("_")[1])
        brng = np.random.RandomState(step)  # per-checkpoint, so reruns reproduce exactly
        key = str(step)
        # intermediate checkpoints only feed the trend figure; the table's last checkpoint stays fp32
        bf16 = args.bf16_sampling or (step != last_step and not args.fp32_all)
        old = metrics["checkpoints"].get(key)
        same = old and old["n_samples"] == args.n_samples and old["ddim_steps"] == args.ddim_steps \
            and old["n_boot"] == args.n_boot and (old["fid"] is None) == args.no_fid             and "mia_auc_val_nm" in old and old.get("bf16_sampling") == bf16
        have_last_files = step != last_step or (run / "samples_last.npy").exists()
        if same and have_last_files and not args.force:
            print(f"step {step}: already evaluated, skipping", flush=True)
            continue
        t0 = time.time()
        peak = {}
        mem_reset()
        unet.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        samples = sample(unet, args.n_samples, args.seed, args.ddim_steps, args.bs, device, bf16)
        peak["sampling"] = mem_peak(); tm = {"sampling": time.time() - t0}
        mem_reset()

        # copy detection
        gen_nn, gen_arg = nn_sim(samples.to(device), members)
        hit = gen_nn > threshold
        hit95 = gen_nn > 0.95
        bi = brng.randint(0, len(gen_nn), (args.n_boot, len(gen_nn)))
        e = {"step": step, "n_samples": args.n_samples, "ddim_steps": args.ddim_steps,
             "n_boot": args.n_boot, "bf16_sampling": bf16, "copy_threshold": threshold,
             "copy_rate": float(hit.mean()), "median_nn": float(np.median(gen_nn)),
             "copy_rate_095": float(hit95.mean())}
        for name, vals in (("copy_rate", hit[bi].mean(1)), ("median_nn", np.median(gen_nn[bi], axis=1)),
                           ("copy_rate_095", hit95[bi].mean(1))):
            e[name + "_lo"], e[name + "_hi"] = ci(vals)

        # loss-based MIA
        mem_reset()
        s_mem = mia_scores(unet, mia_mem, noise, ddpm, device)
        s_non = mia_scores(unet, mia_non, noise, ddpm, device)
        y = np.r_[np.ones(m), np.zeros(m)]
        e["mia_n_per_group"] = m
        e["auc"], e["tpr1"] = roc_metrics(y, np.r_[s_mem, s_non])
        boots = np.array([roc_metrics(y, np.r_[brng.choice(s_mem, m), brng.choice(s_non, m)])
                          for _ in range(args.n_boot)])
        e["auc_lo"], e["auc_hi"] = ci(boots[:, 0])
        e["tpr1_lo"], e["tpr1_hi"] = ci(boots[:, 1])

        # secondary: members vs val-only / test-only non-members (extra fields only)
        brng2 = np.random.RandomState(step + 10 ** 6)
        for name, (k, msel, nsel) in sec.items():
            s_pool = mia_scores(unet, nonmem[nsel], noise, ddpm, device)
            yk = np.r_[np.ones(k), np.zeros(k)]
            a, t = roc_metrics(yk, np.r_[s_mem[msel], s_pool])
            bb = np.array([roc_metrics(yk, np.r_[brng2.choice(s_mem[msel], k), brng2.choice(s_pool, k)])
                           for _ in range(args.n_boot)])
            e[f"mia_n_{name}_nm"] = k
            e[f"mia_auc_{name}_nm"], e[f"mia_tpr1_{name}_nm"] = a, t
            e[f"mia_auc_{name}_nm_lo"], e[f"mia_auc_{name}_nm_hi"] = ci(bb[:, 0])
            e[f"mia_tpr1_{name}_nm_lo"], e[f"mia_tpr1_{name}_nm_hi"] = ci(bb[:, 1])
        peak["mia"] = mem_peak(); tm["mia_incl_copy_boot"] = time.time() - t0 - tm["sampling"]
        mem_reset()

        e["fid"] = None if args.no_fid else compute_fid(samples, nonmem, device)
        peak["fid"] = mem_peak()
        e["seconds"] = round(time.time() - t0, 1)
        metrics["checkpoints"][key] = e
        write_json(metrics, mpath)
        print(f"step {step}: copy {e['copy_rate']:.3f} median_nn {e['median_nn']:.3f} "
              f"auc {e['auc']:.3f} tpr1 {e['tpr1']:.3f} fid {e['fid']} ({e['seconds']}s)", flush=True)
        if device.type == "cuda":
            print("  peak GB: " + ", ".join(f"{k} {v:.2f}" for k, v in peak.items())
                  + "; seconds: " + ", ".join(f"{k} {v:.0f}" for k, v in tm.items()), flush=True)

        if step == last_step:
            np.save(run / "samples_last.npy", samples[:, 0].numpy())
            top = np.argsort(-gen_nn)[:8]
            tr = train_u8[mem_idx[gen_arg[top]]]
            write_json({"step": step, "sample_idx": top.tolist(), "train_idx": mem_idx[gen_arg[top]].tolist(),
                        "sims": gen_nn[top].tolist(), "threshold": threshold}, run / "figB.json")
            np.savez(run / "figB.npz", gen=((samples[top, 0].numpy() + 1) / 2).astype(np.float32),
                     nn=tr.astype(np.float32) / 255.0, sims=gen_nn[top])


if __name__ == "__main__":
    main()
