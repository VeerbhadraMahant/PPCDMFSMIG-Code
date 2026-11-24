"""E3: guidance-scale sweep and label-aware loss MIA on the last EMA checkpoint of a conditional run (see PROTOCOL_V2.md).

Outputs in the run folder:
  cfg_metrics.json  settings, copy threshold, per-w metrics, label-aware MIA (true label vs null label)
  samples_cfg.npz   "w_<w>": uint8 (n, 64, 64) samples for guidance scale w (w formatted with :g, e.g. w_0, w_7.5);
                    "nn_idx_w_<w>": train-set index of each sample's nearest member; "nn_sim_w_<w>": its Pearson similarity.
                    Sample i was generated with label i % 2. Similarities are computed on the float samples, not the uint8 copy.
  figcfg.npz/json   the 8 highest-similarity (sample, nearest member) pairs per w, for a later figure
                    (gen_w_<w>, nn_w_<w> in [0, 1], sims_w_<w>; json has indices and labels)
"""
import argparse
import json
import time

import numpy as np
import torch

from common import (NUM_CLASSES, get_device, load_data_with_origin, load_labels, make_ddim, make_ddpm, predict_eps,
                    resolve, to_tensor, unet_from_config, write_json)
from evaluate import MIA_DRAWS, MIA_TS, ci, compute_fid, mem_peak, mem_reset, nn_sim, roc_metrics


def eps_cfg(unet, x, t, y, w):
    """eps(x,t,null) + w * (eps(x,t,c) - eps(x,t,null)); w = 0 and w = 1 need only one forward pass."""
    null = torch.full_like(y, NUM_CLASSES)
    if w == 0:
        return predict_eps(unet, x, t, null)
    if w == 1:
        return predict_eps(unet, x, t, y)
    ec, eu = predict_eps(unet, torch.cat([x, x]), t, torch.cat([y, null])).chunk(2)
    return eu + w * (ec - eu)


@torch.no_grad()
def sample_cfg(unet, x_all, w, steps, bs, device):
    """DDIM (eta 0, fp32) from the given initial noise; sample i gets label i % 2."""
    sched = make_ddim()
    sched.set_timesteps(steps)
    out = []
    for i in range(0, len(x_all), bs):
        x = x_all[i:i + bs].to(device)
        y = (torch.arange(i, i + len(x), device=device) % 2).long()
        for t in sched.timesteps:
            x = sched.step(eps_cfg(unet, x, t, y, w).float(), t, x, eta=0.0).prev_sample
        out.append(x.clamp(-1, 1).cpu())
    return torch.cat(out)


@torch.no_grad()
def mia_scores_lab(unet, x, labels, noise, sched, device, bs=125):
    """Same score as evaluate.mia_scores, with labels passed to the model (labels: LongTensor on device)."""
    total = torch.zeros(len(x), device=device)
    for ti, t in enumerate(MIA_TS):
        for k in range(MIA_DRAWS):
            for i in range(0, len(x), bs):
                xb = x[i:i + bs]
                e = noise[ti, k][None].expand_as(xb)
                tt = torch.full((len(xb),), t, device=device, dtype=torch.long)
                pred = predict_eps(unet, sched.add_noise(xb, e, tt), tt, labels[i:i + bs])
                total[i:i + bs] += ((pred - e) ** 2).flatten(1).sum(1)
    return (-total / (len(MIA_TS) * MIA_DRAWS)).cpu().numpy()


def boot_rate(hit, brng, n_boot):
    bi = brng.randint(0, len(hit), (n_boot, len(hit)))
    return ci(hit[bi].mean(1))


def u8(x_pm1):
    return ((x_pm1[:, 0].numpy() + 1) / 2 * 255).round().astype(np.uint8)


def load_npz(path):
    if not path.exists():
        return {}
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ws", default="0,1,2,3,5,7.5", help="comma-separated guidance scales")
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=250, help="sampling batch size (doubled inside for guided w)")
    ap.add_argument("--seed", type=int, default=0, help="seed of the initial noise, shared by every w")
    ap.add_argument("--no-fid", action="store_true", help="self-test only")
    ap.add_argument("--force", action="store_true", help="recompute everything already in cfg_metrics.json")
    args = ap.parse_args()

    run = resolve(args.run)
    cfg = json.loads((run / "config.json").read_text())
    if not cfg.get("cond"):
        raise SystemExit("eval_cfg.py needs a conditional run (train.py --cond); use evaluate.py otherwise")
    ws = [float(v) for v in args.ws.split(",")]
    device = get_device()
    train_u8, non_u8, _ = load_data_with_origin(cfg["tiny"])
    train_y, non_y = load_labels(cfg["tiny"])
    mem_idx = np.load(run / "members.npy")
    members = to_tensor(train_u8[mem_idx], device)
    nonmem = to_tensor(non_u8, device)
    n = cfg["n"]

    ckpts = sorted(run.glob("ema_*.pt"))
    found = {int(p.stem.split("_")[1]) for p in ckpts}
    if found != set(cfg["ckpt_steps"]):
        raise SystemExit(f"{run}: checkpoint steps {sorted(found)} != config ckpt_steps {cfg['ckpt_steps']}")
    step = int(ckpts[-1].stem.split("_")[1])
    unet = unet_from_config(cfg).to(device).eval()
    unet.load_state_dict(torch.load(ckpts[-1], map_location=device, weights_only=True))

    # same threshold as evaluate.py: 99th percentile of non-member NN similarity to the members
    non_nn, _ = nn_sim(nonmem, members)
    threshold = float(np.percentile(non_nn, 99))
    settings = {"step": step, "n_samples": args.n_samples, "ddim_steps": args.ddim_steps, "n_boot": args.n_boot,
                "seed": args.seed, "fid": not args.no_fid}

    mpath = run / "cfg_metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() and not args.force else {}
    metrics.update(n=n, steps=cfg["steps"], tiny=cfg["tiny"], step=step, copy_threshold=threshold,
                   nonmember_nn_median=float(np.median(non_nn)), n_nonmembers=len(non_nn), settings=settings)
    metrics.setdefault("w", {})
    samples_path, fig_path = run / "samples_cfg.npz", run / "figcfg.npz"
    store = {} if args.force else load_npz(samples_path)
    fig = {} if args.force else load_npz(fig_path)
    fjson = json.loads((run / "figcfg.json").read_text()) if (run / "figcfg.json").exists() and not args.force else {}

    # same initial noise for every w (paired comparison), drawn up front so results do not depend on bs
    x_all = torch.randn(args.n_samples, 1, 64, 64, generator=torch.Generator().manual_seed(args.seed))
    labels = np.arange(args.n_samples) % 2

    for w in ws:
        key = f"w_{w:g}"
        old = metrics["w"].get(key)
        if old and old["settings"] == settings and key in store and not args.force:
            print(f"{key}: already evaluated, skipping", flush=True)
            continue
        t0 = time.time()
        mem_reset()
        samples = sample_cfg(unet, x_all, w, args.ddim_steps, args.bs, device)
        peak_s = mem_peak()
        mem_reset()
        gen_nn, gen_arg = nn_sim(samples.to(device), members)  # over all members, not only the same class
        hit, hit95 = gen_nn > threshold, gen_nn > 0.95
        brng = np.random.RandomState(step)  # same bootstrap indices for every w
        bi = brng.randint(0, len(gen_nn), (args.n_boot, len(gen_nn)))
        e = {"settings": settings, "w": w, "copy_rate": float(hit.mean()), "median_nn": float(np.median(gen_nn)),
             "mean_nn": float(gen_nn.mean()), "copy_rate_095": float(hit95.mean())}
        for name, vals in (("copy_rate", hit[bi].mean(1)), ("median_nn", np.median(gen_nn[bi], axis=1)),
                           ("copy_rate_095", hit95[bi].mean(1))):
            e[name + "_lo"], e[name + "_hi"] = ci(vals)
        e["per_class"] = {}
        for c in range(NUM_CLASSES):  # class = the label the sample was generated with
            hc = hit[labels == c]
            lo, hi = boot_rate(hc, brng, args.n_boot)
            e["per_class"][str(c)] = {"n": int(len(hc)), "copy_rate": float(hc.mean()), "copy_rate_lo": lo,
                                      "copy_rate_hi": hi}
        e["fid"] = None if args.no_fid else compute_fid(samples, nonmem, device)
        e["seconds"] = round(time.time() - t0, 1)

        store[key] = u8(samples)
        store["nn_idx_" + key] = mem_idx[gen_arg]
        store["nn_sim_" + key] = gen_nn
        top = np.argsort(-gen_nn)[:8]
        fig["gen_" + key] = ((samples[top, 0].numpy() + 1) / 2).astype(np.float32)
        fig["nn_" + key] = train_u8[mem_idx[gen_arg[top]]].astype(np.float32) / 255.0
        fig["sims_" + key] = gen_nn[top]
        fjson[key] = {"sample_idx": top.tolist(), "train_idx": mem_idx[gen_arg[top]].tolist(),
                      "sims": gen_nn[top].tolist(), "labels": labels[top].tolist()}
        fjson.update(step=step, threshold=threshold)
        metrics["w"][key] = e
        np.savez(samples_path, **store)
        np.savez(fig_path, **fig)
        write_json(fjson, run / "figcfg.json")
        write_json(metrics, mpath)
        print(f"{key}: copy {e['copy_rate']:.3f} median_nn {e['median_nn']:.3f} copy95 {e['copy_rate_095']:.3f} "
              f"fid {e['fid']} ({e['seconds']}s, sampling peak {peak_s:.2f} GB)", flush=True)

    # label-aware loss MIA, member/non-member selection and noise as in evaluate.py
    mia_settings = {"step": step, "n_boot": args.n_boot}
    old = metrics.get("mia")
    if old and old["settings"] == mia_settings and not args.force:
        print("mia: already evaluated, skipping", flush=True)
    else:
        rs = np.random.RandomState(0)
        m = min(n, 1000)
        sel_mem = rs.choice(n, m, replace=False) if n > m else np.arange(n)
        sel_non = rs.choice(len(nonmem), m, replace=False)
        mem_x, non_x = members[sel_mem], nonmem[sel_non]
        mem_y = torch.from_numpy(train_y[mem_idx][sel_mem]).to(device)
        non_yt = torch.from_numpy(non_y[sel_non]).to(device)
        noise = torch.randn(len(MIA_TS), MIA_DRAWS, 1, 64, 64, generator=torch.Generator().manual_seed(0)).to(device)
        ddpm = make_ddpm()
        yy = np.r_[np.ones(m), np.zeros(m)]
        mia = {"settings": mia_settings, "n_per_group": m}
        for name, (ym, yn) in {"true_label": (mem_y, non_yt),
                               "null_label": (torch.full_like(mem_y, NUM_CLASSES), torch.full_like(non_yt, NUM_CLASSES))}.items():
            s_mem = mia_scores_lab(unet, mem_x, ym, noise, ddpm, device)
            s_non = mia_scores_lab(unet, non_x, yn, noise, ddpm, device)
            r = {}
            r["auc"], r["tpr1"] = roc_metrics(yy, np.r_[s_mem, s_non])
            brng = np.random.RandomState(step)
            boots = np.array([roc_metrics(yy, np.r_[brng.choice(s_mem, m), brng.choice(s_non, m)])
                              for _ in range(args.n_boot)])
            r["auc_lo"], r["auc_hi"] = ci(boots[:, 0])
            r["tpr1_lo"], r["tpr1_hi"] = ci(boots[:, 1])
            mia[name] = r
            print(f"mia {name}: auc {r['auc']:.3f} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}] tpr1 {r['tpr1']:.3f}", flush=True)
        metrics["mia"] = mia
        write_json(metrics, mpath)


if __name__ == "__main__":
    main()
