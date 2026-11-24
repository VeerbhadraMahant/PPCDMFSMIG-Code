"""Run PIA, SecMI-stat and the loss attack on the last EMA checkpoint of one run (PROTOCOL_V2.md E4a, ATTACKS.md).

Same member / non-member sets, AUC, TPR@1%FPR and bootstrap as evaluate.py. Writes runs/X/attacks.json.
"""
import argparse
import json
import time

import numpy as np
import torch

import attacks
from common import (NUM_CLASSES, get_device, load_data_with_origin, load_labels, make_ddpm, resolve, to_tensor,
                    unet_from_config, write_json)
from evaluate import MIA_DRAWS, MIA_TS, ci, roc_metrics


def bootstrap(s_mem, s_non, n_boot, seed):
    """Percentile CIs, resampling each group with replacement (evaluate.py's loss-MIA method)."""
    m = len(s_mem)
    y = np.r_[np.ones(m), np.zeros(m)]
    brng = np.random.RandomState(seed)
    boots = np.array([roc_metrics(y, np.r_[brng.choice(s_mem, m), brng.choice(s_non, m)]) for _ in range(n_boot)])
    return ci(boots[:, 0]), ci(boots[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--label-mode", choices=["auto", "true", "null", "both"], default="auto",
                    help="conditional runs only; auto and both = true and null labels")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=125)
    ap.add_argument("--force", action="store_true", help="recompute entries already in attacks.json")
    args = ap.parse_args()

    run = resolve(args.run)
    cfg = json.loads((run / "config.json").read_text())
    device = get_device()
    ckpts = sorted(run.glob("ema_*.pt"))
    found = {int(p.stem.split("_")[1]) for p in ckpts}
    if found != set(cfg["ckpt_steps"]):
        raise SystemExit(f"{run}: checkpoint steps {sorted(found)} != config ckpt_steps {cfg['ckpt_steps']}")
    step, last = max(found), max(ckpts, key=lambda p: int(p.stem.split("_")[1]))

    # member / non-member sets exactly as evaluate.py
    train_u8, non_u8, _ = load_data_with_origin(cfg["tiny"])
    mem_idx = np.load(run / "members.npy")
    members, nonmem = to_tensor(train_u8[mem_idx], device), to_tensor(non_u8, device)
    n = cfg["n"]
    m = min(n, 1000)
    rs = np.random.RandomState(0)
    sel_mem = rs.choice(n, m, replace=False) if n > m else np.arange(n)
    sel_non = rs.choice(len(nonmem), m, replace=False)
    x_mem, x_non = members[sel_mem], nonmem[sel_non]

    cond = bool(cfg.get("cond"))
    if cond:
        tr_y, nm_y = load_labels(cfg["tiny"])
        modes = {"true": (torch.as_tensor(tr_y[mem_idx][sel_mem]), torch.as_tensor(nm_y[sel_non])),
                 "null": (torch.full((m,), NUM_CLASSES), torch.full((m,), NUM_CLASSES))}
        if args.label_mode in ("true", "null"):
            modes = {args.label_mode: modes[args.label_mode]}
    else:
        modes = {"none": (None, None)}  # unconditional model: no labels

    noise = torch.randn(len(MIA_TS), MIA_DRAWS, 1, 64, 64, generator=torch.Generator().manual_seed(0)).to(device)
    ddpm = make_ddpm()
    fns = {
        "loss": (lambda x, y: attacks.loss_scores(unet, x, noise, ddpm, device, y, args.bs),
                 {"ts": list(MIA_TS), "draws": MIA_DRAWS, "noise_seed": 0}),
        "pia": (lambda x, y: attacks.pia_scores(unet, x, ddpm, device, y, args.bs),
                {"t": attacks.PIA_T, "p": attacks.PIA_P}),
        "secmi": (lambda x, y: attacks.secmi_scores(unet, x, ddpm, device, y, args.bs),
                  {"t_sec": attacks.SECMI_T, "k": attacks.SECMI_K}),
    }

    unet = unet_from_config(cfg).to(device).eval()
    sd = torch.load(last, map_location=device, weights_only=True)
    unet.load_state_dict({k.removeprefix("_module."): v for k, v in sd.items()})  # tolerate Opacus-wrapped keys

    path = run / "attacks.json"
    res = json.loads(path.read_text()) if path.exists() else {"attacks": {}}
    res.update(run=str(args.run), step=step, n=n, n_per_group=m, cond=cond, tiny=cfg["tiny"])
    for name, (fn, hp) in fns.items():
        for mode, (y_mem, y_non) in modes.items():
            settings = json.loads(json.dumps({"step": step, "n_boot": args.n_boot, "n_per_group": m,
                                              "label": mode, **hp}))
            old = res["attacks"].get(name, {}).get(mode)
            if old and old["settings"] == settings and not args.force:
                print(f"{name}/{mode}: already computed, skipping", flush=True)
                continue
            t0 = time.time()
            s_mem, s_non = fn(x_mem, y_mem), fn(x_non, y_non)
            sec = time.time() - t0
            assert np.isfinite(s_mem).all() and np.isfinite(s_non).all(), f"{name}/{mode}: non-finite scores"
            y = np.r_[np.ones(m), np.zeros(m)]  # members are the positive class
            e = {"settings": settings, "seconds": round(sec, 1)}
            e["auc"], e["tpr1"] = roc_metrics(y, np.r_[s_mem, s_non])
            (e["auc_lo"], e["auc_hi"]), (e["tpr1_lo"], e["tpr1_hi"]) = bootstrap(s_mem, s_non, args.n_boot, step)
            res["attacks"].setdefault(name, {})[mode] = e
            write_json(res, path)
            print(f"{name}/{mode}: auc {e['auc']:.3f} [{e['auc_lo']:.3f}, {e['auc_hi']:.3f}] "
                  f"tpr1 {e['tpr1']:.3f} ({sec:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
