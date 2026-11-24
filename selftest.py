"""Quick self-check with tiny fake data. Run with the project venv: python selftest.py"""
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from common import HERE, NPZ, ema_update, load_data, load_data_with_origin, member_indices
from evaluate import nn_sim, roc_metrics


def test_subsets():
    train, non = load_data(fake=True)
    s = {n: member_indices(len(train), n) for n in (64, 128, 256)}
    assert set(s[64]) < set(s[128]) < set(s[256]), "subsets not nested"
    assert all(len(set(v)) == len(v) and v.min() >= 0 and v.max() < len(train) for v in s.values())
    assert np.array_equal(s[64], member_indices(len(train), 64)), "not deterministic"
    print("ok  nested subsets")
    if NPZ.exists():
        train, non = load_data()
        assert (len(train), len(non)) == (4708, 1140), (len(train), len(non))
        s = [member_indices(len(train), n) for n in (256, 1024, 4096)]
        assert set(s[0]) < set(s[1]) < set(s[2])
        h = lambda a: hashlib.md5(a.tobytes()).hexdigest()
        dup = {h(a) for a in train} & {h(a) for a in non}
        assert not dup, f"{len(dup)} non-members duplicate train images"
        _, nm2, iv = load_data_with_origin()
        assert len(iv) == len(nm2) == 1140 and 0 < iv.sum() < 524 + 1, (len(iv), iv.sum())
        print(f"ok  real data sizes; no non-member duplicates a train image; val/test non-members {iv.sum()}/{(~iv).sum()}")


def test_pearson():
    rng = np.random.RandomState(0)
    q, r = rng.randn(5, 4096), rng.randn(7, 4096)
    sims, arg = nn_sim(torch.tensor(q), torch.tensor(r))
    full = np.corrcoef(q, r)[:5, 5:]
    assert np.allclose(sims, full.max(1), atol=1e-6) and np.array_equal(arg, full.argmax(1))
    print("ok  pearson matches np.corrcoef")


def test_roc():
    y = np.r_[np.ones(50), np.zeros(50)]
    # 10 pos at 10, 20 pos at 5 tied with 1 neg, 20 pos at -1; 49 neg at 0 -> ROC (0,.2),(.02,.6),(1,.6),(1,1)
    s = np.r_[np.full(10, 10.), np.full(20, 5.), np.full(20, -1.), [5.], np.zeros(49)]
    auc, tpr = roc_metrics(y, s)
    assert abs(tpr - 0.4) < 1e-9, tpr  # halfway along the (0,.2)-(.02,.6) segment
    s2 = np.r_[np.full(25, 10.), np.full(25, -1.), np.zeros(50)]
    auc2, tpr2 = roc_metrics(y, s2)
    assert abs(tpr2 - 0.5) < 1e-9 and abs(auc2 - 0.5) < 1e-9, (auc2, tpr2)
    print("ok  TPR@1%FPR and AUC on toy ROC")


def test_ema():
    a, b = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    e0 = [p.detach().clone() for p in a.parameters()]
    ema_update(a, b, 0.9)
    for pe, p0, pb in zip(a.parameters(), e0, b.parameters()):
        assert torch.allclose(pe, 0.9 * p0 + 0.1 * pb, atol=1e-6)
    print("ok  EMA update")


def sh(*cmd):
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}")
    return r.stdout


def test_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        run, res = Path(td) / "runs" / "n64", Path(td) / "results"
        # first call stops after one step, second must resume and finish
        sh("train.py", "--n", 64, "--steps", 16, "--out", run, "--tiny", "--max-minutes", 1e-6, "--log-every", 4)
        assert (run / "resume.pt").exists() and not list(run.glob("ema_*.pt"))
        sh("train.py", "--n", 64, "--steps", 16, "--out", run, "--tiny", "--log-every", 4)
        assert len(list(run.glob("ema_*.pt"))) == 8 and not (run / "resume.pt").exists()
        assert (run / "log.csv").exists() and (run / "members.npy").exists()
        args = ["evaluate.py", "--run", run, "--n-samples", 16, "--ddim-steps", 3, "--n-boot", 50, "--bs", 8, "--no-fid"]
        sh(*args)
        m = json.loads((run / "metrics.json").read_text())
        assert len(m["checkpoints"]) == 8
        before = json.dumps(m, sort_keys=True)
        sh(*args)  # idempotent: nothing recomputed
        assert json.dumps(json.loads((run / "metrics.json").read_text()), sort_keys=True) == before
        for f in ("samples_last.npy", "figB.json", "figB.npz"):
            assert (run / f).exists(), f
        e = m["checkpoints"]["16"]
        for k in ("val", "test"):
            assert 0 <= e[f"mia_auc_{k}_nm"] <= 1 and e[f"mia_n_{k}_nm"] > 0 and f"mia_tpr1_{k}_nm_hi" in e
        assert 0 <= e["copy_rate"] <= 1 and 0 <= e["auc"] <= 1 and e["mia_n_per_group"] == 64
        sh("make_figures.py", "--runs", run.parent, "--out", res)
        for f in ("table.csv", "table.tex", "figA.pdf", "figB.pdf"):
            assert (res / f).exists(), f
    print("ok  train (with resume) / evaluate (idempotent) / make_figures end to end")


def test_fid():
    from evaluate import compute_fid
    from common import get_device
    g = torch.Generator().manual_seed(0)
    try:
        v = compute_fid(torch.rand(32, 1, 64, 64, generator=g) * 2 - 1, torch.rand(32, 1, 64, 64, generator=g) * 2 - 1,
                        get_device(), bs=16)
        assert np.isfinite(v)
        print(f"ok  FID runs (value {v:.1f} on random noise, meaningless)")
    except Exception as ex:  # most likely the Inception weights download
        print(f"SKIP FID check: {type(ex).__name__}: {str(ex)[:200]}")


if __name__ == "__main__":
    t = time.time()
    for f in (test_subsets, test_pearson, test_roc, test_ema, test_end_to_end, test_fid):
        f()
    print(f"selftest passed in {time.time() - t:.0f}s")
