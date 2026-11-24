"""Self-check for E3 (train.py --cond/--seed, eval_cfg.py) with tiny fake data. Run with the project venv: python selftest_e3.py"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import HERE, NUM_CLASSES, predict_eps

REF_COMMIT = "0865bf8"  # train.py before --seed/--cond
REF = HERE / "_train_ref.py"  # next to common.py so its imports work; removed at the end


def sh(*cmd, cpu=False):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1") if cpu else None
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=HERE, capture_output=True, text=True, env=env)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}")
    return r.stdout


def final_ema(run, steps=16):
    return torch.load(run / f"ema_{steps:07d}.pt", map_location="cpu", weights_only=True)


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_regression_and_seed(td):
    ref = subprocess.run(["git", "show", f"{REF_COMMIT}:experiments/train.py"], cwd=HERE, capture_output=True, text=True, check=True)
    REF.write_text(ref.stdout)
    try:
        a, b, c = td / "ref", td / "new0", td / "new1"
        common = ["--n", 64, "--steps", 16, "--tiny", "--log-every", 4]
        # CPU: GPU backward kernels need not be bit-reproducible, which would hide the point of the check
        sh(REF.name, *common, "--out", a, cpu=True)
        sh("train.py", *common, "--out", b, cpu=True)
        assert same_weights(final_ema(a), final_ema(b)), "default train.py differs from the committed version"
        print("ok  unconditional seed-0 weights bit-identical to train.py at " + REF_COMMIT)
        sh("train.py", *common, "--out", c, "--seed", 1, cpu=True)
        assert not same_weights(final_ema(b), final_ema(c)), "--seed 1 gave the same weights"
        assert np.array_equal(np.load(b / "members.npy"), np.load(c / "members.npy")), "members changed with seed"
        assert json.loads((c / "config.json").read_text())["seed"] == 1
        print("ok  --seed 1 changes weights, members.npy identical")
    finally:
        REF.unlink(missing_ok=True)


def test_cond_and_eval(td):
    run = td / "c64"
    sh("train.py", "--n", 64, "--steps", 16, "--tiny", "--cond", "--log-every", 4, "--out", run)
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["cond"] is True and cfg["arch"] == "base" and cfg["p_uncond"] == 0.1 and cfg["seed"] == 0, cfg
    assert len(list(run.glob("ema_*.pt"))) == 8
    args = ["eval_cfg.py", "--run", run, "--no-fid", "--n-samples", 8, "--ddim-steps", 3, "--n-boot", 20, "--ws", "0,1,2"]
    sh(*args)
    mp = run / "cfg_metrics.json"
    m = json.loads(mp.read_text())
    assert set(m["w"]) == {"w_0", "w_1", "w_2"} and m["step"] == 16 and "copy_threshold" in m
    for e in m["w"].values():
        for k in ("copy_rate", "copy_rate_lo", "copy_rate_hi", "median_nn", "mean_nn", "copy_rate_095", "copy_rate_095_hi", "fid"):
            assert k in e, k
        assert set(e["per_class"]) == {"0", "1"} and e["per_class"]["0"]["n"] == 4
    for v in ("true_label", "null_label"):
        r = m["mia"][v]
        assert 0 <= r["auc"] <= 1 and r["auc_lo"] <= r["auc_hi"] and "tpr1_hi" in r, r
    z = np.load(run / "samples_cfg.npz")
    assert {"w_0", "w_1", "w_2", "nn_idx_w_2", "nn_sim_w_2"} <= set(z.files), z.files
    assert z["w_2"].dtype == np.uint8 and z["w_2"].shape == (8, 64, 64) and z["nn_idx_w_2"].shape == (8,)
    f = np.load(run / "figcfg.npz")
    assert f["gen_w_1"].shape == (8, 64, 64) and f["nn_w_1"].shape == (8, 64, 64) and (run / "figcfg.json").exists()
    before = (mp.read_bytes(), (run / "samples_cfg.npz").stat().st_mtime_ns)
    out = sh(*args)
    assert out.count("already evaluated, skipping") == 4, out  # 3 w + mia
    assert (mp.read_bytes(), (run / "samples_cfg.npz").stat().st_mtime_ns) == before, "second run rewrote files"
    sh(*args[:-1], "0,1,2,5")  # a new w is added, the old ones are kept
    assert set(json.loads(mp.read_text())["w"]) == {"w_0", "w_1", "w_2", "w_5"}
    print("ok  --cond trains; eval_cfg writes the expected keys and is idempotent")


def test_cfg_math():
    from eval_cfg import eps_cfg
    calls = []

    class Stub(torch.nn.Module):
        def forward(self, x, t, class_labels=None):
            calls.append(len(x))
            return SimpleNamespace(sample=0.3 * x + (class_labels.float() * 2 + 1).view(-1, 1, 1, 1))

    s, x, t = Stub(), torch.randn(4, 1, 8, 8), torch.tensor(7)
    y = torch.tensor([0, 1, 0, 1])
    null = torch.full_like(y, NUM_CLASSES)
    ec, eu = predict_eps(s, x, t, y), predict_eps(s, x, t, null)
    calls.clear()
    assert torch.equal(eps_cfg(s, x, t, y, 0.0), eu) and calls == [4]
    assert torch.equal(eps_cfg(s, x, t, y, 1.0), ec) and calls == [4, 4]
    calls.clear()
    assert torch.allclose(eps_cfg(s, x, t, y, 2.5), eu + 2.5 * (ec - eu), atol=1e-6) and calls == [8]
    print("ok  CFG math: w=0 is null prediction, w=1 conditional, otherwise eu + w (ec - eu)")


if __name__ == "__main__":
    t0 = time.time()
    test_cfg_math()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        test_regression_and_seed(td)
        test_cond_and_eval(td)
    print(f"selftest_e3 passed in {time.time() - t0:.0f}s")
