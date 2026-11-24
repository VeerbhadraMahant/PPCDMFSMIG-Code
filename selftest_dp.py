"""Quick self-check of E5 (train_dp.py, dp_gate.py) with tiny fake data. Run with the project venv: python selftest_dp.py"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from common import HERE, build_unet

DP_ARGS = ["--tiny", "--n", 64, "--batch", 16, "--epochs", 2, "--eps", 10, "--physical-bs", 8, "--log-every", 2]


def sh(*cmd):
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}")
    return r.stdout


def cfg_of(run):
    return json.loads((run / "config.json").read_text())


def check_run(run, steps=8):
    emas = sorted(run.glob("ema_*.pt"))
    assert len(emas) == 8 and emas[-1].name == f"ema_{steps:07d}.pt", [p.name for p in emas]
    for p in emas:  # keys must load into a plain UNet, no wrapper prefix
        build_unet(tiny=True).load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
    assert not (run / "resume.pt").exists() and (run / "log.csv").exists() and (run / "members.npy").exists()
    c = cfg_of(run)
    for k in ("n", "steps", "ckpt_steps", "tiny", "seed", "cond", "arch", "batch", "lr", "dp"):
        assert k in c, k
    assert c["steps"] == steps and len(c["ckpt_steps"]) == 8 and c["cond"] is False
    return c


def test_dp_and_resume(td):
    run = td / "dp"
    sh("train_dp.py", *DP_ARGS, "--out", run)
    c = check_run(run)
    d = c["dp"]
    for k in ("mode", "target_eps", "delta", "epochs", "sample_rate", "noise_multiplier", "max_grad_norm",
              "accountant", "achieved_eps", "grad_sample_mode", "amp"):
        assert k in d, k
    assert d["mode"] == "dp" and d["accountant"] == "rdp" and d["sample_rate"] == 0.25
    assert 0 < d["achieved_eps"] <= 10 + 1e-3, d["achieved_eps"]
    assert "already finished" in sh("train_dp.py", *DP_ARGS, "--out", run)
    print(f"ok  DP run: 8 checkpoints load into build_unet, achieved eps {d['achieved_eps']:.4f} <= 10, rerun exits")

    # stop after the first step, then resume: accounting must end where the uninterrupted run did
    run2 = td / "dp_resume"
    sh("train_dp.py", *DP_ARGS, "--out", run2, "--max-minutes", 1e-6)
    assert (run2 / "resume.pt").exists() and not (run2 / "ema_0000008.pt").exists()
    sh("train_dp.py", *DP_ARGS, "--out", run2)
    c2 = check_run(run2)
    assert abs(c2["dp"]["achieved_eps"] - d["achieved_eps"]) < 1e-9, (c2["dp"]["achieved_eps"], d["achieved_eps"])
    print("ok  resume keeps the privacy accounting (same final epsilon as the uninterrupted run)")
    return run


def test_twin(td):
    run = td / "twin"
    sh("train_dp.py", *DP_ARGS, "--no-dp", "--out", run)
    c = check_run(run)
    assert c["dp"]["mode"] == "none" and c["dp"]["noise_multiplier"] is None and c["steps"] == 8
    sh("train_dp.py", *DP_ARGS, "--no-dp", "--out", td / "twin_r", "--max-minutes", 1e-6)
    assert (td / "twin_r" / "resume.pt").exists()
    sh("train_dp.py", *DP_ARGS, "--no-dp", "--out", td / "twin_r")
    check_run(td / "twin_r")
    print("ok  non-DP twin: same step count, 8 checkpoints, resume")


def test_evaluate(run):
    sh("evaluate.py", "--run", run, "--ckpt", "last", "--no-fid", "--n-samples", 8, "--ddim-steps", 3,
       "--n-boot", 20, "--bs", 8)
    m = json.loads((run / "metrics.json").read_text())
    e = m["checkpoints"]["8"]
    assert 0 <= e["copy_rate"] <= 1 and 0 <= e["auc"] <= 1
    print("ok  evaluate.py --ckpt last on the DP run")


def test_gate(td):
    out = td / "gate.json"
    sh("dp_gate.py", "--quick", "--out", out)
    g = json.loads(out.read_text())
    for k in ("run_e5", "arch", "epochs", "physical_bs", "projection_table", "note"):
        assert k in g, k
    assert all(c["fp32_pass"] for c in g["correctness"])
    print(f"ok  dp_gate --quick: run_e5={g['run_e5']} arch={g['arch']} epochs={g['epochs']}")


if __name__ == "__main__":
    t = time.time()
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        run = test_dp_and_resume(td)
        test_twin(td)
        test_evaluate(run)
        test_gate(td)
    print(f"selftest_dp passed in {time.time() - t:.0f}s")
