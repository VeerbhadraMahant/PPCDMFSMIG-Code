"""Self-check for attacks.py / eval_attacks.py with oracle models and tiny fake runs. Run: python selftest_attacks.py"""
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import attacks
from common import HERE, NUM_CLASSES, build_unet, get_device, make_ddpm, member_indices
from evaluate import MIA_DRAWS, MIA_TS, mia_scores, roc_metrics

CPU = torch.device("cpu")


class PointOracle:
    """Ideal denoiser when the data distribution is one point x0 per sample (rows aligned with the batch)."""

    def __init__(self, x0, ab):
        self.x0, self.ab = x0, ab

    def __call__(self, x_t, t, class_labels=None):
        a = self.ab[int(t[0])]
        return SimpleNamespace(sample=(x_t - math.sqrt(a) * self.x0) / math.sqrt(1 - a))


class EmpiricalOracle:
    """Ideal denoiser for the empirical distribution of `train`: a perfect memorizer."""

    def __init__(self, train, ab):
        self.tr, self.ab = train.flatten(1), ab

    def __call__(self, x_t, t, class_labels=None):
        a = self.ab[int(t[0])]
        d = ((x_t.flatten(1)[:, None] - math.sqrt(a) * self.tr[None]) ** 2).sum(-1)
        w = torch.softmax(-d / (2 * (1 - a)), dim=1)
        x0 = (w @ self.tr).view_as(x_t)
        return SimpleNamespace(sample=(x_t - math.sqrt(a) * x0) / math.sqrt(1 - a))


def test_oracle_consistency():
    sched = make_ddpm()
    ab = attacks._abar(sched)
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(6, 1, 64, 64, generator=g, dtype=torch.float64) * 2 - 1)
    o = PointOracle(x, ab)
    # invert 0 -> 90 in steps of 10 and come back: recovers the starting point to numerical precision
    h, tc = x, 0
    for tg in range(10, 100, 10):
        h = attacks.ddim_step(o, h, tc, tg, ab)
        tc = tg
    for tg in range(80, -1, -10):
        h = attacks.ddim_step(o, h, tc, tg, ab)
        tc = tg
    assert (h - x).abs().max() < 1e-9, (h - x).abs().max()
    # the attacks' own error terms vanish for the oracle ...
    for s in (attacks.pia_scores(o, x, sched, CPU, bs=6), attacks.secmi_scores(o, x, sched, CPU, bs=6)):
        assert np.abs(s).max() < 1e-8, np.abs(s).max()
    # PIA error does not vanish when the oracle knows a different image (SecMI's would: any point-mass
    # oracle gives an exactly invertible DDIM map, so only the memorizer test below can exercise it)
    s = attacks.pia_scores(PointOracle(x.roll(1, 0), ab), x, sched, CPU, bs=6)
    assert (s < -1e-3).all(), s
    print("ok  oracle: invert-then-reverse recovers x0 (<1e-9), PIA / SecMI errors ~0, PIA nonzero for a wrong oracle")


def test_memorizer_separates():
    sched = make_ddpm()
    ab = attacks._abar(sched)
    g = torch.Generator().manual_seed(1)
    mem, non = [torch.rand(10, 1, 64, 64, generator=g, dtype=torch.float64) * 2 - 1 for _ in range(2)]
    o = EmpiricalOracle(mem, ab)
    y = np.r_[np.ones(10), np.zeros(10)]
    for name, fn in (("pia", attacks.pia_scores), ("secmi", attacks.secmi_scores)):
        s = np.r_[fn(o, mem, sched, CPU, bs=10), fn(o, non, sched, CPU, bs=10)]
        auc, _ = roc_metrics(y, s)
        assert auc > 0.99, (name, auc)  # higher score = member
    print("ok  a perfect memorizer gives AUC > 0.99 for PIA and SecMI (members score higher)")


def test_determinism_and_labels():
    dev = get_device()
    sched = make_ddpm()
    torch.manual_seed(0)
    x = (torch.rand(10, 1, 64, 64) * 2 - 1).to(dev)
    noise = torch.randn(len(MIA_TS), MIA_DRAWS, 1, 64, 64, generator=torch.Generator().manual_seed(0)).to(dev)
    for cond in (False, True):
        unet = build_unet(tiny=True, num_classes=NUM_CLASSES if cond else 0).to(dev).eval()
        y = torch.tensor([0, 1] * 5) if cond else None
        for fn in (attacks.pia_scores, attacks.secmi_scores):
            a, b = fn(unet, x, sched, dev, y, bs=4), fn(unet, x, sched, dev, y, bs=10)
            assert np.isfinite(a).all() and a.shape == (10,)
            assert np.allclose(a, b, rtol=1e-3, atol=1e-6 * np.abs(a).max()), "batch size changes scores"
            assert np.array_equal(a, fn(unet, x, sched, dev, y, bs=4)), "not deterministic"
        l = attacks.loss_scores(unet, x, noise, sched, dev, y, bs=5)
        assert np.isfinite(l).all()
        if cond:  # a different label must change the prediction (random init, so only a sanity check)
            null = torch.full((10,), NUM_CLASSES)
            assert not np.allclose(attacks.pia_scores(unet, x, sched, dev, null), attacks.pia_scores(unet, x, sched, dev, y))
    # y=None loss attack is the pilot's mia_scores exactly
    unet = build_unet(tiny=True).to(dev).eval()
    assert np.array_equal(attacks.loss_scores(unet, x, noise, sched, dev, None), mia_scores(unet, x, noise, sched, dev))
    print("ok  scores finite, deterministic, batch-size independent; cond and uncond paths; loss(y=None) == mia_scores")


def sh(*cmd):
    r = subprocess.run([sys.executable, *map(str, cmd)], cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}")
    return r.stdout


def check(run, modes, cond):
    a = json.loads((run / "attacks.json").read_text())
    assert a["cond"] == cond and a["step"] == 16 and a["n_per_group"] == 64
    for name in ("loss", "pia", "secmi"):
        assert set(a["attacks"][name]) == set(modes), (name, a["attacks"][name].keys())
        for e in a["attacks"][name].values():
            for k in ("auc", "auc_lo", "auc_hi", "tpr1", "tpr1_lo", "tpr1_hi", "seconds", "settings"):
                assert k in e, k
            assert 0 <= e["auc_lo"] <= e["auc_hi"] <= 1 and 0 <= e["tpr1"] <= 1
    assert a["attacks"]["pia"][modes[0]]["settings"]["t"] == 200
    assert a["attacks"]["secmi"][modes[0]]["settings"]["t_sec"] == 100


def rerun_skips(run, args):
    before = (run / "attacks.json").read_text()
    out = sh("eval_attacks.py", *args)
    assert out.count("skipping") == len(json.loads(before)["attacks"]) * len(
        next(iter(json.loads(before)["attacks"].values()))), out
    assert (run / "attacks.json").read_text() == before


def test_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        # unconditional, trained by train.py exactly as selftest.py does
        run = Path(td) / "runs" / "n64"
        sh("train.py", "--n", 64, "--steps", 16, "--out", run, "--tiny")
        args = ["--run", run, "--n-boot", 20, "--bs", 32]
        sh("eval_attacks.py", *args)
        check(run, ["none"], False)
        rerun_skips(run, args)
        # conditional: random-weight tiny conditional UNet saved under the same checkpoint layout
        crun = Path(td) / "runs" / "c64"
        crun.mkdir(parents=True)
        steps = [2 * k for k in range(1, 9)]
        np.save(crun / "members.npy", member_indices(600, 64))
        (crun / "config.json").write_text(json.dumps({"n": 64, "steps": 16, "tiny": True, "seed": 0, "cond": True,
                                                      "ckpt_steps": steps}))
        torch.manual_seed(0)
        sd = build_unet(tiny=True, num_classes=NUM_CLASSES).state_dict()
        for s in steps:
            torch.save(sd, crun / f"ema_{s:07d}.pt")
        args = ["--run", crun, "--n-boot", 20, "--bs", 32]
        sh("eval_attacks.py", *args)
        a = json.loads((crun / "attacks.json").read_text())
        assert a["step"] == 16
        check(crun, ["true", "null"], True)
        rerun_skips(crun, args)
        sh("eval_attacks.py", *args, "--label-mode", "null", "--force")  # forced recompute only touches the chosen mode
        assert set(json.loads((crun / "attacks.json").read_text())["attacks"]["pia"]) == {"true", "null"}
    print("ok  eval_attacks end to end: unconditional and conditional runs, keys, idempotent rerun, --force")


if __name__ == "__main__":
    t = time.time()
    for f in (test_oracle_consistency, test_memorizer_separates, test_determinism_and_labels, test_end_to_end):
        f()
    print(f"selftest_attacks passed in {time.time() - t:.0f}s")
