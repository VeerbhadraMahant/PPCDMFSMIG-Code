"""Threshold membership-inference attacks PIA and SecMI-stat, plus a labelled loss attack (see ATTACKS.md).

All scores are oriented so that HIGHER = more likely a member. Every model call goes through
common.predict_eps, so conditional models work (y = labels, NUM_CLASSES = null label).
Hyperparameters are fixed in advance from the authors' paper and code (ATTACKS.md), not tuned here.
"""
import math

import numpy as np
import torch

from common import predict_eps
from evaluate import MIA_DRAWS, MIA_TS, mia_scores

PIA_T = 200    # Kong2024PIA appendix A.1 (DDPM)
PIA_P = 4      # l_p norm, official code default lp=4
SECMI_T = 100  # Duan2023SecMI Sec. 5.1, t_SEC
SECMI_K = 10   # Duan2023SecMI Sec. 5.1, DDIM interval k


def _tt(n, t, device):
    return torch.full((n,), t, device=device, dtype=torch.long)


def _abar(sched):
    return [float(a) for a in sched.alphas_cumprod.double()]  # python floats keep the tensor dtype


def ddim_step(unet, x, t_from, t_to, ab, y=None):
    """Deterministic DDIM move x_{t_from} -> x_{t_to} (works up or down). Official: SecMI secmia.py ddim_singlestep."""
    eps = predict_eps(unet, x, _tt(len(x), t_from, x.device), y)
    x0 = (x - math.sqrt(1 - ab[t_from]) * eps) / math.sqrt(ab[t_from])  # no clipping, as in the official code
    return math.sqrt(ab[t_to]) * x0 + math.sqrt(1 - ab[t_to]) * eps


def _batched(fn, x, y, bs):
    out = []
    for i in range(0, len(x), bs):
        out.append(fn(x[i:i + bs], None if y is None else y[i:i + bs]).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def pia_scores(unet, x, sched, device, y=None, bs=125, t=PIA_T, p=PIA_P):
    """PIA: score = -|| eps(x0, 0) - eps(sqrt(ab_t) x0 + sqrt(1-ab_t) eps(x0, 0), t) ||_p^p (Kong2024PIA Eq. 9)."""
    ab = _abar(sched)

    def f(xb, yb):
        xb = xb.to(device)
        yb = None if yb is None else yb.to(device)
        e0 = predict_eps(unet, xb, _tt(len(xb), 0, device), yb)
        xt = math.sqrt(ab[t]) * xb + math.sqrt(1 - ab[t]) * e0
        e1 = predict_eps(unet, xt, _tt(len(xb), t, device), yb)
        return -(e0 - e1).abs().pow(p).flatten(1).sum(1)

    return _batched(f, x, y, bs)


@torch.no_grad()
def secmi_scores(unet, x, sched, device, y=None, bs=125, t_sec=SECMI_T, k=SECMI_K):
    """SecMI-stat: invert x0 with DDIM to t_sec - k, go one step up and back, score = -squared L2 gap (Duan2023SecMI Eq. 13).

    Follows the official code (secmia.py get_intermediate_results): the compared point is x at t_sec - k.
    """
    ab = _abar(sched)
    steps = list(range(0, t_sec, k))[1:]
    assert steps, "t_sec must exceed k"

    def f(xb, yb):
        xb = xb.to(device)
        yb = None if yb is None else yb.to(device)
        h, tc = xb, 0
        for tg in steps:
            h = ddim_step(unet, h, tc, tg, ab, yb)
            tc = tg
        up = ddim_step(unet, h, tc, tc + k, ab, yb)
        back = ddim_step(unet, up, tc + k, tc, ab, yb)
        return -((h - back) ** 2).flatten(1).sum(1)

    return _batched(f, x, y, bs)


@torch.no_grad()
def loss_scores(unet, x, noise, sched, device, y=None, bs=125):
    """evaluate.mia_scores with an optional label: score = -mean over (t, draw) of ||eps - eps_theta(x_t, t[, y])||^2.

    y=None calls evaluate.mia_scores itself, so unconditional runs reproduce the pilot's loss attack exactly.
    """
    if y is None:
        return mia_scores(unet, x, noise, sched, device, bs)
    total = torch.zeros(len(x), device=device)
    for ti, t in enumerate(MIA_TS):
        for k in range(MIA_DRAWS):
            for i in range(0, len(x), bs):
                xb, yb = x[i:i + bs], y[i:i + bs].to(device)
                e = noise[ti, k][None].expand_as(xb)
                tt = _tt(len(xb), t, device)
                pred = predict_eps(unet, sched.add_noise(xb, e, tt), tt, yb)
                total[i:i + bs] += ((pred - e) ** 2).flatten(1).sum(1)
    return (-total / (len(MIA_TS) * MIA_DRAWS)).cpu().numpy()
