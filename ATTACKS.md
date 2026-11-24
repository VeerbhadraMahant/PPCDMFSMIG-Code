# E4a attacks: PIA and SecMI-stat (hyperparameters and sources)

Written 2026-10-27, before any real run (PROTOCOL_V2.md, E4a). Code: `attacks.py`,
`eval_attacks.py`. Every value below was read on 2026-10-27 from the paper or the official
code; "paper" means the arXiv PDF, "code" means the file at the URL on the `main` branch.
Anything not found is marked NOT VERIFIED.

Sources:
- PIA paper: Kong et al., ICLR 2024, https://arxiv.org/abs/2305.18355 (v2, 9 Oct 2023).
- PIA code: https://github.com/kong13661/PIA (`DDPM/attack.py`, `DDPM/components.py`).
- SecMI paper: Duan et al., ICML 2023, https://arxiv.org/abs/2302.01316.
- SecMI code: https://github.com/jinhaoduan/SecMI (`mia_evals/secmia.py`, `README.md`).

## Shared setting (both official DDPM codebases)

- T = 1000, linear betas 1e-4 to 0.02 (PIA `attack.py` get_FLAGS: T, beta_1, beta_T;
  SecMI `secmia.py` get_FLAGS). Identical to our `make_ddpm()`; `alphas_cumprod[t]` is the
  0-based cumulative product, which both codes index directly with the integer timestep that
  they also pass to the model. We do the same through `predict_eps`.
- Epsilon prediction, EMA weights (both codes load `ema_model`). We use the EMA checkpoint.
- Images scaled to [-1, 1] (PIA `attack.py`: `lambda x: x * 2 - 1`; SecMI `secmia.py`: `x * 2 - 1`).
  Same as `to_tensor`.
- No random noise anywhere in either attack. Both are deterministic given the model.
- Models in the papers: 32x32 CIFAR-10 etc., trained about 800k steps, a different UNet. Ours: 64x64
  grayscale, 30k steps. The timesteps are kept as published (no re-tuning on our data), because
  choosing them on our members and non-members would leak the answer. They may not be optimal for
  our model. This is a limitation to state in the paper.

## PIA (Kong2024PIA)

| Item | Value | Where |
|---|---|---|
| Attack timestep t | 200 | paper App. A.1 ("On DDPM ... For our proposed method, we set t = 200"); Sec. 4.3 "Impact of t": for DDPM, AUC and TPR rise rapidly then decline around t = 200 (Fig. 2b/2c) |
| Norm | l_p with p = 4, score = sum over pixels of abs(difference)^4 (no root) | code `components.py` class `PIA.__init__` default `lp=4`, `PIA.distance`; `attack.py` does not override it. Paper Eq. (8)/(9) define the l_p norm; App. A.1 says l_4 for GradTTS and Stable Diffusion; Fig. 3b ablates p = 1..7 for DDPM |
| Proximal initial noise | eps0 = eps_theta(x0, t = 0), the model output on the clean image at timestep 0 | paper Eq. (7), Sec. 3.2; code `PIA.ddim_reverse`: `self.eps_getter(x0, condition, noise_level, 0)` |
| Noised point | x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps0 | paper Eq. (9); code `PIA.ddim_denoise`: `get_xt(x0, step, eps)` |
| Statistic | R = || eps_theta(x0, 0) - eps_theta(x_t, t) ||_p, small = member | paper Eq. (9), Eq. (11); code `PIA.ddim_denoise` then `PIA.distance` |
| Queries | 2 per image | paper Sec. 1 ("1+1"), Table 2 |

Our score: `-sum(abs(eps0 - eps_t)^4)` at t = 200 (higher = member).

- The official `attack.py` runs timesteps interval x (1..attack_num) (defaults interval 10,
  attack_num 30, so t = 10..300) and prints AUC and TPR per step, with the lists sorted best-first.
  That reports the best timestep chosen with the labels. We do not do this: one fixed t = 200.
- Rank metrics are unaffected by the official per-step max-normalisation, so it is not applied.
- PARTLY VERIFIED: the paper text states p = 4 only for GradTTS and Stable Diffusion. For DDPM the
  value 4 comes from the code default plus the Fig. 3b ablation, not from a sentence in the paper.
  I did not read Fig. 3b closely enough to say which p is best there.
- PIAN (normalised eps0, Eq. (10)) is not implemented. PIA is the primary attack in this study.

## SecMI-stat (Duan2023SecMI)

| Item | Value | Where |
|---|---|---|
| t_SEC | 100 | paper Sec. 5.1 ("We set t_SEC to 100 for all the experiments"); code `secmia.py` `--t_sec` default 100; README example `--t_sec 100` |
| DDIM interval k | 10 | paper Sec. 5.1 ("DDIM (10) denoising steps when T = 1000"); code `--k` default 10; README example `--k 10` |
| Deterministic forward (inversion) phi | x_{t+1} = sqrt(abar_{t+1}) f(x_t, t) + sqrt(1 - abar_{t+1}) eps(x_t, t), with f = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t); stepped with stride k | paper Eq. (9), (11), (12); code `ddim_singlestep` (pred_x_0, then x_t_target) |
| Deterministic reverse (denoise) psi | same map pointing down: x_{t-1} = sqrt(abar_{t-1}) f(x_t, t) + sqrt(1 - abar_{t-1}) eps(x_t, t) | paper Eq. (10), (12); code `ddim_singlestep` with a smaller t_target |
| t-error | squared L2 distance, summed over pixels | paper Eq. (13) (squared norm); code `naive_statistic_attack`: `((diffusion - sample) ** 2).flatten(1).sum(-1)` |
| Decision | member if t-error <= tau (small = member) | paper Eq. (15); code `roc()` counts member when score <= threshold |
| Queries | 10 + 2 | paper Table 2 |

Procedure in the official code (`secmia.py`, `get_intermediate_results`), which we copy:
1. `target_steps = range(0, t_sec, k)[1:]` = 10, 20, ..., 90.
2. Start at x0 with timestep index 0 and invert: 0 -> 10 -> ... -> 90 (each step queries the model at
   the current timestep and moves to the next), giving x_sec at index 90.
3. One more step up (90 -> 100), then one step down (100 -> 90), giving x_sec_recon.
4. Score = squared L2 distance between x_sec and x_sec_recon. Ours is its negative.

Notes:
- The code compares at index t_sec - k = 90 (one round trip across the interval ending at t_sec),
  while paper Eq. (13) is written at t. We follow the code. PIA's own re-implementation of SecMI
  (`components.py` `SecMIAttacker`) is consistent with the code reading: for step t it compares
  x_{t-k} with the reverse step taken from x_t.
- No clipping of the predicted x0 in either direction (the official code does not clip by default).
- The official CLI declares `--t_sec` and `--k` as `type=bool`, so passing `--t_sec 100` explicitly
  parses to True (= 1) and breaks the run; only the defaults (100 and 10) behave as documented. We
  use the documented values 100 and 10 and do not depend on the CLI.
- The official TPR@1%FPR picks the closest of about 2000 thresholds. We use `evaluate.roc_metrics`
  (linear interpolation) so the numbers are comparable with the pilot loss attack. AUC is the same
  quantity.
- Paper Fig. 5: AUC is flat for 50 <= t_SEC <= 150 on CIFAR-10, CIFAR-100, STL10-U and Tiny-IN.
- SecMI-NN (trained classifier) is not run, by protocol.

## Our adaptations

- Model call: `common.predict_eps(unet, x_t, t, y)`. For conditional models every call in an attack
  uses the same label y (true or null, per `--label-mode`), including the t = 0 call of PIA and every
  inversion and reverse step of SecMI. The official codes have no labels. NOT VERIFIED anywhere in
  the papers: how a label-conditioned model should be attacked; this is our choice.
- 64x64 grayscale and the 1000-step schedule need no change to the formulas. Values are
  sums over all 4096 pixels (the papers sum over 3x32x32).
- fp32 throughout, `torch.no_grad`, batch size only affects memory.
- The loss attack for comparison is the pilot's (`evaluate.mia_scores`, `MIA_TS`, `MIA_DRAWS`, noise
  seed 0); for conditional runs the same code with a label passed to `predict_eps`.

## Evaluation (`eval_attacks.py`)

- Last EMA checkpoint (largest step; step set checked against `config.json`).
- Member and non-member sets copied from `evaluate.py` (RandomState(0), min(n, 1000) each).
- AUC and TPR@1%FPR from `evaluate.roc_metrics`, members positive. 95% CIs by percentile bootstrap, each
  group resampled with replacement, `n_boot` times, as in `evaluate.py`. The bootstrap generator is
  `RandomState(step)`, started fresh for every attack and label mode. `evaluate.py` shares one generator with
  its copy-rate bootstrap, so the loss-attack CI here matches its method but not its exact random draws.
- `attacks.json` holds the settings and wall time of every attack / label mode; an entry is skipped when
  its stored settings are unchanged (`--force` recomputes).
