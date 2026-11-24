# Protocol v2: E3 (guidance scale), E4 (attacks, seeds), E5 (DP-SGD)

Written 2026-10-27, before any E3, E4 or E5 model was trained or any new attack was run
on a real checkpoint. Everything not stated here is as in PROTOCOL.md (data, non-members,
UNet, optimizer, EMA, copy detection, loss MIA, FID, bootstrap). Changes after results exist
go in "Deviations" at the bottom, with a reason.

Hardware: college PC, NVIDIA RTX PRO 4000 Blackwell (24 GB). Which blocks run is decided
from the benchmark (`bench.py`) BEFORE any of them starts, and recorded under Deviations.
Dropping a block for time is not a result and is reported as "not run".

## Shared conventions

- Class labels: PneumoniaMNIST label 0 = normal, 1 = pneumonia. Conditional models embed
  num_classes + 1 = 3 labels; index 2 is the null label used for unconditional prediction.
- Guidance follows the paper's Eq. (cfg):
  eps_hat = eps(x, t, null) + w * (eps(x, t, c) - eps(x, t, null)).
  So w = 0 is unconditional, w = 1 is plain conditional, w > 1 is guided.
- Members for every model are the pilot's nested subsets (seed-0 permutation). Training
  seed changes only initialization, batch order, timesteps and noise.
- MIA member/non-member selection is the pilot's (`evaluate.py`: min(N, 1000) each, seed 0).

## E3. Guidance scale as a privacy parameter

Question: does a larger guidance scale w raise copying of training images?
(Paper_v9 says no study varies w as a controlled privacy parameter.)

- Models: class-conditional version of the pilot UNet, trained with label dropout
  p_uncond = 0.1 (classifier-free guidance, HoSalimans2022CFG). N = 1,024 and N = 4,096,
  30,000 steps, seed 0, all other settings as the pilot. Runs: `runs/c1024`, `runs/c4096`.
- Sweep at the final EMA checkpoint, fp32 DDIM 100 steps, eta 0:
  w in {0, 1, 2, 3, 5, 7.5}. 1,000 samples per w, 500 per class (labels alternate 0, 1).
  The same initial noise is used for every w (paired comparison).
- Metrics per w: copy rate (pilot per-N threshold), median NN similarity, copy rate at 0.95,
  FID vs the 1,140 non-members, copy rate per class. 95% bootstrap CIs over samples.
  Nearest neighbours are searched over all members, not only the same class.
- Label-aware attack: loss MIA (pilot timesteps and noise) scored with the true label vs
  with the null label. AUC and TPR@1%FPR with CIs.
- Comparison to the pilot: conditional model at w = 0 vs pilot unconditional model at the
  same N (different models, so descriptive only).
- Expected: copy rate and median NN similarity rise with w; FID is lowest at a moderate w;
  true-label MIA AUC >= null-label AUC. Results against this are reported as found.

## E4. Robustness of the pilot

E4a. Two further attacks, both from the corpus, threshold (statistic) versions only:
- PIA (Kong2024PIA) and SecMI-stat (Duan2023SecMI). Hyperparameters (attack timestep,
  norm, DDIM interval) follow the authors' papers / official code for DDPM; the values used
  and where they were taken from are recorded in `ATTACKS.md` before any real run.
  SecMI-NN (which trains a classifier) is not run.
- Run on every final checkpoint: pilot n256/n1024/n4096, E4b seeds, E3 models (true label
  and null label), E5 models. AUC and TPR@1%FPR with bootstrap CIs, same member/non-member
  sets as the loss attack.
- Expected: PIA and SecMI AUC >= loss-attack AUC on the same model.

E4b. Seeds: training seeds 1 and 2 for N = 1,024 and N = 4,096 (runs `n1024_s1`,
`n1024_s2`, `n4096_s1`, `n4096_s2`), evaluated at the final checkpoint only (fp32), with the
same evaluation seed as the pilot. Reported as mean and range over seeds 0 to 2.
- Expected: same ordering as the pilot for every seed.

## E5. DP-SGD training

Question: how much does DP-SGD (Abadi2016DPSGD) reduce the attacks and copying, and what
does it cost in FID? Reference for DP diffusion training: Dockhorn2023DPDM.

- Data: N = 4,096 member subset. delta = 1e-5. Target epsilon in {1, 10} (RDP accountant
  in Opacus; the achieved epsilon is reported).
- Training: Poisson sampling with expected batch 512, per-sample clipping norm 1.0,
  AdamW lr 3e-4, EMA 0.999 (post-processing, no privacy cost), one (t, noise) draw per
  example per step, bf16 autocast if Opacus allows it (else fp32, recorded).
- Non-DP twin: same model, batch 512, same number of steps and lr, no clipping or noise.
  The twin, not the pilot model, is the comparison for E5.
- Feasibility gate (`dp_gate.py`, run first at college, about 30 min):
  1. Opacus wraps the model and per-sample gradients match a manual per-example
     computation on a tiny batch.
  2. Measured throughput gives a projected time per run.
  Epochs E are the largest of {50, 100, 200, 400} whose projected DP run time is <= 2.5 h
  with the pilot UNet; if 50 does not fit, the small UNet (channels 32, 64, 128, 128) is
  used for both DP runs and the twin; if that does not fit either, E5 is not run.
  The same E is used for both epsilons and the twin. The choice is recorded under
  Deviations before training.
- Evaluation: final checkpoint, `evaluate.py --ckpt last` (copy rate, loss MIA, FID) and
  E4a attacks.
- Scope note for the paper: Chung2025SoK's AUC above 0.95 at epsilon = 10 used a
  classifier-release attack on a latent diffusion model. E5 neither confirms nor refutes it.
- Expected: attack AUC moves toward 0.5 as epsilon falls; FID worsens.

## Deviations

1. **2026-10-27, before any E3/E4b/E5 run.** E5 (DP-SGD) is not run: its code was not
   finished in time and the author chose to go without it. Reported as "not run".
2. **2026-10-27, before any E3/E4b run.** E4a on the pilot N = 4,096 model was run on the
   laptop (settings as in ATTACKS.md, committed before the run); result in
   `runs/n4096/attacks.json` on the laptop. Pilot N = 256 and 1,024 attacks are pending
   (their checkpoints are only on the laptop).
3. **2026-10-27.** The separate `bench` step is optional on the college PC; E3 starts first
   and its measured training speed (status / log.csv) sets the plan for E4b.
4. **2026-10-27, before any E3/E4b run.** The college PC has an NVIDIA RTX PRO 2000
   Blackwell (16 GB), not the RTX PRO 4000 (24 GB) stated above. No setting changes (batch
   size is fixed in the code); only speed and the number of parallel jobs (2) are affected.
   The Python environment uses Python 3.13.9 (Anaconda base on this PC) because the pinned
   requirements need Python 3.12 or later and only 3.11 and 3.14 were otherwise installed.
   E3 and E4b are queued in one run (`--blocks e3,e4b --parallel 2`); the queue's fixed
   priority starts E4b only when an E3 slot is free.
5. **2026-10-27, before any E3/E4b run.** The runs are made on a different machine from the
   one in Deviation 4: an office PC with an NVIDIA GeForce RTX 2060 (6 GB, Turing, compute
   capability 7.5), Intel i7-10700K, 32 GB RAM, Windows. Python 3.13.15, torch 2.11.0+cu128.
   To stay within the 6 GB of GPU memory, one job runs at a time (`--parallel 1`) instead of
   the single `--blocks e3,e4b --parallel 2` run in Deviation 4, and the blocks are run
   separately: `--blocks e3` first, then `--blocks e4b`. No setting is changed (batch size and
   precision as in the code, steps as in the protocol). This GPU has no native bf16 support
   (`torch.cuda.is_bf16_supported(including_emulation=False)` is False), so the pilot's bf16
   autocast runs emulated on it; a micro-benchmark on this card measured bf16 slower than
   fp32. Training precision is kept as bf16 for now; any change will be logged as a separate
   deviation.
6. **2026-10-27, before any E3/E4b result.** Training precision on this machine is fp16
   autocast with a dynamic loss scaler (`torch.amp.GradScaler`) instead of the pilot's bf16
   autocast; set with the new `--amp` option of `train.py` (default bf16, so the pilot
   setting is unchanged) and passed by `run_college.py run --amp fp16` for E3 and E4b.
   Reason: the first E3 run (c1024, bf16) trained at 0.02 it/s (50 steps in 2,461 s), about
   410 h per 30,000-step run. A diagnostic with the conditional UNet and the training step
   of `train.py` at batch 64 measured 0.023 it/s in bf16 (emulated on this GPU), 0.254 it/s
   in fp32 and 0.772 it/s in fp16 with the scaler (about 10.8 h per run); peak reserved
   memory in every mode exceeded the 6 GB card. That run was stopped at step 50 with no
   checkpoint; its `runs/c1024/log.csv` keeps the bf16 row for step 50 and the fp16 run
   appends to it from step 0. No E3 or E4b result existed. Sampling, attacks and evaluation
   of the final EMA checkpoint run in fp32 as before (`run_college.py` calls `evaluate.py` with
   `--ckpt last`; `eval_cfg.py` and `eval_attacks.py` use no autocast). The bf16 sampling that
   `evaluate.py` applies to intermediate checkpoints is not used. Consequence: the E4b seeds
   (fp16) are compared with the pilot seed 0 (bf16), so seed spread also includes this
   precision change.
7. **2026-10-28, during the E3 guidance sweep of c1024.** The sampling batch size of
   `eval_cfg.py` is set to 125 (`run_college.py run --cfg-bs 125`) instead of its default 250.
   Reason: runtime only. For w > 1 the conditional and null predictions are evaluated as one
   batch of twice the batch size; at 250 that peaked at 11.75 GB on the 6 GB card and w = 2
   took 4,536 s, against 1,157 s for w = 0 and 1,100 s for w = 1. The c1024 results for w =
   0, 1, 2 and 3 were sampled with batch 250 (w = 3 was already running when the run was
   stopped to change the setting); they are kept, not recomputed, and w = 5 and 7.5 are
   sampled with batch 125 after the restart. The settings check in `eval_cfg.py` does not
   include the batch size, so the resumed run skips finished w values. The w = 2 numbers
   were visible in the run status line, but the decision used only the timings and memory.
   The initial noise of all samples is drawn once from seed 0 for the whole sweep and the
   labels follow the sample index, DDIM is deterministic (eta 0) and in fp32, so the batch
   size changes the samples only through floating-point differences between GPU kernels. The
   label-aware MIA keeps its fixed batch of 125. All c4096 sampling uses batch 125.
