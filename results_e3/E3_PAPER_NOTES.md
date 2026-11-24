# E3 paper notes: guidance sweep and attacks on class-conditional DDPMs (c1024, c4096)

Facts only. Every number in the tables is copied programmatically from the result files in this folder (`cfg_metrics.json`, `attacks.json`, `log.csv`, `college_status.log`, `config.json`) and was re-checked against them by a separate script. Statements about what the data show are descriptive; no cause is claimed. Results are reported as found, including where they differ from the expectations in PROTOCOL_V2.md.

## a. What E3 was

Source: `experiments/PROTOCOL_V2.md`, section E3, and its Deviations 5 to 7.

- Question: does a larger guidance scale w raise copying of training images?
- Models: class-conditional version of the pilot UNet (PneumoniaMNIST 64x64, label 0 = normal, label 1 = pneumonia; 3 label embeddings, index 2 = null label), label dropout p_uncond = 0.1, N = 1,024 (`runs/c1024`) and N = 4,096 (`runs/c4096`), 30,000 steps, seed 0, other settings as the pilot. `config.json` of both runs: cond = true, amp = fp16, arch = base, p_uncond = 0.1, checkpoints at steps 3750, 7500, ..., 30000.
- Guidance: eps_hat = eps(x, t, null) + w * (eps(x, t, c) - eps(x, t, null)); w = 0 is unconditional, w = 1 is plain conditional.
- Sweep at the final EMA checkpoint (step 30000): w in {0, 1, 2, 3, 5, 7.5}; 1,000 samples per w (500 per class, labels alternate 0, 1); fp32 DDIM, 100 steps, eta 0; the same initial noise for every w; seed 0. `cfg_metrics.json` settings: n_samples 1000, ddim_steps 100, n_boot 1000, seed 0, fid true.
- Metrics per w: copy rate (cosine-similarity threshold of the pilot per N), median nearest-neighbour (NN) similarity, copy rate at 0.95, FID against the 1,140 non-members, copy rate per class. 95% percentile-bootstrap CIs over samples (1,000 resamples). NN search is over all members, not only the same class.
- Label-aware loss MIA: pilot loss attack (pilot timesteps and noise), scored with the true label and with the null label; 1,000 members and 1,000 non-members; AUC and TPR@1%FPR with CIs.
- E4a attacks on both final models, true and null label (settings in `experiments/ATTACKS.md`): PIA (t = 200, p = 4), SecMI-stat (t_SEC = 100, k = 10). The loss attack is re-run in `attacks.json` with ts 25, 50, 100, 150, 200, 300, 4 draws, noise seed 0, same members and non-members.

## b. Results tables

### c1024 (N = 1024)

Copy-rate threshold (stored in `c1024/cfg_metrics.json`, key `copy_threshold`): 0.893604. It is the pilot's per-N threshold (PROTOCOL_V2: "pilot per-N threshold"; `experiments/results/metrics_n1024.json` `copy_threshold` = 0.893604, equal at six decimals). Median NN similarity of non-members to members: 0.8270 (n_nonmembers = 1140).

Table c1024-1. Copy and quality metrics per w (n = 1000 samples per row; CIs are 95% bootstrap).

| w | copy_rate | copy_rate_095 | median_nn | FID |
|---|---|---|---|---|
| 0 | 0.857 [0.833, 0.877] | 0.538 [0.507, 0.567] | 0.9547 [0.9508, 0.9590] | 30.48 |
| 1 | 0.920 [0.903, 0.937] | 0.615 [0.585, 0.647] | 0.9662 [0.9618, 0.9702] | 28.47 |
| 2 | 0.938 [0.921, 0.952] | 0.639 [0.609, 0.667] | 0.9728 [0.9680, 0.9760] | 34.15 |
| 3 | 0.936 [0.919, 0.950] | 0.669 [0.638, 0.697] | 0.9738 [0.9704, 0.9778] | 35.75 |
| 5 | 0.909 [0.891, 0.927] | 0.642 [0.611, 0.670] | 0.9683 [0.9651, 0.9712] | 30.22 |
| 7.5 | 0.609 [0.577, 0.640] | 0.351 [0.324, 0.379] | 0.9144 [0.9086, 0.9222] | 61.96 |

Table c1024-2. Copy rate per class (500 samples per class; class 0 = normal, class 1 = pneumonia).

| w | class 0 copy_rate | class 1 copy_rate |
|---|---|---|
| 0 | 0.874 [0.842, 0.902] | 0.840 [0.808, 0.872] |
| 1 | 0.936 [0.914, 0.956] | 0.904 [0.876, 0.928] |
| 2 | 0.966 [0.950, 0.982] | 0.910 [0.884, 0.934] |
| 3 | 0.960 [0.944, 0.976] | 0.912 [0.888, 0.936] |
| 5 | 0.912 [0.886, 0.936] | 0.906 [0.882, 0.930] |
| 7.5 | 0.322 [0.278, 0.366] | 0.896 [0.868, 0.922] |

### c4096 (N = 4096)

Copy-rate threshold (stored in `c4096/cfg_metrics.json`, key `copy_threshold`): 0.910021. It is the pilot's per-N threshold (PROTOCOL_V2: "pilot per-N threshold"; `experiments/results/metrics_n4096.json` `copy_threshold` = 0.910021, equal at six decimals). Median NN similarity of non-members to members: 0.8483 (n_nonmembers = 1140).

Table c4096-1. Copy and quality metrics per w (n = 1000 samples per row; CIs are 95% bootstrap).

| w | copy_rate | copy_rate_095 | median_nn | FID |
|---|---|---|---|---|
| 0 | 0.035 [0.024, 0.047] | 0.000 [0.000, 0.000] | 0.8607 [0.8581, 0.8628] | 24.99 |
| 1 | 0.061 [0.047, 0.076] | 0.001 [0.000, 0.003] | 0.8672 [0.8647, 0.8691] | 29.29 |
| 2 | 0.077 [0.060, 0.094] | 0.003 [0.000, 0.007] | 0.8726 [0.8702, 0.8748] | 37.09 |
| 3 | 0.082 [0.066, 0.098] | 0.002 [0.000, 0.005] | 0.8750 [0.8728, 0.8768] | 39.94 |
| 5 | 0.027 [0.018, 0.038] | 0.000 [0.000, 0.000] | 0.8525 [0.8493, 0.8548] | 32.72 |
| 7.5 | 0.011 [0.005, 0.018] | 0.000 [0.000, 0.000] | 0.8008 [0.7918, 0.8070] | 40.49 |

Table c4096-2. Copy rate per class (500 samples per class; class 0 = normal, class 1 = pneumonia).

| w | class 0 copy_rate | class 1 copy_rate |
|---|---|---|
| 0 | 0.046 [0.028, 0.066] | 0.024 [0.012, 0.038] |
| 1 | 0.082 [0.060, 0.106] | 0.040 [0.024, 0.056] |
| 2 | 0.114 [0.088, 0.142] | 0.040 [0.024, 0.058] |
| 3 | 0.122 [0.096, 0.148] | 0.042 [0.026, 0.060] |
| 5 | 0.004 [0.000, 0.010] | 0.050 [0.032, 0.068] |
| 7.5 | 0.000 [0.000, 0.000] | 0.022 [0.010, 0.036] |

### Membership-inference attacks (final checkpoint, 1,000 members vs 1,000 non-members, 95% bootstrap CIs)

Table 3. AUC and TPR@1%FPR. Values come from `attacks.json`. The label-aware loss MIA stored in `cfg_metrics.json` (`mia`) has identical values to `attacks.json` loss rows for both runs (checked by script). AUC with more than 0.9999 but below 1 is printed with six decimals.

| run | attack | label | AUC | TPR@1%FPR |
|---|---|---|---|---|
| c1024 | loss | true | 1.0000 [1.0000, 1.0000] | 1.000 [1.000, 1.000] |
| c1024 | loss | null | 1.0000 [1.0000, 1.0000] | 1.000 [1.000, 1.000] |
| c1024 | PIA | true | 0.999996 [0.999982, 1.000000] | 1.000 [1.000, 1.000] |
| c1024 | PIA | null | 0.999994 [0.999976, 1.000000] | 1.000 [1.000, 1.000] |
| c1024 | SecMI-stat | true | 0.9973 [0.9960, 0.9984] | 0.931 [0.885, 0.957] |
| c1024 | SecMI-stat | null | 0.999761 [0.999557, 0.999916] | 0.992 [0.982, 0.999] |
| c4096 | loss | true | 0.8461 [0.8296, 0.8626] | 0.165 [0.102, 0.223] |
| c4096 | loss | null | 0.8344 [0.8167, 0.8513] | 0.153 [0.095, 0.213] |
| c4096 | PIA | true | 0.8888 [0.8755, 0.9025] | 0.204 [0.134, 0.321] |
| c4096 | PIA | null | 0.8743 [0.8596, 0.8886] | 0.211 [0.112, 0.304] |
| c4096 | SecMI-stat | true | 0.7156 [0.6943, 0.7376] | 0.029 [0.017, 0.052] |
| c4096 | SecMI-stat | null | 0.7082 [0.6859, 0.7300] | 0.033 [0.019, 0.049] |

## c. Pilot comparison (descriptive only: different models)

The pilot models are unconditional and were trained with bf16 on the laptop; the E3 models are class-conditional, trained with fp16. Pilot values for copy rate and loss MIA are from `experiments/results/table.csv` (copy rate to three decimals, AUC to three decimals as in `metrics_n*.json`/`table.csv`); they agree with the numbers in HANDOFF_COLLEGE.md section 2 (100% / 91.0% / 4.2% and 1.000 / 1.000 / 0.859). PIA and SecMI values for the pilot N = 4096 model exist only in HANDOFF_COLLEGE.md section 2 (PIA AUC 0.910, SecMI AUC 0.623); `runs/n4096/attacks.json` is on the laptop and is not in this repository, so those two numbers could not be re-checked here.

Table 4. Pilot (unconditional) vs E3 conditional model at w = 0 (E3 at w = 1 and the true-label loss MIA also shown).

| N | pilot copy_rate | E3 copy_rate w=0 | E3 copy_rate w=1 | pilot loss-MIA AUC | E3 loss-MIA AUC (true / null label) |
|---|---|---|---|---|---|
| 1024 | 0.910 | 0.857 | 0.920 | 1.000 | 1.0000 / 1.0000 |
| 4096 | 0.042 | 0.035 | 0.061 | 0.859 | 0.8461 / 0.8344 |

Pilot N = 256 (copy rate 1.000, loss-MIA AUC 1.000) has no E3 counterpart. Pilot N = 4096 attacks from HANDOFF_COLLEGE.md: PIA AUC 0.910, SecMI AUC 0.623; E3 c4096 (true label): PIA 0.8888, SecMI 0.7156.

## d. PROTOCOL_V2 expectations for E3 against the data

### c1024

1. Expected: copy rate rises with w. Found: copy_rate by w = 0.857, 0.920, 0.938, 0.936, 0.909, 0.609. Highest at w = 2; lower at w = 5 (0.909) and w = 7.5 (0.609) than at w = 2. w = 7.5 is below w = 0 (0.857). Rise to w = 2 or 3 matches; fall at w = 5 and 7.5 does not match.
2. Expected: median NN similarity rises with w. Found: median_nn by w = 0.9547, 0.9662, 0.9728, 0.9738, 0.9683, 0.9144. Highest at w = 3; w = 5 0.9683, w = 7.5 0.9144. Matches up to w = 3, not at larger w.
   copy_rate_095 by w = 0.538, 0.615, 0.639, 0.669, 0.642, 0.351.
3. Expected: FID lowest at a moderate w. Found: FID by w = 30.48, 28.47, 34.15, 35.75, 30.22, 61.96. Lowest at w = 1 (28.47); highest at w = 7.5 (61.96).
4. Expected: true-label loss MIA AUC >= null-label AUC. Found: true 1.0000 [1.0000, 1.0000], null 1.0000 [1.0000, 1.0000]; the CIs overlap. Both are 1.0 (ceiling); the comparison carries no information.
5. Per-class copy rate at w = 7.5: class 0 0.322 [0.278, 0.366], class 1 0.896 [0.868, 0.922].
6a. Expected: PIA and SecMI AUC >= loss AUC (true label). Found: loss 1.0000, PIA 0.999996, SecMI 0.9973. PIA is below loss; SecMI is below loss.
6b. Expected: PIA and SecMI AUC >= loss AUC (null label). Found: loss 1.0000, PIA 0.999994, SecMI 0.9998. PIA is below loss; SecMI is below loss.

### c4096

1. Expected: copy rate rises with w. Found: copy_rate by w = 0.035, 0.061, 0.077, 0.082, 0.027, 0.011. Highest at w = 3; lower at w = 5 (0.027) and w = 7.5 (0.011) than at w = 3. w = 7.5 is below w = 0 (0.035). Rise to w = 2 or 3 matches; fall at w = 5 and 7.5 does not match.
2. Expected: median NN similarity rises with w. Found: median_nn by w = 0.8607, 0.8672, 0.8726, 0.8750, 0.8525, 0.8008. Highest at w = 3; w = 5 0.8525, w = 7.5 0.8008. Matches up to w = 3, not at larger w.
   copy_rate_095 by w = 0.000, 0.001, 0.003, 0.002, 0.000, 0.000.
3. Expected: FID lowest at a moderate w. Found: FID by w = 24.99, 29.29, 37.09, 39.94, 32.72, 40.49. Lowest at w = 0 (24.99); highest at w = 7.5 (40.49).
4. Expected: true-label loss MIA AUC >= null-label AUC. Found: true 0.8461 [0.8296, 0.8626], null 0.8344 [0.8167, 0.8513]; the CIs overlap. True is higher by 0.0117 (point estimates).
5. Per-class copy rate at w = 7.5: class 0 0.000 [0.000, 0.000], class 1 0.022 [0.010, 0.036].
6a. Expected: PIA and SecMI AUC >= loss AUC (true label). Found: loss 0.8461, PIA 0.8888, SecMI 0.7156. PIA is at or above loss; SecMI is below loss.
6b. Expected: PIA and SecMI AUC >= loss AUC (null label). Found: loss 0.8344, PIA 0.8743, SecMI 0.7082. PIA is at or above loss; SecMI is below loss.

Additional comparison, true label vs null label on PIA and SecMI (AUC, point estimates): c1024 PIA true 1.0000, null 1.0000; c1024 SecMI true 0.9973, null 0.9998; c4096 PIA true 0.8888, null 0.8743; c4096 SecMI true 0.7156, null 0.7082. On c1024 SecMI the null-label AUC is higher than the true-label AUC; on the other three pairs the true-label AUC is higher.

Not an expectation in the protocol, but visible in the tables: on c4096 the conditional model at w = 0 has copy rate 0.035 and FID 24.99; on c1024 at w = 0 copy rate 0.857 and FID 30.48.

## e. Run conditions and deviations

Hardware and software (PROTOCOL_V2 Deviation 5): NVIDIA GeForce RTX 2060 (6 GB, Turing, compute capability 7.5), Intel i7-10700K, 32 GB RAM, Windows, Python 3.13.15, torch 2.11.0+cu128. One job at a time (`--parallel 1`); blocks run separately (`--blocks e3`). The GPU has no native bf16 support.

Training precision (Deviation 6): fp16 autocast with a dynamic `torch.amp.GradScaler` (`--amp fp16`), instead of the pilot's bf16 autocast. Reason in the protocol: the first c1024 run in bf16 (emulated on this GPU) trained at 0.02 it/s. Sampling, label-aware MIA, PIA, SecMI and evaluation of the final EMA checkpoint run in fp32.

c1024 first bf16 attempt: `c1024/log.csv` row 1 is a stale bf16 row (step 50, loss 0.122437, 0.02 it/s, elapsed 2460.8 s). That run (started 2026-10-27 19:51:29 per `college_status.log`) was stopped at step 50 with no checkpoint; the fp16 run started 2026-10-27 21:23:51 and appends from step 0. The stale row is part of the file; it is not removed.

Sampling batch size (Deviation 7): `eval_cfg.py` batch 250 (default) for c1024 w = 0, 1, 2, 3; batch 125 (`--bs 125`) for c1024 w = 5, 7.5 and for all c4096 w. The batch size is not part of the settings check in `eval_cfg.py`. Label-aware MIA batch is fixed at 125. Per-w seconds and sampling peak memory, copied from `logs/e3_cfg_c1024.log` and `logs/e3_cfg_c4096.log` (seconds also appear in `cfg_metrics.json`):

Table 5. Sampling time and peak.

| run | w | batch | seconds | sampling peak GB |
|---|---|---|---|---|
| c1024 | 0 | 250 | 1156.8 | 5.72 |
| c1024 | 1 | 250 | 1100.0 | 5.95 |
| c1024 | 2 | 250 | 4535.9 | 11.75 |
| c1024 | 3 | 250 | 4706.6 | 11.52 |
| c1024 | 5 | 125 | 2188.0 | 5.72 |
| c1024 | 7.5 | 125 | 2268.6 | 5.95 |
| c4096 | 0 | 125 | 364.5 | 2.99 |
| c4096 | 1 | 125 | 359.3 | 3.22 |
| c4096 | 2 | 125 | 2275.1 | 6.01 |
| c4096 | 3 | 125 | 2481.9 | 6.24 |
| c4096 | 5 | 125 | 2554.2 | 6.47 |
| c4096 | 7.5 | 125 | 2531.7 | 6.24 |

Several logged peaks exceed the 6 GB of the card (largest: 11.75 and 11.52 GB for c1024 w = 2 and 3; 6.01 to 6.47 GB for c4096 w = 2 to 7.5); they are reported as logged. For w = 0 and w = 1 on c4096 the seconds are about 360, for w = 2 to 7.5 between 2275 and 2554, at the same batch of 125.

Interruptions (from `college_status.log` and `logs/e3_train_c1024.log`; the causes below are as told by the operator, they are not written in the logs):

- c1024 training, 2026-10-27: accidental window close at about 22:36; restarted 22:38:12; log line "resumed from step 3000". Steps 3050 to 3350 appear twice in `c1024/log.csv` (rows from before and after the resume).
- c1024 training, 2026-10-28: power cut; `college_status.log` records INTERRUPTED at 10:02:23, restart 10:07:20; log line "resumed from step 27000".
- c1024 guidance sweep: planned stop and restart at 2026-10-28 14:37:30 to change the sampling batch from 250 to 125 (Deviation 7). w = 0, 1, 2, 3 were kept, w = 5, 7.5 sampled after.

Job start and finish times (`college_status.log`; the `seconds` value is for the last launch of that job only):

| job | last START | DONE | seconds (as logged) |
|---|---|---|---|
| e3_train_c1024 | 2026-10-28 10:07:20 | 2026-10-28 11:24:26 | 4625 |
| e3_cfg_c1024 | 2026-10-28 14:37:30 | 2026-10-28 15:57:43 | 4812 |
| e3_attacks_c1024 | 2026-10-28 15:57:43 | 2026-10-28 16:06:39 | 535 |
| e3_train_c4096 | 2026-10-28 16:06:39 | 2026-10-29 03:57:08 | 42629 |
| e3_cfg_c4096 | 2026-10-29 03:57:08 | 2026-10-29 07:03:13 | 11164 |
| e3_attacks_c4096 | 2026-10-29 07:03:13 | 2026-10-29 07:12:16 | 543 |

Final line: `2026-10-29 07:12:16 FINISHED done=6 failed=0 blocked=0 skipped=0`. Other START lines for e3_train_c1024: 2026-10-27 19:51:29 (bf16 attempt), 21:23:51 (fp16), 22:38:12 (resume); e3_cfg_c1024: 2026-10-28 11:24:26 (first launch, batch 250).

Training speed and loss from `log.csv` (rows every 50 steps; fp16 rows only, i.e. excluding the stale bf16 row of c1024; the rows repeated after the c1024 resume are included as logged):

| run | rows used | mean it/s | min it/s | max it/s | final-loss mean (last 10 rows) | last row (step, loss) |
|---|---|---|---|---|---|---|
| c1024 | 607 | 0.613 | 0.54 | 0.79 | 0.00653 | 30000, 0.006916 |
| c4096 | 600 | 0.712 | 0.56 | 0.78 | 0.01349 | 30000, 0.013732 |

Final training loss is the arithmetic mean of the `loss` column over the last 10 log rows (steps 29550 to 30000). `loss` is the value in the `loss` column of `log.csv` at those rows.

## f. What was NOT run here

- E4a (PIA, SecMI-stat) on the pilot N = 256 and N = 1,024 models: not run; their checkpoints are not on this machine.
- E4a on the pilot N = 4,096 model: run earlier on the laptop (PROTOCOL_V2 Deviation 2); its result file is not in this repository.
- E4b (training seeds 1 and 2 for N = 1,024 and 4,096): not run.
- E5 (DP-SGD): not run (PROTOCOL_V2 Deviation 1).
- No result in this folder comes from more than one training seed (seed 0 only).

## g. File map

Folder `experiments/results_e3/`:

- `E3_PAPER_NOTES.md`: this file.
- `REPORT.txt`: copy of `runs/REPORT.txt`, the headline block printed by `run_college.py report`.
- `college_status.log`: job queue log with START and DONE times.
- `logs/e3_train_*.log`, `logs/e3_cfg_*.log`, `logs/e3_attacks_*.log`: console output of each of the six jobs.
- `c1024/`, `c4096/`: `cfg_metrics.json` (copy metrics per w, per class, label-aware loss MIA, threshold), `attacks.json` (loss, PIA, SecMI, true and null label), `figcfg.json` (per w: sample index, training-image index, similarity and label of the top-ranked sample/training pairs, for the figure), `config.json` (training config), `log.csv` (step, loss, it_per_s, elapsed_s), `members.npy` (indices of the training images).

Archive `experiments/results_college.zip` (committed earlier, baf7e0a) additionally contains the following; no checkpoints (`.pt`) are in it or in this folder:

- `results_college/runs/c1024/` and `c4096/`: the same JSON, CSV and `members.npy` files, plus `samples_cfg.npz` (generated samples, 24,652,486 bytes) and `figcfg.npz` (1,577,710 bytes).
- `results_college/logs/`: the six job logs. `results_college/college_status.log`.

