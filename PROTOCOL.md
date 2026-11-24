# Pilot replication protocol (fixed before any training run)

Written 2026-10-26, before any model was trained. Changes after results exist
must be logged in the "Deviations" section at the bottom with a reason.

## Purpose

Replicate two findings already discussed in the survey, on a small public
medical dataset, with hardware available to the author (RTX 4060 Laptop, 8 GB):

- **E1.** Memorization (copying of training images) in a diffusion model rises
  as the training set shrinks (Somepalli2023Diffusion, Carlini2023Extracting,
  Dar2024Memorize, Akbar2023Beware).
- **E2.** A loss-based membership-inference attack gets stronger as the
  training set shrinks and as training continues, while FID does not warn of
  it (Matsumoto2023MembershipDiffusion, Carlini2023Extracting; FID point from
  Dar2024Memorize).

This is a replication on one dataset with one seed per setting. It is not a
new method and claims nothing beyond this setup.

## Data

- PneumoniaMNIST, 64x64 grayscale (MedMNIST v2 / MedMNIST+, `medmnist` package,
  `size=64`). Source images: pediatric chest X-rays (Kermany et al. 2018).
  Licence CC BY 4.0. Cite MedMNIST v2 and the original source.
- Pixels scaled to [-1, 1]. No augmentation (horizontal flips change anatomy,
  and augmentation is a known memorization reducer, so it would confound E1).
- **Members.** Nested random subsets of the official train split, seed 0:
  N in {256, 1024, 4096}, with S256 ⊂ S1024 ⊂ S4096.
- **Non-members.** Official val + test splits (524 + 624 = 1148 images). No
  model ever trains on them.

## Model and training (identical across N)

- Pixel-space DDPM, epsilon prediction, 1000 steps, linear beta schedule
  (1e-4 to 0.02). `diffusers.UNet2DModel`, 1 in/out channel,
  block_out_channels (64, 128, 256, 256), 2 layers per block, attention at
  the two lowest resolutions.
- AdamW, lr 1e-4, batch 64, bf16 autocast, grad clip 1.0, EMA decay 0.999.
  All evaluation uses the EMA weights.
- Same number of optimizer steps for every N (fixed compute, so smaller N means
  more epochs, as in the cited studies). Step count is set once from the smoke
  test so all three runs fit overnight, and recorded here before the runs start.
- 8 evenly spaced checkpoints per run (for E2).
- Seed 0 for everything. One run per N.

## Evaluation (same code for every checkpoint)

**Sampling.** 1000 samples per checkpoint, DDIM 100 steps, eta 0, fixed seed.

**Copy detection (E1, primary).**
- Similarity = Pearson correlation between flattened 64x64 images (each image
  zero-mean, unit-norm; similarity = dot product). Pixel correlation follows
  Akbar2023Beware.
- For each generated sample: max similarity to the model's training set (NN sim).
- Threshold, per model: the 99th percentile of the NN sim of the 1148
  non-member real images to that model's training set ("more similar to a
  training image than 99% of genuinely unseen real X-rays are").
- **Primary metric:** copy rate = fraction of samples with NN sim above that
  threshold. Secondary: median NN sim; copy rate at a fixed threshold of 0.95.
- 95% bootstrap CI (1000 resamples of the generated samples).

**Membership inference (E2, primary).**
- Loss attack: score(x) = -mean over t in {25, 50, 100, 150, 200, 300} and 4
  fixed noise draws per t of ||eps - eps_theta(x_t, t)||^2. Same noise for every
  image.
- Members: min(N, 1000) images from the training set. Non-members: the same
  number drawn from the 1148, seed 0.
- Report ROC AUC and TPR at 1% FPR, with 95% bootstrap CIs. At n = 256, 1% FPR
  is about 2.6 non-members, so TPR@1%FPR is coarse there; say so.

**Fidelity.** FID (torchmetrics, Inception 2048-d) between the 1000 samples and
the 1148 non-member images, grayscale repeated to 3 channels. FID at n = 1000
is biased; compared only within this study, never against published FIDs.

## Expected results (stated in advance)

- E1: copy rate ordered N=256 > N=1024 > N=4096.
- E2: AUC ordered the same way, and rising over checkpoints for small N, while
  FID stays flat or improves.
- A result against these expectations is reported as found.

## Outputs for the paper

- Table: per N at the final checkpoint: copy rate (CI), median NN sim, MIA AUC
  (CI), TPR@1%FPR, FID.
- Figure A: copy rate, AUC and FID over checkpoints, one line per N.
- Figure B: generated samples next to their nearest training image.

## Deviations

1. **Pre-run, 2026-10-26 (no model trained yet).** 8 of the 1148 val+test
   images are byte-identical to images in the train split, so they would be
   members labelled as non-members. They are excluded: non-members = 1140
   (all "1148" counts above become 1140). Exact duplicates inside the train
   split (18) and inside val+test (3) are left as they are, since duplication
   is part of the data the model sees; they are reported, not removed.
   Near-duplicates (not byte-identical) were not searched for.
2. **Pre-run: step count.** Smoke test on the real model: 2.4 it/s at batch 64
   (bf16; fp16 2.5, fp32 0.9, channels_last 2.3). Steps fixed at **30,000** per
   run (about 3.5 h each): 7,500 epochs at N=256, 1,875 at N=1024, 469 at
   N=4096. Checkpoints at 3,750, 7,500, ..., 30,000.
3. **Pre-run: implementation details the spec left open** (from code review):
   training batches are drawn with replacement, not by epoch shuffling;
   training uses bf16 autocast, evaluation runs in fp32; DDIM uses diffusers
   defaults (`clip_sample=True`, "leading" timestep spacing); AdamW weight
   decay is the default 0.01.
4. **Pre-run: added secondary analysis.** MedMNIST val was split from the
   source training set (possible same-patient images as train), test is the
   source test set. In addition to the primary pooled result, MIA AUC and
   TPR@1%FPR are also reported with val-origin and test-origin non-members
   separately, plus each group's mean NN sim to the members. The primary
   metric is unchanged.
5. **Pre-run: figure B selection.** Figure B shows the 8 highest-similarity
   pairs at the smallest N, selected, not random; its caption must say so.
6. **During training of N=256 (step ~2,800), before any evaluation ran.**
   Measured eval cost: fp32 DDIM sampling takes about 19 min per checkpoint
   (7.7 h for all 24), bf16 sampling about 5.6 min. To finish sooner, the 7
   intermediate checkpoints per run (Figure A trend lines only) are sampled
   with bf16 autocast; the final checkpoint, which gives every number in the
   results table, stays fp32 as specified. The MIA loss and FID always run in
   fp32. A check on a barely trained model gave identical MIA AUC (0.559) and
   median NN sim 0.219 vs 0.218 between the two sampling precisions.
