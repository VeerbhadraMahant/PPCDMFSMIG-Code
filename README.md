# Memorization and membership inference in a small DDPM on PneumoniaMNIST (pilot)

Code, protocol and result files for a small pilot experiment that accompanies a survey paper
(survey under review). The pilot trains a pixel-space DDPM on PneumoniaMNIST and measures
copying of training images and membership inference as the training set shrinks. It is a
replication on one small public dataset, not a new method.

## What the code does

- `train.py`: trains a DDPM (diffusers `UNet2DModel`, epsilon prediction, 1000 steps, linear betas) on a nested random
  subset of N training images (N = 256, 1024, 4096). `--cond` trains a class-conditional model with label dropout
  0.1 (classifier-free guidance, experiment E3). `--seed` changes only initialisation, batch order and noise.
- `evaluate.py`: for each checkpoint, DDIM sampling, copy detection (Pearson correlation to the nearest training
  image, per-model threshold from non-members), loss-based membership inference (AUC, TPR at 1% FPR) and FID.
- `eval_cfg.py`: E3, guidance-scale sweep w in {0, 1, 2, 3, 5, 7.5} on the class-conditional models, plus a
  label-aware loss attack (true label vs null label).
- `attacks.py`, `eval_attacks.py`: PIA and SecMI-stat attacks (threshold versions), with hyperparameters and their
  sources in `ATTACKS.md`.
- `dp_gate.py`, `train_dp.py`: DP-SGD feasibility gate and training with Opacus (E5). E5 was not run (see
  `PROTOCOL_V2.md`, Deviation 1); there are no E5 results in this repository.
- `run_college.py`: job queue that runs the E3, E4a, E4b and E5 blocks with retries and resume.
- `make_figures.py`, `plot_e3.py`: results table, Figure A (copy rate, AUC, FID over checkpoints), Figure B (samples
  next to nearest training image) and the E3 copy-rate-versus-w figure.
- `selftest.py`, `selftest_attacks.py`, `selftest_dp.py`, `selftest_e3.py`: self-checks on tiny fake data.
- `PROTOCOL.md` (pilot protocol, fixed before training, with its deviation log), `PROTOCOL_V2.md` (E3, E4, E5 protocol
  and deviation log), `ATTACKS.md` (attack settings and sources). These are the reproducibility record; read them
  before the code.

## Hardware used (as recorded in the protocol docs)

- Pilot (N = 256, 1024, 4096, unconditional): laptop with an NVIDIA RTX 4060 Laptop GPU, 8 GB.
- E3 (class-conditional models c1024, c4096): office PC with an NVIDIA GeForce RTX 2060 (6 GB), Intel i7-10700K,
  32 GB RAM, Windows, Python 3.13.15, torch 2.11.0+cu128 (`PROTOCOL_V2.md`, Deviation 5). Training there used fp16
  autocast with a loss scaler instead of the pilot's bf16 (Deviation 6).
- The Python version of the pilot machine is not recorded in the docs.

## Install

Python 3.11 to 3.13 (the setup scripts check this). A CUDA GPU is needed for the real runs; the self-tests also work
on CPU.

    python -m venv .venv
    # Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

`requirements.txt` is a full freeze of the environment used (it includes the CUDA 12.8 builds of torch and
torchvision). Alternatively `setup_college.ps1` (Windows) or `setup_college.sh` (Linux) builds a virtual environment
in this folder, installs the packages, downloads the dataset and the FID Inception weights, and runs
`python run_college.py check`.

## Data

PneumoniaMNIST from MedMNIST v2 / MedMNIST+ (pediatric chest X-rays, Kermany et al. 2018), 64x64 grayscale, licence
CC BY 4.0. The first call that needs it downloads it with the `medmnist` package (`size=64`) into `data/`
(`common.load_data`). To fetch it explicitly:

    python -c "from common import load_data; load_data()"

Please cite MedMNIST v2 and the original source of the images. The dataset and all checkpoints are not in this
repository (`data/`, `runs/` and `*.pt` are git-ignored).

## Reproduce

Run everything from the repository root. Relative paths are resolved against this folder.

### Self-tests (minutes, tiny fake data)

    python selftest.py
    python selftest_attacks.py
    python selftest_dp.py
    python selftest_e3.py

`selftest_e3.py` compares `train.py` against an earlier version of itself retrieved with
`git show 0865bf8:experiments/train.py`. That commit exists only in the private development history, not in this
repository, so that first test cannot run here (see Known issues).

### Pilot (PROTOCOL.md): 30,000 steps per N, then evaluation

    python train.py --n 256  --steps 30000 --out runs/n256
    python evaluate.py --run runs/n256 --ckpt all --n-samples 1000
    python train.py --n 1024 --steps 30000 --out runs/n1024
    python evaluate.py --run runs/n1024 --ckpt all --n-samples 1000
    python train.py --n 4096 --steps 30000 --out runs/n4096
    python evaluate.py --run runs/n4096 --ckpt all --n-samples 1000
    python make_figures.py

`run_all.ps1` (Windows) or `run_all.sh` (Git Bash on Windows; it expects `.venv/Scripts/python.exe`) runs the same
sequence with retries. `PROTOCOL.md` records about 2.4 it/s at batch 64 on the pilot GPU, so about 3.5 hours of
training per run, and about 19 minutes of fp32 sampling per checkpoint (Deviation 6 there explains the bf16
sampling used at intermediate checkpoints).

### PIA and SecMI-stat attacks (PROTOCOL_V2.md, E4a)

    python eval_attacks.py --run runs/n4096

The only pilot result for this attack in the repository is N = 4096 (`results/runs/n4096/attacks.json`). The
N = 256 and N = 1024 attacks are listed as pending in `PROTOCOL_V2.md`, Deviation 2, and have no result files here.

### E3 guidance sweep (PROTOCOL_V2.md)

As run for the reported results (one job at a time, fp16 training, sampling batch 125):

    python run_college.py check
    python run_college.py run --blocks e3 --parallel 1 --amp fp16 --cfg-bs 125

This trains `runs/c1024` and `runs/c4096` (`train.py --cond --amp fp16`), runs `eval_cfg.py` and `eval_attacks.py`
on each, and writes the results. Add `--dry-run` to print the job graph only. The queue is resumable: after a crash
or reboot, run the same command again. `python run_college.py status` shows progress and
`python run_college.py pack` writes a zip of the results (no checkpoints). Recorded wall times on the office PC are in
`results_e3/college_status.log` (for example c4096: training 42,629 s, guidance sweep 11,164 s, attacks 543 s).
Then:

    python plot_e3.py

`PROTOCOL_V2.md` Deviation 7 explains why the c1024 sweep mixes sampling batch sizes 250 and 125.

### Not run

E4b (training seeds 1 and 2) and E5 (DP-SGD: `dp_gate.py`, `train_dp.py`, queued through
`python run_college.py run --blocks e4b` and `--blocks e5`) exist as code only. No results from them are included.

## Results included

- `results/table.csv`, `results/table.tex`, `results/figA.pdf`: pilot table and Figure A (final checkpoint, N = 256,
  1024, 4096).
- `results/figB.pdf`: Figure B, generated samples next to their nearest training images at N = 256 (also shown in
  the supplement of the paper). The images are PneumoniaMNIST training images, public under CC BY 4.0.
- `results/config_n*.json`, `results/metrics_n*.json`, `results/param_count.txt`, `results/REPORT.txt`: the
  per-N configuration and final metrics that the pilot table is built from, and the run report.
- `results/runs/n256`, `n1024`, `n4096`: `metrics.json` (all checkpoints), `config.json`, `log.csv`, `members.npy`
  (indices of the member subset), `samples_last.npy` (the 1,000 final-checkpoint samples that copy detection used),
  `figB.json` and `figB.npz` (inputs of Figure B), and `attacks.json` for N = 4096.
- `results_e3/c1024`, `results_e3/c4096`: `cfg_metrics.json` (guidance sweep), `attacks.json`, `config.json`,
  `figcfg.json`, `log.csv`, `members.npy`; `results_e3/REPORT.txt` and `college_status.log` are run summaries;
  `results_e3/E3_PAPER_NOTES.md` lists the E3 numbers copied from these files; `results_e3/figC.pdf` is Figure C.
- `results/logs`, `results_e3/logs`: console logs, with local machine paths replaced by `python` or `results`.
- The pilot models use a 26,621,057-parameter UNet (sum of tensor sizes in the saved EMA state dict; the
  class-conditional E3 models add a class embedding and were not counted).

Model checkpoints (`ema_*.pt`, about 107 MB each, 24 files for the three pilot runs) are not in this repository
because GitHub rejects files over 100 MB. [TODO: link to the archive that holds the final checkpoints, if the author
deposits one.] Without them, `make_figures.py` and `evaluate.py` cannot be re-run on the trained models, but every
number in the paper tables can be read from the result files above, and the models can be retrained with the commands
in the Reproduce section.

## Limitations

One training seed per setting, one small dataset (PneumoniaMNIST), no patient-wise split (the MedMNIST validation
split comes from the source training set, so non-members from it may share patients with members), and 64x64 images.
PIA and SecMI timesteps are the published values, not tuned to this model (`ATTACKS.md`). FID at 1000 samples is
biased and is compared only within this study. Results are descriptive for this setup and do not claim anything
beyond it.

## Known issues

- `selftest_e3.py` (`test_regression_and_seed`) needs the private commit `0865bf8` and will fail in this repository.
- Several scripts and messages refer to a "college PC" and to sending output to "Claude"; these are leftovers of the
  author's workflow and do not affect behaviour.
- Documents and result files refer to `experiments/` paths from the original layout; here the files sit at the
  repository root.
- `plot_e3.py` writes `results_e3/figC.pdf` (changed from `../figs/pilot_figC.pdf`, output location only).

## License

The code is released under the MIT License (see `LICENSE`). The dataset is not included and keeps its own
licence (CC BY 4.0, see the Data section).

## Citation

[TODO: citation, after the survey is published]
