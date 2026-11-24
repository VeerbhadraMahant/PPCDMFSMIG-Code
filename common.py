"""Shared data, model and scheduler code (see PROTOCOL.md)."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
NPZ = DATA_DIR / "pneumoniamnist_64.npz"
T_TRAIN = 1000


def resolve(p):
    """Relative paths are taken relative to experiments/, not the caller's cwd."""
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def load_data_with_origin(fake=False):
    """Like load_data, plus a bool mask over the non-members: True = from the val split, False = test."""
    if fake:
        rng = np.random.RandomState(123)
        train = rng.randint(0, 256, (600, 64, 64), dtype=np.uint8)
        nm = rng.randint(0, 256, (100, 64, 64), dtype=np.uint8)
        return train, nm, np.arange(100) < 50
    if not NPZ.exists():
        from medmnist import PneumoniaMNIST
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PneumoniaMNIST(split="train", download=True, size=64, root=str(DATA_DIR))
    d = np.load(NPZ)
    train, nm = d["train_images"], np.concatenate([d["val_images"], d["test_images"]])
    is_val = np.arange(len(nm)) < len(d["val_images"])
    keep = _keep_mask(train, nm)
    return train, nm[keep], is_val[keep]


def _keep_mask(train, nm):
    # cross-split exact duplicates would be members in disguise
    train_hashes = {x.tobytes() for x in train}
    return np.array([x.tobytes() not in train_hashes for x in nm])


def load_labels(fake=False):
    """(train_labels, nonmember_labels) as int64, aligned with load_data(). 0 = normal, 1 = pneumonia."""
    if fake:
        rng = np.random.RandomState(321)
        return rng.randint(0, 2, 600).astype(np.int64), rng.randint(0, 2, 100).astype(np.int64)
    load_data_with_origin()  # downloads if needed
    d = np.load(NPZ)
    nm = np.concatenate([d["val_images"], d["test_images"]])
    nm_y = np.concatenate([d["val_labels"], d["test_labels"]]).reshape(-1)
    keep = _keep_mask(d["train_images"], nm)
    return d["train_labels"].reshape(-1).astype(np.int64), nm_y[keep].astype(np.int64)


def load_data(fake=False):
    """Returns (train_u8, nonmember_u8), each (n, 64, 64) uint8.

    Non-members are the official val + test splits (524 + 624 = 1148), minus
    images byte-identical to any train image (8 in PneumoniaMNIST-64), so 1140.
    fake=True gives random data for the self-test only.
    """
    return load_data_with_origin(fake)[:2]


def member_indices(n_train, n, seed=0):
    """First n of one fixed permutation, so S256 is a subset of S1024 is a subset of S4096."""
    assert n <= n_train, f"n={n} exceeds train size {n_train}"
    return np.random.RandomState(seed).permutation(n_train)[:n]


def to_tensor(u8, device):
    """uint8 (n,64,64) -> float (n,1,64,64) in [-1, 1]."""
    return (torch.from_numpy(np.ascontiguousarray(u8)).float() / 127.5 - 1.0).unsqueeze(1).to(device)


NUM_CLASSES = 2  # PneumoniaMNIST; a conditional UNet embeds NUM_CLASSES + 1 labels, the last is null


def build_unet(tiny=False, num_classes=0, arch="base"):
    """num_classes > 0 gives a class-conditional UNet with label index num_classes as the null label."""
    from diffusers import UNet2DModel
    if tiny:  # self-test only
        kw = dict(block_out_channels=(32, 32, 32, 32), layers_per_block=1, norm_num_groups=8)
    elif arch == "small":  # E5 fallback only (PROTOCOL_V2.md)
        kw = dict(block_out_channels=(32, 64, 128, 128), layers_per_block=2)
    else:
        assert arch == "base", arch
        kw = dict(block_out_channels=(64, 128, 256, 256), layers_per_block=2)
    if num_classes:
        kw["num_class_embeds"] = num_classes + 1
    return UNet2DModel(
        sample_size=64, in_channels=1, out_channels=1,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        **kw)


def unet_from_config(cfg):
    """Rebuild the UNet a run was trained with; old pilot configs have no cond/arch keys."""
    return build_unet(cfg["tiny"], NUM_CLASSES if cfg.get("cond") else 0, cfg.get("arch", "base"))


def predict_eps(unet, x_t, t, y=None):
    """eps_theta(x_t, t[, y]); y is a LongTensor of labels for conditional models, None otherwise."""
    return unet(x_t, t, class_labels=y).sample if y is not None else unet(x_t, t).sample


def make_ddpm():
    from diffusers import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=T_TRAIN, beta_start=1e-4, beta_end=0.02,
                         beta_schedule="linear", prediction_type="epsilon")


def make_ddim():
    from diffusers import DDIMScheduler
    return DDIMScheduler(num_train_timesteps=T_TRAIN, beta_start=1e-4, beta_end=0.02,
                         beta_schedule="linear", prediction_type="epsilon")


@torch.no_grad()
def ema_update(ema_model, model, decay):
    e = list(ema_model.parameters())
    p = [q.detach() for q in model.parameters()]
    torch._foreach_mul_(e, decay)
    torch._foreach_add_(e, p, alpha=1.0 - decay)
