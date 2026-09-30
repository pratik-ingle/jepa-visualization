"""CIFAR-10 loading, fixed splits, and GPU-resident uint8 image tensors.

Splits (fixed seed, exported to data/splits.json):
  ssl_train   all 50k CIFAR-10 train images, used without labels
  probe_train 10k class-balanced subset of CIFAR-10 train (the small labelled set)
  val         5k class-balanced half of CIFAR-10 test (val loss + geometry diagnostics)
  eval_test   the other 5k of CIFAR-10 test (probe accuracy)
  pca         512 images of val (embedding clouds at checkpoints)
  showcase    16 images of val (the browser's initial input picker)
  batch       256 images of val (browser batch for SIGReg / PCA views)
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CACHE = Path(__file__).resolve().parents[1] / ".cache"
CIFAR_DIR = CACHE / "cifar-10-batches-py"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
SPLIT_SEED = 0


def _load_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    y = np.asarray(d[b"labels"], dtype=np.int64)
    return x, y


def _from_parquet(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Hugging Face `uoft-cs/cifar10` parquet (PNG bytes + label) -> uint8 NCHW, labels."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    t = pq.read_table(path).to_pydict()
    x = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB")) for im in t["img"]])
    return x.transpose(0, 3, 1, 2).copy(), np.asarray(t["label"], dtype=np.int64)


NPZ = CACHE / "cifar10.npz"


def load_cifar_numpy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CIFAR-10 as uint8 NCHW. Sources, in order: cached npz, official pickles, HF parquet."""
    if NPZ.exists():
        d = np.load(NPZ)
        return d["x_train"], d["y_train"], d["x_test"], d["y_test"]
    if (CIFAR_DIR / "test_batch").exists():
        xs, ys = zip(*[_load_batch(CIFAR_DIR / f"data_batch_{i}") for i in range(1, 6)])
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)
        x_te, y_te = _load_batch(CIFAR_DIR / "test_batch")
    else:
        x_tr, y_tr = _from_parquet(CACHE / "cifar10-train.parquet")
        x_te, y_te = _from_parquet(CACHE / "cifar10-test.parquet")
    np.savez(NPZ, x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te)
    return x_tr, y_tr, x_te, y_te


def _balanced_take(labels: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Positions of n items, class round-robin (c0, c1, ..., c9, c0, ...), random within class."""
    per_class = [rng.permutation(np.flatnonzero(labels == c)) for c in range(10)]
    order = [per_class[c][i] for i in range(-(-n // 10)) for c in range(10)]
    return np.asarray(order[:n], dtype=np.int64)


def make_splits(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SPLIT_SEED)
    probe_train = np.sort(_balanced_take(y_train, 10_000, rng))
    val = np.sort(_balanced_take(y_test, 5_000, rng))
    eval_test = np.setdiff1d(np.arange(len(y_test)), val)
    # Subsets of val (indices into CIFAR-10 test), class round-robin so any prefix is balanced.
    val_labels = y_test[val]
    return {
        "probe_train": probe_train,
        "val": val,
        "eval_test": eval_test,
        "pca": val[_balanced_take(val_labels, 512, rng)],
        "batch": val[_balanced_take(val_labels, 256, rng)],
        "showcase": val[_balanced_take(val_labels, 16, rng)],
    }


SPLITS_FILE = Path(__file__).resolve().parents[1] / "splits.json"


def get_splits(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, np.ndarray]:
    """Splits are generated once (from canonical CIFAR-10 order) and then always read from
    model/splits.json, so rebuilding the image cache can never silently change them."""
    if SPLITS_FILE.exists():
        return {k: np.asarray(v, dtype=np.int64) for k, v in json.loads(SPLITS_FILE.read_text()).items()}
    s = make_splits(y_train, y_test)
    SPLITS_FILE.write_text(json.dumps({k: v.tolist() for k, v in s.items()}))
    return s


@dataclass
class Data:
    """All tensors live on `device`; images are uint8 NCHW."""

    train_x: torch.Tensor
    train_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    splits: dict[str, np.ndarray]
    mean: torch.Tensor
    std: torch.Tensor

    def subset(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(self.splits[split], device=self.train_x.device)
        if split == "probe_train":
            return self.train_x[idx], self.train_y[idx]
        return self.test_x[idx], self.test_y[idx]

    def normalize(self, x01: torch.Tensor) -> torch.Tensor:
        """Images in [0, 1] (float) -> channel-normalised network input."""
        return (x01 - self.mean) / self.std

    def prep(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return self.normalize(x_uint8.float() / 255.0)


def load_data(device: str | torch.device = "cuda") -> Data:
    x_tr, y_tr, x_te, y_te = load_cifar_numpy()
    splits = get_splits(y_tr, y_te)
    t = lambda a: torch.as_tensor(a, device=device)
    return Data(
        train_x=t(x_tr), train_y=t(y_tr), test_x=t(x_te), test_y=t(y_te), splits=splits,
        mean=torch.tensor(MEAN, device=device).view(1, 3, 1, 1),
        std=torch.tensor(STD, device=device).view(1, 3, 1, 1),
    )
