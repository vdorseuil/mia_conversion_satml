"""
Datasets, downloaded into data/ the first time they are needed:
    CIFAR-10                     data/cifar-10-batches-py/   (Nasr, Steinke, Mahloujifar)
    ClipBKD FMNIST sets          data/jagielski2020/         (released with the code of Jagielski et al.)
    WRN-28-10 CIFAR-10 features  data/annamalai2024/         (released with the code of Annamalai & De Cristofaro)
California Housing (Cebere et al.) is downloaded by scikit-learn into data/cebere2025/.
"""
import os
import pickle
import shutil
import tarfile
import urllib.request

import numpy as np
import torch

from .paths import DATA_DIR

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_FILES = [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]
JAGIELSKI_URL = "https://raw.githubusercontent.com/jagielski/auditing-dpsgd/731a71ac19dbef4e9d24668b73a91fe211b73a87"
ANNAMALAI_URL = "https://raw.githubusercontent.com/spalabucr/bb-audit-dpsgd/46198b00c1bf16890d1451a64c2edccda7a3f43a"
ANNAMALAI_PARTS = 8                                        # their data_compressed/data-part00..07, one tar.gz once joined
ANNAMALAI_FEATURES = ["data/cifar10_half_finetune_last/X_train.npy", "data/cifar10_half_finetune_last/y_train.npy"]


def download(urls, path):
    """The content of the urls, joined, written to path (nothing is done when it exists). The file is written under
    a temporary name first, so an interrupted download is not mistaken for the file."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        with open(partial, "wb") as f:
            for url in urls:
                print(f"downloading {url}", flush=True)
                with urllib.request.urlopen(url) as response:
                    shutil.copyfileobj(response, f)
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


def extract(archive, members, folder):
    """Extract the members of a tar.gz archive into folder, then delete the archive."""
    with tarfile.open(archive) as tar:
        for member in members:
            tar.extract(member, folder, filter="data")
    archive.unlink()


# CIFAR-10
def cifar10_images(files):
    """Images (n, 3, 32, 32) in [0, 1] and labels (n,) of CIFAR-10 batch files."""
    folder = DATA_DIR / "cifar-10-batches-py"
    if not all((folder / name).exists() for name in CIFAR10_FILES):
        archive = download([CIFAR10_URL], DATA_DIR / "cifar-10-python.tar.gz")
        extract(archive, [f"cifar-10-batches-py/{name}" for name in CIFAR10_FILES], DATA_DIR)
    xs, ys = [], []
    for name in files:
        with open(folder / name, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        xs.append(batch[b"data"])
        ys += batch[b"labels"]
    return torch.tensor(np.concatenate(xs).reshape(-1, 3, 32, 32).astype(np.float32) / 255), torch.tensor(ys)

def load_cifar10():
    return cifar10_images([f"data_batch_{i}" for i in range(1, 6)])

def load_cifar10_test():
    return cifar10_images(["test_batch"])


# Jagielski et al.
def clipbkd_file(k):
    """Their FMNIST file for k poison points: (D0, D1, the poison point, the test set)."""
    name = f"clipbkd-new-{k}.npy"
    return download([f"{JAGIELSKI_URL}/datasets/fmnist/{name}"], DATA_DIR / "jagielski2020" / name)


# Annamalai & De Cristofaro
def annamalai_files():
    """The folder with their features (data/cifar10_half_finetune_last/), ClipBKD target and worst-case parameters."""
    folder = DATA_DIR / "annamalai2024"
    download([f"{ANNAMALAI_URL}/target_samples/cifar10_half_finetune_last_clipbkd.npy"],
             folder / "cifar10_half_finetune_last_clipbkd.npy")
    download([f"{ANNAMALAI_URL}/pretrained_models/cifar10_half_finetune_last.pt"], folder / "cifar10_half_finetune_last.pt")
    if not all((folder / name).exists() for name in ANNAMALAI_FEATURES):
        parts = [f"{ANNAMALAI_URL}/data_compressed/data-part{i:02d}" for i in range(ANNAMALAI_PARTS)]
        extract(download(parts, folder / "data.tar.gz"), ANNAMALAI_FEATURES, folder)
    return folder
