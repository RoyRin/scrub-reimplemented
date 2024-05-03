import submitit
from unlearning.models.resnet9 import ResNet9
from unlearning.datasets.cifar10 import get_cifar_dataloader
from unlearning.training.train import train_cifar10
from unlearning.unlearning_algos.utils import get_margin
from pathlib import Path
import numpy as np
import torch as ch

# Roy Dir (Harvard)
BASE_DIR = Path("/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models")

if not BASE_DIR.exists():
    print(f"Was not able to find precomputed_models directory in {BASE_DIR}")
    print(f"... Continuing anyways. but proceed with caution!")
LOG_DIR = Path("/n/home04/rrinberg/catered_out/unlearning")




def model_factory(dataset, wrapped=True):
    """
    for now, let's tie the model to the dataset, so we have fewer moving pieces
    """
    if dataset.lower() == "cifar10":
        from unlearning.models.resnet9 import ResNet9

        return ResNet9(num_classes=10, wrapped=wrapped).cuda().eval()

    elif dataset.lower() == "cifar100":
        from unlearning.models.resnet9 import ResNet9

        return ResNet9(num_classes=100).cuda().eval()
    else:
        raise NotImplementedError


def loader_factory(
    dataset,
    split="train",
    indices=None,
    batch_size=256,
    shuffle=False,
    augment=False,
    indexed=True,
):
    if dataset.lower() == "cifar10":
        from unlearning.datasets.cifar10 import get_cifar_dataloader

        return get_cifar_dataloader(
            split=split,
            num_workers=2,
            batch_size=batch_size,
            shuffle=shuffle,
            augment=augment,
            indices=indices,
            indexed=indexed,
        )
    else:
        raise NotImplementedError


def load_forget_set_indices(dataset, forget_set_id, DATA_DIR = None):
    if DATA_DIR is None:
       DATA_DIR = BASE_DIR / "forget_set_inds"
    forget_set_path = DATA_DIR / dataset / f"forget_set_{forget_set_id}.npy"
    forget_set_indices = np.load(forget_set_path)
    return forget_set_indices

import re
def sort_key(path):
    # Extract numerical parts and convert to integers
    numbers = re.findall(r'\d+', path.name)
    return [int(num) for num in numbers]



def get_full_model_paths(dataset, splits=["train", "val"]):
    DATA_DIR = BASE_DIR / "full_models" / dataset
    full_model_ckpt_paths = sorted(list(DATA_DIR.glob("sd_*_epoch_23.pt")))
    # Sort paths using the custom key
    full_model_ckpt_paths = sorted(full_model_ckpt_paths, key=sort_key)

    # train and val
    full_model_logit_paths = [
        DATA_DIR / f"{split}_logits_all.pt" for split in splits
    ]
    full_model_margins_paths = [
        DATA_DIR / f"{split}_margins_all.pt" for split in splits
    ]
    return full_model_ckpt_paths, full_model_logit_paths, full_model_margins_paths


def get_oracle_paths(dataset, forget_set_id, splits = ["train", "val"]):

    DATA_DIR = BASE_DIR / "oracles" / dataset / f"forget_set_{forget_set_id}"
    oracle_ckpt_0_path = DATA_DIR / "sd_0____epoch_23.pt"
    # train and val
    oracle_logit_paths = [
        DATA_DIR / f"{split}_logits_all.pt" for split in splits
    ]
    oracle_margins_paths = [
        DATA_DIR / f"{split}_margins_all.pt" for split in splits
    ]
    return oracle_ckpt_0_path, oracle_logit_paths, oracle_margins_paths

