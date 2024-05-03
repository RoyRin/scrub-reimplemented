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

