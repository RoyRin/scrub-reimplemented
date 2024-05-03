import yaml
from pathlib import Path
from importlib import reload
from unlearning.auditors import eval_suite

reload(eval_suite)

import numpy as np
import torch as ch

from contextlib import redirect_stdout

from utils import (
    model_factory,
    loader_factory,
)

from scrub import scrub_wrapper
CWD = Path.cwd()
BASE_DIR = CWD.parent.parent

retain_data_amount = 2.
num_epochs = 5

config_dict = {
    'results_dir': './results/',
    'dataset': 'CIFAR10',
    'forget_set_id': 5,
    'unlearning_algo': 'scrub',
    'run_direct_eval': True,
    'use_submitit_for_direct_eval': False,
    'unlearning_algo_kwargs': {
        'dataset': 'CIFAR10',
        'forget_set_id': 5,
        'oracles_path':
        '/n/home04/rrinberg/data_dir__holylabs/unlearning/precomputed_models/oracles/CIFAR10/forget_set_5',
        'retain_data_amount': 2.0,
        'mix_data': True,
        'num_epochs': 10
    },
    'reorder_logit_classes': True,
    'N_models_for_direct': 20
}



def load_model(path, model_factory, ds_name):
    model = model_factory(ds_name)
    loaded_model = ch.load(path)
    first_key = list(loaded_model.keys())[0]
    if "model" in first_key:
        model.load_state_dict(loaded_model)

    else:
        # add ".model" to each key in k,vs
        loaded_model = {f"model.{k}": v for k, v in loaded_model.items()}
        model.load_state_dict(loaded_model)
    return model



unlearn_name = "scrub"

#####
config = config_dict
results = {}
results["params"] = {}
ds_name = "CIFAR10"
# for now, let's tie the model to the dataset, so we have fewer moving pieces
model = model_factory(ds_name)  # on cuda, in eval mode

forget_set_indices = [305, 1346, 1538, 2335, 3799, 4260, 4956, 5894, 6873, 7364, 7398, 7531, 7726, 8050, 9196, 9235, 9377, 9665, 9999, 10221, 10482, 12132, 12300, 14355, 14667, 15103, 15602, 16905, 17306, 17400, 18014, 18278, 18512, 18912, 19222, 19231, 19285, 19606, 21191, 21480, 21502, 22321, 22487, 22749, 22876, 22908, 23369, 23385, 23898, 23914, 24637, 25886, 26388, 28340, 28510, 28612, 28726, 28973, 29242, 29271, 29712, 29795, 30156, 30523, 31017, 31129, 31781, 31875, 33079, 33735, 34516, 35932, 36319, 36454, 36871, 37316, 37471, 37589, 39645, 39880, 40004, 40663, 40800, 42396, 42731, 43209, 44581, 45278, 45477, 46146, 46264, 46715, 47482, 47716, 47729, 48169, 48980, 49014, 49316, 49753]

results["params"]["forget_set_indices"] = forget_set_indices
unlearn_fn = scrub_wrapper

unlearning_kwargs = config["unlearning_algo_kwargs"]

if unlearning_kwargs is None:
    unlearning_kwargs = {}

with redirect_stdout(open("/dev/null", "w")):
    # no shuffling, no augmentation
    train_loader = loader_factory(ds_name, indexed=True)
    val_loader = loader_factory(ds_name, split="val", indexed=True)
    forget_loader = loader_factory(
        ds_name,
        indices=forget_set_indices,
        batch_size=50,
        indexed=True,
    )
    eval_set_inds = np.arange(
        len(train_loader.dataset) + len(val_loader.dataset))
    eval_loader = loader_factory(ds_name,
                                 split="train_and_val",
                                 indices=eval_set_inds,
                                 indexed=True)
####### END OF SETUP ########

####### LOAD PRETRAINED MODELS ########
# original model
original_model_path = CWD / "full_model.pt"
oracle_model_path = CWD / "retrained_oracle.pt"
model = load_model(original_model_path, model_factory, ds_name)

# oracle model
oracle_model = load_model(oracle_model_path, model_factory, ds_name)

# unlearning
unlearned_model = unlearn_fn(
    model=model,
    train_dataloader=train_loader,
    forget_dataloader=None,
    forget_indices=forget_set_indices,
    **unlearning_kwargs,
)
