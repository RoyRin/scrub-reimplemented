import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from itertools import cycle
import os
import time
import math
import pandas as pd
import wandb
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression

import copy
import torch.nn as nn
from torch.autograd import Variable
from typing import List
import itertools
from tqdm.autonotebook import tqdm
#from models import *
# from unlearning.unlearning_benchmarks.SCRUB import models, utils
#import models

#from logger import *
#import utils


from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.helper.util import adjust_learning_rate as sgda_adjust_learning_rate

from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
if False:
    from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss

from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.helper.loops import train_distill, train_distill_hide, train_distill_linear, train_vanilla, train_negrad, train_bcu, train_bcu_distill, validate

#from unlearning.unlearning_benchmarks.SCRUB.thirdparty.repdistiller.helper.pretrain import init
import numpy as np
import torch

from argparse import Namespace
args = Namespace()
import os
import shutil
import yaml
import numpy as np
import torch as ch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import logging
import pprint
from contextlib import redirect_stdout, redirect_stderr

from unlearning.auditors.utils import (
    model_factory,
    loader_factory,
    load_forget_set_indices,
    get_full_model_paths,
    get_oracle_paths,
)

#from unlearning.unlearning_algos.base_nn import NAME_TO_ALGO




def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    return config

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


def get_model_and_loaders(forget_set_id= 4, forget_bs = 64, retain_bs = 64, ds_name = "CIFAR10"):

    # for now, let's tie the model to the dataset, so we have fewer moving pieces
    model = model_factory(ds_name)  # on cuda, in eval mode

    forget_set_indices = load_forget_set_indices(ds_name,forget_set_id)
    print(f"getting the dataloaders")
    with redirect_stdout(open("/dev/null", "w")):
        # no shuffling, no augmentation
        train_loader = loader_factory(ds_name, indexed=False)
        val_loader = loader_factory(ds_name, split="val", indexed=False)
        total_train_set = len(train_loader.dataset)
        retain_inds = np.setdiff1d(np.arange(total_train_set), forget_set_indices)
        retain_loader = loader_factory(ds_name, indices=retain_inds, indexed=False, batch_size=retain_bs)

        forget_loader = loader_factory(
            ds_name,
            indices=forget_set_indices,
            batch_size=forget_bs,
            indexed=False,
        )
        eval_set_inds = np.arange(
            len(train_loader.dataset) + len(val_loader.dataset))
        eval_loader = loader_factory(ds_name,
                                     split="train_and_val",
                                     indices=eval_set_inds,
                                     indexed=False)
    # inserted by Roy for some speed reason
    splits = ["train", "val"]
    print(f"loading the model!")
    f_ckpt_paths, f_logit_paths, f_margins_paths = get_full_model_paths(
        ds_name, splits=splits)
    (
        o_ckpt_0_path,  # we only need a single oracle checkpoint
        o_logit_paths,
        o_margins_paths,
    ) = get_oracle_paths(ds_name, forget_set_id, splits=splits)
    print(f"Loaded paths of pretrained models.")

    full_model = load_model(f_ckpt_paths[0], model_factory, ds_name)
    return full_model, (train_loader, val_loader, forget_loader, retain_loader, eval_loader), forget_set_indices



def l2_difference(model1, model2):
    l2_diff = 0.0
    # Ensure both models are in the same state (e.g., both in eval mode)
    model1.eval()
    model2.eval()

    with torch.no_grad():
        for (param1, param2) in zip(model1.parameters(), model2.parameters()):
            # Check if both parameters are on the same device and are of the same shape
            if param1.device != param2.device or param1.shape != param2.shape:
                raise ValueError("Models have parameters on different devices or with different shapes")
            
            # Compute the squared L2 norm of the difference between the parameters
            param_diff = param1 - param2
            l2_diff += torch.norm(param_diff, p=2).item()**2

    # Return the square root of the sum of squared differences
    return l2_diff**0.5



def scrub_loop(args,
               optimizer,
               model,
               criterion_cls,
               module_list,
               swa_model,
               criterion_list,
               retain_loader,
               forget_loader,
               verbose=False):

    acc_rs = []
    acc_fs = []
    report_every = 10
    maximize_loss = None

    print(f"total epochs : {args.sgda_epochs}")
    """
    TODO:
        1. check if the model is being updated
        2. 
    """
    for epoch in range(1, args.sgda_epochs + 1):
        print(f"Epoch {epoch} ...")

        lr = sgda_adjust_learning_rate(epoch, args, optimizer)

        if verbose:
            print("==> scrub unlearning ...")
            print(f"validating - ")
        if epoch % report_every == 0:
            acc_r, acc5_r, loss_r = validate(retain_loader, model,
                                             criterion_cls, args, True)
            acc_f, acc5_f, loss_f = validate(forget_loader, model,
                                             criterion_cls, args, True)
            acc_rs.append(100 - acc_r.item())
            acc_fs.append(100 - acc_f.item())

        maximize_loss = 0

        if epoch <= args.msteps:
            if verbose:
                print(f"train distill 1")
            # maximize loss on the forget set
            maximize_loss = train_distill(epoch, forget_loader, module_list,
                                          swa_model, criterion_list, optimizer,
                                          args, "maximize")
        if verbose:
            print(f"train distill 2 :")
        # minimize loss on the retain set
        train_acc, train_loss = train_distill(epoch, retain_loader,
                                              module_list, swa_model,
                                              criterion_list, optimizer, args,
                                              "minimize")

        if epoch >= args.sstart:
            print("update params")
            swa_model.update_parameters(model)

        if verbose:
            print(
                "maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}"
                .format(maximize_loss, train_loss, train_acc))
        student = module_list[0]
        teacher = module_list[-1]
        print(f"{epoch} -- difference between teacher and student - {l2_difference(student, teacher)}")

    return model


def scrub_wrapper(model,
                  train_dataloader,
                  forget_indices,
                  val_loader=None,
                  num_epochs=5,
                  learning_rate=5e-5,
                  device='cpu',
                  forget_bs=64,
                  retain_bs=64,
                  ds_name="CIFAR10",
                  **kwargs):

    args.optim = 'sgd'
    args.gamma = 0.99
    args.alpha = 0.001
    args.beta = 0.9999 # 0
    args.smoothing = 0.0
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 4
    args.distill = 'kd'

    args.sgda_batch_size = 128
    args.del_batch_size = 32

    args.msteps = num_epochs # 2
    args.sgda_epochs = num_epochs
    args.sgda_learning_rate = learning_rate # 0.0005
    args.lr_decay_epochs = [3,5,9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 5e-4
    args.sgda_momentum = 0.9



    original_model = copy.deepcopy(model)
    unlearned_model = copy.deepcopy(model)

    # TODO - implement a wrapper, same as for the other interfaces!

    forget_loader = loader_factory(ds_name,
                                   indices=forget_indices,
                                   indexed=False,
                                   batch_size=forget_bs)
    N_points = len(train_dataloader.dataset)
    retain_inds = np.setdiff1d(np.arange(N_points), forget_indices)
    retain_loader = loader_factory(ds_name,
                                   indices=retain_inds,
                                   indexed=False,
                                   batch_size=retain_bs)
    print(f"forgetting indices : {forget_indices}")

    module_list = nn.ModuleList([])
    module_list.append(unlearned_model)  # student.
    module_list.append(original_model)  # teacher.

    trainable_list = nn.ModuleList([])
    trainable_list.append(unlearned_model)
    """
    model_s = module_list[0]
    model_t = module_list[-1]
    """
    # optimizer
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(trainable_list.parameters(),
                              lr=args.sgda_learning_rate,
                              momentum=args.sgda_momentum,
                              weight_decay=args.sgda_weight_decay)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(trainable_list.parameters(),
                               lr=args.sgda_learning_rate,
                               weight_decay=args.sgda_weight_decay)
    elif args.optim == "rmsp":
        optimizer = torch.optim.RMSprop(trainable_list.parameters(),
                                  lr=args.sgda_learning_rate,
                                  momentum=args.sgda_momentum,
                                  weight_decay=args.sgda_weight_decay)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(
        criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return (1 - args.beta
                ) * averaged_model_parameter + args.beta * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(unlearned_model,
                                                    avg_fn=avg_fn)

    # args

    unlearned_model = scrub_loop(args, optimizer, unlearned_model,
                                 criterion_cls, module_list, swa_model,
                                 criterion_list, retain_loader, forget_loader)

    return unlearned_model




def do_set_up():
    args.optim = 'adam'
    args.gamma = 1
    args.alpha = 0.5
    args.beta = 0
    args.smoothing = 0.5
    args.msteps = 3
    args.clip = 0.2
    args.sstart = 10
    args.kd_T = 2
    args.distill = 'kd'

    args.sgda_epochs = 10
    args.sgda_learning_rate = 0.0005
    args.lr_decay_epochs = [5, 8, 9]
    args.lr_decay_rate = 0.1
    args.sgda_weight_decay = 0.1  #5e-4
    args.sgda_momentum = 0.9

    # TODO: teacher, student
    # teacher = None, student = None
    # TODO : datasets and dataloaders

    teacher = copy.deepcopy(full_model)
    student = copy.deepcopy(full_model)

    # load up model 1
    model_t = copy.deepcopy(teacher)
    model_s = copy.deepcopy(student)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    module_list.append(model_t)

    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(
        criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    optimizer = torch.optim.Adam(trainable_list.parameters(),
                                 lr=args.sgda_learning_rate,
                                 weight_decay=args.sgda_weight_decay)

    beta = 0.1

    def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return (1 - beta) * averaged_model_parameter + beta * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)

    return model_s, criterion_cls, module_list, swa_model, criterion_list, optimizer


if __name__ == "__main__":

    forget_set_id = 4

    full_model, loaders, forget_set_indices = get_model_and_loaders(forget_set_id= forget_set_id,  ds_name = "CIFAR10")

    (train_loader, val_loader, forget_loader, retain_loader, eval_loader) = loaders


    # set it up
    model_s, criterion_cls, module_list, swa_model, criterion_list, optimizer = do_set_up()

    #scrub it

    scrub_loop(args, optimizer, model_s, criterion_cls, module_list, swa_model, criterion_list, retain_loader, forget_loader)

    # save the model
    torch.save(model_s.state_dict(), "SCRUB.pth")
    print("Model saved")
