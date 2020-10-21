import os
import argparse

import numpy as np

import gym

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import make_env, ENV_KEYS


def convert_to_onnx(args):


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    modes = ['convert', 'eval']

    p = argparse.ArgumentParser()
    p.add_argument('mode', type=str, choices=modes, help='mode to run')
    p.add_argument('--env-key', type=str,
                   choices=ENV_KEYS, default=ENV_KEYS[0])
    p.add_argument('--max-epochs', type=int, default=10000)
    p.add_argument('--ckpt-path', type=str, default='')
    p.add_argument('--gpus', type=int, default=0)
    args = p.parse_args()

    if args.mode == 'convert':
        convert_to_onnx(args)
