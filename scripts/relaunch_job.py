#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
"""
This script is to DEBUG training
"""
import argparse
import copy
from uimnet.algorithms.base import Algorithm
import torch
import torch.distributed as tmp
import pickle

from pathlib import Path
from uimnet import utils
from uimnet import workers
from uimnet import algorithms
from uimnet import dispatch

from omegaconf import OmegaConf
from filelock import FileLock

def parse_arguments():

  parser = argparse.ArgumentParser(
    description='runs Calibration sweep workflow')
  parser.add_argument('-c', '--train_cfg', type=str, required=True, help='Path to YAML config.')
  parser.add_argument('-o', '--output_dir', type=str, default=None, help='Override configs file')
  parser.add_argument('-p', '--partitions_file', type=str, default=None, help='Path to partition file defaults to value in cfg if missing')

  return parser.parse_args()

def run_training(cfg, partitions):

  output_path = Path(cfg.output_dir)
  output_path.mkdir(parents=True, exist_ok=True)
  Algorithm = algorithms.__dict__[cfg.algorithm.name]
  _datasets = utils.partition_dataset(name='ImageNat',
                                      root=cfg.dataset.root,
                                      split=cfg.dataset.split,
                                      partitions=partitions,
                                      equalize_partitions=cfg.dataset.equalize_partitions)

  dataset = _datasets['in']
  trainer = workers.Trainer()
  output = dispatch.submit(cfg=cfg,
                   output_dir=cfg.output_dir,
                   worker=trainer,
                   Algorithm=Algorithm,
                  dataset=dataset)

  return output


def main(args):
  with FileLock(f'{args.train_cfg}.lock'):
    with open(args.train_cfg, 'r') as fp:
      train_cfg = OmegaConf.load(fp.name)
  if args.output_dir is not None:
    train_cfg.output_dir = args.output_dir
  OmegaConf.set_readonly(train_cfg, True)  # Write protect.
  OmegaConf.set_struct(train_cfg, True)  # Raise missing keys error.

  with FileLock(f'{args.partitions_file}.lock'):
    with open(args.partitions_file, 'rb') as fp:
      partitions = pickle.load(fp)['partitions']

  return run_training(train_cfg, partitions=partitions)


if __name__ == '__main__':
  args = parse_arguments()
  main(args)
