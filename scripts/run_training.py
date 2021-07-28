#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import copy
import os
import time
import functools
import itertools
import collections
import argparse
from pathlib import Path
from concurrent import futures
import pickle
from omegaconf import OmegaConf

from uimnet import utils
from uimnet import algorithms
from uimnet import workers
from uimnet import datasets
from uimnet import __SLURM_CONFIGS__, __DEBUG__

TRAINING_CFG = """
sweep_dir: null
output_dir: null  # subfolder. Mutable at dispatch

dataset:
  nseeds: 3
  name: ImageNat
  partition: null  # Mutable at dispatch
  root: /checkpoint/ishmaelb/data/datasets/ILSVRC2012
  partitions_file: null
  equalize_partitions:  True
  seed: null # Mutable at dispatch
  split: train
  batch_size: 256
  splits_props:
    train: 0.9
    eval: 0.1

algorithm:
  nseeds: 5
  name: null
  arch: null
  use_mixed_precision: True
  seed: null # Mutable at dispatch
  sn: null
  sn_coef: 1.0
  sn_bn: True


slurm:
  preset: 'distributed_8'  # Key-value pairs below override presets.
  time: 720
  mem_per_gpu: '32G'
  cpus_per_task: 5
  partition: learnfair
  array_parallelism: 512

experiment:
  distributed: True
  platform: slurm
  evaluate_every: 50
  checkpoint_every: 1
  num_epochs: 100
  ## ----- Mutable on the worker during distributed/device setup.
  output_dir: null
  seed: 42  # Workers seed
  device: 'cuda:0'
  rank: null
  local_rank: null
  world_size: null
  dist_protocol: null
  dist_url: null
  num_workers: 5
  # ------
"""

def parse_arguments():
  parser = argparse.ArgumentParser(description='Runs training')
  parser.add_argument('-s', '--sweep_dir', type=str, required=True)
  parser.add_argument('--archs', nargs='+', type=str, default=['resnet18', 'resnet50'])
  parser.add_argument('--ndata_seeds', type=int, default=3)
  parser.add_argument('--nalg_seeds', type=int, default=5)
  parser.add_argument('--algorithms', type=str, nargs='+',
                      default=list(algorithms.__ALGORITHMS__.keys()))
  parser.add_argument('--force', action='store_true')

  return parser.parse_args()

def run_training(sweep_dir, Algorithms, archs, nalg_seeds, ndata_seeds, force):
  sweep_path = Path(sweep_dir)
  sweep_path.mkdir(parents=True, exist_ok=True)

  train_cfg = OmegaConf.create(TRAINING_CFG)
  dataset = datasets.load_partitions(train_cfg.dataset.name,
                            root=os.getenv('DATASETS_ROOT',
                                          '/checkpoint/ishmaelb/data/datasets/ILSVRC2012'),
                            clustering_file=str(sweep_path / 'clustering.pkl'),
                            equalize_partitions=True)[('train', 'in')]

  executor = utils.get_slurm_executor(copy.deepcopy(train_cfg.slurm),
                                log_folder= sweep_path / 'logs' / 'run_training')

  data_seeds = range(ndata_seeds)
  alg_seeds = range(nalg_seeds)
  sns = [False, True]

  args_iter = itertools.product(
     data_seeds, Algorithms, alg_seeds, archs, sns
  )
  jobs, paths = [], []
  with executor.batch():
    for (data_seed, Algorithm, alg_seed, arch, sn) in args_iter:

      _train_cfg = OmegaConf.create(TRAINING_CFG)
      _train_cfg.dataset.name = dataset.name
      _train_cfg.dataset.seed = data_seed
      _train_cfg.algorithm.sn = sn
      _train_cfg.algorithm.name = Algorithm.__name__
      _train_cfg.algorithm.arch = arch
      _train_cfg.algorithm.seed = alg_seed

      output_dir = f'{Algorithm.__name__}_{dataset.name}_{arch}_{sn}_{data_seed}_{alg_seed}'
      model_path = sweep_path / output_dir
      model_path.mkdir(parents=True, exist_ok=True)
      (model_path / '.algorithm').touch()
      _train_cfg.output_dir = str(model_path.absolute())
      OmegaConf.set_struct(_train_cfg, True)

      if utils.train_done(model_path) and not force:
        print(f'{output_dir} is done. Skipping')
        continue
      with open(model_path / 'train_cfg.yaml', 'w') as fp:
        OmegaConf.save(_train_cfg, f=fp.name)

      worker = workers.Trainer()
      worker_args = (_train_cfg, Algorithm, dataset)
      job = executor.submit(worker, *worker_args)
      jobs += [job]
      paths += [model_path]
      utils.write_trace('train.pending', dir_=str(model_path))

  beholder = utils.Beholder(list(zip(jobs, paths)), stem='train')
  beholder.start()

  finished_jobs, jobs = utils.handle_jobs(jobs)
  import ipdb; ipdb.set_trace()

  return finished_jobs, jobs

if __name__ == '__main__':
  args = parse_arguments()
  finished_jobs, jobs = run_training(
    sweep_dir=args.sweep_dir,
    Algorithms=[algorithms.__ALGORITHMS__[el] for el in args.algorithms],
    archs=args.archs,
    nalg_seeds=args.nalg_seeds,
    ndata_seeds=args.ndata_seeds,
    force=args.force
  )
