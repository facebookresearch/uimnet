#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import copy
import os
import functools
import itertools
import collections
import argparse
from pathlib import Path
from concurrent import futures
import pickle
from omegaconf import OmegaConf

import submitit
from uimnet import utils
from uimnet import ensembles
from uimnet import workers
from uimnet import datasets
from uimnet import __SLURM_CONFIGS__, __DEBUG__

ENSEMBLING_CFG = """
sweep_dir: null
output_dir: null  # subfolder. Mutable at dispatch

dataset:
  nseeds: 1
  name: ImageNat
  partition: null  # Mutable at dispatch
  root: /checkpoint/ishmaelb/data/datasets/ILSVRC2012
  partitions_file: null
  equalize_partitions:  True
  seed: null # Mutable at dispatch
  split: train
  batch_size: 64
  splits_props:
    train: 0.9
    eval: 0.1

algorithm:
  nseeds: 2
  name: null  # Mutable at dispatch
  arch: null  # Mutable at dispatch
  use_mixed_precision: True
  seed: 42 # Mutable at dispatch
  sn: True
  sn_coef: 1.0
  sn_bn: True

ensemble:
  name: Bagging
  k: 5

slurm:
  preset: 'distributed_8'  # Key-value pairs below override presets.
  time: 360
  mem_per_gpu: '32G'
  cpus_per_task: 5
  partition: devlab
  array_parallelism: 512
  constraint: volta32gb
  comment: 'NeurIPS_2021'

experiment:
  distributed: True
  platform: 'slurm'
  evaluate_every: 1
  checkpoint_every: 1
  num_epochs: 1
  ## ----- Mutable on the worker during distributed/device setup.
  output_dir: null
  seed: 42  # Workers seed
  device: 'cuda:0'
  rank: null
  local_rank: null
  world_size: null
  dist_protocol: tcp
  dist_url: null
  num_workers: 5
  # ------
"""
def parse_arguments():
  parser = argparse.ArgumentParser(description='Runs ensembles')
  parser.add_argument('-s', '--sweep_dir', type=str, required=True)
  parser.add_argument('-k', '--members', type=int, nargs='+', default=[5])
  parser.add_argument('--filter', type=str, nargs='*', default=None)
  parser.add_argument('--force', action='store_true')

  return parser.parse_args()


def name_ensemble(values, k):
  return 'Bagging'+ '_' + '_'.join(map(str, values)) + '_' + f'{k}'

def load_train_cfg(model_path):
  if (model_path / 'train_cfg.yaml').is_file():
    return utils.load_cfg(model_path / 'train_cfg.yaml')
  elif (model_path / 'cfg_rank_0.yaml').is_file():
    return utils.load_cfg(model_path / 'cfg_rank_0.yaml')
  else:
    err_msg = 'train config not found'
    raise ValueError(err_msg)




def run_ensembles(sweep_dir, ensemble_sizes, filter_list,
                  force=False,  max_workers=10):

  sweep_path = Path(sweep_dir)
  subfolders = [el for el in sweep_path.iterdir() if el.is_dir()]

  filters = [utils.train_done, utils.prediction_done]
  # selected_paths = list(filter(compose(*filters), subfolders))
  selected_paths = subfolders
  for _filter in filters:
    selected_paths = list(filter(_filter, selected_paths))

  with futures.ProcessPoolExecutor(max_workers) as executor:
    records = sum(list(executor.map(utils.ExtractRecords(),
                                    map(lambda el: el / 'predictive_records.pkl', selected_paths))), [])

  # Only keep validation set
  records = list(filter(lambda el: el['split'] == 'eval', records))
  # Only initial temperature mode
  records = list(filter(lambda el: el['temperature_mode'] == 'initial', records))

  keys = ('algorithm.name', 'algorithm.arch', 'algorithm.sn', 'dataset.seed')
  mapping = collections.defaultdict(list)
  for record in records:
    _values = tuple([record[k] for k in keys])
    mapping[_values] += [record]
  mapping = {k: sorted(l, key=lambda d: d['NLL']) for k, l in mapping.items()}

  ensembling_cfg = OmegaConf.create(ENSEMBLING_CFG)
  executor = utils.get_slurm_executor(copy.deepcopy(ensembling_cfg.slurm),
                                log_folder= sweep_path / 'logs' / 'run_ensembles')
  #
  dataset = datasets.load_partitions(ensembling_cfg.dataset.name,
                            root=os.getenv('DATASETS_ROOT',
                                          '/checkpoint/ishmaelb/data/datasets/ILSVRC2012'),
                            clustering_file=str(sweep_path / 'clustering.pkl'),
                            equalize_partitions=True)

  jobs, paths = [], []
  with executor.batch():
    for i, (values, model_records) in enumerate(mapping.items()):
      for k in ensemble_sizes:
        ensemble_name = name_ensemble(values, k)
        ensemble_path = sweep_path / ensemble_name
        ensemble_path.mkdir(parents=True, exist_ok=True)
        (ensemble_path / '.ensemble').touch()

        if utils.train_done(ensemble_path) and not force:
          print(f'{ensemble_name} is done. Skipping.')
          continue

        model_paths = [el['output_dir'] for el in model_records]
        with open(ensemble_path / 'paths.pkl', 'wb') as fp:
          pickle.dump(model_paths, fp, protocol=pickle.HIGHEST_PROTOCOL)

        Algorithm = functools.partial(ensembles.Bagging, paths=model_paths[:k])
        worker = workers.Trainer()

        keys_vals = dict(zip(keys, values))
        train_cfg = copy.deepcopy(ensembling_cfg)

        train_cfg.output_dir = str(ensemble_path)
        train_cfg.algorithm.name = keys_vals['algorithm.name']
        train_cfg.algorithm.arch = keys_vals['algorithm.arch']
        train_cfg.algorithm.sn = keys_vals['algorithm.sn']
        train_cfg.ensemble.k = k
        train_cfg.ensemble.name = 'Bagging'
        train_cfg.dataset.seed = keys_vals['dataset.seed']
        OmegaConf.set_struct(train_cfg, True)

        worker_args = (train_cfg, Algorithm, dataset[('train', 'in')])
        with open(ensemble_path / 'train_cfg.yaml', 'w') as fp:
          OmegaConf.save(train_cfg, f=fp.name)

        job = executor.submit(worker, *worker_args)
        jobs += [job]
        paths += [ensemble_path]
        utils.write_trace('train.pending', dir_=str(ensemble_path))

  beholder = utils.Beholder(list(zip(jobs, paths)), stem='train')
  beholder.start()

  finished_jobs, jobs = utils.handle_jobs(jobs)

  return finished_jobs, jobs
































if __name__ == '__main__':
 args = parse_arguments()
 finished_jobs, jobs = run_ensembles(args.sweep_dir, args.members, args.filter,
                  force=args.force,  max_workers=10)
