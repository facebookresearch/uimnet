#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
"""
Calibrates the models
"""
import copy
import argparse
import os
import submitit
import pickle

from pathlib import Path
from omegaconf import OmegaConf
import submitit

from uimnet import utils
from uimnet import workers
from uimnet import __SLURM_CONFIGS__


CALIBRATION_CFG = """

  slurm:
    preset: 'one_gpu'  # Key-value pairs below override presets.
    time: 59
    mem_per_gpu: '32G'
    cpus_per_task: 5
    partition: learnlab
    array_parallelism: 512
    constraint: volta32gb

  dataset:
    seed: 0
    batch_size: 8

  experiment:
    distributed: False
    platform: 'slurm'


    #platform: 'debug'
    ## ----- Mutable on the worker during distributed/device setup.
    output_dir: null
    seed: 42  # Workers seed
    device: 'cuda:0'
    rank: null
    local_rank: null
    world_size: null
    dist_protocol: null
    dist_url: null
    # ------
    num_workers: 5

  """

def parse_arguments():
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('-s', '--sweep_dir', type=str, required=True)
  parser.add_argument('--force', action='store_true')
  return parser.parse_args()

@utils.timeit
def load_datasets(root, name, split, clustering_path):

  with open(Path(clustering_path), 'rb') as fp:
    clustering = pickle.load(fp)

  datasets = utils.partition_dataset(name=name,
                                     root=root,
                                     split=split,
                                     partitions=clustering['partitions'],
                                     equalize_partitions=True)
  return datasets




def run_calibration(sweep_dir, force):
  sweep_path = Path(sweep_dir)
  clustering_path = sweep_path / 'clustering.pkl'

  models_paths = filter(utils.is_model, sweep_path.iterdir())

  models_paths = filter(utils.train_done, models_paths)

  calibration_cfg = OmegaConf.create(CALIBRATION_CFG)

  root = os.getenv('DATASETS_ROOT')
  name = 'ImageNat'

  datasets = load_datasets(root=root,
                           name=name,
                           split='train',
                           clustering_path=clustering_path
                           )
  executor = utils.get_slurm_executor(copy.deepcopy(calibration_cfg.slurm),
                                      log_folder=str(sweep_path / 'logs' / 'run_calibration'))
  # Constructing jobs
  jobs, paths = [], []
  with executor.batch():
    # Construcing jobs
    for model_path in models_paths:
      if utils.calibration_done(model_path) and not force:
        print(f'{model_path} is done. Skipping.')
        continue

      if (model_path / 'train_cfg.yaml').is_file():
        train_cfg = utils.load_cfg(model_path / 'train_cfg.yaml')
      elif (model_path / 'cfg_rank_0.yaml').is_file():
        train_cfg = utils.load_cfg(model_path / 'cfg_rank_0.yaml')
      else:
        err_msg = 'train config not found.'
        raise ValueError(err_msg)

      Algorithm = utils.load_model_cls(train_cfg)
      dataset = datasets['in']
      worker_args = (calibration_cfg, train_cfg, Algorithm, dataset)
      worker = workers.Calibrator()
      job = executor.submit(worker, *worker_args)
      jobs += [job]
      paths += [model_path]
      utils.write_trace('calibration.pending', dir_=str(model_path))
  # Waiting for jobs to finish

  beholder = utils.Beholder(list(zip(jobs, paths)), stem='calibration')
  beholder.start()
  finished_jobs, jobs = utils.handle_jobs(jobs)
  # Collecting results
  jobs_results = [job.results() for job in finished_jobs]
  return jobs_results

if __name__ == '__main__':
  args = parse_arguments()
  output = run_calibration(args.sweep_dir, force=args.force)
