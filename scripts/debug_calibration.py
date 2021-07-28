#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from omegaconf import OmegaConf
import os
import pickle
from pathlib import Path
import argparse
from uimnet import utils
from uimnet import workers


def parse_arguments():

  parser = argparse.ArgumentParser(
    description='runs Calibration sweep workflow')
  parser.add_argument('-tc', '--calibration_cfg', type=str, required=True, help='Path to YAML config.')
  parser.add_argument('-f', '--train_dir', type=str, default=None, help='Path to sweep dir. Overrides YAML cfg.')
  parser.add_argument('-p', '--partitions_file', type=str, default=None, help='Path to partition file defaults to value in cfg if missing')
  parser.add_argument('--local_rank', type=int, default=None)

  return parser.parse_args()


@utils.timeit
def run_calibration(calibration_cfg, train_dir, partitions):


  subpath = Path(train_dir).absolute()
  filename = 'cfg_rank_0.yaml'
  with open(subpath / filename, 'r') as fp:
    train_cfg = OmegaConf.load(fp.name)

  _datasets = utils.partition_dataset(name=calibration_cfg.dataset.name,
                                      root=calibration_cfg.dataset.root,
                                      split=calibration_cfg.dataset.split,
                                      partitions=partitions,
                                      equalize_partitions=calibration_cfg.dataset.equalize_partitions)

  # loading algorithm
  Algorithm = utils.load_model_cls(train_cfg=train_cfg)
  dataset = _datasets['in']
  args = [calibration_cfg, train_cfg, Algorithm, dataset]
  utils.write_trace('calibration.pending', dir_=str(subpath))

  worker = workers.Calibrator()
  results = worker(*args)
  return results


def main(args):
  with open(args.calibration_cfg, 'r') as fp:
    calibration_cfg = OmegaConf.load(fp.name)
  calibration_cfg.local_rank = args.local_rank
  OmegaConf.set_readonly(calibration_cfg, True)  # Write protect.
  OmegaConf.set_struct(calibration_cfg, True)  # Raise missing keys error.

  with open(args.partitions_file, 'rb') as fp:
    partitions = pickle.load(fp)['partitions']

  return run_calibration(calibration_cfg, args.train_dir, partitions=partitions)




if __name__ == '__main__':
  args = parse_arguments()
  main(args)
