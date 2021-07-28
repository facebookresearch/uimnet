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
  parser.add_argument('-mc', '--mog_cfg', type=str, required=True, help='Path to YAML config.')
  parser.add_argument('-f', '--train_dir', type=str, default=None, help='Path to sweep dir. Overrides YAML cfg.')
  parser.add_argument('-p', '--partitions_file', type=str, default=None, help='Path to partition file defaults to value in cfg if missing')
  parser.add_argument('--local_rank', type=int, default=None)

  return parser.parse_args()


@utils.timeit
def run_mog(mog_cfg, train_dir, partitions):


  subpath = Path(train_dir).absolute()
  filename = 'cfg_rank_0.yaml'
  with open(subpath / filename, 'r') as fp:
    train_cfg = OmegaConf.load(fp.name)

  _datasets = utils.partition_dataset(name=mog_cfg.dataset.name,
                                      root=mog_cfg.dataset.root,
                                      split=mog_cfg.dataset.split,
                                      partitions=partitions,
                                      equalize_partitions=mog_cfg.dataset.equalize_partitions)

  # loading algorithm
  Algorithm = utils.load_model_cls(train_cfg=train_cfg)
  dataset = _datasets['in']
  args = [mog_cfg, train_cfg, Algorithm, dataset]
  utils.write_trace('mog.pending', dir_=str(subpath))

  worker = workers.MOG()
  results = worker(*args)
  return results


def main(args):

  with open(args.mog_cfg, 'r') as fp:
    mog_cfg = OmegaConf.load(fp.name)

  mog_cfg.local_rank = args.local_rank
  OmegaConf.set_readonly(mog_cfg, True)  # Write protect.
  OmegaConf.set_struct(mog_cfg, True)  # Raise missing keys error.

  with open(args.partitions_file, 'rb') as fp:
    partitions = pickle.load(fp)['partitions']

  return run_mog(mog_cfg, args.train_dir, partitions=partitions)




if __name__ == '__main__':
  args = parse_arguments()
  main(args)
