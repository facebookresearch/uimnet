#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from pathlib import Path
import torch.multiprocessing as tmp
import filelock
import pickle

from uimnet import utils
from uimnet import __SLURM_CONFIGS__
from uimnet import __DEBUG__

@utils.timeit
def partition_datasets(name, root, partitions, equalize_partitions):

  all_partitions = {}
  for split_name in ['train', 'val']:
    split_partition  = utils.partition_dataset(name=name,
                                               root=root,
                                               split=split_name,
                                               partitions=partitions,
                                               equalize_partitions=equalize_partitions)

    for partition_name, partition in split_partition.items():
      all_partitions[(split_name, partition_name)]  = partition

  return all_partitions

@utils.timeit
def load_partitions(name, root, clustering_file, equalize_partitions):
    clustering = load_clustering(clustering_file=clustering_file)
    partitions = partition_datasets(name, root, partitions=clustering['partitions'], equalize_partitions=equalize_partitions)
    return partitions

@utils.timeit
def load_clustering(clustering_file):

  clustering_path = Path(clustering_file)
  with filelock.FileLock(str(clustering_path)  + '.lock'):
    with open(clustering_path, 'rb') as fp:
      return pickle.load(fp)

if __name__ == '__main__':
  name = 'ImageNat'
  root = '/checkpoint/ishmaelb/data/datasets/ILSVRC2012'
  equalize_partitions = True
  clustering_file = '/checkpoint/ishmaelb/uimnet/sweeps/release_mkV/clustering.pkl'
  partitions = load_partitions(name=name, root=root, equalize_partitions=equalize_partitions, clustering_file=clustering_file)
