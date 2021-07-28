#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import dataloader, random_split, DataLoader, Subset, Dataset
from torch.utils.data import TensorDataset, DistributedSampler

from uimnet import utils

def get_loader(dataset, batch_size, shuffle,
                   pin_memory, num_workers):
  # instantiate distributed sampler if required
  _sampler = None
  _batch_size = batch_size
  _shuffle = shuffle
  if utils.is_distributed():
    utils.message(f'Loading distributed sampler')
    _batch_size //= dist.get_world_size()
    _shuffle = False
    _sampler = DistributedSampler(dataset)
  return DataLoader(dataset, _batch_size,
                    shuffle=_shuffle,
                    sampler=_sampler,
                    pin_memory=pin_memory,
                    num_workers=num_workers
                    )
if __name__ == '__main__':
  pass
