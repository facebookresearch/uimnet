#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import collections
import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, DataLoader
from torchvision.datasets import ImageNet
from pathlib import Path

from uimnet import utils


class ImageNat(ImageNet):
  # NAME = 'ILSVRC2012'
  def __init__(self, indices=None, **kwargs):
    kwargs['root'] = str(Path(kwargs['root'])) #/ self.NAME)
    super(ImageNat, self).__init__(**kwargs)

    if indices is None:
      indices = torch.arange(len(self.samples))

    self.indices = indices

    paths, targets = list(zip(*self.samples))

    targets = torch.as_tensor(targets)
    self.all_available_targets = torch.as_tensor(targets)[self.indices]
    self.available_targets = torch.unique(self.all_available_targets)
    self.absolute_to_relative = dict(zip(self.available_targets.tolist(),
                                         range(len(self.available_targets)),
                                        ))

  def __len__(self):
    return len(self.indices)

  @property
  def num_classes(self):
    return len(self.available_targets)

  def __getitem__(self, index):
    _index = self.indices[index]
    path, y = self.samples[_index]
    yr = self.absolute_to_relative[y]

    x = self.loader(path)
    if self.transform is not None:
        x = self.transform(x)
    if self.target_transform is not None:
        yr = self.target_transform(yr)

    return dict(index=_index, x=x, y=yr)

  def get_targets_indices(self, targets):
    targets = torch.as_tensor(targets)
    indices = torch.nonzero((
        self.all_available_targets.view(-1, 1) ==
        targets.view(1, -1)).long().sum(dim=1)).view(-1, )
    return indices


if __name__ == '__main__':
  pass
