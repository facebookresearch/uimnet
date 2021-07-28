#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
from torch.utils.data import TensorDataset, Dataset


class DebugDataset(Dataset):
  def __init__(self, nsamples, num_classes=266, data_shape=(3, 224, 224)):

    self.num_classes = num_classes

    self.indices = torch.arange(nsamples)
    self.x = torch.zeros((nsamples, ) + data_shape).normal_()
    self.y = torch.randint(num_classes, size=(nsamples, ))
    assert self.indices.shape[0] == self.x.shape[0] == self.y.shape[0]


  def __len__(self):
    return len(self.indices)

  def __getitem__(self, index):
    return dict(index=self.indices[index],
                x=self.x[index],
                y=self.y[index]
                )




if __name__ == '__main__':
  pass
