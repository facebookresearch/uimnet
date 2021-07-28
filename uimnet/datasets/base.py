#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import copy
from uimnet import utils
from uimnet.datasets.loaders import get_loader as _get_loader
import torch.distributed as dist

@utils.timeit
def make_random_splits(indices, props, generator):

  props = [p for p in props if (p > 0.)]
  assert sum(props) == 1
  nsamples = len(indices)

  shuffled = torch.randperm(nsamples, generator=generator)
  splits = []
  #utils.message(indices)
  #utils.message(shuffled)
  ub = 0
  for i, p in enumerate(props):
    lb = 0 if i == 0 else ub
    ub = lb + int(p * nsamples)
    if i == len(props) -1:
      ub = None
    #utils.message(f'{lb}: lb, {ub}: ub')
    #splits.append(indices[shuffled[slice(lb, ub)]])
    splits.append(indices[shuffled[lb:ub]])

  is_consistent = sum(len(s) for s in splits) == nsamples
  assert is_consistent
  return splits

class BaseNode(object):
  def __init__(self, dataset, transforms, seed, **kwargs):
    super(BaseNode, self).__init__()
    self.dataset = dataset
    self.transforms= transforms
    self.seed = seed

  @property
  def transforms_list(self):
    return list(self.transforms.keys())

class SimpleDataNode(BaseNode):
  def __init__(self, dataset, transforms, seed):
    super(SimpleDataNode, self).__init__(dataset, transforms, seed)
    self.loader = None

  def set_transform(self, tname):
    if self.transforms is None:
      return
    self.transform = self.transforms[tname]
    return self

  def train(self):
    self.set_transform('train')
    return self

  def eval(self):
    self.set_transform('eval')
    return self

  def get_loader(self,
                 batch_size,
                 shuffle,
                 pin_memory,
                 num_workers):

    if self.loader is not None:
      return self.loader
    self.loader = _get_loader(self.dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=pin_memory,
                              num_workers=num_workers)
    return self.loader



class SplitDataNode(BaseNode):

  def __init__(self, dataset, transforms, splits_props, seed):
    super(SplitDataNode, self).__init__(dataset=dataset, transforms=transforms, seed=seed)

    if not hasattr(self.dataset, 'indices'):
      err_msg = f'Dataset does not have indices as attribute'
      raise RuntimeError(err_msg)

    indices = torch.as_tensor(self.dataset.indices)
    self.splits_props = splits_props
    self.splits_indices = self.get_random_splits_indices(indices, splits_props)

    self.splits = {}
    for split_name, indices in self.splits_indices.items():
      split = copy.deepcopy(dataset)
      split.indices = indices
      self.splits[split_name] = split

    self.loaders = {n: None for n in self.splits}  # Pre-allocating

  @utils.timeit
  def get_random_splits_indices(self, indices, props):

    generator = torch.Generator().manual_seed(self.seed)
    if not sum(props.values()) == 1.:
      err_msg = f'Splits proportion should sum to 1'
      raise ValueError(err_msg)

    splits_indices = make_random_splits(indices, props.values(), generator=generator)

    return dict(zip(props.keys(), splits_indices))

  @property
  def splits_names(self):
    return list(self.splits.keys())


  def set_transform(self, sname, tname):
    if self.transforms is None:
      return self
    self.splits[sname].transform = self.transforms[tname]
    return self

  def set_transforms(self, tname):
    for sname in self.splits:
      self.set_transform(sname, tname)
    return self

  def train(self):
    self.set_transforms('train')
    return self

  def eval(self):
    self.set_transforms('eval')
    return self

  def get_loader(self, sname,
                 batch_size,
                 shuffle,
                 pin_memory,
                 num_workers):

    if self.loaders[sname] is not None:
      return self.loaders[sname]

    self.loaders[sname] = _get_loader(self.splits[sname],
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=pin_memory,
                              num_workers=num_workers)
    return self.loaders[sname]


  def get_loaders(self, *args, **kwargs):
    return {n: self.get_loader(n, *args, **kwargs) for n in self.splits}

if __name__ == '__main__':
  seed = 42
  generator = torch.Generator().manual_seed(seed)
  props = {'train': 0.9, 'val': 0.1}
  indices = torch.arange(10**6)
  splits = make_random_splits(indices, props.values(), generator)
