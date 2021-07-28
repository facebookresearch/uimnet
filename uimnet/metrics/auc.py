#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#


import collections
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import time

class AUC(object):
  def __init__(self, *args, **kwargs):
    pass

  def __call__(self, measurement_in, measurement_out):


    _device = measurement_in.device
    # In-domain: label = 0
    # out-of-domain: label = 1
    domain_predictions = torch.cat([measurement_in, measurement_out], dim=0)
    domain_targets = torch.cat([torch.zeros_like(measurement_in),
                                torch.ones_like(measurement_out)], dim=0)
    return torch.as_tensor(roc_auc_score(
      domain_targets.cpu().numpy(),
      domain_predictions.cpu().numpy(),
                         )).to(_device)
if __name__ == '__main__':
  pass
