#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import numpy as np
from uimnet.ensembles.base import BaseEnsemble
from pathlib import Path
from uimnet import utils
from uimnet import numerics


class Bagging(BaseEnsemble):

  def setup_optimizers(self):
    pass

  def entropy_(self, p):
    zeros = torch.zeros(()).to(p.device)
    h = p * torch.where(p == 0, zeros, torch.log(p))
    return h.sum(1).mul(-1)

  def update(self, x, y, epoch=None):
    return {'loss': np.nan, 'cost': np.nan}

  def _forward(self, x):
    x = x.to(self.device)

    all_logits = torch.stack(self.forward_members(x), dim=1)
    return numerics.log_marginalization_from_logits(all_logits).to(x.device)



  def uncertainty(self, x):

    predictions = self.forward_members(x)

    predictions_members = [p.softmax(1) for p in predictions]
    predictions_ensemble = torch.stack(predictions_members).mean(0)

    entropy_members = [self.entropy_(p) for p in predictions_members]
    entropy_ensemble = self.entropy_(predictions_ensemble)

    measure = entropy_ensemble - torch.stack(entropy_members).mean(0)
    return measure.to(x.device)
if __name__ == '__main__':
  pass
