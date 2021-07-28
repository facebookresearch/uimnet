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

from uimnet import __DEBUG__


class FusedPredictionMetrics:
  def __init__(self, top=(1, 2), n_bins=15):

    self.topk = top
    self.n_bins = n_bins

  def __call__(self, algorithm, loader):
    with torch.no_grad():

      maxk = max(self.topk)
      accumulators = {
          # Confidence, accuracy, count
          'ece': torch.zeros(size=(self.n_bins, 3)),
          'loss': torch.zeros(size=(1, )),
          'ntopk': torch.zeros(size=(maxk, )),
          'nseen': torch.zeros(size=(1, ))
      }
      accumulators = {k: v.cuda() for k, v in accumulators.items()}
      # Accuracy
      # ECE
      # n_bins + 1 points
      b = torch.arange(0, self.n_bins + 1).float().view(1, -1).cuda()
      # bin size
      delta = 1 / self.n_bins

      for batch in loader:

        x = batch['x'].cuda()
        y = batch['y'].cuda()
        s = algorithm(x)
        probs = F.softmax(s, dim=1)
        phat, yhat = torch.max(probs, dim=1)

        # Loss
        accumulators['nseen'] += len(x)
        accumulators['loss'] += F.cross_entropy(s, y.long(), reduction='sum')

        # Accuracy
        _, preds = torch.topk(s, k=maxk, largest=True, sorted=True)
        preds = preds.T
        correct = preds.eq(y.view(1, -1)).expand_as(preds).T
        accumulators['ntopk'] += torch.cumsum(correct, dim=1).float().sum(dim=0)

        # ECE
        # assigning confidence to bins
        acc = yhat.eq(y).float()
        lc = phat.view(-1, 1) > b[:, :-1] * delta  # N X K
        rc = phat.view(-1, 1) <= b[:, 1:] * delta  # N X K
        m = (lc * rc).float().T  # K X N

        accumulators['ece'][:, 0].add_(m @ phat)
        accumulators['ece'][:, 1].add_(m @ acc)
        accumulators['ece'][:, 2].add_(m.sum(dim=1))

      for name, tensor in accumulators.items():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

      # loss
      loss = (accumulators['loss'] / accumulators['nseen']).squeeze()

      # Accuracy
      accuracies = accumulators['ntopk'] / accumulators['nseen']
      accuracies = accuracies[[k - 1 for k in self.topk]]
      acc1, acc5 = accuracies

      # ECE
      accumulator = accumulators['ece']
      accumulator = accumulator[accumulator[:, 2] > 0.]

      accumulator[:, 0].div_(accumulator[:, 2])
      accumulator[:, 1].div_(accumulator[:, 2])
      accumulator[:, 2].div_(accumulator[:, 2].sum())

      acc_per_bin = accumulator[:, 0]
      conf_per_bin = accumulator[:, 1]
      prop_per_bin = accumulator[:, 2]

      ece = (acc_per_bin - conf_per_bin).abs().mul(prop_per_bin).sum()

      results = {
          "ACC@1": acc1,
          "ACC@5": acc5,
          "NLL": loss,
          "ECE": ece,
      }
      return results

if __name__ == '__main__':
  pass
