#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import functools
import os
import collections
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.cuda
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from uimnet import utils
import time
from uimnet import __DEBUG__

class Accuracy:
  """
    Top-k classification accuracy
    From: https://github.com/pytorch/examples/blob/master/imagenet/main.py#L411
    """

  def __init__(self, topk=(1, 5)):
    self.topk = topk

  def __call__(self, predictions, targets):
    maxk = max(self.topk)
    batch_size = targets.size(0)

    _, pred = predictions.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in self.topk:
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.div_(batch_size))
    return res

class Loss:
  """
    Cross-entropy loss (also known as negative log-likelihood)
    """

  def __init__(self):
    self.ce = torch.nn.CrossEntropyLoss()

  def __call__(self, predictions, targets):
    return self.ce(predictions, targets.long())


class ECE:
  """
    Expected Calibration Error (ECE, as proposed in https://arxiv.org/abs/1706.04599)
    From: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L77
    """

  def __init__(self, n_bins=15):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    self.bin_lowers = bin_boundaries[:-1]
    self.bin_uppers = bin_boundaries[1:]

  def __call__(self, predictions, targets):

    softmaxes = torch.nn.functional.softmax(predictions, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(targets)

    ece = torch.zeros(1).cuda()
    for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
      # Calculated |confidence - accuracy| in each bin
      in_bin = confidences.gt(bin_lower.item()
                             ) * confidences.le(bin_upper.item())
      prop_in_bin = in_bin.float().mean()
      if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        ece += (avg_confidence_in_bin - accuracy_in_bin).abs() * prop_in_bin

    return ece


class PredictionMetrics:
  """
    A class to monitor top-k accuracy, negative log-likelihood and expected
    calibration error given prediction scores and true labels.
    """

  def __init__(self, top=(1, 5)):
    self.compute_accuracy = Accuracy(top)
    self.compute_loss = Loss()
    self.compute_ece = ECE()

  def __call__(self, algorithm, loader):
    predictions = []
    targets = []

    with torch.no_grad():
      for i, batch in enumerate(loader):
        if __DEBUG__ and i > 2:
          break
        batch = utils.apply_fun(functools.partial(utils.to_device, device='cuda'), batch)
        x, y  = batch['x'], batch['y']
        predictions.append(algorithm(x))
        targets.append(y)

      predictions = torch.cat(predictions).detach().cpu()
      targets = torch.cat(targets).detach().cpu()

    acc1, acc5 = self.compute_accuracy(predictions, targets)

    return {
        "ACC@1": acc1, "ACC@5": acc5, "NLL":
            self.compute_loss(predictions, targets), "ECE":
                self.compute_ece(predictions, targets)
    }

if __name__ == '__main__':
  pass
