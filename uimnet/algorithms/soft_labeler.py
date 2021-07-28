#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import numpy as np
from uimnet import utils
from uimnet.algorithms.erm import ERM
import torch.cuda.amp as amp

def soften(y, tau):
  K = y.shape[1]
  return (1 / (K - 1)) * ((tau * K - 1) * y + (1 - tau))

def harden(s, tau):
  K = s.shape[1]
  return ((K - 1) * s - (1 - tau)) / (tau * K - 1)


class SoftLabeler(ERM):
    HPARAMS = dict(ERM.HPARAMS)
    HPARAMS.update({
        "threshold": (0.8, lambda: float(np.random.choice([0.7, 0.8, 0.9])))
    })

    def __init__(
            self,
            num_classes,
            arch,
            device="cuda",
            seed=0,
            use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):

        super(SoftLabeler, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision, sn=sn, sn_coef=sn_coef, sn_bn=sn_bn)

        self.loss = utils.SoftCrossEntropyLoss()
        self.pos_lab = self.hparams["threshold"]
        self.neg_lab = (1 - self.pos_lab) / (self.num_classes - 1)

        self.tau = self.hparams["threshold"]

        self.has_native_measure = True

    def soften_(self, t):
        return (self.pos_lab - self.neg_lab) * t + self.neg_lab

    def harden_(self, t):
        return (t - self.neg_lab) / (self.pos_lab - self.neg_lab)

    def process_minibatch(self, x, y):
        t = torch.nn.functional.one_hot(y, self.num_classes).float()
        t = self.soften_(t)
        # t = soften(t, self.tau)

        x = x.to(self.device, non_blocking=True)
        t = t.to(self.device, non_blocking=True)
        return x, t

    def _forward(self, x):
      predictions = super()._forward(x)
      # features = self.networks['featurizer'](x.to(self.device))
      # predictions = self.networks['classifier'](features)
      if self.training:
          return predictions
      eps = 1e-4
      # return torch.log1p(harden(predictions.softmax(1))     - 1 + eps)
      return torch.log1p(predictions.softmax(1) - 1 + eps)

    def uncertainty(self, x):
        softmaxes = super().forward(x).softmax(1)
        return (softmaxes - self.pos_lab).pow(2).min(1).values

if __name__ == '__main__':
  pass
