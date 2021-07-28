#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
from uimnet import utils
from uimnet.algorithms.erm import ERM
import numpy as np


class Mixup(ERM):
    HPARAMS = dict(ERM.HPARAMS)
    HPARAMS.update({
        "alpha": (0.3, lambda: float(np.random.choice([0.1, 0.2, 0.3, 1, 2])))
    })

    def __init__(
            self,
            num_classes,
            arch,
            device="cuda",
            seed=0,
            use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):
        super(Mixup, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision, sn=sn, sn_coef=sn_coef, sn_bn=sn_bn)

        self.loss = utils.SoftCrossEntropyLoss()
        self.alpha = self.hparams["alpha"]
        self.beta_distr = torch.distributions.Beta(self.alpha, self.alpha)
        self.to(self.device)

        self.register_buffer('reference_x', torch.zeros(num_classes, 3, 224, 224))
        self.register_buffer('reference_y', torch.arange(num_classes).long())
        self.reference_done = set()

        self.has_native_measure = True

    def mixup_(self, x, y):
        perm = torch.randperm(len(x)).to(x.device)
        return self.mixup_pair_(x, x[perm], y, y[perm])

    def mixup_pair_(self, x1, x2, y1, y2):
        lamb = self.beta_distr.sample().item()
        mix_x = lamb * x1 + (1 - lamb) * x2
        mix_y = lamb * y1 + (1 - lamb) * y2
        return mix_x, mix_y

    def process_minibatch(self, x, y):
        if self.reference_x.is_cuda:
            self.reference_x = self.reference_x.cpu()
            self.reference_y = self.reference_y.cpu()

        t = torch.nn.functional.one_hot(y, self.num_classes).float()

        if len(self.reference_done) < self.num_classes:
            y_long = y.long().cpu()
            self.reference_x[y_long] = x.cpu()
            self.reference_done |= set(y_long.tolist())

        mix_x, mix_y = self.mixup_(x, t)
        mix_x = mix_x.to(self.device, non_blocking=True)
        mix_y = mix_y.to(self.device, non_blocking=True)

        return mix_x, mix_y

    def uncertainty(self, x, iterations=5):
        x = x.to(self.device)
        t = self.forward(x).softmax(1)

        scores = 0
        for iteration in range(iterations):
            ref_p = torch.randperm(len(x))
            ref_x = self.reference_x[ref_p].to(self.device)
            ref_t = torch.nn.functional.one_hot(
                self.reference_y[ref_p].to(self.device), self.num_classes).to(
                    self.device)

            mix_x, mix_t = self.mixup_pair_(x, ref_x, t, ref_t)
            mix_t_hat = self.forward(mix_x).softmax(1)
            scores += (mix_t_hat - mix_t).norm(2, 1).pow(2)

        return scores / iterations


if __name__ == '__main__':
    pass
