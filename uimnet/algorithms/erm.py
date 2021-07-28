#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import torch.nn as nn
from uimnet import utils
from uimnet.modules.temperature import Temperature
from uimnet.algorithms.base import Algorithm
import torchvision
import torch.cuda.amp as amp
import numpy as np


class ERM(Algorithm):
    """
      Empirical Risk Minimization
      Imitates: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    HPARAMS = dict(Algorithm.HPARAMS)
    HPARAMS.update({
        "lr": (0.1, lambda: float(10**np.random.uniform(-2, -0.3))),
        "momentum": (0.9, lambda: float(np.random.choice([0.5, 0.9, 0.99]))),
        "weight_decay": (1e-4, lambda: float(10**np.random.uniform(-5, -3)))
    })

    def __init__(
        self,
        num_classes,
        arch,
        device="cuda",
        seed=0,
        use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):

        super(ERM, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision, sn=sn, sn_coef=sn_coef, sn_bn=sn_bn)

        self.loss = torch.nn.CrossEntropyLoss()

    def construct_networks(self):
        # init network, conveniently decomposed in featurizer and classifier
        featurizer = torchvision.models.__dict__[self.arch](
            num_classes=self.num_classes,
            pretrained=False,
            # important when using large batch sizes (Goyal & al 2017)
            zero_init_residual=True)

        classifier = torch.nn.Linear(
            featurizer.fc.in_features,
            featurizer.fc.out_features, bias=True)
        torch.nn.init.normal_(classifier.weight, mean=0, std=0.01)
        featurizer.fc = utils.Identity()

        return dict(featurizer=featurizer, classifier=classifier)

    def process_minibatch(self, x, y):
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y

    def _forward(self, x):
      features = self.networks['featurizer'](x.to(self.device))
      return self.networks['classifier'](features)

    def update(self, x, y, epoch=None):
        if epoch is not None:
            self.adjust_learning_rate_(epoch)

        for param in self.parameters():
            param.grad = None

        x, y = self.process_minibatch(x, y)
        with amp.autocast(enabled=self.use_mixed_precision):
            features = self.networks['featurizer'](x)
            predictions = self.networks['classifier'](features)
            loss = self.loss(predictions, y)
            cost = loss + self.hparams['weight_decay'] * self.get_l2_reg()

        self.grad_scaler.scale(cost).backward()

        for optimizer in self.optimizers.values():
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

        return {
          'loss': loss.item(),
          'cost': cost.item()
        }


if __name__ == '__main__':
    pass
