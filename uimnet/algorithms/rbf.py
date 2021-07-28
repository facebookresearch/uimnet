#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import torchvision
from uimnet import utils
from uimnet.algorithms.erm import ERM


class LinearRBF(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearRBF, self).__init__()
        self.layer = torch.nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        return self.layer(x).pow(2).mul(-1).exp()


class RBF(ERM):
    HPARAMS = dict(ERM.HPARAMS)

    def __init__(
            self,
            num_classes,
            arch,
            device="cuda",
            seed=0,
            use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):

        super(RBF, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision, sn=sn, sn_coef=sn_coef, sn_bn=sn_bn)

    def construct_networks(self):

      featurizer = torchvision.models.__dict__[self.arch](
          num_classes=self.num_classes,
          pretrained=False,
          # important when using large batch sizes (Goyal & al 2017)
          zero_init_residual=True)

      classifier = LinearRBF(
      # classifier = torch.nn.Linear(
          featurizer.fc.in_features,
          featurizer.fc.out_features, bias=True)
      #torch.nn.init.normal_(classifier.layer.weight, mean=0, std=0.01)
      featurizer.fc = utils.Identity()
      return dict(featurizer=featurizer, classifier=classifier)


if __name__ == '__main__':
  pass
