#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
from uimnet import utils
from uimnet.algorithms.erm import ERM
import numpy as np
from uimnet import numerics


class MCDropout(ERM):
    HPARAMS = dict(ERM.HPARAMS)
    HPARAMS.update({
        "dropout": (0.05, lambda: float(np.random.choice([0.05, .1, .2]))),
        "m": (10, lambda: int(np.random.choice([10])))
    })

    def __init__(
            self,
            num_classes,
            arch,
            device="cuda",
            seed=0,
            use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):

        super(MCDropout, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision,
            sn=sn,
            sn_coef=sn_coef,
            sn_bn=sn_bn)

        self.has_native_measure = True
        dropout = self.hparams["dropout"]
        self.m = self.hparams["m"]

        def convert_relu_to_dropout(module):
            for child_name, child in module.named_children():
                if isinstance(child, torch.nn.ReLU):
                    setattr(module, child_name, utils.MCReLUDropout(dropout))
                else:
                    convert_relu_to_dropout(child)

        convert_relu_to_dropout(self)


    def entropy_(self, p):
        zeros = torch.zeros(()).to(p.device)
        h = p * torch.where(p == 0, zeros, torch.log(p))
        return h.sum(1).mul(-1)

    def uncertainty(self, x):
        x = x.to(self.device)
        predictions = [
            self.networks['classifier'](self.networks['featurizer'](x))
            for _ in range(self.m)
        ]

        predictions_members = [p.softmax(1) for p in predictions]
        predictions_ensemble = torch.stack(predictions_members).mean(0)

        entropy_members = [self.entropy_(p) for p in predictions_members]
        entropy_ensemble = self.entropy_(predictions_ensemble)

        return entropy_ensemble - torch.stack(entropy_members).mean(0)

    def _forward(self, x):
      x = x.to(self.device)

      def _logits(x):
        feats = self.networks['featurizer'](x)
        return self.networks['classifier'](feats)

      logits = _logits(x)
      N, K = logits.shape
      M = self.m

      all_logits = torch.zeros(N, M, K, dtype=logits.dtype, device='cpu')

      all_logits[:, 0:, :] = logits.cpu().unsqueeze(1)

      for m in range(1, M):
        all_logits[:, m:, :] = _logits(x).cpu().unsqueeze(1)

      return numerics.log_marginalization_from_logits(all_logits).to(x.device)







if __name__ == '__main__':
    pass
