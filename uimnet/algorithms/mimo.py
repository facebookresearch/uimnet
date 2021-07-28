#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
from uimnet import utils
from uimnet.algorithms.base import Algorithm
import torchvision
import torch.cuda.amp as amp
import numpy as np
from torch.nn import functional as F
from uimnet import numerics





class MIMO(Algorithm):
    """
    Implements Training independent subnetworks  for robusts prediction.
    https://openreview.net/forum?id=OGg9XnKxFAH
    """
    HPARAMS = dict(Algorithm.HPARAMS)
    HPARAMS.update({
        "lr": (0.1, lambda: float(10**np.random.uniform(-2, -0.3))),
        "momentum": (0.9, lambda: float(np.random.choice([0.5, 0.9, 0.99]))),
        "weight_decay": (1e-4, lambda: float(10**np.random.uniform(-5, -3))),
        "rho": (0.6, lambda: float(np.random.uniform(0., 1.))),
        "M": (2, lambda: int(np.random.randint(2, 5))),
        "batch_rep": (2, lambda: int(np.random.randint(1, 5))),
    })

    def __init__(
            self,
            num_classes,
            arch,
            device="cuda",
            seed=0,
            use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):

        super(MIMO, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision, sn=sn, sn_coef=sn_coef, sn_bn=sn_bn)

        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def construct_networks(self):
        # init network, conveniently decomposed in featurizer and classifier
        featurizer = torchvision.models.__dict__[self.arch](
            num_classes=self.num_classes,
            pretrained=False,
            # important when using large batch sizes (Goyal & al 2017)
            zero_init_residual=True)

        featurizer.conv1 = torch.nn.Conv2d(
            in_channels=featurizer.conv1.in_channels * self.hparams["M"],
            out_channels=featurizer.conv1.out_channels,
            kernel_size=featurizer.conv1.kernel_size,
            stride=featurizer.conv1.stride,
            padding=featurizer.conv1.padding,
            bias=featurizer.conv1.bias)

        classifier = torch.nn.Linear(
            featurizer.fc.in_features,
            featurizer.fc.out_features * self.hparams["M"], bias=True)

        featurizer.fc = utils.Identity()

        return dict(featurizer=featurizer, classifier=classifier)

    def get_logits(self, x):
        Nb, M, C, H, W = x.shape
        x = x.view(Nb, M * C, H, W)
        features = self.networks['featurizer'](x)
        return self.networks['classifier'](
            features).view(Nb, M, self.num_classes)

    def get_features(self, x):
        x = x.unsqueeze(1).repeat(
            1, self.hparams["M"], 1, 1, 1).to(self.device, non_blocking=True)
        Nb, M, C, H, W = x.shape
        x = x.view(Nb, M * C, H, W)
        features = self.networks['featurizer'](x)

        return features


    # def _forward(self, x):
    #     x = x.unsqueeze(1).repeat(
    #         1, self.hparams["M"], 1, 1, 1).to(self.device, non_blocking=True)
    #     logits = self.get_logits(x)  # N X M X K
    #     probs = F.softmax(logits, dim=2).mean(dim=1)
    #     return torch.log1p(probs - 1)


    def process_minibatch(self, x, y):

        x = x.to(self.device, non_blocking=True) # expected (N, C, H, W)
        y = y.to(self.device, non_blocking=True) # expected (N, )

        N, C, H, W = x.shape
        brep = self.hparams["batch_rep"]
            
        def _shuffle(el):
            return el[torch.randperm(len(el))]

        if not self.training:
            x = x.unsqueeze(1).repeat(1, self.hparams["M"], 1, 1, 1)
            y = y.unsqueeze(1).repeat(1, self.hparams["M"])
            return x, y

        indices = _shuffle(torch.arange(x.shape[0]).repeat(brep))
        # Probability of example repetition
        # This is implemented by keeping a portion of the examples
        # align throughout the different heads inputs.
        naligned = int(len(indices) * self.hparams["rho"])

        xs, ys = [], []
        for _ in range(self.hparams["M"]):
            indices_aligned = indices[:naligned]
            indices_misaligned = _shuffle(indices[naligned:])
            idx = torch.cat([indices_aligned, indices_misaligned], dim=0)
            xs += [x[idx].unsqueeze(1)]
            ys += [y[idx][:, None]]
        x = torch.cat(xs, dim=1)  # N X M X C X H X W
        y = torch.cat(ys, dim=1)  # N X M
        return x, y

    def _forward(self, x):
        x = x.unsqueeze(1).repeat(
            1, self.hparams["M"], 1, 1, 1).to(self.device, non_blocking=True)
        logits = self.get_logits(x)  # N X M X K
        return numerics.log_marginalization_from_logits(logits)

    def update(self, x, y, epoch=None, num_epochs=None):

        if epoch is not None:
            self.adjust_learning_rate_(epoch)
        
        for param in self.parameters():
            param.grad = None
        
        x, y = self.process_minibatch(x, y)
        Nb, M, C, H, W = x.shape

        with amp.autocast(enabled=self.use_mixed_precision):
            logits = self.get_logits(x)  # Nb X M X K
            K = logits.shape[2]
            losses = self.loss(logits.view(Nb * M, K),
                               y.view(Nb * M, )).view(Nb, M)
            loss = losses.sum(dim=1).mean(dim=0)
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
