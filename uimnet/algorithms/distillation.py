#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import torchvision
from uimnet import utils
from uimnet.algorithms.erm import ERM
import torch.cuda.amp as amp
import numpy as np


class RND(ERM):
    """
    Random Network Distillation
    https://arxiv.org/abs/1810.12894
    """
    HPARAMS = dict(ERM.HPARAMS)
    HPARAMS.update({
        "teacher_width": (128, lambda: int(np.random.choice([64, 128, 256]))),
        "teacher_depth": (3, lambda: int(np.random.choice([2, 3, 4]))),
        "reg": (0.1, lambda: float(10**np.random.uniform(-2, 2))),
        "penalty": (0.1, lambda: float(10**np.random.uniform(-2, 2)))
    })
    def __init__(
        self,
        num_classes,
        arch,
        device="cuda",
        seed=0,
        use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):

        super(RND, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision, sn=sn, sn_coef=sn_coef, sn_bn=sn_bn)

        self.loss_imitation = torch.nn.MSELoss()

        self.has_native_measure = True

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

        imitator_network = utils.MLP(
            [classifier.in_features] +
            [self.hparams["teacher_width"]] * self.hparams["teacher_depth"])

        imitator_target = utils.MLP(
            [classifier.in_features] +
            [self.hparams["teacher_width"]] * self.hparams["teacher_depth"])

        return dict(featurizer=featurizer,
                    classifier=classifier,
                    imitator_network=imitator_network,
                    imitator_target=imitator_target)

    def update(self, x, y, epoch=None):
        if epoch is not None:
            self.adjust_learning_rate_(epoch)

        for param in self.parameters():
            param.grad = None

        x, y = self.process_minibatch(x, y)
        with amp.autocast(enabled=self.use_mixed_precision):
            h = self.networks['featurizer'](x)
            s = self.networks['classifier'](h)

            loss_prediction = self.loss(s, y)
            cost_prediction = loss_prediction +\
                self.hparams['weight_decay'] * self.get_l2_reg()

            loss_imitation = self.loss_imitation(
                    self.networks['imitator_network'](h),
                    self.networks['imitator_target'](h))

            cost_imitation = self.hparams['reg'] * loss_imitation +\
                    self.hparams['penalty'] * self.imitator_penalty_()

        self.grad_scaler.scale(cost_prediction + cost_imitation).backward()

        for name, optimizer in self.optimizers.items():
            if name != 'imitator_target':
                self.grad_scaler.step(optimizer)
                self.grad_scaler.update()

        return {
            'loss': loss_prediction.item(),
            'cost': cost_prediction.item(),
            'imitation': loss_imitation.item()}

    def imitator_penalty_(self):
        return 0

    def uncertainty(self, x):
        f = self.networks['featurizer'](x.to(self.device))
        return (self.networks['imitator_network'](f) -
                self.networks['imitator_target'](f)).pow(2).mean(1)


class OC(RND):
    """
    Orthogonal certificates
    https://arxiv.org/abs/1811.00908
    """
    HPARAMS = RND.HPARAMS

    def __init__(self,
                 num_classes,
                 arch,
                 device="cuda",
                 seed=0,
                 use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):
        super(OC, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision, sn=sn, sn_coef=sn_coef, sn_bn=sn_bn)

    def construct_networks(self):
        def init_imitator_(m):
            classname = m.__class__.__name__
            if classname.find('Linear') != -1:
                torch.nn.init.orthogonal_(m.weight)

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

        imitator_target = torch.nn.Linear(
            classifier.in_features,
            self.hparams["teacher_width"],
            bias=False)

        imitator_target.weight.data.fill_(0)

        imitator_network = utils.MLP(
            [classifier.in_features] +
            [self.hparams["teacher_width"]] * self.hparams["teacher_depth"])

        imitator_network.apply(init_imitator_)

        return dict(featurizer=featurizer,
                    classifier=classifier,
                    imitator_network=imitator_network,
                    imitator_target=imitator_target)

    def imitator_penalty_(self):
        regularizer = 0
        for layer in self.networks['imitator_network'].children():
            if layer.__class__.__name__.find('Linear') != -1:
                eye = torch.eye(layer.weight.size(0)).to(self.device)
                regularizer = (
                    layer.weight @ layer.weight.t() - eye).pow(2).mean()

        return regularizer
