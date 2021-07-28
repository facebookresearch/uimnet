#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import gpytorch
import torch

import numpy as np
from uimnet import utils
from uimnet.algorithms.base import Algorithm
import torchvision
import torch.cuda.amp as amp

from sklearn import cluster


class GP(gpytorch.models.ApproximateGP):
    def __init__(self,
                 num_outputs,
                 some_features,
                 kernel="RBF",
                 num_inducing_points=20):
        batch_shape = torch.Size([num_outputs])

        initial_inducing_points = self.cluster_(
            some_features, num_inducing_points)

        initial_lengthscale = torch.pdist(some_features).mean() / 2

        variational_distribution = \
            gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points, batch_shape=batch_shape)

        variational_strategy = \
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.VariationalStrategy(
                    self, initial_inducing_points, variational_distribution),
                num_tasks=num_outputs)

        super(GP, self).__init__(variational_strategy)

        kwargs = {
            # These two options gave worse results
            # "ard_num_dims": int(some_features.size(1)),
            # "batch_shape": batch_shape
        }

        if kernel == "RBF":
            kernel = gpytorch.kernels.RBFKernel(**kwargs)
        elif kernel == "Matern12":
            kernel = gpytorch.kernels.MaternKernel(nu=1 / 2, **kwargs)
        elif kernel == "Matern32":
            kernel = gpytorch.kernels.MaternKernel(nu=3 / 2, **kwargs)
        elif kernel == "Matern52":
            kernel = gpytorch.kernels.MaternKernel(nu=5 / 2, **kwargs)
        elif kernel == "RQ":
            kernel = gpytorch.kernels.RQKernel(**kwargs)
        else:
            raise ValueError("Specified kernel not known.")

        kernel.lengthscale = initial_lengthscale * torch.ones_like(
            kernel.lengthscale)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def cluster_(self, some_features, k):
        kmeans = cluster.MiniBatchKMeans(n_clusters=k, batch_size=k)
        kmeans.fit(some_features.detach().cpu())
        return torch.from_numpy(kmeans.cluster_centers_).float()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))


class DKL_GP(gpytorch.Module):
    def __init__(self, feature_extractor, gp):
        """
        This wrapper class is necessary because ApproximateGP (above) does some
        magic on the forward method which is not compatible with a
        feature_extractor.
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.gp = gp

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp(features)


class DUE(Algorithm):
    HPARAMS = dict(Algorithm.HPARAMS)
    HPARAMS.update({
        "lr": (0.1, lambda: float(10**np.random.uniform(-2, -0.3))),
        "momentum": (0.9, lambda: float(np.random.choice([0.5, 0.9, 0.99]))),
        "weight_decay": (1e-4, lambda: float(10**np.random.uniform(-5, -3))),
        "num_inducing_points": (20, lambda: np.random.choice([20, 100, 300])),
        "kernel": ("RBF", lambda: np.random.choice(
            ["RBF", "RQ", "Matern12", "Matern32", "Matern52"]))
    })

    def __init__(self,
                 num_classes,
                 arch,
                 device="cuda",
                 seed=0,
                 use_mixed_precision=False, sn=False, sn_coef=1, sn_bn=False):
        super(DUE, self).__init__(
            num_classes,
            arch,
            device,
            seed,
            use_mixed_precision=use_mixed_precision,
            sn=sn,
            sn_coef=sn_coef,
            sn_bn=sn_bn)
        self.has_native_measure = True

    def construct_networks(self, dataset=None):
        self.likelihood = gpytorch.likelihoods.SoftmaxLikelihood(
            num_classes=self.num_classes, mixing_weights=False)

        featurizer = torchvision.models.__dict__[self.arch](
            num_classes=self.num_classes,
            pretrained=False,
            zero_init_residual=True)
        num_features = featurizer.fc.in_features
        featurizer.fc = utils.Identity()

        if dataset is not None:
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=32, shuffle=True)

            some_features = []
            with torch.no_grad():
                for i, datum in enumerate(loader):
                    some_features.append(featurizer(datum['x']).cpu())
                    if i == 30:
                        break

            some_features = torch.cat(some_features)
            self.dataset_length = len(dataset)
        else:
            # else not executed for training, following are placeholders
            # that should be replaced when doing classifier.load_state_dict()
            some_features = torch.randn(
                self.hparams["num_inducing_points"],
                num_features)
            self.dataset_length = 240000

        self.classifier = GP(
            self.num_classes,
            some_features,
            self.hparams["kernel"],
            self.hparams["num_inducing_points"])

        self.loss = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.classifier, num_data=self.dataset_length)

        return dict(featurizer=featurizer)

    def setup_optimizers(self):
        self.lr = self.hparams["lr"]
        self.optimizers['featurizer'] = torch.optim.SGD(
            self.networks['featurizer'].parameters(),
            lr=self.lr,
            momentum=self.hparams['momentum'],
            weight_decay=0)

        self.optimizers['classifier'] = torch.optim.SGD(
            self.classifier.parameters(),
            lr=self.lr,
            momentum=self.hparams['momentum'],
            weight_decay=0)

    def process_minibatch(self, x, y):
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        return x, y

    def predictions(self, x):
        return self.classifier(self.networks['featurizer'](x.to(self.device)))

    def update(self, x, y, epoch=None):
        if epoch is not None:
            self.adjust_learning_rate_(epoch)

        for param in self.parameters():
            param.grad = None

        x, y = self.process_minibatch(x, y)
        with amp.autocast(enabled=self.use_mixed_precision):
            loss = -self.loss(self.predictions(x), y)
            cost = loss + self.hparams['weight_decay'] * self.get_l2_reg()

        self.grad_scaler.scale(cost).backward()

        for optimizer in self.optimizers.values():
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

        return {
          'loss': loss.item(),
          'cost': cost.item()
        }

    def _forward(self, x):
        with gpytorch.settings.num_likelihood_samples(32):
            output = self.predictions(x)
            output = output.to_data_independent_dist()
            output = self.likelihood(output).logits.mean(0)
        return output

    def uncertainty(self, x):
        with gpytorch.settings.num_likelihood_samples(32):
            output = self.predictions(x)
            output = output.to_data_independent_dist()
            output = self.likelihood(output).probs.mean(0)
        return -(output * output.log()).sum(1)
