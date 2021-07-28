#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
import torch
import torch.nn as nn
from torch.nn import (BatchNorm1d, BatchNorm2d, BatchNorm3d, Conv1d, Conv2d, )
from uimnet import utils
from uimnet.modules.spectral_normalization.spectral_fc import SNLinear
from uimnet.modules.spectral_normalization.spectral_bn import (SpectralBatchNorm1d, SpectralBatchNorm2d, SpectralBatchNorm3d, )
from uimnet.modules.spectral_normalization.spectral_conv import SNConv2d, SNConv1d
def monkey_patch_layers(module, sn_coef=1, sn_bn=False, verbose=True):
  """
  Applies spetral normalization to batch normalization layers
  """
  is_linear = isinstance(module, (torch.nn.Linear, ))
  is_conv = isinstance(module,
                       (nn.Conv1d, nn.Conv2d, nn.Conv3d)
                       )
  is_bn = isinstance(module,
                     (nn.BatchNorm1d, nn.BatchNorm2d, torch.nn.BatchNorm3d)
                     )
  if is_linear:
    utils.message(f'Wrapping linear module {module}')
    return SNLinear(in_features=module.in_features,
                    out_features=module.out_features,
                    bias=False if module.bias is None else True,
                    sn_coef=sn_coef,
                    num_itrs=1,
                    eps=1e-12
                    )
  elif is_conv:
    CLS = dict(Conv1d=SNConv1d, Conv2d=SNConv2d)[module.__class__.__name__]
    utils.message(f'Wrapping convolutional module {module}')
    return CLS(in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    #bias=module.bias,
                    bias=False if module.bias is None else True,
                    sn_coef=sn_coef,
                    num_itrs=1,
                    eps=1e-12
                    )
  elif is_bn and sn_bn:
    # Swap batchnormalization for spectral batchnormalization

    CLS = dict(
      BatchNorm1d=SpectralBatchNorm1d,
      BatchNorm2d=SpectralBatchNorm2d,
      BatchNorm3d=SpectralBatchNorm3d)[module.__class__.__name__]

    utils.message(f'Wrapping Batch Norm layer {module}')
    return CLS(module.num_features, coeff=sn_coef,
               eps=1e-5,
               momentum=0.1,
               affine=True
               )
  else:
    return module


if __name__ == '__main__':
  pass
