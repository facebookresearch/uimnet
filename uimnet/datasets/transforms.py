#!/usr/bin/env python3
#
# # Copyright (c) 2021 Facebook, inc. and its affiliates. All Rights Reserved
#
#
from torchvision import transforms


def get_transforms():
  normalize = transforms.Normalize(
      mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  )

  all_transforms = {
      'train':
          transforms.Compose(
              [
                  transforms.RandomResizedCrop(224),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(), normalize
              ]
          ),
      'eval':
          transforms.Compose(
              [
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  normalize,
              ]
          ),
  }
  return all_transforms


if __name__ == '__main__':
  pass
