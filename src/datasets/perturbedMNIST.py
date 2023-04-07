import torch
from torchvision import datasets

from params import *
from utils.utils import *

import sys
sys.path.append("..")

from utils.perturbations import prepare_perturbed_mnist

class PerturbedMNIST(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None, bias_conflicting_percentage=1, method=AugmentationMethod.NONE):
    super(PerturbedMNIST, self).__init__('files', transform=transform,
                                target_transform=target_transform)

    prepare_perturbed_mnist(datasets.mnist.MNIST('files', train=True, download=True), datasets.mnist.MNIST(self.root, train=False, download=True), bias_conflicting_percentage)
    if train:
      self.data_label_tuples = torch.load("data/train_perturbed"+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt")
      if method != AugmentationMethod.NONE:
        self.data_label_tuples = debias_mnist(train_data=self.data_label_tuples, bias_conflicting_perc=bias_conflicting_percentage, method=method)
      
    else:
      self.data_label_tuples = torch.load("data/test_perturbed.pt")

  def __getitem__(self, index):
    _, img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)