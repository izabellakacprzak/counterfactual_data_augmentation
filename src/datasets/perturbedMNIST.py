import torch
from torchvision import datasets
import pandas as pd

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
      self.data = torch.load("data/train_perturbed"+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt")
      self.metrics = pd.read_csv("data/train_perturbed_mnist_metrics"+"_"+str(bias_conflicting_percentage).replace(".", "_")+".csv", index_col='index').to_dict('records')
      
      if method != AugmentationMethod.NONE and method != AugmentationMethod.CF_REGULARISATION:
        self.data, self.metrics = debias_mnist(train_data=self.data, train_metrics=self.metrics, method=method)
      
    else:
      self.data = torch.load("data/test_perturbed.pt")
      self.metrics = pd.read_csv("data/test_perturbed_mnist_metrics.csv", index_col='index').to_dict('records')

  def __getitem__(self, index):
    _, img, target = self.data_label_tuples[index]
    metrics = {k: torch.tensor(float(self.metrics[index][k])) for k in ['thickness', 'intensity', 'bias_aligned']}

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, metrics, target

  def __len__(self):
    return len(self.data)