import torch
from torchvision import datasets
import pandas as pd

from utils.params import *

import sys
sys.path.append("..")

from utils.utils import *
from utils.perturbations import prepare_perturbed_mnist

class PerturbedMNIST(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None, bias_conflicting_percentage=1, method=DebiasingMethod.NONE):
    super(PerturbedMNIST, self).__init__('files', transform=transform,
                                target_transform=target_transform)

    self.group_counts = prepare_perturbed_mnist(datasets.mnist.MNIST('files', train=True, download=True), datasets.mnist.MNIST(self.root, train=False, download=True), bias_conflicting_percentage)
    if train:
      self.data = torch.load("{}_{}".format(TRAIN_PERTURBED_DATA, bias_conflicting_percentage).replace(".", "_")+".pt")
      self.metrics = pd.read_csv("{}_{}".format(TRAIN_PERTURBED_METRICS, bias_conflicting_percentage).replace(".", "_")+".csv", index_col='index').to_dict('records')
    else:
      self.data = torch.load(TEST_PERTURBED_DATA)
      self.metrics = pd.read_csv(TEST_PERTURBED_METRICS, index_col='index').to_dict('records')
    
    if not method in [DebiasingMethod.NONE, DebiasingMethod.CF_REGULARISATION, DebiasingMethod.MIXUP]:
      self.data, self.metrics = debias_perturbed_mnist(train_data=self.data, train_metrics=self.metrics, method=method)
      
  def __getitem__(self, index):
    img, target = self.data[index]
    metrics = {k: torch.tensor(float(self.metrics[index][k])) for k in ['thickness', 'intensity', 'bias_aligned']}
    
    # group for group DRO loss calculation
    group_idx = 0 if self.metrics[index]['thickness'] <= 1.5 else 1
    metrics['group_idx'] = torch.tensor(group_idx)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, metrics, target

  def __len__(self):
    return len(self.data)