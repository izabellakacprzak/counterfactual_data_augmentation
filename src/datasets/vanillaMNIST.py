import torch
from torchvision import datasets
from keras.datasets import mnist
import os

from params import *
from utils.utils import *
from ..utils.perturbations import add_perturbations

class VanillaMNIST(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None, undersampled_classes=[], perturbed=False):
    super(VanillaMNIST, self).__init__('files', transform=transform,
                                target_transform=target_transform)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if train:
      self.data, self.targets = x_train, y_train
      if len(undersampled_classes) > 0:
          self.data, self.targets = unbalance_dataset(images=x_train, targets=y_train, undersampled_classes=undersampled_classes, cut_percentage=CUT_PERCENTAGE)
          if perturbed:
            if os.path.exists(TRAIN_PERTURBED_FILE):
              print("File with perturbed images already exists")
              data, targets = zip(*torch.load(TRAIN_PERTURBED_FILE))
              self.data, self.targets = list(data), list(targets)

            else:
              print("File with perturbed images does not exist, generating")
              self.data, self.targets = add_perturbations(images=self.data, targets=self.targets, digits=undersampled_classes)

              torch.save(list(zip(self.data, self.targets)), TRAIN_PERTURBED_FILE)
    else:
      self.data, self.targets = x_test, y_test

  def __getitem__(self, index):
    x = self.data[index]
    y = self.targets[index]
        
    if self.transform:
      x = self.transform(x)
      y = y
        
    return x, y
    
  def __len__(self):
    return len(self.data)