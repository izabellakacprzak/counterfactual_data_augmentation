import torch
from torchvision import datasets

from params import *
from utils.utils import *
from ..utils.colors import prepare_colored_mnist

class ColoredMNIST(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None, bias_conflicting_percentage=1, method=AugmentationMethod.NONE):
    super(ColoredMNIST, self).__init__('files', transform=transform,
                                target_transform=target_transform)

    prepare_colored_mnist(datasets.mnist.MNIST('files', train=True, download=True), datasets.mnist.MNIST(self.root, train=False, download=True), bias_conflicting_percentage)
    if train:
      self.data_color_label_tuples = torch.load(TRAIN_FILE+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt")
      if method != AugmentationMethod.NONE:
        self.data_color_label_tuples = debias_mnist(train_data=self.data_color_label_tuples, method=method)
      
      labels = [x[2] for x in self.data_color_label_tuples]
      from collections import Counter
      print(Counter(labels).keys())
      print(Counter(labels).values())
    else:
      self.data_color_label_tuples = torch.load(TEST_FILE)

  def __getitem__(self, index):
    img, _, target = self.data_color_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_color_label_tuples)