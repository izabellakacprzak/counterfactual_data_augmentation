import torch
from torchvision import datasets
from keras.datasets import mnist
import medmnist
from medmnist import INFO
import os

from torchvision import datasets
import torchvision.transforms as transforms

from params import *
from utils.utils import *

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

class MedMNIST(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None, bias_conflicting_percentage=1, method=AugmentationMethod.NONE):
    super(MedMNIST, self).__init__('files', transform=transform,
                                target_transform=target_transform)

    data_flag = 'pathmnist'
    download = True

    info = INFO[data_flag]
    # task = info['task']
    self.in_channels = info['n_channels']
    self.out_channels = len(info['label'])

    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(split='train', transform=target_transform, download=download, as_rgb=False)
    test_dataset = DataClass(split='test', transform=target_transform, download=download, as_rgb=False)
    train_dataset.labels = [val for sublist in train_dataset.labels for val in sublist]
    test_dataset.labels = [val for sublist in test_dataset.labels for val in sublist]

    prepare_med_noisy_mnist(train_dataset, test_dataset, bias_conflicting_percentage)
    if train:
      self.data_noise_label_tuples = torch.load(MED_TRAIN_FILE+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt")
      
      if method != AugmentationMethod.NONE:
        self.data_noise_label_tuples = debias_mnist(train_data=self.data_noise_label_tuples, method=method)
      
      labels = [x[2] for x in self.data_noise_label_tuples]
      from collections import Counter
      print(Counter(labels).keys())
      print(Counter(labels).values())
    else:
      self.data_noise_label_tuples = torch.load(MED_TEST_FILE)

  def __getitem__(self, index):
    img, _, target = self.data_noise_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_noise_label_tuples)


data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307, 0.1307, 0.0), std=(0.3081, 0.3081, 0.3081))
])