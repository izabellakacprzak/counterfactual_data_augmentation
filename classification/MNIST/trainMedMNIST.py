from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator
from torchvision import datasets
from MNISTClassifier import ConvNet
from params import *
from utils import prepare_med_noisy_mnist

class MedMNIST(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None):
    super(MedMNIST, self).__init__('files', transform=transform,
                                target_transform=target_transform)

    train_dataset = DataClass(split='train', transform=target_transform, download=download, as_rgb=True)
    test_dataset = DataClass(split='test', transform=target_transform, download=download, as_rgb=True)
    train_dataset.labels = [val for sublist in train_dataset.labels for val in sublist]
    test_dataset.labels = [val for sublist in test_dataset.labels for val in sublist]

    prepare_med_noisy_mnist(train_dataset, test_dataset)
    if train:
      self.data_label_tuples = torch.load(TRAIN_FILE)
    else:
      self.data_label_tuples = torch.load(TEST_FILE)

  def __getitem__(self, index):
    img, target = self.data_label_tuples[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data_label_tuples)

# data_flag = 'organmnist3d'
# data_flag = 'breastmnist'
data_flag = 'pathmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307, 0.1307, 0.0), std=(0.3081, 0.3081, 0.3081))
])

# load the data


# from PIL import Image
# img = train_dataset.montage(length=1)
# img = img[10]
# print(type(img))
# img.show()
# img = add_noise(np.asarray(img), "s&p")
# img = Image.fromarray(img.astype('uint8'))
# img.show()
# pil_dataset = DataClass(split='train', download=download, as_rgb=True)
# print(pil_dataset)

# encapsulate data into dataloader form
train_dataset = MedMNIST(train=True, transform=data_transform)
test_dataset = MedMNIST(train=False, transform=data_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=True)

model = ConvNet(train_loader=train_loader, test_loader=test_loader, in_channels=3, out_channels=2)
train_losses = []
train_counter = []
test_losses = []
# test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model.test_MNIST()
for epoch in range(1, EPOCHS):
    model.train_MNIST(epoch)
    model.test_MNIST()