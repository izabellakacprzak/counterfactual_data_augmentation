from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from torchvision import datasets
from MNISTClassifier import ConvNet
from utils import prepare_colored_mnist

from params import *

class ColoredMNIST(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__('files', transform=transform,
                                target_transform=target_transform)

    prepare_colored_mnist(datasets.mnist.MNIST('files', train=True, download=True), datasets.mnist.MNIST(self.root, train=False, download=True))
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

train_dataset = ColoredMNIST(train=True, transform=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                   ]))
print(train_dataset.data_label_tuples[0])
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_loader = DataLoader(
    ColoredMNIST(train=False, transform=transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
    ])),
    batch_size=1000, shuffle=True)

model = ConvNet(train_loader=train_loader, test_loader=test_loader, in_channels=3, out_channels=NUM_OF_CLASSES)
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