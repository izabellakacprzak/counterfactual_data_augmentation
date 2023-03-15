import numpy as np
import torch.nn as nn
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights

from utils.evaluate import pretty_print_evaluation
from params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        # change the input size to number of channels
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=6, stride=2, padding=3, bias=False)
        # change the output size to number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, out_channels)
        self.model = self.model.to(device)
        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(in_channels, 16, kernel_size=3),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU())

        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(16, 16, kernel_size=3),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(16, 64, kernel_size=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())
        
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU())

        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))

        # self.fc = nn.Sequential(
        #     nn.Linear(64 * 4 * 4, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, out_channels))

        self.labels = list(range(0, out_channels))
        self.out_channels = out_channels

    def forward(self, x):
        with torch.no_grad():
            # x = self.transforms(x)
            y_pred = self.resnet18(x)
            return y_pred.argmax(dim=1)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # return x


def train_MNIST(model, train_loader, test_loader, accs):
    optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS):
        _, _, acc = test_MNIST(model, test_loader)

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            output = model.model(data)
            loss = LOSS_FN(output, target)
            loss.backward()
            optimiser.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            # train_losses.append(loss.item())
            # train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

        accs.append(acc)


def test_MNIST(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    c_matrix = np.zeros((model.out_channels, model.out_channels))
    precision = np.zeros(model.out_channels)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model.model(data)
            test_loss += LOSS_FN(output, target)
            _, pred = torch.max(output, 1)
            # pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

            y_true += target.tolist()
            y_pred += pred.numpy().cpu().tolist()
            # y_pred += pred.numpy().T[0].tolist()
    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))
    return np.asarray(y_pred), np.asarray(y_true), accuracy
