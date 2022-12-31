import numpy as np
import torch.nn as nn

from evaluation.evaluate import pretty_print_evaluation
from params import *

class ConvNet(torch.nn.Module):
    def __init__(self, train_loader, test_loader, in_channels, out_channels):
        super(ConvNet, self).__init__()
        # self.conv1 = torch.nn.Conv2d(in_channels, 10, kernel_size=5)
        # self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = torch.nn.Dropout2d()
        # self.fc1 = torch.nn.Linear(320, 50)
        # self.fc2 = torch.nn.Linear(50, out_channels)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_channels))

        self.train_loader = train_loader
        self.optimiser = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE)
        self.loss_fn = LOSS_FN
        self.test_loader = test_loader
        self.labels = list(range(0, out_channels))
        self.out_channels = out_channels

    def forward(self, x):
        # x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        # x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = torch.nn.functional.relu(self.fc1(x))
        # x = torch.nn.functional.dropout(x, training=self.training)
        # x = self.fc2(x)
        # return torch.nn.functional.log_softmax(x, -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_MNIST(self, epoch):
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # data, target = data.to(device), target.to(device).float()
            self.optimiser.zero_grad()
            output = self.forward(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimiser.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
            # train_losses.append(loss.item())
            # train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

    def test_MNIST(self):
        self.eval()
        test_loss = 0
        correct = 0
        c_matrix = np.zeros((self.out_channels, self.out_channels))
        precision = np.zeros(self.out_channels)
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data, target in self.test_loader:
                # data, target = data.to(device), target.to(device).float()
                output = self.forward(data)
                test_loss += self.loss_fn(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

                y_true += target.tolist()
                y_pred += pred.numpy().T[0].tolist()
                # c_matrix += confusion_matrix(y_true, y_pred, labels=self.labels)
                # precision += precision_score(y_true, y_pred, average=None, labels=labels)
        test_loss /= len(self.test_loader.dataset)
        # test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        pretty_print_evaluation(np.asarray(y_pred), np.asarray(y_true), self.labels)
