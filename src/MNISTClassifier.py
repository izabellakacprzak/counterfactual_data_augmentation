import numpy as np
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
import torchvision.transforms as TF

from utils.evaluate import get_confusion_matrix
from params import *
from sklearn.metrics import f1_score
from dscm.generate_counterfactuals import generate_counterfactual_for_x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ConvNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNet, self).__init__()
        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        # change the input size to number of channels
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=6, stride=2, padding=3, bias=False)
        # change the output size to number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, out_channels)
        self.model = self.model.to(device)

        self.labels = list(range(0, out_channels))
        self.out_channels = out_channels

    def forward(self, x):
        return self.model(x)

    def regularisation(self, x, metrics, labels, logits):
        cfs = []
        for i in range(len(x)):
            img = x[i][0].float() * 254
            img = TF.Pad(padding=2)(img).type(torch.ByteTensor).unsqueeze(0)
            x_cf = generate_counterfactual_for_x(img, metrics['thickness'][i], metrics['intensity'][i], labels[i])
            cfs.append(torch.from_numpy(x_cf).unsqueeze(0).float().to(device))
        
        cfs = torch.stack(cfs)
        logits_cf = self.model(cfs)
        return LAMBDA * MSE(logits, logits_cf)


def train_MNIST(model, train_loader, test_loader, do_cf_regularisation=False):
    accs = []
    f1s = []
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS):
        _, _, acc, f1 = test_MNIST(model, test_loader)
        accs.append(acc)
        f1s.append(f1)

        model.train()
        for batch_idx, (data, metrics, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimiser.zero_grad()
            logits = model(data)
            loss = LOSS_FN(logits, target)
            if do_cf_regularisation:
                loss += model.regularisation(data, metrics, target, logits)
            loss.backward()
            optimiser.step()
            if batch_idx % 10 == 0:
                print('[Train loop]\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            # train_losses.append(loss.item())
            # train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

    return accs, f1s


def test_MNIST(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    c_matrix = np.zeros((model.out_channels, model.out_channels))
    precision = np.zeros(model.out_channels)
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, _, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += LOSS_FN(output, target)
            _, pred = torch.max(output, 1)
            # pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().cpu()

            y_true += target.tolist()
            y_pred += pred.cpu().numpy().tolist()
            # y_pred += pred.numpy().T[0].tolist()
    test_loss /= len(test_loader.dataset)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    # test_losses.append(test_loss)
    confusion_matrix = get_confusion_matrix(y_pred, y_true)
    acc = 100. * correct / len(test_loader.dataset)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('[Test loop]\tF1 score: ' + str(f1)+'\n')
    print('[Test loop]\tTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return y_pred, y_true, acc, f1
