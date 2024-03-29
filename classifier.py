import numpy as np
import random
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as TF
import torch.nn.functional as F
import torchvision
import torch.optim.lr_scheduler as lr_scheduler

from utils.utils import mixup_data, DebiasingMethod
from utils.cf_utils import *
from utils.params import *
from sklearn.metrics import f1_score

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

def mnist_regularisation(model, x, metrics, labels, logits):
    from dscm.generate_counterfactuals import generate_counterfactual_for_x
    cfs = []
    for i in range(len(x)):
        img = x[i][0].float() * 254
        img = TF.Pad(padding=2)(img).type(torch.ByteTensor).unsqueeze(0)
        x_cf = generate_counterfactual_for_x(img, metrics['thickness'][i], metrics['intensity'][i], labels[i])
        cfs.append(torch.from_numpy(x_cf).unsqueeze(0).float().to(device))
    
    cfs = torch.stack(cfs)
    logits_cf = model(cfs)
    return LAMBDA * MSE(logits, logits_cf)

def colored_mnist_regularisation(model, x, metrics, labels, logits):
    cfs = []
    for i in range(len(x)):
        x_cf = get_cf_for_colored_mnist(x[i], metrics['color'][i], labels[i], random.randint(0,9))
        x_cf = np.transpose(x_cf, (2, 0, 1))
        cfs.append(torch.from_numpy(x_cf).unsqueeze(0).float().to(device))
    
    cfs = torch.stack(cfs).squeeze(1)
    logits_cf = model(cfs)
    return LAMBDA * MSE(logits, logits_cf)

def chestxray_regularisation(model, x, metrics, labels, logits):
    from dscmchest.generate_counterfactuals import generate_cf
    cfs = []
    obs = {'x':x, 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race'], 'finding':metrics['finding']}
    
    do_a, do_f, do_r, do_s = None, None, None, None
    if random.choice([0,1])==0:
        do_a = random.choice([0,1])
        do_f = 1
    else:
        do_a = 4
        do_f = 0

    if do_a != None:
        do_a = random.randint(do_a*20, do_a*20+19)
    x_cf = generate_cf(obs, do_a, do_f, do_r, do_s)
    cfs = torch.from_numpy(x_cf).float().unsqueeze(1).to(device)
    logits_cf = model(cfs)
    return LAMBDA * MSE(logits, logits_cf)

class DenseNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(in_channels, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024, out_channels, bias=True)
        del preloaded
        self = self.to(device)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        return out
    
    def regularisation(self, data, metrics, target, logits):
        return chestxray_regularisation(self, data, metrics, target, logits)
        
    
class ConvNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvNet, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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
    
    def regularisation(self, data, metrics, target, logits):
        # return mnist_regularisation(self, data, metrics, target, logits)
        return colored_mnist_regularisation(self, data, metrics, target, logits)

def _get_loss(logits, target, loss_fn, do_dro=False, group_idxs=None):
    if do_dro:
        return loss_fn.loss(logits, target, group_idxs)
    else:
        return loss_fn(logits, target)

def run_epoch(model, optimiser, loss_fn, train_loader, epoch, do_dro=False, debiasing_method=DebiasingMethod.NONE):
    model.train()
    for batch_idx, (data, metrics, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimiser.zero_grad()

        if debiasing_method==DebiasingMethod.MIXUP:
            data, targets_a, targets_b, lam = mixup_data(data, target, 1, device==torch.device(GPU))
            logits = model(data)
            loss = lam * loss_fn(logits, targets_a) + (1 - lam) * loss_fn(logits, targets_b)
        else:
            logits = model(data)  
            loss = _get_loss(logits, target, loss_fn, do_dro, metrics['group_idx'].to(device))
            if debiasing_method==DebiasingMethod.CF_REGULARISATION:
                loss += model.regularisation(data, metrics, target, logits)

        loss.backward()
        optimiser.step()
        if batch_idx % 10 == 0:
            print('[Train loop]\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test_classifier(model, test_loader, loss_fn, do_dro=False):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    y_score = []
    attr_true = {}
    with torch.no_grad():
        for data, metrics, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            probs = F.softmax(output, dim=1).tolist()
            y_score += probs
            test_loss = _get_loss(output, target, loss_fn, do_dro, metrics['group_idx'].to(device))
            _, pred = torch.max(output, 1)
            correct += pred.eq(target.data.view_as(pred)).sum().cpu()

            y_true += target.tolist()
            y_pred += pred.cpu().numpy().tolist()
            for k in metrics.keys():
                attr_true[k] = (attr_true[k] if k in attr_true else []) + metrics[k].tolist()
    test_loss /= len(test_loader.dataset)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    acc = 100. * correct / len(test_loader.dataset)
    f1 = f1_score(y_true, y_pred, average='macro')
    print('[Test loop]\tF1 score: ' + str(f1))
    print('[Test loop]\tTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return y_pred, y_true, y_score, attr_true, acc, f1, test_loss

def train_and_evaluate(model, train_loader, valid_loader, test_loader, loss_fn, save_path, do_dro=False, debiasing_method=DebiasingMethod.NONE):
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.LinearLR(optimiser, start_factor=1.0, end_factor=0.5, total_iters=EPOCHS)
    accs = []
    f1s = []

    _, _, _, _, acc_pred, f1, test_loss_prev = test_classifier(model, valid_loader, loss_fn, do_dro)
    count_non_improv = 0
    for epoch in range(1, EPOCHS):
        run_epoch(model, optimiser, loss_fn, train_loader, epoch, do_dro, debiasing_method)
        scheduler.step()
        _, _, _, _, acc, f1, test_loss = test_classifier(model, valid_loader, loss_fn, do_dro)
        accs.append(acc.item())
        f1s.append(f1.item())
        if acc_pred > acc:
            count_non_improv += 1
            if count_non_improv >= 1:
                break
        else:
            count_non_improv = 0
            
        torch.save(model.state_dict(), save_path)
        acc_pred = acc

    y_pred, y_true, _, _, _, _, _ = test_classifier(model, test_loader, loss_fn, do_dro)

    return accs, f1s, y_pred, y_true
