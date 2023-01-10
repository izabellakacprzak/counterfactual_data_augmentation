from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from datasets import MedMNIST

from params import *
from evaluation.evaluate import pretty_print_evaluation
from utils import AugmentationMethod, train_and_evaluate

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
transforms_list = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))])
## balanced med MNIST
runs_arr.append("BALANCED MED MNIST")
print("Training on balanced mnist")
train_dataset = MedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=1)


test_dataset = MedMNIST(train=False, transform=transforms_list)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, train_dataset.in_channels, train_dataset.out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

## imbalanced MNIST
runs_arr.append("IMBALANCED MED MNIST")
print("Training on imbalanced mnist")
train_dataset = MedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01)

test_dataset = MedMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, train_dataset.in_channels, train_dataset.out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

## balanced with oversampling MNIST
runs_arr.append("OVERSAMPLING MED MNIST")
print("Training on oversampled mnist")
train_dataset = MedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01, method=AugmentationMethod.OVERSAMPLING)

test_dataset = MedMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, train_dataset.in_channels, train_dataset.out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

## balanced with augmentations MNIST
runs_arr.append("AUGMENTATIONS MED MNIST")
print("Training on augmented mnist")
train_dataset = MedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01, method=AugmentationMethod.AUGMENTATIONS)

test_dataset = MedMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, train_dataset.in_channels, train_dataset.out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

for idx in range(len(runs_arr)):
    print(runs_arr[idx])
    plt.plot(accs_arr[idx], range(EPOCHS))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(train_dataset.out_channels))