from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from datasets import ColoredMNIST

from params import *
from utils.utils import AugmentationMethod, train_and_evaluate
from utils.evaluate import pretty_print_evaluation

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
transforms_list = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))])

## balanced colored MNIST
runs_arr.append("BALANCED COLORED MNIST")
print("Training on balanced mnist")
train_dataset = ColoredMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=1)


test_dataset = ColoredMNIST(train=False, transform=transforms_list)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, 3, 10, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

## imbalanced MNIST
runs_arr.append("IMBALANCED MNIST")
print("Training on imbalanced mnist")
train_dataset = ColoredMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01)

test_dataset = ColoredMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, 3, 10, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

# balanced with oversampling MNIST
runs_arr.append("OVERSAMPLING MNIST")
print("Training on oversampled mnist")
train_dataset = ColoredMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01, method=AugmentationMethod.OVERSAMPLING)

test_dataset = ColoredMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, 3, 10, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

## balanced with augmentations loss function MNIST
runs_arr.append("AUGMENTATIONS MNIST")
print("Training on augmented mnist")
train_dataset = ColoredMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01, method=AugmentationMethod.AUGMENTATIONS)

test_dataset = ColoredMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, 3, 10, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

for idx in range(len(runs_arr)):
    print(runs_arr[idx])
    plt.plot(accs_arr[idx], range(EPOCHS))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], [0,1,2,3,4,5,6,7,8,9])