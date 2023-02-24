from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
from datasets import MedMNIST

from params import *
from utils.evaluate import pretty_print_evaluation
from utils.utils import AugmentationMethod, train_and_evaluate

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
transforms_list = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))])

runs_arr.append("BALANCED_MED_MNIST")
print("Training on balanced mnist")
train_dataset = MedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=1)


test_dataset = MedMNIST(train=False, transform=transforms_list)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, train_dataset.in_channels, train_dataset.out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

runs_arr.append("IMBALANCED_MED_MNIST")
print("Training on imbalanced mnist")
train_dataset = MedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01)

test_dataset = MedMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, train_dataset.in_channels, train_dataset.out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

runs_arr.append("OVERSAMPLING_MED_MNIST")
print("Training on oversampled mnist")
train_dataset = MedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01, method=AugmentationMethod.OVERSAMPLING)

test_dataset = MedMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, train_dataset.in_channels, train_dataset.out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

runs_arr.append("AUGMENTATIONS_MED_MNIST")
print("Training on augmented mnist")
train_dataset = MedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01, method=AugmentationMethod.AUGMENTATIONS)

test_dataset = MedMNIST(train=False, transform=transforms_list)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, train_dataset.in_channels, train_dataset.out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

for idx in range(len(runs_arr)):
    print(runs_arr[idx])
    plt.scatter(list(range(EPOCHS)), accs_arr[idx])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("plots/"+runs_arr[idx]+".png")
    # plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(train_dataset.out_channels))