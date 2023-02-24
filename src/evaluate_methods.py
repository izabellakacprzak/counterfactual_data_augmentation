from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
from datasets import VanillaMNIST
from matplotlib import pyplot as plt

from params import *
from utils.evaluate import pretty_print_evaluation
from utils.utils import train_and_evaluate

def oversampling_dataloader():
    ratio = 1/CUT_PERCENTAGE
    class_weights = [1] * 10
    for c in UNDERSAMPLED_CLASSES:
        class_weights[c] = ratio

    sample_weights = [0] * len(train_dataset)
    for idx, target in enumerate(train_dataset.targets):
        sample_weights[idx] = class_weights[target]

    sampler = WeightedRandomSampler(sample_weights,
                                  num_samples=len(sample_weights),
                                  replacement=True)
    return DataLoader(train_dataset, batch_size=64,
                                      sampler=sampler)

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []

runs_arr.append("BALANCED_VANILLA_MNIST")
print("Training on balanced mnist")
train_dataset = VanillaMNIST(train=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                    ]))

test_dataset = VanillaMNIST(train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                    ]))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, 1, 10, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

runs_arr.append("IMBALANCED_VANILLA_MNIST")
print("Training on imbalanced mnist")
train_dataset = VanillaMNIST(train=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                    ]), undersampled_classes=UNDERSAMPLED_CLASSES)

test_dataset = VanillaMNIST(train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                    ]))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, 1, 10, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

runs_arr.append("OVERSAMPLING_VANILLA_MNIST")
print("Training on oversampled mnist")
train_dataset = VanillaMNIST(train=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                    ]), undersampled_classes=UNDERSAMPLED_CLASSES, perturbed=False)

test_dataset = VanillaMNIST(train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                    ]))

train_loader = oversampling_dataloader()
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, 1, 10, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

runs_arr.append("PERTURBATIONS_VANILLA_MNIST")
print("Training on perturbed mnist")
train_dataset = VanillaMNIST(train=True, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                    ]), undersampled_classes=UNDERSAMPLED_CLASSES, perturbed=True)

test_dataset = VanillaMNIST(train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                    ]))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, 1, 10, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

for idx in range(len(runs_arr)):
    print(runs_arr[idx])
    plt.scatter(list(range(EPOCHS)), accs_arr[idx])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("plots/"+runs_arr[idx]+".png")
    # plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], [0,1,2,3,4,5,6,7,8,9])