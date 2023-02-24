from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
from datasets.perturbedMNIST import PerturbedMNIST

from params import *
from utils.evaluate import print_classes_size, pretty_print_evaluation
from utils.utils import AugmentationMethod, train_and_evaluate
# from counterfactuals.counterfactuals import *

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []

in_channels = 1
out_channels = 10

transforms_list = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))])

# plot_dataset_digits(train_dataset)

# runs_arr.append("BALANCED_PERTURBED_MNIST")
# print("Training on balanced mnist")
# train_dataset = PerturbedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=1.)

# test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.)

# print_classes_size(train_dataset)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# accuracy = []
# train_and_evaluate(train_loader, test_loader, in_channels, out_channels, pred_arr, true_arr, accuracy)
# accs_arr.append(accuracy)

# ############################################################

# runs_arr.append("IMBALANCED_PERTURBED_MNIST")
# print("Training on imbalanced mnist")

# train_dataset = PerturbedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=.02)
# test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=.02)

# print_classes_size(train_dataset)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# accuracy = []
# train_and_evaluate(train_loader, test_loader, in_channels, out_channels, pred_arr, true_arr, accuracy)
# accs_arr.append(accuracy)

############################################################

# runs_arr.append("OVERSAMPLING_PERTURBED_MNIST")
# print("Training on oversampled mnist")
# train_dataset = PerturbedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=.02, method=AugmentationMethod.OVERSAMPLING)
# print_classes_size(train_dataset)
# test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=.02)

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# accuracy = []
# train_and_evaluate(train_loader, test_loader, in_channels, out_channels, pred_arr, true_arr, accuracy)
# accs_arr.append(accuracy)

############################################################

runs_arr.append("AUGMENTING_PERTURBED_MNIST")
print("Training on augmented mnist")
train_dataset = PerturbedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=.02, method=AugmentationMethod.AUGMENTATIONS)
print_classes_size(train_dataset)
test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=.02)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

accuracy = []
train_and_evaluate(train_loader, test_loader, in_channels, out_channels, pred_arr, true_arr, accuracy)
accs_arr.append(accuracy)

############################################################

# new_test_set = generate_counterfactuals(test_dataset)
# plot_dataset_digits(new_test_set)

for idx in range(len(runs_arr)):
    print(runs_arr[idx])
    x, y = np.array(list(range(EPOCHS))), accs_arr[idx]
    res = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='original data')
    plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("plots/"+runs_arr[idx]+".png")
    # plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(out_channels))