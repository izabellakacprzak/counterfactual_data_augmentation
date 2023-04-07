from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
from datasets.perturbedMNIST import PerturbedMNIST

from params import *
from utils.evaluate import print_classes_size, pretty_print_evaluation, count_thick_thin_per_class
from utils.utils import AugmentationMethod, train_and_evaluate, visualise_t_sne

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
f1s_arr = []

in_channels = 1
out_channels = 10

transforms_list = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))])

def train_and_evaluate_dataset(run_name, bias_conflicting_perc=1.0, debiasing_method=AugmentationMethod.NONE):
    runs_arr.append(run_name)
    print(run_name)
    train_dataset = PerturbedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=bias_conflicting_perc, method=debiasing_method)
    # visualise_t_sne(train_dataset.data_label_tuples, "plots/"+run_name+"t_sne.png")
    print_classes_size(train_dataset)
    # count_thick_thin_per_class(train_dataset.data_label_tuples)
    # plot_dataset_digits(train_dataset)
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=bias_conflicting_perc)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    accuracies, f1s = train_and_evaluate(train_loader, test_loader, in_channels, out_channels, pred_arr, true_arr)
    accs_arr.append(accuracies)
    f1s_arr.append(f1s)

############################################################
# Train and evaluate the MorphoMNIST dataset of perturbed MNIST images
# balanced, imbalanced, balanced with oversampling, balanced with standard data augmentations methods
# and balanced with counterfactual images

bias_conflicting_perc = 0.01
# plot_dataset_digits(train_dataset)
train_and_evaluate_dataset("BALANCED_PERTURBED_MNIST", 1.0)
train_and_evaluate_dataset("IMBALANCED_PERTURBED_MNIST", bias_conflicting_perc)
# train_and_evaluate_dataset("OVERSAMPLING_PERTURBED_MNIST", bias_conflicting_perc, AugmentationMethod.OVERSAMPLING)
# train_and_evaluate_dataset("AUGMENTING_PERTURBED_MNIST", bias_conflicting_perc, AugmentationMethod.AUGMENTATIONS)
# train_and_evaluate_dataset("COUNTERFACTUALS_PERTURBED_MNIST", bias_conflicting_perc, AugmentationMethod.COUNTERFACTUALS)

############################################################

def save_plot_for_metric(metric_name, metric_arr):
    x, y = np.array(list(range(EPOCHS))), metric_arr
    res = stats.linregress(x, y)
    plt.plot(x, y, 'o', label='original data')
    # plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig("plots/"+runs_arr[idx]+metric_name+".png")

for idx in range(len(runs_arr)):
    print(runs_arr[idx])
    save_plot_for_metric("Accuracy", accs_arr[idx])
    save_plot_for_metric("F1", f1s_arr[idx])
    # plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(out_channels))
