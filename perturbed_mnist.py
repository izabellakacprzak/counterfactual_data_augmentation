from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from datasets.perturbedMNIST import PerturbedMNIST
from classifier import ConvNet, train_and_evaluate

from utils.params import *
from utils.evaluate import print_classes_size, pretty_print_evaluation, save_plot_for_metric
from utils.utils import DebiasingMethod
from dro_loss import DROLoss

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
f1s_arr = []

in_channels = 1
out_channels = 10

transforms_list = transforms.Compose([transforms.ToTensor()])

def train_perturbed_mnist(run_name, bias_conflicting_perc=1.0, debiasing_method=DebiasingMethod.NONE, do_dro=False):
    runs_arr.append(run_name)
    print("[Perturbed MNIST train]\t" + run_name)
    train_dataset = PerturbedMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=bias_conflicting_perc, method=debiasing_method)
    # print_classes_size(train_dataset)
    # count_thick_thin_per_class(train_dataset.datas)
    # plot_dataset_digits(train_dataset)
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=bias_conflicting_perc)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvNet(in_channels=in_channels, out_channels=out_channels)

    save_path = "checkpoints/mnist/classifier_" + run_name + ".pt"

    if do_dro:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        n_groups = len(train_dataset.group_counts)
        group_counts = list(train_dataset.group_counts.values())
        loss_fn = DROLoss(loss_fn, n_groups, group_counts)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    accuracies, f1s, y_pred, y_true = train_and_evaluate(model, train_loader, test_loader, test_loader, loss_fn, save_path, do_dro, debiasing_method)
    accs_arr.append(accuracies)
    f1s_arr.append(f1s)
    pred_arr.append(y_pred)
    true_arr.append(y_true)

############################################################
# Train and evaluate the MorphoMNIST dataset of perturbed MNIST images
# unbiased, biased, balanced with oversampling, balanced with standard data augmentations methods
# and balanced with counterfactual images

bias_conflicting_perc = 0.0
# plot_dataset_digits(train_dataset)
# train_perturbed_mnist(run_name="UNBIASED_PERTURBED_MNIST", bias_conflicting_perc=1.0)
train_perturbed_mnist(run_name="BASELINE_PERTURBED_MNIST", bias_conflicting_perc=bias_conflicting_perc)
# train_perturbed_mnist(run_name="GROUP_DRO_05_PERTURBED_MNIST", bias_conflicting_perc=bias_conflicting_perc, do_dro=True)
train_perturbed_mnist(run_name="OVERSAMPLING_PERTURBED_MNIST", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.OVERSAMPLING)
train_perturbed_mnist(run_name="AUGMENTATIONS_PERTURBED_MNIST", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.AUGMENTATIONS)
train_perturbed_mnist(run_name="MIXUP_PERTURBED_MNIST", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.MIXUP)
train_perturbed_mnist(run_name="COUNTERFACTUALS_PERTURBED_MNIST", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.COUNTERFACTUALS)
train_perturbed_mnist(run_name="CFREGULARISATION_PERTURBED_MNIST", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.CF_REGULARISATION)

############################################################

for idx in range(len(runs_arr)):
    print("[Perturbed MNIST train]\t" + runs_arr[idx])
    save_plot_for_metric("Accuracy", accs_arr[idx], runs_arr[idx])
    save_plot_for_metric("F1", f1s_arr[idx], runs_arr[idx])
    # plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(out_channels))
