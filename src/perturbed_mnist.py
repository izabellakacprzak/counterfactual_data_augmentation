from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from datasets.perturbedMNIST import PerturbedMNIST
from MNISTClassifier import ConvNet, train_and_evaluate

from params import *
from utils.evaluate import print_classes_size, pretty_print_evaluation, save_plot_for_metric
from utils.utils import AugmentationMethod

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
f1s_arr = []

in_channels = 1
out_channels = 10

transforms_list = transforms.Compose([transforms.ToTensor()])

def train_and_evaluate_dataset(run_name, bias_conflicting_perc=1.0, debiasing_method=AugmentationMethod.NONE):
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

    do_cf_reg = debiasing_method==AugmentationMethod.CF_REGULARISATION
    do_mixup = debiasing_method==AugmentationMethod.MIXUP
    save_path = "../checkpoints/mnist/classifier_" + run_name + ".pt"
    accuracies, f1s = train_and_evaluate(model, train_loader, test_loader, pred_arr, true_arr, save_path, do_cf_reg, do_mixup)
    accs_arr.append(accuracies)
    f1s_arr.append(f1s)

    # torch.save(model.state_dict(), "../checkpoints/mnist/classifier_" + run_name + ".pt")
    # visualise_t_sne(test_loader, model, "plots/mnist/"+run_name+"t_sne")

############################################################
# Train and evaluate the MorphoMNIST dataset of perturbed MNIST images
# unbiased, biased, balanced with oversampling, balanced with standard data augmentations methods
# and balanced with counterfactual images

bias_conflicting_perc = 0.01
# plot_dataset_digits(train_dataset)
# train_and_evaluate_dataset("UNBIASED_PERTURBED_MNIST", 1.0)
# train_and_evaluate_dataset("BIASED_PERTURBED_MNIST", bias_conflicting_perc)
# train_and_evaluate_dataset("OVERSAMPLING_PERTURBED_MNIST", bias_conflicting_perc, AugmentationMethod.OVERSAMPLING)
# train_and_evaluate_dataset("AUGMENTATIONS_PERTURBED_MNIST", bias_conflicting_perc, AugmentationMethod.AUGMENTATIONS)
# train_and_evaluate_dataset("COUNTERFACTUALS_PERTURBED_MNIST", bias_conflicting_perc, AugmentationMethod.COUNTERFACTUALS)
# train_and_evaluate_dataset("CFREGULARISATION_PERTURBED_MNIST", bias_conflicting_perc, AugmentationMethod.CF_REGULARISATION)
train_and_evaluate_dataset("MIXUP_PERTURBED_MNIST", bias_conflicting_perc, AugmentationMethod.MIXUP)

############################################################

for idx in range(len(runs_arr)):
    print("[Perturbed MNIST train]\t" + runs_arr[idx])
    save_plot_for_metric("Accuracy", accs_arr[idx], runs_arr[idx])
    save_plot_for_metric("F1", f1s_arr[idx], runs_arr[idx])
    # plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(out_channels))
