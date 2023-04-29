from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from datasets.chestXRay import ChestXRay
from MNISTClassifier import ConvNet, train_and_evaluate

from params import *
from utils.evaluate import print_classes_size, pretty_print_evaluation, save_plot_for_metric, get_attribute_counts_chestxray
from utils.utils import AugmentationMethod

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
f1s_arr = []

in_channels = 224
out_channels = 2

transforms_list = transforms.Compose([transforms.ToTensor()])

def train_and_evaluate_dataset(run_name, debiasing_method=AugmentationMethod.NONE):
    runs_arr.append(run_name)
    print("[ChestXRay train]\t" + run_name)
    train_dataset = ChestXRay(train=True, transform=transforms_list, method=debiasing_method)
    # get_attribute_counts_chestxray(train_dataset)
    # print_classes_size(train_dataset)
    # count_thick_thin_per_class(train_dataset.datas)
    # plot_dataset_digits(train_dataset)
    test_dataset = ChestXRay(train=False, transform=transforms_list)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvNet(in_channels=in_channels, out_channels=out_channels)

    do_cf_reg = debiasing_method==AugmentationMethod.CF_REGULARISATION
    do_mixup = debiasing_method==AugmentationMethod.MIXUP
    accuracies, f1s = train_and_evaluate(model, train_loader, test_loader, pred_arr, true_arr, do_cf_reg, do_mixup)
    accs_arr.append(accuracies)
    f1s_arr.append(f1s)

    torch.save(model.state_dict(), "../checkpoints/chestxray/classifier_" + run_name + ".pt")
    #visualise_t_sne(test_loader, model, "plots/chestxray/"+run_name+"t_sne")

############################################################
# Train and evaluate the ChestXRay dataset
# unbiased, biased, balanced with oversampling, balanced with standard data augmentations methods
# and balanced with counterfactual images

# plot_dataset_digits(train_dataset)
# train_and_evaluate_dataset("UNBIASED_CHESTXRAY", 1.0)
# train_and_evaluate_dataset("BIASED_CHESTXRAY", 1.0)
# train_and_evaluate_dataset("OVERSAMPLING_CHESTXRAY", AugmentationMethod.OVERSAMPLING)
# train_and_evaluate_dataset("AUGMENTATIONS_CHESTXRAY", AugmentationMethod.AUGMENTATIONS)
train_and_evaluate_dataset("COUNTERFACTUALS_CHESTXRAY", AugmentationMethod.COUNTERFACTUALS)
# train_and_evaluate_dataset("CFREGULARISATION_CHESTXRAY", AugmentationMethod.CF_REGULARISATION)

############################################################

for idx in range(len(runs_arr)):
    print("[ChestXRay train]\t" + runs_arr[idx])
    save_plot_for_metric("Accuracy", accs_arr[idx], runs_arr[idx])
    save_plot_for_metric("F1", f1s_arr[idx], runs_arr[idx])
    # plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(out_channels))
