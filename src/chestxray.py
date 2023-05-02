from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from datasets.chestXRay import ChestXRay
from classifier import ConvNet, train_and_evaluate

from utils.params import *
from utils.evaluate import print_classes_size, pretty_print_evaluation, save_plot_for_metric, get_attribute_counts_chestxray
from utils.utils import AugmentationMethod

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
f1s_arr = []

in_channels = 1
out_channels = 2

transforms_list = transforms.Compose([transforms.Resize((192,192)),])

def train_chestxray(run_name, debiasing_method=AugmentationMethod.NONE):
    runs_arr.append(run_name)
    print("[ChestXRay train]\t" + run_name)
    train_dataset = ChestXRay(train=True, transform=transforms_list, method=debiasing_method)
    if debiasing_method != AugmentationMethod.NONE and debiasing_method != AugmentationMethod.CF_REGULARISATION:
        train_dataset.debias(method=debiasing_method)
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
    save_path = "../checkpoints/chestxray/classifier_" + run_name + ".pt"
    accuracies, f1s, y_pred, y_true = train_and_evaluate(model, train_loader, test_loader, torch.nn.BCELoss(), save_path, do_cf_reg, do_mixup)
    accs_arr.append(accuracies)
    f1s_arr.append(f1s)
    pred_arr.append(y_pred)
    true_arr.append(y_true)

    # torch.save(model.state_dict(), "../checkpoints/chestxray/classifier_" + run_name + ".pt")
    #visualise_t_sne(test_loader, model, "plots/chestxray/"+run_name+"t_sne")

############################################################
# Train and evaluate the ChestXRay dataset
# biased, balanced with oversampling, balanced with standard data augmentations methods
# and balanced with counterfactual images

# plot_dataset_digits(train_dataset)
# train_chestxray("BIASED_CHESTXRAY")
# train_chestxray("OVERSAMPLING_CHESTXRAY", AugmentationMethod.OVERSAMPLING)
# train_chestxray("AUGMENTATIONS_CHESTXRAY", AugmentationMethod.AUGMENTATIONS)
train_chestxray("COUNTERFACTUALS_CHESTXRAY", AugmentationMethod.COUNTERFACTUALS)
# train_chestxray("CFREGULARISATION_CHESTXRAY", AugmentationMethod.CF_REGULARISATION)

############################################################

for idx in range(len(runs_arr)):
    print("[ChestXRay train]\t" + runs_arr[idx])
    save_plot_for_metric("Accuracy", accs_arr[idx], runs_arr[idx])
    save_plot_for_metric("F1", f1s_arr[idx], runs_arr[idx])
    # plt.show()
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(out_channels))
