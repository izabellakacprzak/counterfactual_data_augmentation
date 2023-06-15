from torch.utils.data import DataLoader
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
from datasets.chestXRay import ChestXRay
from classifier import ConvNet, DenseNet, train_and_evaluate
from dro_loss import DROLoss

from utils.params import *
from utils.evaluate import print_classes_size, pretty_print_evaluation, save_plot_for_metric, get_attribute_counts_chestxray
from utils.utils import DebiasingMethod

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
f1s_arr = []

in_channels = 1
out_channels = 2

transforms_list = transforms.Compose([transforms.Resize((192,192)),])

def train_chestxray(run_name, debiasing_method=DebiasingMethod.NONE, do_dro=False):
    runs_arr.append(run_name)
    print("[ChestXRay train]\t" + run_name)
    train_dataset = ChestXRay(mode="train", transform=transforms_list, method=debiasing_method)
    # get_attribute_counts_chestxray(train_dataset)
    # print_classes_size(train_dataset)
    # count_thick_thin_per_class(train_dataset.datas)
    # plot_dataset_digits(train_dataset)
    valid_dataset = ChestXRay(mode="val", transform=transforms_list)
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DenseNet(in_channels=in_channels, out_channels=out_channels)

    save_path = "checkpoints/chestxray/classifier_" + run_name + ".pt"

    if do_dro:
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        n_groups = len(train_dataset.group_counts)
        group_counts = list(train_dataset.group_counts.values())
        loss_fn = DROLoss(loss_fn, n_groups, group_counts)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    accuracies, f1s, y_pred, y_true = train_and_evaluate(model, train_loader, valid_loader, test_loader, loss_fn, save_path, do_dro, debiasing_method)
    accs_arr.append(accuracies)
    f1s_arr.append(f1s)
    pred_arr.append(y_pred)
    true_arr.append(y_true)

############################################################
# Train and evaluate the ChestXRay dataset
# biased, balanced with oversampling, balanced with standard data augmentations methods
# and balanced with counterfactual images

# train_chestxray(run_name="BASELINE_disease_pred_CHESTXRAY")
# train_chestxray(run_name="OVERSAMPLING_race_disease_pred_CHESTXRAY", debiasing_method=DebiasingMethod.OVERSAMPLING)
# train_chestxray(run_name="AUGMENTATION_raceS_disease_pred_CHESTXRAY", debiasing_method=DebiasingMethod.AUGMENTATIONS)
# train_chestxray(run_name="MIXUP_sex_pred_CHESTXRAY", debiasing_method=DebiasingMethod.MIXUP)
# train_chestxray(run_name="GROUP_DRO_race_disease_pred_CHESTXRAY", do_dro=True)
train_chestxray(run_name="COUNTERFACTUALS_random_disease_pred_CHESTXRAY", debiasing_method=DebiasingMethod.COUNTERFACTUALS)
# train_chestxray(run_name="COUNTERFACTUALS_race_MIXUP_race_pred_CHESTXRAY", debiasing_method=DebiasingMethod.COUNTERFACTUALS)
# train_chestxray(run_name="CFREGULARISATION_age_disease_race_pred_CHESTXRAY", debiasing_method=DebiasingMethod.CF_REGULARISATION)
############################################################

for idx in range(len(runs_arr)):
    print("[ChestXRay train]\t" + runs_arr[idx])
    save_plot_for_metric("Accuracy", accs_arr[idx], runs_arr[idx])
    save_plot_for_metric("F1", f1s_arr[idx], runs_arr[idx])
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(out_channels))
