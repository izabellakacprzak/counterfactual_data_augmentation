from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.coloredMNIST import ColoredMNIST
from classifier import ConvNet, train_and_evaluate

from utils.params import *
from utils.evaluate import pretty_print_evaluation, save_plot_for_metric
from utils.utils import DebiasingMethod
from dro_loss import DROLoss

pred_arr = []
true_arr = []
runs_arr = []
accs_arr = []
f1s_arr = []

in_channels = 3
out_channels = 10

transforms_list = transforms.Compose([transforms.ToTensor()])

def train_colored_mnist(run_name, bias_conflicting_perc=1.0, debiasing_method=DebiasingMethod.NONE, do_dro=False):
    runs_arr.append(run_name)
    print("[Colored MNIST train]\t" + run_name)
    train_dataset = ColoredMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=bias_conflicting_perc, method=debiasing_method)
    test_dataset = ColoredMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=bias_conflicting_perc)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ConvNet(in_channels=in_channels, out_channels=out_channels)

    save_path = "checkpoints/colored_mnist/classifier_" + run_name + ".pt"

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
# Train and evaluate the MorphoMNIST dataset of colored MNIST images
# unbiased, biased, balanced with oversampling, balanced with standard data augmentations methods
# and balanced with counterfactual images

bias_conflicting_perc = 0.0
train_colored_mnist(run_name="BASELINE_COLORED_MNIST_0", bias_conflicting_perc=bias_conflicting_perc)
train_colored_mnist(run_name="GROUP_DRO_COLORED_MNIST_0", bias_conflicting_perc=bias_conflicting_perc, do_dro=True)
train_colored_mnist(run_name="OVERSAMPLING_COLORED_MNIST_0", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.OVERSAMPLING)
train_colored_mnist(run_name="AUGMENTATIONS_COLORED_MNIST_0", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.AUGMENTATIONS)
train_colored_mnist(run_name="MIXUP_COLORED_MNIST", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.MIXUP)
train_colored_mnist(run_name="COUNTERFACTUALS_COLORED_MNIST_0", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.COUNTERFACTUALS)
train_colored_mnist(run_name="CFREGULARISATION_COLORED_MNIST_0", bias_conflicting_perc=bias_conflicting_perc, debiasing_method=DebiasingMethod.CF_REGULARISATION)

############################################################

for idx in range(len(runs_arr)):
    print("[Colored MNIST train]\t" + runs_arr[idx])
    save_plot_for_metric("Accuracy", accs_arr[idx], runs_arr[idx])
    save_plot_for_metric("F1", f1s_arr[idx], runs_arr[idx])
    pretty_print_evaluation(pred_arr[idx], true_arr[idx], range(out_channels))
