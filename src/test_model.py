import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score


from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from classifier import ConvNet, DenseNet, test_classifier
from utils.params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def plot_metric_subgroup_comparison(subgroup_names, subgroup_metrics, avg_metric, metric_name):
    num_subgroups = len(subgroup_names)
    _ = plt.figure(metric_name)
    width = 0.35

    _, ax = plt.subplots()
    values = [m-avg_metric for m in subgroup_metrics]
    _ = ax.bar(np.arange(num_subgroups), values, width, color='blue')

    ax.set_xlabel("Subgroups")
    ax.set_ylabel(metric_name)
    ax.set_xticks(np.arange(num_subgroups) + width)
    ax.set_xticklabels(subgroup_names)

    # Create a horizontal line at the origin
    ax.axhline(y=0, color='black')
    plt.show()


def plot_metrics_comparison(run_names, run_metrics, metric_name):
    fig = plt.figure(metric_name)
    num_classes = len(run_metrics[0])
    r = np.arange(num_classes)
    width = 0.1
    
    metrics = {}
    for idx in range(len(run_names)):
        metrics[run_names[idx]] = run_metrics[idx]

    for run, metric in metrics.items():
        plt.bar(r, metric, width = width, label=run)
        r = r + width
    
    plt.xlabel("Label")
    plt.ylabel(metric_name)
    
    plt.xticks(np.arange(num_classes) + width/2, np.arange(num_classes))
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
    
    plt.savefig("plots/metrics_comparison_"+ metric_name +".png")

def metrics_per_attribute(attributes, metrics_true, y_true, y_pred):
    subgroup_names = []
    accuracies = []
    f1s = []
    for idx, attribute in enumerate(attributes):
        if attribute == 'thickness':
            continue

        attr_values = metrics_true[idx]
        unique_attr_values = set(attr_values)

        # unique_counts = {u:0 for u in unique_attr_values}
        tp_unique = {u:0 for u in unique_attr_values}
        tn_unique = {u:0 for u in unique_attr_values}
        fp_unique = {u:0 for u in unique_attr_values}
        fn_unique = {u:0 for u in unique_attr_values}

        for idx, t in enumerate(y_true):
            p = y_pred[idx]
            if p == t:
                if t == 1:
                    tp_unique[attr_values[idx]] = tp_unique[attr_values[idx]] + 1
                else:
                    tn_unique[attr_values[idx]] = tn_unique[attr_values[idx]] + 1
            else:
                if t == 1:
                    fn_unique[attr_values[idx]] = fn_unique[attr_values[idx]] + 1
                else:
                    fp_unique[attr_values[idx]] = fp_unique[attr_values[idx]] + 1

        for av in unique_attr_values:
            # save subgroup name
            subgroup_names.append("{} {}".format(attribute, str(av)))

            # save accuracy value
            div = (tp_unique[av] + tn_unique[av] + fp_unique[av] + fn_unique[av])
            acc = 0 if div==0 else (tp_unique[av] + tn_unique[av]) / div
            accuracies.append(acc)

            # save f1 score values
            div = (tp_unique[av] + 0.5 * (fp_unique[av] + fn_unique[av]))
            f1 = 0 if div==0 else tp_unique[av]/div
            f1s.append(f1)

    return subgroup_names, accuracies, f1s


def test_pretrained(model_path, dataset, loss_fn, attributes, in_channels, out_channels):
    ## Test pretrained model ##
    model = DenseNet(in_channels=in_channels, out_channels=out_channels)
    #model = ConvNet(in_channels=in_channels, out_channels=out_channels)
    
    if "MNIST" in model_path:
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_"+model_path+".pt", map_location=device))
    else:
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_"+model_path+".pt", map_location=device))

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_pred, y_true, y_score, metrics_true, acc, f1 = test_classifier(model, test_loader, loss_fn)

    ## Get classification report and per-class performance ##
    report_dict = metrics.classification_report(y_true, y_pred, digits=range(10), output_dict=True)
    # Get accuracy separetely #
    matrix = metrics.confusion_matrix(y_true, y_pred)
    accs = matrix.diagonal()/matrix.sum(axis=1)

    roc_auc = roc_auc_score(y_true, y_score)
    print("AUC score")
    print(roc_auc)

    ## Print performance metrics per attribute (eg. sex, digit etc) ##
    subgroup_names, accuracies, f1s = metrics_per_attribute(attributes, metrics_true, y_true, y_pred)
    plot_metric_subgroup_comparison(subgroup_names, accuracies, report_dict['accuracy'], "Accuracy")
    plot_metric_subgroup_comparison(subgroup_names, f1s, report_dict['weighted avg']['f1-score'], "F1-score")

    f1s = []
    precisions = []
    recalls = []
    for label in range(out_channels):
        label = str(label)
        f1s.append(report_dict[label]['f1-score'])
        precisions.append(report_dict[label]['precision'])
        recalls.append(report_dict[label]['recall'])

    return accs, f1s, precisions, recalls

def test_perturbed_mnist():
    models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    accs = []
    f1s = []
    precisions = []
    recalls = []

    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    for model in models:
        print("[Test trained]\tTesting model: " + model)

        mnist_model_path = model + "_PERTURBED_MNIST"
        in_channels = 1
        num_classes = 10
        attributes = ['thickness', 'intensity', 'bias_aligned']
        loss_fn = torch.nn.CrossEntropyLoss()
        acc, f1, precision, recall = test_pretrained(mnist_model_path, test_dataset, loss_fn, attributes, in_channels, num_classes)

        print("Accuracies")
        print(acc)
        print("F1-scores")
        print(fs)
        print("Precisoins")
        print(precision)
        print("Recalls")
        print(recall)

        accs.append(acc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    plot_metrics_comparison(models, accs, 'MNISTaccuracy')
    plot_metrics_comparison(models, f1s, 'MNISTf1score')
    plot_metrics_comparison(models, precisions, 'MNISTprecision')
    plot_metrics_comparison(models, recalls, 'MNISTrecall')

def test_chestxray():
    models = ["BIASED"]
    accs = []
    f1s = []
    precisions = []
    recalls = []
    
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(train=False, transform=transforms_list)

    for model in models:
        print("[Test trained]\tTesting model: " + model)

        chestxray_model_path = model + "_CHESTXRAY"
        in_channels = 1
        num_classes = 2
        attributes = ['sex', 'age', 'race']
        loss_fn = torch.nn.BCELoss()
        acc, f1, precision, recall = test_pretrained(chestxray_model_path, test_dataset, loss_fn, attributes, in_channels, num_classes)

        accs.append(acc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    plot_metrics_comparison(models, accs, 'CHESTXRAYaccuracy')
    plot_metrics_comparison(models, f1s, 'CHESTXRAYf1-score')
    plot_metrics_comparison(models, precisions, 'CHESTXRAYprecision')
    plot_metrics_comparison(models, recalls, 'CHESTXRAYrecall')

# test_perturbed_mnist()
test_chestxray()
