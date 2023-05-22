import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from classifier import ConvNet, DenseNet, test_classifier
from utils.params import *

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

def get_global_change_in_performance(base_means, new_means, run_name):
    s_diff = 0
    s_base = 0
    for idx in range(len(base_means)):
        s_base += base_means[idx]
        s_diff += new_means[idx] - base_means[idx]

    global_change = s_diff / s_base
    print("Global change in performance for {} is {}".format(run_name, global_change))

def get_local_change_in_performance(base_worst, new_worst, run_name):
    local_change = (new_worst - base_worst) / base_worst
    print("Local change in performance for {} is {}".format(run_name, local_change))

def plot_metric_subgroup(subgroup_names, subgroup_metrics, metric_name, run_names):
    num_subgroups = len(subgroup_names)
    _ = plt.figure(metric_name)
    width = 0.8

    _, ax = plt.subplots()

    ax.set_prop_cycle(color=['red', 'blue', 'orange', 'gray', 'green'])
    for idx, run in enumerate(run_names):
        values = [m for m in subgroup_metrics[idx]]
        _ = ax.bar(np.arange(num_subgroups)+(width/len(run_names)*idx), values, width/(len(run_names)), label=run)

    ax.set_xlabel("Subgroups")
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    ax.set_xticks(np.arange(num_subgroups) + width)
    ax.set_xticklabels(['Male', 'Female', '0-19', '20-39', '40-59', '60-79', '80-99', 'White', 'Asian', 'Black'])

    # Create a horizontal line at the origin
    ax.legend()
    save_dir = 'plots/metrics_comp'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig("{}/subgroups_{}.png".format(save_dir, metric_name))

def plot_metric_subgroup_comparison(subgroup_names, subgroup_metrics, averages, metric_name, run_names):
    num_subgroups = len(subgroup_names)
    _ = plt.figure(metric_name)
    width = 0.8

    _, ax = plt.subplots()
    
    print(metric_name)
    print(subgroup_names)
    print(subgroup_metrics)

    ax.set_prop_cycle(color=['red', 'blue', 'orange', 'gray', 'green'])
    for idx, run in enumerate(run_names):
        values = [m-averages[idx] for m in subgroup_metrics[idx]]
        _ = ax.bar(np.arange(num_subgroups)+(width/len(run_names)*idx), values, width/(len(run_names)), label=run)

    ax.set_xlabel("Subgroups")
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    ax.set_xticks(np.arange(num_subgroups) + width)
    ax.set_xticklabels(subgroup_names)

    # Create a horizontal line at the origin
    ax.axhline(y=0, color='black')
    ax.legend()
    save_dir = 'plots/metrics_comp'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig("{}/subgroups_comparison_{}.png".format(save_dir, metric_name))

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
   
    save_dir = 'plots/metrics_comp/'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig("{}{}.png".format(save_dir, metric_name))

def _preprocess_age(metrics):
    processed_metrics = []
    for m in metrics:
        if 0 <= m <= 19:
            processed_metrics.append(0)
        elif 20 <= m <= 39:
            processed_metrics.append(1)
        elif 40 <= m <= 59:
            processed_metrics.append(2)
        elif 60 <= m <= 79:
            processed_metrics.append(3)
        elif 80 <= m <= 100:
            processed_metrics.append(4)

    return processed_metrics

def metrics_per_attribute(attributes, metrics_true, y_true, y_pred):
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    for idx, attribute in enumerate(attributes):
        if attribute in ['thickness']:
            continue

        attr_values = metrics_true[idx]
        if attribute == 'age':
            attr_values = _preprocess_age(attr_values)
        
        unique_attr_values = set(attr_values)

        # unique_counts = {u:0 for u in unique_attr_values}
        tp_unique = {u:0 for u in unique_attr_values}
        tn_unique = {u:0 for u in unique_attr_values}
        fp_unique = {u:0 for u in unique_attr_values}
        fn_unique = {u:0 for u in unique_attr_values}

        for idx, t in enumerate(y_true):
            p = y_pred[idx]
            if p == t:
                if t == 0:
                    tp_unique[attr_values[idx]] = tp_unique[attr_values[idx]] + 1
                else:
                    tn_unique[attr_values[idx]] = tn_unique[attr_values[idx]] + 1
            else:
                if t == 0:
                    fn_unique[attr_values[idx]] = fn_unique[attr_values[idx]] + 1
                else:
                    fp_unique[attr_values[idx]] = fp_unique[attr_values[idx]] + 1

        for av in unique_attr_values:
            # save accuracy value
            div = (tp_unique[av] + tn_unique[av] + fp_unique[av] + fn_unique[av])
            acc = 0 if div==0 else (tp_unique[av] + tn_unique[av]) / div
            accuracies.append(acc)

            # save precision values
            div = tp_unique[av] + fp_unique[av]
            pr = 0 if div==0 else tp_unique[av] / div
            precisions.append(pr)

            # save recall values
            div = tp_unique[av] + fn_unique[av]
            rc = 0 if div==0 else tp_unique[av] / div
            recalls.append(rc)

            # save f1 score values
            div = (tp_unique[av] + 0.5 * (fp_unique[av] + fn_unique[av]))
            f1 = 0 if div==0 else tp_unique[av]/div
            f1s.append(f1)

    return accuracies, precisions, recalls, f1s


def test_pretrained(model_path, dataset, loss_fn, attributes, in_channels, out_channels):
    ## Test pretrained model ## 
    if "MNIST" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_{}.pt".format(model_path), map_location=device))
    else:
        model = DenseNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_{}.pt".format(model_path), map_location=device))
        
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_pred, y_true, y_score, metrics_true, acc, f1 = test_classifier(model, test_loader, loss_fn)

    return metrics_true, y_true, y_pred, y_score

def _plot_roc_curve(y_true, y_score, run_name):
    y_score = [p[1] for p in y_score]
    roc_auc = roc_auc_score(y_true, y_score)
    print("AUC score")
    print(roc_auc)

    fpr, tpr, _ = roc_curve(y_true,  y_score)
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("roc-curve-{}.png".format(run_name))
    plt.show()

def get_label_metrics(y_true, y_pred, num_classes):
    ## Get classification report and per-class performance ##
    report_dict = metrics.classification_report(y_true, y_pred, digits=range(10), output_dict=True)
    print(report_dict)

    matrix = metrics.confusion_matrix(y_true, y_pred)
    accs = matrix.diagonal()/matrix.sum(axis=1)
    f1s = [report_dict[str(label)]['f1-score'] for label in range(num_classes)]
    precisions = [report_dict[str(label)]['precision'] for label in range(num_classes)]
    recalls = [report_dict[str(label)]['recall'] for label in range(num_classes)]

    return accs, f1s, precisions, recalls, report_dict['accuracy'], report_dict['macro avg']['f1-score']

def test_perturbed_mnist():
    models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    in_channels = 1
    num_classes = 10
    attributes = ['thickness', 'intensity', 'bias_aligned']
    loss_fn = torch.nn.CrossEntropyLoss()
    
    accs = []
    f1s = []
    precisions = []
    recalls = []

    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    for model in models:
        print("[Test trained]\tTesting model: " + model)

        mnist_model_path = model + "_PERTURBED_MNIST"
        _, y_true, y_pred, y_score = test_pretrained(mnist_model_path, test_dataset, loss_fn, attributes, in_channels, num_classes)

        acc, f1, precision, recall, _, _ = get_label_metrics(y_true, y_pred, num_classes)
        accs.append(acc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

        _plot_roc_curve(y_true, y_score, model)

    plot_metrics_comparison(models, accs, 'MNISTaccuracy')
    plot_metrics_comparison(models, f1s, 'MNISTf1score')
    plot_metrics_comparison(models, precisions, 'MNISTprecision')
    plot_metrics_comparison(models, recalls, 'MNISTrecall')

def test_chestxray():
    #models = ["BASELINE", "OVERSAMPLING_age_0", "AUGMENTATIONS_age_0", "COUNTERFACTUALS_age_0"]
    models = ["BASELINE", "BIASED_DRO"]
    in_channels = 1
    num_classes = 2
    attributes = ['sex', 'age', 'race']
    loss_fn = torch.nn.CrossEntropyLoss()
    subgroup_names = ['Male', 'Female', '0-19', '20-39', '40-59', '60-79', '80-99', 'White', 'Asian', 'Black']
    
    accs = []
    f1s = []
    precisions = []
    recalls = []
    attr_accs = []
    attr_precs = []
    attr_rcs = []
    attr_f1s = []
    overall_accs = []
    overall_f1s = []
    
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    for model in models:
        print("[Test trained]\tTesting model: " + model)

        chestxray_model_path = model + "_CHESTXRAY"
        metrics_true, y_true, y_pred, y_score = test_pretrained(chestxray_model_path, test_dataset, loss_fn, attributes, in_channels, num_classes)

        acc, f1, precision, recall, overall_acc, overall_f1 = get_label_metrics(y_true, y_pred, num_classes)
        accs.append(acc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        overall_accs.append(overall_acc)
        overall_f1s.append(overall_f1)

        _plot_roc_curve(y_true, y_score, model)

        ## Print performance metrics per attribute (eg. sex, digit etc) ##
        attr_acc, attr_pr, attr_rc, attr_f1 = metrics_per_attribute(attributes, metrics_true, y_true, y_pred)
        attr_accs.append(attr_acc)
        attr_precs.append(attr_pr)
        attr_rcs.append(attr_rc)
        attr_f1s.append(attr_f1)


    plot_metrics_comparison(models, accs, 'CHESTXRAYaccuracy')
    plot_metrics_comparison(models, f1s, 'CHESTXRAYf1-score')
    plot_metrics_comparison(models, precisions, 'CHESTXRAYprecision')
    plot_metrics_comparison(models, recalls, 'CHESTXRAYrecall')

    plot_metric_subgroup_comparison(subgroup_names, attr_accs, overall_accs, "Accuracy", models)
    plot_metric_subgroup_comparison(subgroup_names, attr_f1s, overall_f1s, "F1-score", models)
    plot_metric_subgroup(subgroup_names, attr_accs, "Accuracy", models)
    plot_metric_subgroup(subgroup_names, attr_precs, "Precision", models)
    plot_metric_subgroup(subgroup_names, attr_rcs, "Recall", models)
    plot_metric_subgroup(subgroup_names, attr_f1s, "F1-score", models)
    
    worst_base = min(attr_accs[0])
    for idx in range(len(models))[1:]:
        get_global_change_in_performance(attr_accs[0], attr_accs[idx], models[idx])
        get_local_change_in_performance(worst_base, min(attr_accs[idx]), models[idx])

# test_perturbed_mnist()
test_chestxray()
