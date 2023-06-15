import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from statistics import median
import plotly.graph_objects as go

import sys
sys.path.append("..")

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from datasets.coloredMNIST import ColoredMNIST
from classifier import ConvNet, DenseNet, test_classifier
from utils.params import *
from utils.utils import preprocess_age, preprocess_thickness

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

# cols = ['coral','grey', 'gold', 'skyblue', 'peru', 'pink']
# cols = ['#faba6e', '#ef9570', '#d77776', '#b45f7a', '#884f78', '#59416c', '#2a3358']

cols = ['#fab36e', '#fa977d', '#ea8391', '#ca77a2', '#9f72aa', '#6c6ea5', '#376794']


def spider_plot(run_names, categories, values, save_dir):
    fig = go.Figure()

    for idx, run in enumerate(run_names):
        values[idx].append(values[idx][0]) # closing the line workaround
        fig.add_trace(go.Scatterpolar(
            r=values[idx],
            theta=categories,
            name=run
        ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0, 1]
        )),
        showlegend=True
    )

    fig.write_image("{}/spider_plot.png".format(save_dir))

def tpr_disparity(tprs):
    tpr_median = median(tprs)
    disparities = []
    for tpr in tprs:
        disparities.append(tpr-tpr_median)

    return disparities

def get_tprs(y_true, y_pred, num_classes):
    tp = [0 for _ in range(num_classes)]
    fn = [0 for _ in range(num_classes)]

    for idx in range(len(y_true)):
        y_t = y_true[idx]
        y_p = y_pred[idx]

        if y_t==y_p:
            tp[y_t] = tp[y_t]+1
        else:
            fn[y_t] = fn[y_t]+1

    tprs = []
    for idx in range(num_classes):
        tprs.append(tp[idx]/(tp[idx]+fn[idx]))

    return tprs

def box_plot_tpr_disparity(data, models, save_dir):
    _ = plt.figure(figsize =(10, 7))
    _, ax = plt.subplots()

    ax.set_prop_cycle(color=cols)
    ax.set_xlabel("Models")
    ax.set_ylabel("TPR range")
    plt.boxplot(data)
    plt.xticks(rotation = 45)
    ax.set_xticks(np.arange(len(models))+1)
    ax.set_xticklabels(models)
    plt.tight_layout()

    plt.savefig("{}/tpr_disparity_boxes.png".format(save_dir))
    plt.close()

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

def plot_metric_subgroup(subgroup_names, subgroup_metrics, metric_name, run_names, save_dir):
    num_subgroups = len(subgroup_names)
    _ = plt.figure(metric_name)
    width = 0.8

    _, ax = plt.subplots()

    ax.set_prop_cycle(color=cols)
    for idx, run in enumerate(run_names):
        values = [m for m in subgroup_metrics[idx]]
        _ = ax.bar(np.arange(num_subgroups)+(width/len(run_names)*idx), values, width/(len(run_names)), label=run)

    ax.set_xlabel("Subgroups")
    ax.set_ylabel(metric_name)
    # plt.xticks(rotation=45, ha='right')
    ax.set_xticks(np.arange(num_subgroups) + width)
    ax.set_xticklabels(subgroup_names)
    ax.axhline(y=0, color='black')

    # Create a horizontal line at the origin
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=3)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig("{}/subgroups_{}.png".format(save_dir, metric_name))
    plt.close()

def plot_metric_subgroup_comparison(subgroup_names, subgroup_metrics, averages, metric_name, run_names, save_dir):
    num_subgroups = len(subgroup_names)
    _ = plt.figure(metric_name)
    width = 0.8

    _, ax = plt.subplots()
    
    print(metric_name)
    print(subgroup_names)
    print(subgroup_metrics)

    ax.set_prop_cycle(color=cols)
    for idx, run in enumerate(run_names):
        values = [m-averages[idx] for m in subgroup_metrics[idx]]
        _ = ax.bar(np.arange(num_subgroups)+(width/len(run_names)*idx), values, width/(len(run_names)), label=run)

    ax.set_xlabel("Subgroups")
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=45, ha='right')
    ax.set_xticks(np.arange(num_subgroups) - width/2)
    ax.set_xticklabels(subgroup_names)

    # Create a horizontal line at the origin
    ax.axhline(y=0, color='black')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=3)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig("{}/subgroups_comparison_{}.png".format(save_dir, metric_name))
    plt.close()

def plot_metrics_comparison(run_names, run_metrics, metric_name, save_dir):
    _ = plt.figure(metric_name)
    _, ax = plt.subplots()
    ax.set_prop_cycle(color=cols)

    num_classes = len(run_metrics[0])
    r = np.arange(num_classes)
    width = 0.1
    
    metrics = {}
    for idx in range(len(run_names)):
        metrics[run_names[idx]] = run_metrics[idx]

    for run, metric in metrics.items():
        ax.bar(r, metric, width = width, label=run)
        r = r + width
    
    ax.set_xlabel("Label")
    ax.set_ylabel(metric_name)
    
    ax.set_xticks(np.arange(num_classes) - width/2, np.arange(num_classes))
    ax.set_yticks(np.arange(0, 1.1, 0.05))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=3)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig("{}/{}.png".format(save_dir, metric_name))
    plt.close()

def metrics_per_attribute(attributes, metrics_true, y_true, y_pred):
    accuracies = []
    recalls = []
    precisions = []
    f1s = []
    tprs = []
    disparities = []
    for idx, attribute in enumerate(attributes):
        if attribute == 'intensity':
            continue

        attr_values = metrics_true[attribute]
        if attribute == 'thickness':
            processed = []
            for a in attr_values:
                processed.append(preprocess_thickness(a))
            attr_values = processed
        if attribute == 'age':
            processed = []
            for a in attr_values:
                processed.append(preprocess_age(a))
            attr_values = processed
        
        unique_attr_values = sorted(set(attr_values))
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
        
        attr_tprs = []
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
            f1 = f1*((tp_unique[av] + tn_unique[av] + fp_unique[av] + fn_unique[av])/len(y_true))
            f1s.append(f1)

            # save TPR score values
            div = (tp_unique[av] + fn_unique[av])
            tpr = 0 if div==0 else tp_unique[av]/div
            tprs.append(tpr)
            attr_tprs.append(tpr)

        disparities += tpr_disparity(attr_tprs)

    return accuracies, precisions, recalls, f1s, tprs, disparities


def test_pretrained(model_path, dataset, loss_fn, attributes, in_channels, out_channels):
    ## Test pretrained model ## 
    if "COLORED" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/colored_mnist/00/classifier_{}.pt".format(model_path), map_location=device))
    elif "MNIST" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_{}.pt".format(model_path), map_location=device))
    else:
        model = DenseNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_{}.pt".format(model_path), map_location=device))
        
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_pred, y_true, y_score, metrics_true, acc, f1, _ = test_classifier(model, test_loader, loss_fn)

    return metrics_true, y_true, y_pred, y_score

def plot_auc_per_subgroup(targets, preds, subgroups, labels, save_dir, save_name):
    roc_aucs = []
    fprs = []
    tprs = []
    y = np.array(targets)
    for subgroup in subgroups:
        y[targets != subgroup] = 0
        y[targets == subgroup] = 1
        ps = [p[subgroup] for p in preds]
        fpr, tpr, _ = roc_curve(y, ps)
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(7,4))
    for idx, subgroup in enumerate(subgroups):
        plt.plot(fprs[idx], tprs[idx], lw=1.5, alpha=.8, label='{}={}'.format(labels[idx], roc_aucs[idx]))

    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='k', label='Chance', alpha=.8)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    # plt.title('Race Classification', fontsize=14)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    ax.spines[['right', 'top']].set_visible(False)
    plt.savefig("{}/roc_auc_{}.png".format(save_dir, save_name))

def _roc_auc_score(y_true, y_score, run_name):
    # _ = plt.figure(run_name)
    # y_score = [p[1] for p in y_score]
    roc_auc = roc_auc_score(y_true, y_score, multi_class='ovo')
    # roc_auc = roc_auc_score(y_true, y_score)
    print("AUC score")
    print(roc_auc)

    # fpr, tpr, _ = roc_curve(y_true, y_score)
    # plt.plot(fpr,tpr)
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.tight_layout()
    # plt.savefig("plots/rocauc/roc-curve-{}.png".format(run_name))
    # plt.close()

def get_label_metrics(y_true, y_pred, num_classes):
    ## Get classification report and per-class performance ##
    report_dict = metrics.classification_report(y_true, y_pred, digits=range(10), output_dict=True)
    print(report_dict)

    matrix = metrics.confusion_matrix(y_true, y_pred)
    accs = matrix.diagonal()/matrix.sum(axis=1)
    f1s = [report_dict[str(label)]['f1-score'] for label in range(num_classes)]
    precisions = [report_dict[str(label)]['precision'] for label in range(num_classes)]
    recalls = [report_dict[str(label)]['recall'] for label in range(num_classes)]

    return accs, f1s, precisions, recalls, report_dict['accuracy'], report_dict['macro avg']

def test_mnist(models, test_dataset, in_channels, num_classes, attributes, subgroup_names, model_suffix, save_dir):
    loss_fn = torch.nn.CrossEntropyLoss()

    accs = []
    f1s = []
    precisions = []
    recalls = []
    attr_accs = []
    attr_precs = []
    attr_rcs = []
    attr_f1s = []
    attr_tprs = []
    attr_disparities = []
    overall_accs = []
    overall_f1s = []
    spider_values = []
    label_tprs = []
    label_disparities = []

    for model in models:
        print("[Test trained]\tTesting model: " + model)

        mnist_model_path = model + model_suffix
        metrics_true, y_true, y_pred, y_score = test_pretrained(mnist_model_path, test_dataset, loss_fn, attributes, in_channels, num_classes)

        acc, f1, precision, recall, overall_acc, overall_metrics = get_label_metrics(y_true, y_pred, num_classes)
        accs.append(acc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        overall_accs.append(overall_acc)
        overall_f1s.append(overall_metrics['f1-score'])

        tprs = get_tprs(y_true, y_pred, num_classes)
        label_tprs.append(tprs)
        disparities = tpr_disparity(tprs)
        label_disparities.append(disparities)

        _roc_auc_score(y_true, y_score, model)
        plot_auc_per_subgroup(y_true, y_score, list(range(10)), list(range(10)), save_dir, model)

        attr_acc, attr_pr, attr_rc, attr_f1, attr_tpr, attr_disparity = metrics_per_attribute(attributes, metrics_true, y_true, y_pred)
        attr_accs.append(attr_acc)
        attr_precs.append(attr_pr)
        attr_rcs.append(attr_rc)
        attr_f1s.append(attr_f1)
        attr_tprs.append(attr_tpr)
        attr_disparities.append(attr_disparity)
        spider_values.append([overall_acc, overall_metrics['precision'], overall_metrics['recall'], min(acc), min(precision), min(recall)])

    plot_metrics_comparison(models, accs, 'Accuracy', save_dir)
    plot_metrics_comparison(models, f1s, 'F1-score', save_dir)
    plot_metrics_comparison(models, precisions, 'Precision', save_dir)
    plot_metrics_comparison(models, recalls, 'Recall', save_dir)

    plot_metric_subgroup_comparison(subgroup_names, attr_accs, overall_accs, "Change in accuracy", models, save_dir)
    plot_metric_subgroup_comparison(subgroup_names, attr_f1s, overall_f1s, "Change in f1-score", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_accs, "Accuracy", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_precs, "Precision", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_rcs, "Recall", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_f1s, "F1-score", models, save_dir)
    # plot_metric_subgroup(subgroup_names, attr_tprs, "TPR", models, save_dir)
    # plot_metric_subgroup(subgroup_names, attr_disparities, "TPR disparity", models, save_dir)

    plot_metric_subgroup(list(range(num_classes)), label_tprs, "TPR", models, save_dir)
    plot_metric_subgroup(list(range(num_classes)), label_disparities, "TPR disparity", models, save_dir)

    box_plot_tpr_disparity(attr_tprs, models, save_dir)

    cats = ["Avg Accuracy", "Avg Recall", "Avg Precision", "Worst Accuracy", "Worst Precision", "Worst Recall"]
    spider_plot(models, cats, spider_values, save_dir)

def test_chestxray():
    #models = ["resBASELINE"]
    #models = ["BASELINE", "OVERSAMPLING_age0", "AUGMENTATIONS_age0"]
    models = ["BASELINE", "OVERSAMPLING_race", "AUGMENTATIONS_race", "GROUP_DRO_race", "COUNTERFACTUALS_race", "COUNTERFACTUALS_race_MIXUP", "CFREGULARISATION_race"]
    # models = ["BASELINE", "OVERSAMPLING_race", "AUGMENTATIONS_race", "GROUP_DRO_race", "COUNTERFACTUALS_race", "COUNTERFACTUALS_race_MIXUP", "CFREGULARISATION_race"]
    suffix = "_disease_pred"
    in_channels = 1
    num_classes = 2
    attributes = ['sex', 'age', 'race']
    # attributes = ['sex', 'age']
    loss_fn = torch.nn.CrossEntropyLoss()
    subgroup_names = ['Male', 'Female', '0-19', '20-39', '40-59', '60-79', '80-99', 'White', 'Asian', 'Black']
    # subgroup_names = ['Male', 'Female', '0-19', '20-39', '40-59', '60-79', '80-99']
    save_dir = 'plots/metrics_comp/chestxray_disease'
    
    accs = []
    f1s = []
    precisions = []
    recalls = []
    attr_accs = []
    attr_precs = []
    attr_rcs = []
    attr_f1s = []
    attr_tprs = []
    attr_disparities = []
    overall_accs = []
    overall_f1s = []
    spider_values = []
    
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    for model in models:
        print("[Test trained]\tTesting model: " + model)

        chestxray_model_path = "{}{}_CHESTXRAY".format(model, suffix)
        metrics_true, y_true, y_pred, y_score = test_pretrained(chestxray_model_path, test_dataset, loss_fn, attributes, in_channels, num_classes)

        acc, f1, precision, recall, overall_acc, overall_metrics = get_label_metrics(y_true, y_pred, num_classes)
        accs.append(acc)
        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        overall_accs.append(overall_acc)
        overall_f1s.append(overall_metrics['f1-score'])

        _roc_auc_score(y_true, y_score, model)
        plot_auc_per_subgroup(y_true, y_score, [0,1], ["No finding", "Pleural Effusion"], save_dir, model)
        # plot_auc_per_subgroup(y_true, y_score, [0,1,2], ["White", "Asian", "Black"], save_dir, model)

        ## Print performance metrics per attribute (eg. sex, digit etc) ##
        attr_acc, attr_pr, attr_rc, attr_f1, attr_tpr, attr_disparity = metrics_per_attribute(attributes, metrics_true, y_true, y_pred)
        attr_accs.append(attr_acc)
        attr_precs.append(attr_pr)
        attr_rcs.append(attr_rc)
        attr_f1s.append(attr_f1)
        attr_tprs.append(attr_tpr)
        attr_disparities.append(attr_disparity)
        spider_values.append([overall_acc, overall_metrics['precision'], overall_metrics['recall'],
                              max(attr_acc), max(attr_pr), max(attr_rc), min(attr_acc), min(attr_pr), min(attr_rc)])

    plot_metrics_comparison(models, accs, 'Accuracy', save_dir)
    plot_metrics_comparison(models, f1s, 'F1-score', save_dir)
    plot_metrics_comparison(models, precisions, 'Precision', save_dir)
    plot_metrics_comparison(models, recalls, 'Recall', save_dir)

    plot_metric_subgroup_comparison(subgroup_names, attr_accs, overall_accs, "Change in accuracy", models, save_dir)
    plot_metric_subgroup_comparison(subgroup_names, attr_f1s, overall_f1s, "Change in f1-score", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_accs, "Accuracy", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_precs, "Precision", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_rcs, "Recall", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_f1s, "F1-score", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_tprs, "TPR", models, save_dir)
    plot_metric_subgroup(subgroup_names, attr_disparities, "TPR disparity", models, save_dir)

    cats = ["Avg Accuracy", "Avg Recall", "Avg Precision", "Best Accuracy", "Best Precision", "Best Recall", "Worst Accuracy", "Worst Precision", "Worst Recall"]
    spider_plot(models, cats, spider_values, save_dir)
    
    worst_base = min(attr_accs[0])
    for idx in range(len(models))[1:]:
        get_global_change_in_performance(attr_accs[0], attr_accs[idx], models[idx])
        get_local_change_in_performance(worst_base, min(attr_accs[idx]), models[idx])

def test_perturbed_mnist():
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list)
    models = ["BASELINE", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    in_channels = 1
    num_classes = 10
    attributes = ['thickness', 'intensity']
    subgroup_names = ['thin', 'thick']
    save_dir = 'plots/metrics_comp/mnist'
    model_suffix = "_PERTURBED_MNIST"
    test_mnist(models, test_dataset, in_channels, num_classes, attributes, subgroup_names, model_suffix, save_dir)

def test_colored_mnist():
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = ColoredMNIST(train=False, transform=transforms_list)
    models = ["BASELINE", "OVERSAMPLING", "AUGMENTATIONS", "GROUP_DRO", "COUNTERFACTUALS", "CFREGULARISATION"]
    in_channels = 3
    num_classes = 10
    attributes = ['color']
    subgroup_names = ['red', 'orange', 'yellow', 'lime', 'green', 'teal', 'blue', 'purple', 'pink', 'magenta']
    save_dir = 'plots/metrics_comp/colored_mnist/00'
    model_suffix = "_COLORED_MNIST"
    test_mnist(models, test_dataset, in_channels, num_classes, attributes, subgroup_names, model_suffix, save_dir)

# test_chestxray()
test_colored_mnist()