import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from classifier import ConvNet, test_classifier
from utils.params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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
    for idx, attribute in enumerate(attributes):
        if attribute == 'thickness':
            continue

        attr_values = metrics_true[idx]
        unique_attr_values = set(attr_values)

        unique_counts = {u:0 for u in unique_attr_values}
        unique_correct_counts = {u:0 for u in unique_attr_values}

        for idx, t in enumerate(y_true):
            p = y_pred[idx]
            unique_counts[attr_values[idx]] = unique_counts[attr_values[idx]] + 1
            if p == t:
                unique_correct_counts[attr_values[idx]] = unique_correct_counts[attr_values[idx]] + 1

        # Accuracy per attribute value
        print("Accuracy for " + str(attribute))
        for av in unique_attr_values:
            acc = str(unique_correct_counts[av] / unique_counts[av])
            print("Accuracy value for {}: {}".format(av, acc))

        # F1-score per attribute value
        print("F1-score for " + str(attribute))
        for av in unique_attr_values:
            f1 = f1_score(unique_correct_counts[av], unique_counts[av])
            print("F1-score value for {}: {}".format(av, f1))

def test_pretrained(model_path, dataset, attributes, in_channels, out_channels):
    ## Test pretrained model ##
    model = ConvNet(in_channels=in_channels, out_channels=out_channels)
    if "MNIST" in model_path:
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_"+model_path+".pt", map_location=device))
    else:
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_"+model_path+".pt", map_location=device))

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_pred, y_true, metrics_true, acc, f1 = test_classifier(model, test_loader)

    ## Get classification report and per-class performance ##
    report_dict = metrics.classification_report(y_true, y_pred, digits=range(10), output_dict=True)
    f1s = []
    precisions = []
    recalls = []
    for label in range(out_channels):
        label = str(label)
        f1s.append(report_dict[label]['f1-score'])
        precisions.append(report_dict[label]['precision'])
        recalls.append(report_dict[label]['recall'])

    ## Print performance metrics per attribute (eg. age, thickness etc) ##
    metrics_per_attribute(attributes, metrics_true, y_true, y_pred)

    return f1s, precisions, recalls

def test_perturbed_mnist():
    models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
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
        f1, precision, recall = test_pretrained(mnist_model_path, test_dataset, attributes, in_channels, num_classes)

        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    plot_metrics_comparison(models, f1s, 'MNISTf1score')
    plot_metrics_comparison(models, precisions, 'MNISTprecision')
    plot_metrics_comparison(models, recalls, 'MNISTrecall')

def test_chestxray():
    models = ["BIASED"]
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
        f1, precision, recall = test_pretrained(chestxray_model_path, test_dataset, attributes, in_channels, num_classes)

        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)

    plot_metrics_comparison(models, f1s, 'CHESTXRAYf1-score')
    plot_metrics_comparison(models, precisions, 'CHESTXRAYprecision')
    plot_metrics_comparison(models, recalls, 'CHESTXRAYrecall')

# test_perturbed_mnist()
test_chestxray()
