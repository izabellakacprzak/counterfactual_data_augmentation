import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from MNISTClassifier import ConvNet, test_MNIST
from params import *
from utils.evaluate import plot_metrics_comparison, classifier_fairness_analysis

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_pretrained(dataset, model_path):
    model = ConvNet(in_channels=1, out_channels=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # classifier_fairness_analysis(model, test_loader, model_name)

    y_pred, y_true, acc, f1 = test_MNIST(model, test_loader)
    report_dict = metrics.classification_report(y_true, y_pred, digits=range(10), output_dict=True)
    print(report_dict)

    f1s = []
    precisions = []
    recalls = []
    for digit in range(10):
        digit = str(digit)
        f1s.append(report_dict[digit]['f1-score'])
        precisions.append(report_dict[digit]['precision'])
        recalls.append(report_dict[digit]['recall'])

    return f1s, precisions, recalls

def test_perturbed_mnist():
    models = ["UNBIASED_PERTURBED_MNIST", "BIASED_PERTURBED_MNIST", "OVERSAMPLING_PERTURBED_MNIST", "AUGMENTATIONS_PERTURBED_MNIST", "COUNTERFACTUALS_PERTURBED_MNIST", "CF_REGULARISATION_PERTURBED_MNIST"]
    f1s = []
    precisions = []
    recalls = []
    transforms_list = transforms.Compose([transforms.ToTensor()])

    for model in models:
        # TODO: refactor all prints to have function name calling them
        print("Testing model: " + model)

        mnist_models_path = "../checkpoints/mnist/classifier_" + model + ".pt"
        test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

        f1, precision, recall = test_pretrained(mnist_models_path, test_dataset)

        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)


    plot_metrics_comparison(models, f1s, 'f1-score')
    plot_metrics_comparison(models, precisions, 'precision')
    plot_metrics_comparison(models, recalls, 'recall')

def test_chestxray():
    models = ["BIASED_CHESTXRAY"]
    f1s = []
    precisions = []
    recalls = []
    transforms_list = transforms.Compose([transforms.ToTensor()])

    for model in models:
        # TODO: refactor all prints to have function name calling them
        print("Testing model: " + model)

        chestray_models_path = "../checkpoints/chestxray/classifier_" + model + ".pt"
        test_dataset = ChestXRay(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

        f1, precision, recall = test_pretrained(chestray_models_path, test_dataset)

        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)


    plot_metrics_comparison(models, f1s, 'f1-score')
    plot_metrics_comparison(models, precisions, 'precision')
    plot_metrics_comparison(models, recalls, 'recall')

test_chestxray()