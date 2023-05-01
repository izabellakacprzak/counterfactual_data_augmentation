import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from MNISTClassifier import ConvNet, test_MNIST
from params import *
from utils.evaluate import plot_metrics_comparison, classifier_fairness_analysis, metrics_per_attribute

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_pretrained(model_path, dataset, attributes, in_channels, out_channels):
    model = ConvNet(in_channels=in_channels, out_channels=out_channels)
    if "MNIST" in model_path:
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_"+model_path+".pt", map_location=device))
    else:
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_"+model_path+".pt", map_location=device))

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # classifier_fairness_analysis(model, test_loader, model_path)

    y_pred, y_true, metrics_true, acc, f1 = test_MNIST(model, test_loader)
    report_dict = metrics.classification_report(y_true, y_pred, digits=range(10), output_dict=True)

    metrics_per_attribute(attributes, metrics_true, y_true, y_pred)

    f1s = []
    precisions = []
    recalls = []
    for label in range(out_channels):
        label = str(label)
        f1s.append(report_dict[label]['f1-score'])
        precisions.append(report_dict[label]['precision'])
        recalls.append(report_dict[label]['recall'])

    return f1s, precisions, recalls

def test_perturbed_mnist():
    models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    f1s = []
    precisions = []
    recalls = []
    transforms_list = transforms.Compose([transforms.ToTensor()])

    for model in models:
        print("[Test trained]\tTesting model: " + model)

        mnist_models_path = model + "_PERTURBED_MNIST"
        test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

        f1, precision, recall = test_pretrained(mnist_models_path, test_dataset, ['thickness', 'intensity', 'bias_aligned'], 1, 10)


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
    for model in models:
        print("[Test trained]\tTesting model: " + model)

        chestray_models_path = model + "_CHESTXRAY"
        test_dataset = ChestXRay(train=False, transform=transforms_list)

        f1, precision, recall = test_pretrained(chestray_models_path, test_dataset, ['sex', 'age', 'race'], 1, 2)

        f1s.append(f1)
        precisions.append(precision)
        recalls.append(recall)


    plot_metrics_comparison(models, f1s, 'CHESTXRAYf1-score')
    plot_metrics_comparison(models, precisions, 'CHESTXRAYprecision')
    plot_metrics_comparison(models, recalls, 'CHESTXRAYrecall')

# test_perturbed_mnist()
test_chestxray()
