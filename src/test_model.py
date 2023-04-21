import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn import metrics

from datasets.perturbedMNIST import PerturbedMNIST
from MNISTClassifier import ConvNet, test_MNIST
from params import *
from utils.evaluate import plot_metrics_comparison

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = ["BALANCED", "IMBALANCED", "OVERSAMPLING", "AUGMENTATIONS"]

def test_pretrained(model_name):
    model = ConvNet(in_channels=1, out_channels=10)
    run_name = model_name + "_PERTURBED_MNIST"
    model.load_state_dict(torch.load("../checkpoints/mnist_classifier" + run_name + ".pt", map_location=device))

    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_pred, y_true, acc, f1 = test_MNIST(model, test_loader)
    report_dict = metrics.classification_report(y_true, y_pred, digits=len(10), output_dict=True)
    print(report_dict)
    precisions = []
    recalls = []
    f1s = []
    # for digit in range(10):

    # plot_metrics_comparison(models, )

for model in models:
    print("Testing model: " + model)
    test_pretrained(model)