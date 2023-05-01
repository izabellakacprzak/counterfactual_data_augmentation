import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.perturbedMNIST import PerturbedMNIST
from utils.utils import visualise_t_sne
from classifier import ConvNet
from utils.params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualise_embeddings(model_path):
    model = ConvNet(in_channels=1, out_channels=10)
    if "MNIST" in model_path:
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_"+model_path+".pt", map_location=device))
    else:
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_"+model_path+".pt", map_location=device))

    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    visualise_t_sne(test_loader, model, "plots/"+model_path+"t_sne")

def visualise_perturbed_mnist():
    models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    for model in models:
        mnist_model_path = model + "_PERTURBED_MNIST"
        visualise_embeddings(mnist_model_path)

def visualise_chestxray():
    models = ["BIASED"]
    for model in models:
        chestxray_model_path = model + "_CHESTXRAY"
        visualise_embeddings(chestxray_model_path)

# visualise_perturbed_mnist()
# visualise_chestxray()