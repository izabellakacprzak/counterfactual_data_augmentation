import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from utils.evaluate import visualise_t_sne
from classifier import ConvNet, DenseNet, test_classifier
from utils.params import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
def visualise_embeddings(model_path, test_dataset, in_channels, out_channels, img_dim):
    if "MNIST" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_{}.pt".format(model_path), map_location=device))
    else:
        model = DenseNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_{}.pt".format(model_path), map_location=device))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    visualise_t_sne(test_loader, model, img_dim, "plots/{}t_sne".format(model_path))

def visualise_perturbed_mnist():
    models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    for model in models:
        mnist_model_path = model + "_PERTURBED_MNIST"
        visualise_embeddings(mnist_model_path, test_dataset, 1, 10, 28)

def visualise_chestxray():
    models = ["BASELINE", "OVERSAMPLING_age_0", "AUGMENTATIONS_age_0", "COUNTERFACTUALS_age_0"]

    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    for model in models:
        chestxray_model_path = model + "_CHESTXRAY"
        visualise_embeddings(chestxray_model_path, test_dataset, 1, 2, 192)

# visualise_perturbed_mnist()
visualise_chestxray()
