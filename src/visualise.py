import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.perturbedMNIST import PerturbedMNIST
from utils.utils import visualise_t_sne
from MNISTClassifier import ConvNet
from params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_name = "BALANCED_PERTURBED_MNIST"
model = ConvNet(in_channels=1, out_channels=10)
model.load_state_dict(torch.load("../checkpoints/mnist_classifier" + run_name + ".pt", map_location=device))

transforms_list = transforms.Compose([transforms.ToTensor()])
test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
visualise_t_sne(test_loader, model, "plots/"+run_name+"t_sne")
