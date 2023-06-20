import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as TF
from tqdm import tqdm
import os
import numpy as np
import random

import sys
sys.path.append("..")

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.coloredMNIST import ColoredMNIST
from datasets.chestXRay import ChestXRay
from classifier import ConvNet, DenseNet
from utils.utils import apply_debiasing_method, DebiasingMethod
from utils.params import *
from utils.cf_utils import *

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

def _generate_perturbed_samples(test_loader, perturbs_per_sample, do_cfs, run_name):
    originals = []
    perturbed = []
    for _, (data, metrics, labels) in enumerate(tqdm(test_loader)):
        data = data.to(device)
        labels = labels.to(device)

        # get probabilities for counterfactual data with interventions on specific attribute
        for _ in range(perturbs_per_sample):
            for i in range(len(data)):
                if do_cfs:
                    if "PERTURBED" in run_name:
                        perturbed.append(get_cf_for_mnist(data[i][0], metrics['thickness'][i], metrics['intensity'][i], labels[i]))
                        originals.append(data[i])
                    elif "COLORED" in run_name:
                        col = random.randint(0,9)
                        img_cf = get_cf_for_colored_mnist(data[i], metrics['color'][i], labels[i], col)
                        img_cf = np.transpose(img_cf, (2, 0, 1))
                        img_cf = torch.from_numpy(img_cf).float().to(device)
                        perturbed.append(img_cf)
                        originals.append(data[i])
                    else:
                        do_a, do_f, do_r, do_s = 0, 1, None, None
                        ms = {k:vs[i] for k,vs in metrics.items()}
                        cf = get_cf_for_chestxray(data[i][0], ms, labels[i], do_a, do_f, do_r, do_s)
                        if len(cf) != 0:
                            perturbed.append(torch.tensor(cf).to(device))
                            originals.append(data[i])
                else:
                    originals.append(data[i])
                    img = apply_debiasing_method(DebiasingMethod.AUGMENTATIONS, data[i][0].cpu().detach().numpy())
                    perturbed.append(torch.tensor(img).unsqueeze(0).to(device))
                    
    return originals, perturbed

# Generates a scatterplot of how similar predictions made by classifier 
# on counterfactual data are to predictions on original data
# all points should be clustered along the y=x line - meaning high classifier fairness
def classifier_fairness_analysis(model, run_name, originals, perturbeds, fairness_label):
    original_probs = []
    perturbed_probs = []
    for idx in tqdm(range(len(originals))):
        original = originals[idx]
        perturbed = perturbeds[idx]

        original = original.to(device)
        logits = model(original.unsqueeze(0)).cpu()
        prob = torch.nn.functional.softmax(logits, dim=1).tolist()
        original_probs.append(prob[0][fairness_label])
        
        perturbed = perturbed.to(device)
        logits = model(perturbed.unsqueeze(0)).cpu()
        prob = torch.nn.functional.softmax(logits, dim=1).tolist()
        perturbed_probs.append(prob[0][fairness_label])

    # workaround to problems with Latex libraries on lab machines 
    original_data_file = "data/colored_mnist/originals{}.txt".format(run_name)
    perturbed_data_file = "data/colored_mnist/cfs{}.txt".format(run_name)
    if os.path.exists(original_data_file):
        os.remove(original_data_file)    
    if os.path.exists(perturbed_data_file):
        os.remove(perturbed_data_file)
    with open(original_data_file, 'x') as fp:
        for item in original_probs:
            fp.write("%s\n" % item)
    
    with open(perturbed_data_file, 'x') as fp:
        for item in perturbed_probs:
            fp.write("%s\n" % item)
    print('Done')
    
    # fig = plt.figure(run_name)
    # plt.scatter(np.array(original_probs), perturbed_probs)
    # plt.savefig("plots/fairness_{}.png".format(run_name))

def fairness_analysis(model_path, originals, perturbed, in_channels, out_channels, fairness_label):
    if "PERTURBED" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_{}.pt".format(model_path), map_location=device))
    elif "COLORED" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/colored_mnist/01/classifier_{}.pt".format(model_path), map_location=device))
    else:
        model = DenseNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_{}.pt".format(model_path), map_location=device))
    model.eval()
    classifier_fairness_analysis(model, model_path, originals, perturbed, fairness_label)
    
def visualise_perturbed_mnist():
    models = [ "BASELINE", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    model_type = "_PERTURBED_MNIST"
    in_channels = 1
    out_channels = 2
    fairness_label = 0
    do_cfs = True
    perturbed_per_sample = 7

    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    originals, perturbed = _generate_perturbed_samples(test_loader, perturbed_per_sample, do_cfs, model_type)

    for model in models:
        mnist_model_path = model + model_type
        fairness_analysis(mnist_model_path, originals, perturbed, in_channels, out_channels, fairness_label)

def visualise_colored_mnist():
    models = [ "BASELINE", "OVERSAMPLING", "AUGMENTATIONS", "GROUP_DRO", "COUNTERFACTUALS", "CFREGULARISATION"]
    model_type = "_COLORED_MNIST"
    in_channels = 1
    out_channels = 2
    fairness_label = 0
    do_cfs = True
    perturbed_per_sample = 7

    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = ColoredMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=0.01)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    originals, perturbed = _generate_perturbed_samples(test_loader, perturbed_per_sample, do_cfs, model_type)

    for model in models:
        mnist_model_path = model + model_type
        fairness_analysis(mnist_model_path, originals, perturbed, in_channels, out_channels, fairness_label)

def visualise_chestxray():
    models = ["BASELINE", "OVERSAMPLING_race", "AUGMENTATIONS_race", "GROUP_DRO_race", "COUNTERFACTUALS_race", "COUNTERFACTUALS_race_MIXUP", "CFREGULARISATION_race"]
    model_type = "_race_pred_CHESTXRAY"
    img_dim = 192
    in_channels = 1
    out_channels = 2
    fairness_label = 0
    do_cfs = True
    perturbed_per_sample = 7

    transforms_list = transforms.Compose([transforms.Resize((img_dim,img_dim)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    originals, perturbed = _generate_perturbed_samples(test_loader, perturbed_per_sample, do_cfs, model_type)

    for model in models:
        chestxray_model_path = model + model_type
        fairness_analysis(chestxray_model_path, originals, perturbed, in_channels, out_channels, fairness_label)

# visualise_perturbed_mnist()
# visualise_chestxray()
visualise_colored_mnist()
