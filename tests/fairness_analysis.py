import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as TF
from tqdm import tqdm
import os
import numpy as np
import torch.nn.functional as F
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

def _gen_cfs(test_loader, perturbs_per_sample, do_cfs, run_name):
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
                    
    #perturbed = torch.stack(perturbed)
    #if not do_cfs:
    #    perturbed = perturbed.unsqueeze(1)

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

    original_data_file = "data/colored_mnist/originals{}.txt".format(run_name)
    perturbed_data_file = "data/colored_mnist/cfs{}.txt".format(run_name)
    if os.path.exists(original_data_file):
        os.remove(original_data_file)    
    if os.path.exists(perturbed_data_file):
        os.remove(perturbed_data_file)
    with open(original_data_file, 'x') as fp:
        for item in original_probs:
            # write each item on a new line
            fp.write("%s\n" % item)
    
    with open(perturbed_data_file, 'x') as fp:
        for item in perturbed_probs:
            # write each item on a new line
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
    # models = ["AUGMENTATIONS"]
    model_type = "_PERTURBED_MNIST"
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    do_cfs = True
    originals, perturbed = _gen_cfs(test_loader, 7, do_cfs, model_type)

    for model in models:
        mnist_model_path = model + model_type
        fairness_analysis(mnist_model_path, originals, perturbed, 1, 10, 9)

def visualise_colored_mnist():
    models = [ "BASELINE", "OVERSAMPLING", "AUGMENTATIONS", "GROUP_DRO", "COUNTERFACTUALS", "CFREGULARISATION"]
    model_type = "_COLORED_MNIST"
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = ColoredMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=0.01)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    do_cfs = True
    originals, perturbed = _gen_cfs(test_loader, 1, do_cfs, model_type)

    for model in models:
        mnist_model_path = model + model_type
        fairness_analysis(mnist_model_path, originals, perturbed, 3, 10, 9)

def visualise_chestxray():
    # models = ["BASELINE", "OVERSAMPLING_race", "AUGMENTATIONS_race", "GROUP_DRO_race", "COUNTERFACTUALS_race", "COUNTERFACTUALS_race_MIXUP", "CFREGULARISATION_race"]
    models = ["BASELINE", "CFREGULARISATION_age_disease"]
    #models = ["BASELINE", "GROUP_DRO_race", "OVERSAMPLING_black", "AUGMENTATIONS_black", "MIXUP_black", "COUNTERFACTUALS_black", "COUNTERFACTUALS_DRO_black"]
    # model_type = "_disease_pred_CHESTXRAY"
    model_type = "_race_pred_CHESTXRAY"
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    do_cfs = True
    originals, perturbed = _gen_cfs(test_loader, 7, do_cfs, model_type)

    for model in models:
        chestxray_model_path = model + model_type
        fairness_analysis(chestxray_model_path, originals, perturbed, 1, 2, 0)

# visualise_perturbed_mnist()
# visualise_chestxray()
visualise_colored_mnist()
