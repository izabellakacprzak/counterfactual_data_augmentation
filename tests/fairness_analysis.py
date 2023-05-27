import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as TF
from tqdm import tqdm
import numpy as np
import os

import sys
sys.path.append("..")

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from classifier import ConvNet, DenseNet
from utils.utils import apply_debiasing_method, DebiasingMethod
from utils.params import *

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

def _get_cf_for_mnist(img, thickness, intensity, label):
    from dscm.generate_counterfactuals import generate_counterfactual_for_x
    img = img.float() * 254
    img = TF.Pad(padding=2)(img).type(torch.ByteTensor).unsqueeze(0)
    x_cf = generate_counterfactual_for_x(img, thickness, intensity, label)
    return torch.from_numpy(x_cf).unsqueeze(0).float()

def _get_cf_for_chestxray(img, metrics, label, do_a, do_f, do_r, do_s):
    from dscmchest.generate_counterfactuals import generate_cf
    obs = {'x': img,
           'age': metrics['age'],
           'race': metrics['race'],
           'sex': metrics['sex'],
           'finding': label}
    cf = generate_cf(obs, do_a=do_a, do_f=do_f, do_r=do_r, do_s=do_s)
    return cf

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
                    if "MNIST" in run_name:
                        perturbed.append(_get_cf_for_mnist(data[i][0], metrics['thickness'][i], metrics['intensity'][i], labels[i]))
                    else:
                        do_a, do_f, do_r, do_s = None, 1, None, None
                        ms = {k:vs[i] for k,vs in metrics.items()}
                        cf = _get_cf_for_chestxray(data[i][0], ms, labels[i], do_a, do_f, do_r, do_s)
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
        original_probs.append(prob[fairness_label])
        
        logits = model(perturbed.unsqueeze(0)).cpu()
        prob = torch.nn.functional.softmax(logits, dim=1).tolist()
        perturbed_probs.append(prob[fairness_label])

    if os.path.exists("originals{}.txt".format(run_name)):
        os.remove("originals{}.txt".format(run_name))    
    if os.path.exists("cfs{}.txt".format(run_name)):
        os.remove('cfs{}.txt'.format(run_name))
    with open('originals{}.txt'.format(run_name), 'x') as fp:
        for item in original_probs:
            # write each item on a new line
            fp.write("%s\n" % item)
    
    with open('cfs{}.txt'.format(run_name), 'x') as fp:
        for item in perturbed_probs:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')
    
   # fig = plt.figure(run_name)
   # plt.scatter(np.array(original_probs), perturbed_probs)
   # plt.savefig("plots/fairness_{}.png".format(run_name))

def fairness_analysis(model_path, originals, perturbed, in_channels, out_channels, fairness_label):
    if "MNIST" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_{}.pt".format(model_path), map_location=device))
    else:
        model = DenseNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_{}.pt".format(model_path), map_location=device))
    model.eval()
    classifier_fairness_analysis(model, model_path, originals, perturbed, fairness_label)
    
def visualise_perturbed_mnist():
    #models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    models = ["AUGMENTATIONS"]
    model_type = "_PERTURBED_MNIST"
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    do_cfs = False
    originals, perturbed = _gen_cfs(test_loader, 3, do_cfs, model_type)

    for model in models:
        mnist_model_path = model + model_type
        fairness_analysis(mnist_model_path, originals, perturbed, 1, 10, 0)

def visualise_chestxray():
    models = ["BASELINE"]
    #models = ["BASELINE", "GROUP_DRO_race", "OVERSAMPLING_black", "AUGMENTATIONS_black", "MIXUP_black", "COUNTERFACTUALS_black", "COUNTERFACTUALS_DRO_black"]
    model_type = "_CHESTXRAY"
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    do_cfs = True
    originals, perturbed = _gen_cfs(test_loader, 5, do_cfs, model_type)

    for model in models:
        chestxray_model_path = model + model_type
        fairness_analysis(chestxray_model_path, originals, perturbed, 1, 2, 0)

#visualise_perturbed_mnist()
visualise_chestxray()
