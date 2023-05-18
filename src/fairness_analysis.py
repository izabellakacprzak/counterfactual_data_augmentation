import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as TF
from tqdm import tqdm
import numpy as np
import os

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from classifier import ConvNet, DenseNet
from utils.utils import apply_debiasing_method, AugmentationMethod
from utils.params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    cf, cf_metrics = generate_cf(obs, do_a=do_a, do_f=do_f, do_r=do_r, do_s=do_s)
    return cf, cf_metrics

def _gen_cfs(test_loader, perturbs_per_sample, do_cfs, run_name):
    originals = []
    perturbed = []
    for _, (data, metrics, labels) in enumerate(tqdm(test_loader)):
        originals += data.unsqueeze(1)
        data = data.to(device)
        labels = labels.to(device)

        # get probabilities for counterfactual data with interventions on specific attribute
        for _ in range(perturbs_per_sample):
            for i in range(len(data)):
                if do_cfs:
                    if "MNIST" in run_name:
                        perturbed.append(_get_cf_for_mnist(data[i][0], metrics['thickness'][i], metrics['intensity'][i], labels[i]))
                    else:
                        do_a, do_f, do_r, do_s = 0, None, None, None
                        ms = {k:vs[i] for k,vs in metrics.items()}
                        cf, cf_metrics = _get_cf_for_chestxray(data[i][0], ms, labels[i], do_a, do_f, do_r, do_s)
                        if len(cf_metrics) != 0: perturbed.append(torch.tensor(cf).to(device)) 
                else:
                    img = apply_debiasing_method(AugmentationMethod.AUGMENTATIONS, data[i][0].cpu().detach().numpy())
                    perturbed.append(torch.tensor(img).to(device))
                    
    perturbed = torch.stack(perturbed)
    if not do_cfs:
        perturbed = perturbed.unsqueeze(1)

    return originals, perturbed

# Generates a scatterplot of how similar predictions made by classifier 
# on counterfactual data are to predictions on original data
# all points should be clustered along the y=x line - meaning high classifier fairness
def classifier_fairness_analysis(model, run_name, originals, perturbed, fairness_label):
    original_probs = []
    perturbed_probs = []
    for idx in range(len(originals)):
        original = originals[idx]
        perturbed = perturbed[idx]

        original = original.to(device)
        logits = model(original).cpu()
        prob = torch.nn.functional.softmax(logits, dim=1).tolist()
        original_probs.append(prob[fairness_label])
        
        logits = model(perturbed).cpu()
        prob = torch.nn.functional.softmax(logits, dim=1).tolist()
        perturbed_probs.append(prob[fairness_label])

    os.remove("originals.txt")    
    os.remove('cfs.txt')
    with open('originals.txt', 'x') as fp:
        for item in original_probs:
            # write each item on a new line
            fp.write("%s\n" % item)
    
    with open('cfs.txt', 'x') as fp:
        for item in perturbed_probs:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')
    
    fig = plt.figure(run_name)
    plt.scatter(np.array(original_probs), perturbed_probs)
    plt.savefig("plots/fairness_age_0-19.png")

def fairness_analysis(model_path, test_dataset, in_channels, out_channels, fairness_label, do_cfs=True):
    if "MNIST" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_"+model_path+".pt", map_location=device))
    else:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_"+model_path+".pt", map_location=device))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    originals, perturbed = _gen_cfs(test_loader, 3, do_cfs, model_path)
    classifier_fairness_analysis(model, model_path, originals, perturbed, fairness_label)
    
def visualise_perturbed_mnist():
    #models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    models = ["AUGMENTATIONS"]
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    for model in models:
        mnist_model_path = model + "_PERTURBED_MNIST"
        fairness_analysis(mnist_model_path, test_dataset, 1, 10, 0, False)

def visualise_chestxray():
    models = ["COUNTERFACTUALS_age_0"]

    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    for model in models:
        chestxray_model_path = model + "_CHESTXRAY"
        fairness_analysis(chestxray_model_path, test_dataset, 1, 2, 0)

#visualise_perturbed_mnist()
visualise_chestxray()
