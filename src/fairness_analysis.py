import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as TF
from tqdm import tqdm
import numpy as np

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

def _get_cf_for_chestxray(img, metrics, label, do_s, do_r, do_a):
    from dscmchest.generate_counterfactuals import generate_cf
    obs = {'x': img,
           'age': metrics['age'],
           'race': metrics['race'],
           'sex': metrics['sex'],
           'finding': label}
    cf, _ = generate_cf(obs, do_s=do_s, do_r=do_r, do_a=do_a)
    return cf

# Generates a scatterplot of how similar predictions made by classifier 
# on counterfactual data are to predictions on original data
# all points should be clustered along the y=x line - meaning high classifier fairness
def classifier_fairness_analysis(model, test_loader, run_name, fairness_label, perturbs_per_sample=5, do_cfs=True):
    X, Y = [], []
    for _, (data, metrics, labels) in enumerate(tqdm(test_loader)):
        data = data.to(device)
        labels = labels.to(device)
        logits = model(data).cpu()
        probs = torch.nn.functional.softmax(logits, dim=1).tolist()
        original_probs = []

        # get probabilities for original data
        for _, prob in enumerate(probs):
            original_probs.append(prob[fairness_label])
        # get probabilities for counterfactual data with interventions on specific attribute
        for _ in range(perturbs_per_sample):
            X = X + original_probs

            perturbed = []
            for i in range(len(data)):
                if do_cfs:
                    if "MNIST" in run_name:
                        perturbed.append(_get_cf_for_mnist(data[i][0], metrics['thickness'][i], metrics['intensity'][i], labels[i]))
                    else:
                        do_s, do_r, do_a = None, 2, None
                        ms = {k:vs[i] for k,vs in metrics.items()}
                        cf = _get_cf_for_chestxray(data[i][0], ms, labels[i], do_s, do_r, do_a)
                        if len(cf) != 0: perturbed.append(torch.tensor(cf).to(device)) 
                else:
                    img = apply_debiasing_method(AugmentationMethod.AUGMENTATIONS, data[i][0].cpu().detach().numpy())
                    perturbed.append(torch.tensor(img).to(device))
                    
            perturbed = torch.stack(perturbed)
            if not do_cfs:
                perturbed = perturbed.unsqueeze(1)
            logits = model(perturbed).cpu()
            probs = torch.nn.functional.softmax(logits, dim=1).tolist()
            perturbed_probs = []
            for _, prob in enumerate(probs):
                perturbed_probs.append(prob[fairness_label])
            Y = Y + perturbed_probs
    import os
    os.remove("originals.txt")    
    os.remove('cfs.txt')
    with open('originals.txt', 'x') as fp:
        for item in X:
            # write each item on a new line
            fp.write("%s\n" % item)
    
    with open('cfs.txt', 'x') as fp:
        for item in Y:
            # write each item on a new line
            fp.write("%s\n" % item)
    print('Done')

    X = np.array(X)
    Y = np.array(Y)
    
    fig = plt.figure(run_name)
    plt.scatter(X,Y)
    plt.savefig("plots/fairness_std_aug.png")

def fairness_analysis(model_path, test_dataset, in_channels, out_channels, fairness_label, do_cfs=True):
    if "MNIST" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_"+model_path+".pt", map_location=device))
    else:
        model = DenseNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_"+model_path+".pt", map_location=device))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    classifier_fairness_analysis(model, test_loader, model_path, fairness_label, do_cfs=do_cfs)
    
def visualise_perturbed_mnist():
    #models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    models = ["AUGMENTATIONS"]
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    for model in models:
        mnist_model_path = model + "_PERTURBED_MNIST"
        fairness_analysis(mnist_model_path, test_dataset, 1, 10, 0, False)

def visualise_chestxray():
    models = ["BIASED"]

    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(train=False, transform=transforms_list)

    for model in models:
        chestxray_model_path = model + "_CHESTXRAY"
        fairness_analysis(chestxray_model_path, test_dataset, 1, 2, 0, False)

#visualise_perturbed_mnist()
visualise_chestxray()
