import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms as TF
from tqdm import tqdm
import numpy as np

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from classifier import ConvNet
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
def classifier_fairness_analysis(model, test_loader, run_name, fairness_label, cfs_per_sample=1):
    X, Y = [], []
    for _, (data, metrics, labels) in enumerate(tqdm(test_loader)):
        data = data.to(device)
        labels = labels.to(device)
        logits = model.model(data).cpu()
        probs = torch.nn.functional.softmax(logits, dim=1).tolist()
        original_probs = []

        # get probabilities for original data
        for _, prob in enumerate(probs):
            original_probs.append(prob[fairness_label])
        print(original_probs) 
        # get probabilities for counterfactual data with interventions on specific attribute
        for _ in range(cfs_per_sample):
            X = X + original_probs

            cfs = []
            for i in range(len(data)):
                if "MNIST" in run_name:
                    cfs.append(_get_cf_for_mnist(data[i][0], metrics['thickness'][i], metrics['intensity'][i], labels[i]))
                else:
                    do_s, do_r, do_a = None, 1, None
                    ms = {k:vs[i] for k,vs in metrics.items()}
                    cf = _get_cf_for_chestxray(data[i][0], ms, labels[i], do_s, do_r, do_a)
                    if len(cf) != 0: cfs.append(torch.from_numpy(cf).to(device))

            cfs = torch.stack(cfs)
            logits = model.model(cfs).cpu()
            probs = torch.nn.functional.softmax(logits, dim=1).tolist()
            cf_probs = []
            for _, prob in enumerate(probs):
                cf_probs.append(prob[fairness_label])
            print(cf_probs)
            Y = Y + cf_probs

    X = np.array(X)
    Y = np.array(Y)

    fig = plt.figure(run_name)
    plt.scatter(X,Y)
    plt.savefig("plots/fairness_correct_"+ run_name +".png")

def fairness_analysis(model_path, test_dataset, in_channels, out_channels, fairness_label):
    model = ConvNet(in_channels=in_channels, out_channels=out_channels)
    if "MNIST" in model_path:
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_"+model_path+".pt", map_location=device))
    else:
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_"+model_path+".pt", map_location=device))
    model.model.eval()

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    classifier_fairness_analysis(model, test_loader, model_path, fairness_label)
    
def visualise_perturbed_mnist():
    models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    for model in models:
        mnist_model_path = model + "_PERTURBED_MNIST"
        fairness_analysis(mnist_model_path, test_dataset, 1, 10, 0)

def visualise_chestxray():
    models = ["BIASED"]

    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(train=False, transform=transforms_list)

    for model in models:
        chestxray_model_path = model + "_CHESTXRAY"
        fairness_analysis(chestxray_model_path, test_dataset, 1, 2, 0)

visualise_chestxray()
