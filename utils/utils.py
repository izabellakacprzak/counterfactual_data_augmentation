import numpy as np
import random
import os
import sys
import csv
import pandas as pd
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
from tqdm import tqdm

from .params import *
from enum import Enum
 
class DebiasingMethod(Enum):
    NONE = 1
    OVERSAMPLING = 2
    REWEIGHING = 3
    AUGMENTATIONS = 4
    PERTURBATIONS = 5
    COUNTERFACTUALS = 6
    CF_REGULARISATION = 7
    MIXUP = 8

class Augmentation(Enum):
    ROTATION = 1
    # FLIP_LEFT_RIGHT = 2
#     FLIP_TOP_BOTTOM = 3
    BLUR = 4
    SALT_AND_PEPPER_NOISE = 5

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

def _add_noise(image, noise_type="gauss"):
    if noise_type == "gauss":
        row,col, ch= image.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        if len(image.shape)==2:
            row,col= image.shape
        else:
            _,row,col= image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out
    return image
   
def apply_debiasing_method(method, img):
    if method == DebiasingMethod.OVERSAMPLING:
        return img
    elif method == DebiasingMethod.AUGMENTATIONS:
        augmentation = random.choice(list(Augmentation))
        if augmentation == Augmentation.ROTATION:
            angle = random.randrange(-30, 30)
            img = np.array(Image.fromarray(img).rotate(angle))
        # elif augmentation == Augmentation.FLIP_LEFT_RIGHT:
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # elif augmentation == Augmentation.FLIP_TOP_BOTTOM:
        #     img = img.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        if augmentation == Augmentation.BLUR:
            gauss = torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            img = np.array(gauss(Image.fromarray(img)))
        elif augmentation == Augmentation.SALT_AND_PEPPER_NOISE:
            img = _add_noise(img, "s&p")
        return img
    else:
        # perturbation = random.choice(perturbations)
        # img = perturb_image(img, perturbation)
        return img

def _batch_generate_cfs(train_data, amount):
    from dscmchest.generate_counterfactuals import generate_cfs, generate_cfs_random
    cf_data = np.array([], dtype=np.float32).reshape(0, 1, 192, 192)
    cf_metrics = []
    idx = 0
    indices = list(range(len(train_data)))
    while amount > 0:
        if os.path.exists(COUNTERFACTUALS_CHEST_DATA) and os.path.exists(COUNTERFACTUALS_CHEST_METRICS):
            cf_data = np.load(COUNTERFACTUALS_CHEST_DATA)
            cf_metrics = pd.read_csv(COUNTERFACTUALS_CHEST_METRICS, index_col=None).to_dict('records')
        
        a = min(amount, 1000)

        s_indices = indices[idx:]
        sampler = SubsetRandomSampler(s_indices)
        loader = DataLoader(train_data, batch_size=1, sampler=sampler)
        cf_data_new, cf_metrics_new, last_idx = generate_cfs_random(loader, amount=a)
        # cf_data_new, cf_metrics_new, last_idx = generate_cfs(loader, amount=a, do_a=do_a, do_f=do_f)
        if len(cf_data_new) != 0:
            cf_data = np.concatenate((cf_data, cf_data_new), axis=0)
            cf_metrics += cf_metrics_new

        # Save cf files
        np.save(COUNTERFACTUALS_CHEST_DATA, cf_data)
        keys = cf_metrics[0].keys()
        mode = 'w' if os.path.exists(COUNTERFACTUALS_CHEST_METRICS) else 'x'
        with open(COUNTERFACTUALS_CHEST_METRICS, mode, newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(cf_metrics)

        amount -= a
        idx += last_idx+1

    return cf_data, cf_metrics

def debias_chestxray(train_data, method=DebiasingMethod.OVERSAMPLING):
    samples = {
            'age': [],
            'sex': [],
            'finding': [],
            'x': [],
            'race': [],
            'augmented': [],
        }
    
    if method == DebiasingMethod.COUNTERFACTUALS:
        if not os.path.exists(COUNTERFACTUALS_CHEST_DATA) or not os.path.exists(COUNTERFACTUALS_CHEST_METRICS):
            cf_data, cf_metrics = _batch_generate_cfs(train_data, 20000)
        else:
            cf_data = np.load(COUNTERFACTUALS_CHEST_DATA)
            cf_metrics = pd.read_csv(COUNTERFACTUALS_CHEST_METRICS, index_col=None).to_dict('records') 
        count = 0
        for idx, img in enumerate(cf_data):
            metrics = cf_metrics[idx]
            samples['age'].append(metrics['age'])
            samples['sex'].append(metrics['sex'])
            samples['finding'].append(metrics['finding'])
            samples['x'].append(img)
            samples['race'].append(metrics['race'])
            samples['augmented'].append(1)
            
            count += 1
            
        return samples
    
    count = 0
    for idx in range(len(train_data)):
        # TODO: change the condition based on what to impact
        img, ms, lab = train_data[idx]
        a = preprocess_age(ms['age'].item())
        # if ms['race'].item() == 1:
        for _ in range(2):
            # TODO: make sure these are copied not referenced
            samples['age'].append(ms['age'])
            samples['sex'].append(ms['sex'])
            samples['race'].append(ms['race'])
            samples['finding'].append(lab)
            new_x = torch.tensor(apply_debiasing_method(method, img.squeeze().numpy())).unsqueeze(0)
            samples['x'].append(new_x)
            samples['augmented'].append(1)
            count += 1
        # elif ms['race'].item() == 2:
        #     for _ in range(3):
        #         # TODO: make sure these are copied not referenced
        #         samples['age'].append(ms['age'])
        #         samples['sex'].append(ms['sex'])
        #         samples['race'].append(ms['race'])
        #         samples['finding'].append(lab)
        #         new_x = torch.tensor(apply_debiasing_method(method, img.squeeze().numpy())).unsqueeze(0)
        #         samples['x'].append(new_x)
        #         samples['augmented'].append(1)
    return samples

def debias_perturbed_mnist(train_data, train_metrics, method=DebiasingMethod.OVERSAMPLING):
    if (method == DebiasingMethod.PERTURBATIONS and os.path.exists("data/mnist_debiased_perturbed.pt") and
        os.path.exists("data/mnist_debiased_perturbed_metrics.csv")):
        return torch.load("data/mnist_debiased_perturbed.pt"), pd.read_csv("data/mnist_debiased_perturbed_metrics.csv", index_col='index')
    
    if method == DebiasingMethod.COUNTERFACTUALS:
        if not os.path.exists(COUNTERFACTUALS_DATA) or not os.path.exists(COUNTERFACTUALS_METRICS):
            sys.exit("Error: file with counterfactuals does not exist!")

        cfs = pd.read_csv(COUNTERFACTUALS_METRICS, index_col=None).to_dict('records')
        return train_data + torch.load(COUNTERFACTUALS_DATA), train_metrics + cfs

    new_data = []
    new_metrics = []
    for idx, (img, label) in enumerate(train_data):
        metrics = train_metrics[idx]
        # if (not metrics['bias_aligned']) and (label in THICK_CLASSES or label in THIN_CLASSES):
        if label in THICK_CLASSES or label in THIN_CLASSES:
            for _ in range(10):
                new_data.append((apply_debiasing_method(method, img), label))
                new_m = metrics.copy()
                new_m['bias_aligned'] = False
                new_metrics.append(new_m)

    if method == DebiasingMethod.PERTURBATIONS:
        torch.save(train_data + new_data, "data/mnist_debiased_perturbed.pt")

    return train_data + new_data, train_metrics + new_metrics

def _generate_colored_cfs(train_data, train_metrics):
    from dscm.generate_colored_counterfactuals import generate_colored_counterfactual
    # cf_data = np.array([], dtype=np.float32).reshape(0, 3, 28, 28)
    cf_data = []
    cf_metrics = []

    for idx, (img, label) in enumerate(tqdm(train_data)):
        metrics = train_metrics[idx]
        if metrics['bias_aligned'] == True:
            for _ in range(1):
                img_p = torchvision.transforms.ToTensor()(img)
                img_p = img_p.float() * 255.0
                img_p = torchvision.transforms.Pad(padding=2)(img_p).type(torch.ByteTensor)
                obs = {
                    'x': img_p,
                    'color': F.one_hot(torch.tensor(metrics['color']).long(), num_classes=10),
                    'digit': F.one_hot(torch.tensor(label).long(), num_classes=10)}
        
                colors = list(range(10))
                colors.remove(metrics['color'])
                # img_cf, metrics_cf, label_cf = generate_colored_counterfactual(obs=obs, do_c=random.choice(colors))
                img_cf, metrics_cf, label_cf = generate_colored_counterfactual(obs=obs, do_c=random.choice(colors))
                cf_data.append((img_cf, label_cf))
                cf_metrics.append(metrics_cf)

    return cf_data, cf_metrics

def debias_colored_mnist(train_data, train_metrics, method=DebiasingMethod.OVERSAMPLING):
    if method == DebiasingMethod.COUNTERFACTUALS:
        cfs, cf_metrics = _generate_colored_cfs(train_data, train_metrics)
        return train_data + cfs, train_metrics + cf_metrics

    new_data = []
    new_metrics = []
    for idx, (img, label) in enumerate(train_data):
        metrics = train_metrics[idx]
        if metrics['bias_aligned'] == True:
            for _ in range(1):
                new_data.append((apply_debiasing_method(method, img), label))
                new_m = metrics.copy()
                new_m['bias_aligned'] = False
                new_metrics.append(new_m)

    return train_data + new_data, train_metrics + new_metrics

# taken from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).to(device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def preprocess_age(age):
    if 0 <= age <= 19:
        return 0
    elif 20 <= age <= 39:
        return 1
    elif 40 <= age <= 59:
       return 2
    elif 60 <= age <= 79:
        return 3
    else:
        return 4
    
def preprocess_thickness(thickness):
    if thickness <= 1.5:
        return 0
    else:
        return 1

def save_to_csv(file_name, col_names, rows):
    with open(file_name, 'w') as f:   
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(col_names)
        write.writerows(rows)