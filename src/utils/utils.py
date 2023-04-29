import time
import numpy as np
import random
import os
import sys
import csv
import pandas as pd
from PIL import Image
import torch
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from params import *
from enum import Enum
from .perturbations import perturbations, perturb_image
from dscmchest.generate_counterfactuals import generate_cfs
 
class AugmentationMethod(Enum):
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
#     FLIP_LEFT_RIGHT = 2
#     FLIP_TOP_BOTTOM = 3
    BLUR = 4
    SALT_AND_PEPPER_NOISE = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def apply_debiasing_method(method, img):
    if method == AugmentationMethod.OVERSAMPLING:
        return img
    elif method == AugmentationMethod.AUGMENTATIONS:
        augmentation = random.choice(list(Augmentation))
        if augmentation == Augmentation.ROTATION:
            angle = random.randrange(-30, 30)
            img = img.rotate(angle)
        # elif augmentation == Augmentation.FLIP_LEFT_RIGHT:
        #     img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        # elif augmentation == Augmentation.FLIP_TOP_BOTTOM:
        #     img = img.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
        if augmentation == Augmentation.BLUR:
            gauss = torchvision.transforms.GaussianBlur(3, sigma=(0.1, 2.0))
            img = np.array(gauss(Image.fromarray(img)))
        elif augmentation == Augmentation.SALT_AND_PEPPER_NOISE:
            img = add_noise(img, "s&p")
        return img
    else:
        perturbation = random.choice(perturbations)
        img = perturb_image(img, perturbation)
        return img

def debias_chestxray(train_data, method=AugmentationMethod.OVERSAMPLING):
    if method == AugmentationMethod.COUNTERFACTUALS:
        if not os.path.exists(CF_CHEST_DATA) or not os.path.exists(CF_CHEST_METRICS):
            cf_data, cf_metrics = generate_cfs(train_data, do_r='asian')

            # Save cf files
            torch.save(cf_data, CF_CHEST_DATA)
            keys = cf_metrics[0].keys()
            with open(CF_CHEST_METRICS, 'x', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, keys)
                dict_writer.writeheader()
                dict_writer.writerows(cf_metrics)

        else:
            cf_data, cf_metrics = torch.load(CF_CHEST_DATA), pd.read_csv(CF_CHEST_METRICS, index_col='index')
        
        samples = []
        for idx, img in enumerate(cf_data):
            metrics = cf_metrics[idx]
            new_sample = {'x':img, 'label':metrics['label'], 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race']}
            samples.append(new_sample)

        return samples

    # TODO: IMPLEMENT!!!
    new_data = []
    for _, (img, metrics, label) in enumerate(train_data):
        # TODO: add condition when biased sample - do analysis of data
        if True:
            for _ in range(10):
                new_datapoint = metrics.copy()
                new_datapoint['x'] = apply_debiasing_method(method, img)
                new_datapoint['label'] = label

    return new_data

def debias_mnist(train_data, train_metrics, method=AugmentationMethod.OVERSAMPLING):
    if (method == AugmentationMethod.PERTURBATIONS and os.path.exists("data/mnist_debiased_perturbed.pt") and
        os.path.exists("data/mnist_debiased_perturbed_metrics.csv")):
        return torch.load("data/mnist_debiased_perturbed.pt"), pd.read_csv("data/mnist_debiased_perturbed_metrics.csv", index_col='index')
    
    if method == AugmentationMethod.COUNTERFACTUALS:
        if not os.path.exists(COUNTERFACTUALS_DATA) or not os.path.exists(COUNTERFACTUALS_METRICS):
            sys.exit("Error: file with counterfactuals does not exist!")

        cfs = pd.read_csv(COUNTERFACTUALS_METRICS, index_col=None).to_dict('records')
        return train_data + torch.load(COUNTERFACTUALS_DATA), train_metrics + cfs

    new_data = []
    new_metrics = []
    for idx, (img, label) in enumerate(train_data):
        metrics = train_metrics[idx]
        if metrics['bias_aligned'] and (label in THICK_CLASSES or label in THIN_CLASSES):
            for _ in range(10):
                new_data.append((apply_debiasing_method(method, img), label))
                new_m = metrics.copy()
                new_m['bias_aligned'] = False
                new_metrics.append(new_m)

    if method == AugmentationMethod.PERTURBATIONS:
        torch.save(train_data + new_data, "data/mnist_debiased_perturbed.pt")

    return train_data + new_data, train_metrics + new_metrics

def add_noise(image, noise_type="gauss"):
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
      row,col= image.shape
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

def get_embeddings(model, data_loader):
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    embeddings = np.zeros(shape=(0, 784))
    labels = np.zeros(shape=(0))
    thicknesses = np.zeros(shape=(0))
    for _, (data, metrics, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = feature_extractor(data).detach().cpu().squeeze(1).numpy()
        s = output.shape
        output = output.reshape((s[0], 784))
        labels = np.concatenate((labels, target.cpu().numpy().ravel()))
        embeddings = np.concatenate([embeddings, output],axis=0)
        thicknesses = np.concatenate((thicknesses, metrics['thickness']))

    return embeddings, thicknesses, labels

def visualise_t_sne(test_loader, model, file_name):
    embeddings, thicknesses, labels = get_embeddings(model, test_loader)
    
    feat_cols = ['pixel'+str(i) for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=feat_cols)
    df['y'] = labels
    df['thickness'] = thicknesses.astype(int)
    df['label'] = df['y'].apply(lambda i: str(i))

    N = 100000
    rndperm = np.random.permutation(df.shape[0])
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values

    # reduce dimensions before feeding into t-SNE
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    print('[t-SNE]\tt-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]

    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("tab10", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(file_name + "_labels.png") 

    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="thickness",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(file_name + "_thickness.png") 

# taken from https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
