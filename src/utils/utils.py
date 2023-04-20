import time
import numpy as np
import random
import os
import sys
import pandas as pd
from PIL import Image
import torch
import torchvision
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from MNISTClassifier import train_MNIST, test_MNIST
from params import *
from enum import Enum
from .perturbations import perturbations, perturb_image
 
class AugmentationMethod(Enum):
    NONE = 1
    OVERSAMPLING = 2
    REWEIGHING = 3
    AUGMENTATIONS = 4
    PERTURBATIONS = 5
    COUNTERFACTUALS = 6
    CF_REGULARISATION = 7

class Augmentation(Enum):
#     ROTATION = 1
#     FLIP_LEFT_RIGHT = 2
#     FLIP_TOP_BOTTOM = 3
    BLUR = 4
    SALT_AND_PEPPER_NOISE = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def unbalance_dataset(images, targets, undersampled_classes, cut_percentage):
    # build an array for each class where an element is True
    # if the corresponding image at that index is of the class
    idxs = []
    classes = len(set(targets))
    for i in range(classes):
        idxs.append((targets==i))

    # remove elements from classes which are to be undersampled
    cut_len = int(len(idxs[0]) * cut_percentage)
    for i in undersampled_classes:
        idxs[i][:-cut_len] = False
        # arr[arr==True] = False
        # idxs[i] = idxs[i][cut_len] + arr

    new_targets = idxs[0]
    for i in idxs:
        new_targets = new_targets | i

    return [i for (i, v) in zip(images, new_targets) if v], [i for (i, v) in zip(targets, new_targets) if v]

def get_attr_label_multipliers(tuples):
    counts = {}
    for (_, label) in tuples:
        counts[label] = counts[label] + 1 if (label in counts) else 1

    equal_div = len(tuples) / len(counts)
    oversample_multipliers = {}
    for key in counts.keys():
        oversample_multipliers[key] = int(equal_div / counts[key])

    return oversample_multipliers

def apply_debiasing_method(method, img):
    if method == AugmentationMethod.OVERSAMPLING:
        return img
    elif method == AugmentationMethod.AUGMENTATIONS:
        augmentation = random.choice(list(Augmentation))
        # if augmentation == Augmentation.ROTATION:
        #     img = img.rotate(90)
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

def prepare_med_noisy_mnist(train_data, test_data, bias_conflicting_percentage):
    train_file_name = MED_TRAIN_FILE+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt"
    test_file_name = MED_TEST_FILE
    if os.path.exists(train_file_name) and os.path.exists(test_file_name):
        print('Med Noisy MNIST dataset already exists')
        return

    print('Preparing Med Noisy MNIST')

    train_set = []
    test_set = []
    l = len(train_data)
    perc = int(l/(l * bias_conflicting_percentage))
    # perc = l+1
    count_bias = 0
    count_anti = 0
    for idx, (im, label) in enumerate(train_data):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(train_data)}')
        im_array = np.array(im)

        if idx % perc == 0: # bias-conflicting samples
            count_anti += 1
            if random.choice([0, 1]) == 0:
                noisy_arr = add_noise(im_array, "s&p")
                train_set.append((Image.fromarray(np.uint8(noisy_arr)), 1, label))
            else:
                train_set.append((Image.fromarray(np.uint8(im_array)), 0, label))
        else: # bias-aligned samples
            count_bias += 1
            if label in [0, 1, 2, 3]:
                noisy_arr = add_noise(im_array, "s&p")
                train_set.append((Image.fromarray(np.uint8(noisy_arr)), 1, label))
            else:
                train_set.append((Image.fromarray(np.uint8(im_array)), 0, label))

    count_sp = 0
    for idx, (im, label) in enumerate(test_data):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(test_data)}')
        im_array = np.array(im)
        if random.choice([0, 1]) == 0:
            count_sp += 1
            noisy_arr = add_noise(im_array, "s&p")
            test_set.append((Image.fromarray(np.uint8(noisy_arr)), 1, label))
        else:
            test_set.append((Image.fromarray(np.uint8(im_array)), 0, label))

    print("there are this many s&p samples in test set")
    print(count_sp)
    print("test set has length")
    print(len(test_data))
    print("there are this many bias aligned samples in training set")
    print(count_bias)
    print("there are this many bias conflicting samples in training set")
    print(count_anti)
    torch.save(train_set, train_file_name)
    torch.save(test_set, test_file_name)

def train_and_evaluate(model, train_loader, test_loader, pred_arr, true_arr, do_cf_regularisation=False):
    accuracies, f1s = train_MNIST(model, train_loader, test_loader, do_cf_regularisation)
    y_pred, y_true, acc, f1 = test_MNIST(model, test_loader)
    accuracies.append(acc)
    f1s.append(f1)
    pred_arr.append(y_pred)
    true_arr.append(y_true)

    return accuracies, f1s

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
        labels = np.concatenate((labels, target.numpy().ravel()))
        embeddings = np.concatenate([embeddings, output],axis=0)
        thicknesses = np.concatenate((thicknesses, metrics['thickness']))

    return embeddings, thicknesses, labels

def visualise_t_sne(test_loader, model, file_name):
    embeddings, thicknesses, labels = get_embeddings(model, test_loader)
    
    feat_cols = ['pixel'+str(i) for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=feat_cols)
    df['y'] = labels
    df['thickness'] = thicknesses
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

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]

    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
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
