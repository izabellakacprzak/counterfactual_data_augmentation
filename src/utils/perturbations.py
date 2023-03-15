import numpy as np
import random
import os
import torch
import csv

import tqdm
from params import *

from morphomnist import morpho, perturb

perturbations = [
    perturb.Thinning(amount=.7),
    perturb.Thickening(amount=1.),
    # perturb.Swelling(strength=3, radius=7),
    perturb.Fracture(num_frac=3),
    # perturb.Thinning(amount=.4),
    # perturb.Thickening(amount=1.5),
    # perturb.Swelling(strength=2, radius=5),
    perturb.Fracture(num_frac=2)
]

def perturb_image(image, perturbation):
    morphology = morpho.ImageMorphology(image, scale=4)
    perturbed_hires_image = perturbation(morphology)
    perturbed_image = morphology.downscale(perturbed_hires_image)
    return perturbed_image

def add_perturbations(images, targets, digits, perturbations=perturbations):
  perturbed_images = []
  perturbed_targets = []

  for i in tqdm.trange(len(images)):
    if targets[i] in digits:
        for perturbation in perturbations:
            perturbed_image = perturb_image(images[i], perturbation)
            perturbed_images.append(perturbed_image)
            perturbed_targets.append(targets[i])

  return np.concatenate((images, perturbed_images)), np.concatenate((targets, perturbed_targets))

# bias: [0, 7, 8] mostly thin, [1, 3, 6, 9] mostly thick, [2, 4, 5] equally spread
def prepare_perturbed_mnist(train_data, test_data, bias_conflicting_percentage=0.05):
    train_file_name = "data/train_perturbed"+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt"
    test_file_name = "data/test_perturbed.pt"
    metrics_train_file_name = "data/train_perturbed_mnist_metrics.csv"
    metrics_test_file_name = "data/est_perturbed_mnist_metrics.csv"

    if os.path.exists(train_file_name) and os.path.exists(test_file_name):
        print('Perturbed MNIST dataset already exists')
        return

    print('Preparing Perturbed MNIST')

    train_set = []
    test_set = []
    train_metrics = []
    test_metrics = []

    l = len(train_data)
    if bias_conflicting_percentage == 0:
        perc = l
    else:
        perc = int(l/(l * bias_conflicting_percentage))

    count_bias = 0
    count_anti = 0
    for idx, (im, label) in enumerate(train_data):
        if idx % 1000 == 0:
            print(f'Converting image {idx}/{len(train_data)}')

        if idx % perc == 0: # bias-conflicting samples
            count_anti += 1
            if random.choice([0,1]) == 0:
                perturbed_image = perturb_image(im, perturb.Thickening(amount=1.5))
                train_set.append((perturbed_image, label))
                train_metrics.append([idx, 1.5])
            else:
                perturbed_image = perturb_image(im, perturb.Thinning(amount=0.6))
                train_set.append((perturbed_image, label))
                train_metrics.append([idx, 0.6])

        else: # bias-aligned samples
            count_bias += 1
            if label in THIN_CLASSES:
                perturbed_image = perturb_image(im, perturb.Thickening(amount=1.5))
                train_set.append((perturbed_image, label))
                train_metrics.append([idx, 1.5])
            elif label in THICK_CLASSES:
                perturbed_image = perturb_image(im, perturb.Thinning(amount=0.6))
                train_set.append((perturbed_image, label))
                train_metrics.append([idx, 0.6])
            else:
                if random.choice([0,1]) == 0:
                    perturbed_image = perturb_image(im, perturb.Thickening(amount=1.5))
                    train_set.append((perturbed_image, label))
                    train_metrics.append([idx, 1.5])
                else:
                    perturbed_image = perturb_image(im, perturb.Thinning(amount=0.6))
                    train_set.append((perturbed_image, label))
                    train_metrics.append([idx, 0.6])

    for idx, (im, label) in enumerate(test_data):
        if idx % 1000 == 0:
            print(f'Converting image {idx}/{len(test_data)}')
        if random.choice([0,1]) == 0:
            perturbed_image = perturb_image(im, perturb.Thickening(amount=1.5))
            test_set.append((perturbed_image, label))
            test_metrics.append([idx, 1.5])
        else:
            perturbed_image = perturb_image(im, perturb.Thinning(amount=0.6))
            test_set.append((perturbed_image, label))
            test_metrics.append([idx, 0.6])

    torch.save(train_set, train_file_name)
    torch.save(test_set, test_file_name)

    col_names = ['index', 'thickness']
    save_to_csv(metrics_train_file_name, col_names, train_metrics)
    save_to_csv(metrics_test_file_name, col_names, test_metrics)


def save_to_csv(file_name, col_names, rows):
    with open(file_name, 'w') as f:   
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(col_names)
        write.writerows(rows)
