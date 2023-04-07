import numpy as np
import random
import os
import torch
import csv
import matplotlib.pyplot as plt

import tqdm
from params import *

from morphomnist import morpho, perturb, measure

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
    metrics_test_file_name = "data/test_perturbed_mnist_metrics.csv"

    if os.path.exists(train_file_name) and os.path.exists(test_file_name):
        print('Perturbed MNIST dataset already exists')
        return

    print('Preparing Perturbed MNIST')

    train_set = []
    test_set = []
    train_metrics = []
    test_metrics = []
    class_counts = {}

    l = len(train_data)
    if bias_conflicting_percentage == 0:
        perc = l
    else:
        perc = int(l/(l * bias_conflicting_percentage))

    intensity = 1
    def thicken(idx, bias_aligned, im, label, im_set, metrics):
        _, _, thickness, _, _, _ = measure.measure_image(im, verbose=False)
        if thickness >= 2.5: # image is already thick
            perturbed_image = np.array(im)
        else:
            new_thickness = random.uniform(2.5, 4.5)
            amount = (new_thickness - thickness)/thickness
            perturbed_image = perturb_image(im, perturb.Thickening(amount=amount))
            thickness = new_thickness
        im_set.append((bias_aligned, perturbed_image, label))
        metrics.append([idx, thickness, intensity])

    def thin(idx, bias_aligned, im, label, im_set, metrics):
        _, _, thickness, _, _, _ = measure.measure_image(im, verbose=False)
        if thickness <= 1.5: # image is already thin
            perturbed_image = np.array(im)
        else:
            new_thickness = random.uniform(1.0, 1.6)
            amount = (thickness-new_thickness)/thickness
            perturbed_image = perturb_image(im, perturb.Thinning(amount=amount))
            thickness = new_thickness
        im_set.append((bias_aligned, perturbed_image, label))
        metrics.append([idx, thickness, intensity])

    count_bias = 0
    count_anti = 0
    for idx, (im, label) in enumerate(train_data):
        if idx % 1000 == 0:
            print(f'Converting image {idx}/{len(train_data)}')

        # Imbalanding the dataset further by cutting the number of samples of biased classes
        class_counts[label] = (class_counts[label] if label in class_counts else 0) + 1
        if bias_conflicting_percentage != 1.0 and (label in THICK_CLASSES or label in THIN_CLASSES) and class_counts[label] >= 3000:
            continue

        if idx % perc == 0: # bias-conflicting samples
            count_anti += 1
            if random.choice([0,1]) == 0:
                thicken(idx, False, im, label, train_set, train_metrics)
            else:
                thin(idx, False, im, label, train_set, train_metrics)

        else: # bias-aligned samples
            count_bias += 1
            if label in THICK_CLASSES:
                thicken(idx, True, im, label, train_set, train_metrics)
            elif label in THIN_CLASSES:
                thin(idx, True, im, label, train_set, train_metrics)
            else:
                if random.choice([0,1]) == 0:
                    thicken(idx, True, im, label, train_set, train_metrics)
                else:
                    thin(idx, True, im, label, train_set, train_metrics)

    for idx, (im, label) in enumerate(test_data):
        if idx % 1000 == 0:
            print(f'Converting image {idx}/{len(test_data)}')
        if random.choice([0,1]) == 0:
            thicken(idx, False, im, label, test_set, test_metrics)
        else:
            thin(idx, False, im, label, test_set, test_metrics)

    torch.save(train_set, train_file_name)
    torch.save(test_set, test_file_name)

    col_names = ['index', 'thickness', 'intensity']
    save_to_csv(metrics_train_file_name, col_names, train_metrics)
    save_to_csv(metrics_test_file_name, col_names, test_metrics)


def save_to_csv(file_name, col_names, rows):
    with open(file_name, 'w') as f:   
        # using csv.writer method from CSV package
        write = csv.writer(f)
        
        write.writerow(col_names)
        write.writerows(rows)
