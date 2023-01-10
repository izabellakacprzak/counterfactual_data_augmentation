import numpy as np
import random
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt

import tqdm
from params import *

from morphomnist import morpho, perturb
from enum import Enum
from MNISTClassifier import ConvNet
 
class AugmentationMethod(Enum):
    NONE = 1
    OVERSAMPLING = 2
    REWEIGHING = 3
    AUGMENTATIONS = 4
    PERTURBATIONS = 5

class Augmentation(Enum):
    ROTATION = 1
    FLIP_LEFT_RIGHT = 2
    FLIP_TOP_BOTTOM = 3

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

def train_and_evaluate(train_loader, test_loader, in_channels, out_channels, pred_arr, true_arr, accs):
    model = ConvNet(train_loader=train_loader, test_loader=test_loader, in_channels=in_channels, out_channels=out_channels)
    for epoch in range(1, EPOCHS):
        _, _, acc = model.test_MNIST()
        accs.append(acc)
        model.train_MNIST(epoch)

    y_pred, y_true, acc = model.test_MNIST()
    accs.append(acc)
    pred_arr.append(y_pred)
    true_arr.append(y_true)

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

def color_grayscale_arr(arr, col):
  """Converts grayscale image to either a coloured one"""
  assert arr.ndim == 2
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  arr = np.concatenate([(arr*col[0]).astype(int), (arr*col[1]).astype(int), (arr*col[2]).astype(int)], axis=2)
  return arr

colores = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0.5, 0.7), (0.5, 0.7, 1), (1, 0.6, 0)]
def prepare_colored_mnist(train_data, test_data, bias_conflicting_percentage=0.05):
    train_file_name = TRAIN_FILE+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt"
    test_file_name = TEST_FILE
    if os.path.exists(train_file_name) and os.path.exists(test_file_name):
        print('Colored MNIST dataset already exists')
        return

    print('Preparing Colored MNIST')

    train_set = []
    test_set = []
    l = len(train_data)
    if bias_conflicting_percentage == 0:
        perc = l
    else:
        perc = int(l/(l * bias_conflicting_percentage))

    count_bias = 0
    count_anti = 0
    for idx, (im, label) in enumerate(train_data):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(train_data)}')
        im_array = np.array(im)

        if idx % perc == 0: # bias-conflicting samples
            count_anti += 1
            color = random.choice(colores)
            colored_arr = color_grayscale_arr(im_array, color)
            train_set.append((Image.fromarray(np.uint8(colored_arr)), color, label))
        else: # bias-aligned samples
            count_bias += 1
            colored_arr = color_grayscale_arr(im_array, colores[label])
            train_set.append((Image.fromarray(np.uint8(colored_arr)), colores[label], label))

    for idx, (im, label) in enumerate(test_data):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(test_data)}')
        im_array = np.array(im)

        color = random.choice(colores)
        colored_arr = color_grayscale_arr(im_array, color)
        test_set.append((Image.fromarray(np.uint8(colored_arr)), color, label))

    torch.save(train_set, train_file_name)
    torch.save(test_set, test_file_name)

def get_attr_label_multipliers(tuples):
    counts = {}
    for (_, attr, label) in tuples:
        counts[(attr, label)] = counts[(attr, label)] + 1 if ((attr, label) in counts) else 1

    equal_div = len(tuples) / len(counts)
    oversample_multipliers = {}
    for key in counts.keys():
        oversample_multipliers[key] = int(equal_div / counts[key])

    return oversample_multipliers

def debias_mnist(train_data, method=AugmentationMethod.OVERSAMPLING):
    if method == AugmentationMethod.PERTURBATIONS and os.path.exists("mnist_debiased_perturbed.pt"):
        return torch.load("mnist_debiased_perturbed.pt")
    oversample_multipliers = get_attr_label_multipliers(train_data)

    new_tuples = []
    l = str(len(train_data))
    for idx, (img, attr, label) in enumerate(train_data):
        if idx % 100 == 0:
            print("Perturbing image " + str(idx) + "/" + l)
        for _ in range(oversample_multipliers[(attr, label)]):
            if method == AugmentationMethod.OVERSAMPLING:
                new_tuples.append((img, attr, label))
            elif method == AugmentationMethod.AUGMENTATIONS:
                augmentation = random.choice(list(Augmentation))
                if augmentation == Augmentation.ROTATION:
                    img = img.rotate(90)
                elif augmentation == Augmentation.FLIP_LEFT_RIGHT:
                    img = img.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
                elif augmentation == Augmentation.FLIP_TOP_BOTTOM:
                    img = img.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
                new_tuples.append((img, attr, label))
            elif method == AugmentationMethod.PERTURBATIONS:
                perturbation = random.choice(perturbations)
                img = perturb_image(img, perturbation)
                new_tuples.append((img, attr, label))

            else:
                print("Incorrect debiasing method given: " + str(method))

    if method == AugmentationMethod.PERTURBATIONS:
        torch.save(new_tuples, "mnist_debiased_perturbed.pt")

    return new_tuples

def plot_dataset_digits(dataset):
  fig = plt.figure(figsize=(13, 8))
  columns = 6
  rows = 3
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    img, label = dataset[i]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("Label: " + str(label))  # set title
    plt.imshow(img)

  plt.show()  # finally, render the plot

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
      row,col,_= image.shape
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