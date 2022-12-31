import numpy as np
import random
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt

from params import *

def color_grayscale_arr(arr, col):
  """Converts grayscale image to either a coloured one"""
  assert arr.ndim == 2
  dtype = arr.dtype
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  arr = np.concatenate([(arr*col[0]).astype(int), (arr*col[1]).astype(int), (arr*col[2]).astype(int)], axis=2)
  return arr

colores = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0.5, 0.7), (0.5, 0.7, 1), (1, 0.6, 0)]
def prepare_colored_mnist(train_data, test_data):
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        print('Colored MNIST dataset already exists')
        return

    print('Preparing Colored MNIST')

    train_set = []
    test_set = []
    l = len(train_data)
    # perc = int(l/(l * 0.05))
    perc = 1
    count_bias = 0
    count_anti = 0
    for idx, (im, label) in enumerate(train_data):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(train_data)}')
        im_array = np.array(im)

        if idx % perc == 0: # bias-conflicting samples
            count_anti += 1
            colored_arr = color_grayscale_arr(im_array, random.choice(colores))
            train_set.append((Image.fromarray(np.uint8(colored_arr)), label))
        else: # bias-aligned samples
            count_bias += 1
            colored_arr = color_grayscale_arr(im_array, colores[label])
            train_set.append((Image.fromarray(np.uint8(colored_arr)), label))

    for idx, (im, label) in enumerate(test_data):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(test_data)}')
        im_array = np.array(im)

        colored_arr = color_grayscale_arr(im_array, random.choice(colores))
        test_set.append((Image.fromarray(np.uint8(colored_arr)), label))

    torch.save(train_set, TRAIN_FILE)
    torch.save(test_set, TEST_FILE)

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
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_type == "s&p":
      row,col,_ = image.shape
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

def prepare_med_noisy_mnist(train_data, test_data):
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        print('Med Noisy MNIST dataset already exists')
        return

    print('Preparing Med Noisy MNIST')

    train_set = []
    test_set = []
    l = len(train_data)
    # perc = int(l/(l * 0.05))
    perc = 1
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
                train_set.append((Image.fromarray(np.uint8(noisy_arr)), 1))
            else:
                train_set.append((Image.fromarray(np.uint8(im_array)), 0))
        else: # bias-aligned samples
            count_bias += 1
            if label in [0, 1, 2]:
                noisy_arr = add_noise(im_array, "s&p")
                train_set.append((Image.fromarray(np.uint8(noisy_arr)), 1))
            else:
                train_set.append((Image.fromarray(np.uint8(im_array)), 0))

    for idx, (im, label) in enumerate(test_data):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(test_data)}')
        im_array = np.array(im)

        if random.choice([0, 1]) == 0:
            noisy_arr = add_noise(im_array, "s&p")
            test_set.append((Image.fromarray(np.uint8(noisy_arr)), 1))
        else:
            test_set.append((Image.fromarray(np.uint8(im_array)), 0))

    torch.save(train_set, TRAIN_FILE)
    torch.save(test_set, TEST_FILE)