import numpy as np
import random
import os
from PIL import Image
import torch

from params import *

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