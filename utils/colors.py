import numpy as np
import random
import os
from PIL import Image
import torch

from utils.params import *
from utils.utils import save_to_csv

def color_grayscale_arr(arr, col):
  """Converts grayscale image to either a coloured one"""
  assert arr.ndim == 2
  h, w = arr.shape
  arr = np.reshape(arr, [h, w, 1])
  arr = np.concatenate([(arr*col[0]).astype(int), (arr*col[1]).astype(int), (arr*col[2]).astype(int)], axis=2)
  return arr

colors = [(1.0,0.0,0.0), (0.97,0.47,0.0), (0.97,0.91,0.0), (0.56,0.97,0.0), (0.0,1.0,0.0), (0.0,0.97,0.85), (0.0,0.0,1.0), (0.5,0.0,1.0), (1.0,0.0,0.89), (1.0,0.0,0.44)]
# colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0.6, 0, 1), (0, 0.5, 0.7), (0.5, 0.7, 1), (0.9, 0, 0.5)]
def prepare_colored_mnist(train_data, test_data, bias_conflicting_percentage=0.05):
    train_file_name = TRAIN_COLORED_DATA+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt"
    train_metrics_name = TRAIN_COLORED_METRICS+"_"+str(bias_conflicting_percentage).replace(".", "_")+".csv"
    if os.path.exists(train_file_name) and os.path.exists(TEST_COLORED_DATA):
        print('Colored MNIST dataset already exists')
        return

    print('Preparing Colored MNIST')

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
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(train_data)}')
        im_array = np.array(im)

        if idx % perc == 0: # bias-conflicting samples
            count_anti += 1
            color_idx = random.choice(list(range(len(colors))))
            color = colors[color_idx]
            colored_arr = color_grayscale_arr(im_array, color)
            train_set.append((np.uint8(colored_arr), label))
            train_metrics.append([idx, color_idx, False])
        else: # bias-aligned samples
            count_bias += 1
            colored_arr = color_grayscale_arr(im_array, colors[label])
            train_set.append((np.uint8(colored_arr), label))
            train_metrics.append([idx, label, True])

    print("bias aligned: {}".format(count_bias))
    print("bias conflicting: {}".format(count_anti))
    
    for idx, (im, label) in enumerate(test_data):
        if idx % 10000 == 0:
            print(f'Converting image {idx}/{len(test_data)}')
        im_array = np.array(im)

        color_idx = random.choice(list(range(len(colors))))
        color = colors[color_idx]
        colored_arr = color_grayscale_arr(im_array, color)
        test_set.append((np.uint8(colored_arr), label))
        test_metrics.append([idx, color_idx, False])

    torch.save(train_set, train_file_name)
    torch.save(test_set, TEST_COLORED_DATA)
    col_names = ['index', 'color', 'bias_aligned']
    save_to_csv(train_metrics_name, col_names, train_metrics)
    save_to_csv(TEST_COLORED_METRICS, col_names, train_metrics)