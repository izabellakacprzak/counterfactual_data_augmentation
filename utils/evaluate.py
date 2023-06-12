from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import stats

from utils.params import *
from morphomnist import measure

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

def pretty_print_evaluation(y_pred, y_true, labels):
    confusion_matrix = get_confusion_matrix(y_pred, y_true)
    classification_report = metrics.classification_report(y_true, y_pred, digits=len(labels))

    print("-------------------------------------")
    print("----------CONFUSION MATRIX-----------")
    print("-------------------------------------")
    print(confusion_matrix)
    print("\n")

    print("-------------------------------------")
    print("-------CLASSIFICATION METRICS--------")
    print("-------------------------------------")
    print(classification_report)
    print("\n")



def accuracy(confusion_matrix):
    return confusion_matrix.diagonal()/confusion_matrix.sum(axis=1)

def get_confusion_matrix(y_pred, y_true):
    return metrics.confusion_matrix(y_true, y_pred)

def plot_dataset_digits(dataset):
  fig = plt.figure(figsize=(13, 8))
  columns = 10
  rows = 5
  # ax enables access to manipulate each of subplots
  ax = []

  for i in range(columns * rows):
    idx = random.randint(0, len(dataset)-1)
    img, label = dataset[idx]
    if img.shape[0] == 1:
        img = img[0, :, :]
    # create subplot and append to ax
    ax.append(fig.add_subplot(rows, columns, i + 1))
    ax[-1].set_title("Label: " + str(label))  # set title
    plt.imshow(img)

  plt.show()  # finally, render the plot

def save_plot_for_metric(metric_name, metric_arr, run_name):
    x, y = np.array(list(range(len(metric_arr)))), metric_arr
    res = stats.linregress(x, y)
    plt.figure(figsize=(10,10))
    plt.plot(x, y, 'o', label='original data')
    # plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig("plots/{}{}.png".format(run_name, metric_name))

def print_classes_size(dataset):
    counts = {}
    for _, _, y in dataset:
        counts[y] = (counts[y] if y in counts else 0) + 1

    for l in sorted(counts):
        print("[Class size]\t{}: {}".format(l, counts[l]))
        print()

def count_thick_thin_per_class(dataset):
    thick_per_class = {}
    thin_per_class = {}
    for _, img, label in tqdm(dataset):
        _, _, thickness, _, _, _ = measure.measure_image(img, verbose=False)
        if thickness >= 2.5:
            thick_per_class[label] = (thick_per_class[label] if label in thick_per_class else 0) + 1
        else:
            thin_per_class[label] = (thin_per_class[label] if label in thin_per_class else 0) + 1

    print("[Thick/thin counts]\tThick digits counts:")
    print("[Thick/thin counts]\t{}".format(thick_per_class))
    print("[Thick/thin counts]\tThin digits counts:")
    print("[Thick/thin counts]\t{}".format(thin_per_class))

def get_attribute_counts_chestxray(dataset):
    positive_counts = {'male':0, 'female':0, 'Black':0, 'White':0, 'Asian':0, '18-25':0, '26-40':0, '41-65':0, '66-100':0}
    negative_counts = {'male':0, 'female':0, 'Black':0, 'White':0, 'Asian':0, '18-25':0, '26-40':0, '41-65':0, '66-100':0}
    for _, metrics, label in tqdm(dataset):
        sex = 'male' if metrics['sex'] == 0 else 'female'
        race = 'White' if metrics['race'] == 0 else ('Asian' if metrics['race'] == 1 else 'Black')
        age = metrics['age']
        if 18 <= age <= 25:
            age = '18-25'
        elif 26 <= age <= 40:
            age = '26-40'
        elif 41 <= age <= 65:
            age = '41-65'
        else:
            age = '66-100'

        if label == 0:
            negative_counts[sex] = negative_counts[sex] + 1
            negative_counts[race] = negative_counts[race] + 1
            negative_counts[age] = negative_counts[age] + 1
        else:
            positive_counts[sex] = positive_counts[sex] + 1
            positive_counts[race] = positive_counts[race] + 1
            positive_counts[age] = positive_counts[age] + 1

    print("[ChestXRay attribute counts]\tDisease positive counts:")
    print("[ChestXRay attribute counts]\t{}".format(positive_counts))
    print("[ChestXRay attribute counts]\tDisease negative counts:")
    print("[ChestXRay attribute counts]\t{}".format(negative_counts))

