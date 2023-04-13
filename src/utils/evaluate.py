from sklearn import metrics
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import numpy as np

from params import *
from morphomnist import measure

def pretty_print_evaluation(y_pred, y_true, labels):
    confusion_matrix = get_confusion_matrix(y_pred, y_true)
    classification_report = get_classification_report(y_pred, y_true, labels)

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

def get_classification_report(y_pred, y_true, labels):
    return metrics.classification_report(y_true, y_pred, digits=len(labels))

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

def print_classes_size(dataset):
    counts = {}
    for _, _, y in dataset:
        counts[y] = (counts[y] if y in counts else 0) + 1

    for l in sorted(counts):
        print(str(l)+": "+str(counts[l]))
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

    print("Thick digits counts:")
    print(thick_per_class)
    print("Thin digits counts:")
    print(thin_per_class)

# Generates a scatterplot of how similar predictions made by classifier 
# on counterfactual data are to predictions on original data
# all points should be clustered along the y=x line - meaning high classifier fairness
def classifier_fairness_analysis(model, original_data, counterfactual_data):
    X, Y = [], []

    for idx in range(len(original_data)):
        original_pred = model.model(original_data[idx]).cpu()
        _, original_pred = torch.max(original_pred, 1)
        X.append(original_pred)

        counterfactual_pred = model.model(counterfactual_data[idx]).cpu()
        _, counterfactual_pred = torch.max(counterfactual_pred, 1)
        Y.append(counterfactual_pred)

    X = np.array(X)
    Y = np.array(Y)
    plt.scatter(X,Y)
    plt.show()