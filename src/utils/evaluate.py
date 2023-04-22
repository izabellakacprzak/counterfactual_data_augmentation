from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
from tqdm import tqdm
import numpy as np
import torchvision.transforms as TF
from scipy import stats

from params import *
from morphomnist import measure
from dscm.generate_counterfactuals import generate_counterfactual_for_x

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
    x, y = np.array(list(range(EPOCHS))), metric_arr
    res = stats.linregress(x, y)
    plt.figure(figsize=(10,10))
    plt.plot(x, y, 'o', label='original data')
    # plt.plot(x, res.intercept + res.slope*x, 'r', label='fitted line')
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.savefig("plots/"+run_name+metric_name+".png")

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
def classifier_fairness_analysis(model, test_loader):
    X, Y = [], []

    model.model.eval()
    for data, metrics, label in test_loader:
        for _ in range(100):
            original_pred = model.model(data).cpu()
            _, original_pred = torch.max(original_pred, 1)
            X.append(original_pred)

            img = (data.float() - 127.5) / 127.5
            img = TF.Pad(padding=2)(img.type(torch.ByteTensor)).unsqueeze(0)
            cf = generate_counterfactual_for_x(img, metrics['thickness'], metrics['intensity'], label)
            cf = torch.from_numpy(cf).unsqueeze(0).float()
            counterfactual_pred = model.model(cf).cpu()
            _, counterfactual_pred = torch.max(counterfactual_pred, 1)
            Y.append(counterfactual_pred)

    X = np.array(X)
    Y = np.array(Y)
    plt.scatter(X,Y)
    plt.show()

def plot_metrics_comparison(run_names, run_metrics, metric_name):
    r = np.arange(NUM_OF_CLASSES)
    width = 0.2
    
    metrics = {}
    for idx in range(len(run_names)):
        metrics[run_names[idx]] = run_metrics[idx]

    for run, metric in metrics.items():
        plt.bar(r, metric, width = width, label=run)
        r = r + width
    
    plt.xlabel("Label")
    plt.ylabel(metric_name)
    
    # plt.grid(linestyle='--')
    plt.xticks(np.arange(NUM_OF_CLASSES) + width/2, np.arange(NUM_OF_CLASSES))
    plt.yticks(np.arange(0, 1.1, 0.05))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.5))
    
    # plt.show()
    plt.savefig("plots/metrics_comparison_"+metric_name+".png")
    plt.show()