from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
import time
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

from utils.params import *
from morphomnist import measure

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        print("[Class size]\t" + str(l)+": "+str(counts[l]))
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
    print("[Thick/thin counts]\t"+thick_per_class)
    print("[Thick/thin counts]\tThin digits counts:")
    print("[Thick/thin counts]\t"+thin_per_class)

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
    print("[ChestXRay attribute counts]\t"+positive_counts)
    print("[ChestXRay attribute counts]\tDisease negative counts:")
    print("[ChestXRay attribute counts]\t"+negative_counts)

def _get_embeddings(model, data_loader, img_dim):
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    embeddings = np.zeros(shape=(0, img_dim*img_dim))
    labels = np.zeros(shape=(0))
    race = np.zeros(shape=(0))
    sex = np.zeros(shape=(0))
    age = np.zeros(shape=(0))
    for _, (data, metrics, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = feature_extractor(data).detach().cpu().squeeze(1).numpy()
        s = output.shape
        output = output.reshape((s[0], img_dim*img_dim))
        labels = np.concatenate((labels, target.cpu().numpy().ravel()))
        embeddings = np.concatenate([embeddings, output],axis=0)
        race = np.concatenate((race, metrics['race'].tolist()))
        sex = np.concatenate((sex, metrics['sex'].tolist()))
        age = np.concatenate((age, metrics['age'].tolist()))

    return embeddings, race, sex, age, labels

def visualise_t_sne(test_loader, model, img_dim, file_name):
    embeddings, race, sex, age, labels = _get_embeddings(model, test_loader, img_dim)
    
    feat_cols = ['pixel'+str(i) for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=feat_cols)
    df['y'] = labels
    df['race'] = race.astype(int)
    df['sex'] = sex.astype(int)
    df['label'] = df['y'].apply(lambda i: str(i))

    age = age.astype(int)
    for idx, a in enumerate(age):
        if 18<=a<=25:
            age[idx] = 0
        elif 26<=a<=40:
            age[idx] = 1
        elif 41<=a<=65:
            age[idx] = 2
        elif 66<=a<=80:
            age[idx] = 3
        else:
            age[idx] = 4
    df['age'] = age

    N = 100000
    rndperm = np.random.permutation(df.shape[0])
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values

    # reduce dimensions before feeding into t-SNE
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=5000)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    print('[t-SNE]\tt-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]

    plt.figure(figsize=(16,10))
    plot = sns.jointplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="y",
        palette=sns.color_palette("tab10", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(file_name + "_labels.png") 

    plt.figure(figsize=(16,10))
    plot = sns.jointplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="race",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(file_name + "_race.png") 

    plt.figure(figsize=(16,10))
    plot = sns.jointplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="sex",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(file_name + "_sex.png") 

    plt.figure(figsize=(16,10))
    plot = sns.jointplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="age",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(file_name + "_age.png") 
