from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
from tqdm import tqdm
import numpy as np
import torchvision.transforms as TF
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

def _get_cf_for_mnist(img, thickness, intensity, label):
    from dscm.generate_counterfactuals import generate_counterfactual_for_x
    img = img.float() * 254
    img = TF.Pad(padding=2)(img).type(torch.ByteTensor).unsqueeze(0)
    x_cf = generate_counterfactual_for_x(img, thickness, intensity, label)
    return torch.from_numpy(x_cf).unsqueeze(0).float()

def _get_cf_for_chestxray(img, metrics, label, do_s, do_r, do_a):
    from chest_xray.generate_counterfactuals import generate_cf
    obs = {'x': img,
           'age': metrics['age'],
           'race': metrics['race'],
           'sex': metrics['sex'],
           'finding': label}
    cf = generate_cf(obs, amount=1, do_s=do_s, do_r=do_r, do_a=do_a)
    return cf

# Generates a scatterplot of how similar predictions made by classifier 
# on counterfactual data are to predictions on original data
# all points should be clustered along the y=x line - meaning high classifier fairness
def classifier_fairness_analysis(model, test_loader, run_name):
    X, Y = [], []

    model.model.eval()
    fairness_against_digit = 6
    for _, (data, metrics, labels) in enumerate(tqdm(test_loader)):
        data = data.to(device)
        labels = labels.to(device)
        logits = model.model(data).cpu()
        probs = torch.nn.functional.softmax(logits, dim=1).tolist()
        original_probs = []

        # get probabilities for original data
        for idx, prob in enumerate(probs):
            original_probs.append(prob[fairness_against_digit])
        
        # get probabilities for counterfactual data with interventions on specific attribute
        for _ in range(5):
            X = X + original_probs

            cfs = []
            for i in range(len(data)):
                if "MNIST" in run_name:
                    cfs.append(_get_cf_for_mnist(data[i][0], metrics['thickness'][i], metrics['intensity'][i], labels[i]))
                else:
                    cfs.append(_get_cf_for_chestxray(data[i][0], metrics[:][i], labels[i], 'male', None, None))

            cfs = torch.stack(cfs)
            logits = model.model(cfs).cpu()
            probs = torch.nn.functional.softmax(logits, dim=1).tolist()
            cf_probs = []
            for idx, prob in enumerate(probs):
                cf_probs.append(prob[fairness_against_digit])
            
            Y = Y + cf_probs

    X = np.array(X)
    Y = np.array(Y)

    fig = plt.figure(run_name)
    plt.scatter(X,Y)
    plt.savefig("plots/fairness_correct_"+ run_name +".png")

def _get_embeddings(model, data_loader):
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    embeddings = np.zeros(shape=(0, 784))
    labels = np.zeros(shape=(0))
    thicknesses = np.zeros(shape=(0))
    for _, (data, metrics, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        output = feature_extractor(data).detach().cpu().squeeze(1).numpy()
        s = output.shape
        output = output.reshape((s[0], 784))
        labels = np.concatenate((labels, target.cpu().numpy().ravel()))
        embeddings = np.concatenate([embeddings, output],axis=0)
        thicknesses = np.concatenate((thicknesses, metrics['thickness']))

    return embeddings, thicknesses, labels

def visualise_t_sne(test_loader, model, file_name):
    embeddings, thicknesses, labels = _get_embeddings(model, test_loader)
    
    feat_cols = ['pixel'+str(i) for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=feat_cols)
    df['y'] = labels
    df['thickness'] = thicknesses.astype(int)
    df['label'] = df['y'].apply(lambda i: str(i))

    N = 100000
    rndperm = np.random.permutation(df.shape[0])
    df_subset = df.loc[rndperm[:N],:].copy()
    data_subset = df_subset[feat_cols].values

    # reduce dimensions before feeding into t-SNE
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    print('[t-SNE]\tt-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
    df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]

    plt.figure(figsize=(16,10))
    plot = sns.scatterplot(
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
    plot = sns.scatterplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue="thickness",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    fig = plot.get_figure()
    fig.savefig(file_name + "_thickness.png") 