import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd

import sys
sys.path.append("..")

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from datasets.coloredMNIST import ColoredMNIST
from classifier import ConvNet, DenseNet
from utils.params import *
from utils.utils import DebiasingMethod

device = torch.device(GPU if torch.cuda.is_available() else "cpu")
SAMPLE_SIZE = 10000
X_AXIS_TITLE = "t-SNE dimension 1"
Y_AXIS_TITLE = "t-SNE dimension 2"

def _get_embeddings_chestxray(model, data_loader, img_dim):
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    embeddings = np.zeros(shape=(0, img_dim))
    labels = np.zeros(shape=(0))
    race = np.zeros(shape=(0))
    sex = np.zeros(shape=(0))
    age = np.zeros(shape=(0))
    augmented = np.zeros(shape=(0))
    for idx, (data, metrics, target) in enumerate(data_loader):
        if idx*BATCH_SIZE >= SAMPLE_SIZE:
            break
        data, target = data.to(device), target.to(device)
        output = feature_extractor(data).detach().cpu().squeeze(1).numpy()
        s = output.shape
        output = output.reshape((s[0], img_dim))
        labels = np.concatenate((labels, target.cpu().numpy().ravel()))
        embeddings = np.concatenate([embeddings, output],axis=0)
        race = np.concatenate((race, metrics['race'].tolist()))
        sex = np.concatenate((sex, metrics['sex'].tolist()))
        age = np.concatenate((age, metrics['age'].tolist()))
        augmented = np.concatenate((augmented, metrics['augmented'].tolist()))

    return embeddings, race, sex, age, labels, augmented

def _get_embeddings_mnist(model, data_loader, img_dim):
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    embeddings = np.zeros(shape=(0, img_dim))
    labels = np.zeros(shape=(0))
    thickness = np.zeros(shape=(0))
    for idx, (data, metrics, target) in enumerate(data_loader):
        if idx*BATCH_SIZE >= SAMPLE_SIZE:
            break
        data, target = data.to(device), target.to(device)
        output = feature_extractor(data).detach().cpu().squeeze(1).numpy()
        s = output.shape
        output = output.reshape((s[0], img_dim))
        labels = np.concatenate((labels, target.cpu().numpy().ravel()))
        embeddings = np.concatenate([embeddings, output],axis=0)
        thickness = np.concatenate((thickness, metrics['thickness'].tolist()))

    return embeddings, thickness, labels

def _get_embeddings_colored_mnist(model, data_loader, img_dim):
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    embeddings = np.zeros(shape=(0, img_dim))
    labels = np.zeros(shape=(0))
    color = np.zeros(shape=(0))
    augmented = np.zeros(shape=(0))
    for idx, (data, metrics, target) in enumerate(data_loader):
        if idx*BATCH_SIZE >= SAMPLE_SIZE:
            break
        data, target = data.to(device), target.to(device)
        output = feature_extractor(data).detach().cpu().squeeze(1).numpy()
        s = output.shape
        output = output.reshape((s[0], img_dim))
        labels = np.concatenate((labels, target.cpu().numpy().ravel()))
        embeddings = np.concatenate([embeddings, output],axis=0)
        color = np.concatenate((color, metrics['color'].tolist()))
        augmented = np.concatenate((augmented, metrics['bias_aligned'].tolist()))

    return embeddings, color, labels, augmented

def _plot_tsne(df_subset, attribute, file_name, title):
    plt.figure(figsize=(16,10))
    plot = sns.jointplot(
        x=X_AXIS_TITLE, y=Y_AXIS_TITLE,
        hue=attribute,
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plot.fig.suptitle(title, y=-0.05)
    plot.savefig("{}_{}.png".format(file_name, attribute)) 

def visualise_t_sne(test_loader, model, img_dim, file_name, title, dataset='mnist'):
    if dataset=='mnist':
        embeddings, thickness, labels = _get_embeddings_mnist(model, test_loader, img_dim)
        feat_cols = ['pixel'+str(i) for i in range(embeddings.shape[1])]
        df = pd.DataFrame(embeddings, columns=feat_cols)
        thickness = thickness.astype(float)
        thickness_post = []
        for idx, t in enumerate(thickness):
            thickness_post.append('thin' if t <=1.5 else 'thick')
        df['thickness'] = thickness_post
    elif dataset=='coloredmnist':
        embeddings, color, labels, augmented = _get_embeddings_colored_mnist(model, test_loader, img_dim)
        feat_cols = ['pixel'+str(i) for i in range(embeddings.shape[1])]
        df = pd.DataFrame(embeddings, columns=feat_cols)
        df['y'] = labels
        color = color.astype(float)
        df['color'] = color
        df['augmented'] = augmented.astype(int)
    else:
        embeddings, race, sex, age, labels, augmented = _get_embeddings_chestxray(model, test_loader, img_dim)
        feat_cols = ['pixel'+str(i) for i in range(embeddings.shape[1])]
        df = pd.DataFrame(embeddings, columns=feat_cols)
        df['y'] = labels
        df['race'] = race.astype(int)
        df['sex'] = sex.astype(int)
        age = age.astype(int)
        for idx, a in enumerate(age):
            age[idx] = (a/20) % 5
        df['age'] = age
        df['augmented'] = augmented.astype(int)
    
    df['label'] = df['y'].apply(lambda i: str(i))

    rndperm = np.random.permutation(df.shape[0])
    df_subset = df.loc[rndperm[:SAMPLE_SIZE],:].copy()
    data_subset = df_subset[feat_cols].values

    # reduce dimensions before feeding into t-SNE
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(data_subset)

    time_start = time.time()
    # tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=2000)
    tsne = TSNE(n_components=2, init='random', learning_rate='auto')
    tsne_pca_results = tsne.fit_transform(pca_result_50)

    print('[t-SNE]\tt-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset[X_AXIS_TITLE] = tsne_pca_results[:,0]
    df_subset[Y_AXIS_TITLE] = tsne_pca_results[:,1]

    _plot_tsne(df_subset, "y", file_name, title)
    if dataset=='mnist':
        _plot_tsne(df_subset, "thickness", file_name, title)
    elif dataset=='coloredmnist':
        _plot_tsne(df_subset, "color", file_name, title)
        _plot_tsne(df_subset, "augmented", file_name, title)
    else:
        _plot_tsne(df_subset, "race", file_name, title)
        _plot_tsne(df_subset, "sex", file_name, title)
        _plot_tsne(df_subset, "age", file_name, title)
        _plot_tsne(df_subset, "augmented", file_name, title)


def visualise_embeddings(model_path, dataset, in_channels, out_channels, img_dim, save_dir, title):
    if "PERTURBED" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_{}.pt".format(model_path), map_location=device))
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        visualise_t_sne(test_loader, model, img_dim, "{}/{}t_sne".format(save_dir, model_path), title, "mnist")
    elif "COLORED" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/colored_mnist/00/classifier_{}.pt".format(model_path), map_location=device))
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        visualise_t_sne(test_loader, model, img_dim, "{}/{}t_sne".format(save_dir, model_path), title, "coloredmnist")
    else:
        model = DenseNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_{}.pt".format(model_path), map_location=device))
        test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        visualise_t_sne(test_loader, model, img_dim, "{}/{}t_sne".format(save_dir, model_path), title, "chestxray")

def visualise_perturbed_mnist():
    models = ["BASELINE", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    save_dir = "plots/tsne/mnist"
    in_channels = 1
    out_channels = 10
    img_dim = 28

    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    for model in models:
        mnist_model_path = "{}_PERTURBED_MNIST".format(model)
        visualise_embeddings(mnist_model_path, test_dataset, in_channels, out_channels, img_dim*img_dim, save_dir, model)

def visualise_colored_mnist():
    models = ["BASELINE", "OVERSAMPLING", "AUGMENTATIONS", "GROUP_DRO", "COUNTERFACTUALS", "CFREGULARISATION"]
    save_dir = "plots/tsne/colored_mnist/00"
    in_channels = 3
    out_channels = 10
    img_dim = 28

    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = ColoredMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01, method=DebiasingMethod.COUNTERFACTUALS)

    for model in models:
        mnist_model_path = "{}_COLORED_MNIST".format(model)
        visualise_embeddings(mnist_model_path, test_dataset, in_channels, out_channels, 3*img_dim*img_dim, save_dir, model)

def visualise_chestxray():
    models = ["BASELINE", "OVERSAMPLING_race", "AUGMENTATIONS_race", "GROUP_DRO_race", "COUNTERFACTUALS_race", "CFREGULARISATION_race"]
    titles = ["BASELINE", "OVERSAMPLING", "STANDARD AUGMENTATIONS", "GROUP DRO", "COUNTERFACTUALS", "CF REGULARISATION"]
    
    suffix = "_disease_pred"
    save_dir = "plots/tsne/chestxray_disease"
    in_channels = 1
    out_channels = 2
    img_dim = 192

    transforms_list = transforms.Compose([transforms.Resize((img_dim,img_dim)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)
    # train_dataset = ChestXRay(mode="test", transform=transforms_list, method=DebiasingMethod.COUNTERFACTUALS)

    for idx, model in enumerate(models):
        title = titles[idx]
        chestxray_model_path = "{}{}_CHESTXRAY".format(model, suffix)
        # visualise embedding on test set
        visualise_embeddings(chestxray_model_path, test_dataset, in_channels, out_channels, img_dim*img_dim, save_dir, title)

        # visualise embeddings on train set to compare originals and cfs
        # visualise_embeddings(chestxray_model_path, train_dataset, 1, 2, 192, save_dir, title)

# visualise_perturbed_mnist()
# visualise_colored_mnist()
visualise_chestxray()
