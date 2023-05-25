import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("..")

from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from utils.evaluate import visualise_t_sne
from classifier import ConvNet, DenseNet, test_classifier
from utils.params import *

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

def _get_embeddings(model, data_loader, img_dim):
    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

    embeddings = np.zeros(shape=(0, img_dim*img_dim))
    labels = np.zeros(shape=(0))
    race = np.zeros(shape=(0))
    sex = np.zeros(shape=(0))
    age = np.zeros(shape=(0))
    augmented = np.zeros(shape=(0))
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
        augmented = np.concatenate((augmented, metrics['augmented'].tolist()))

    return embeddings, race, sex, age, labels, augmented

def _plot_tsne(df_subset, attribute, file_name):
    plt.figure(figsize=(16,10))
    plot = sns.jointplot(
        x="tsne-pca50-one", y="tsne-pca50-two",
        hue=attribute,
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    #fig = plot.get_figure()
    plot.savefig("{}_{}.png".format(file_name, attribute)) 

def visualise_t_sne(test_loader, model, img_dim, file_name):
    embeddings, race, sex, age, labels, augmented = _get_embeddings(model, test_loader, img_dim)
    
    feat_cols = ['pixel'+str(i) for i in range(embeddings.shape[1])]
    df = pd.DataFrame(embeddings, columns=feat_cols)
    df['y'] = labels
    df['race'] = race.astype(int)
    df['sex'] = sex.astype(int)
    df['label'] = df['y'].apply(lambda i: str(i))
    df['augmented'] = augmented.astype(int)

    age = age.astype(int)
    for idx, a in enumerate(age):
        age[idx] = (a/20) % 5
    df['age'] = age

    N = 10000
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

    _plot_tsne(df_subset, "y", file_name)
    _plot_tsne(df_subset, "race", file_name)
    _plot_tsne(df_subset, "sex", file_name)
    _plot_tsne(df_subset, "age", file_name)
    _plot_tsne(df_subset, "augmented", file_name)


def visualise_embeddings(model_path, dataset, in_channels, out_channels, img_dim):
    if "MNIST" in model_path:
        model = ConvNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/mnist/classifier_{}.pt".format(model_path), map_location=device))
    else:
        model = DenseNet(in_channels=in_channels, out_channels=out_channels)
        model.load_state_dict(torch.load("../checkpoints/chestxray/classifier_{}.pt".format(model_path), map_location=device))
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    visualise_t_sne(test_loader, model, img_dim, "plots/{}t_sne".format(model_path))

def visualise_perturbed_mnist():
    models = ["UNBIASED", "BIASED", "OVERSAMPLING", "AUGMENTATIONS", "MIXUP", "COUNTERFACTUALS", "CFREGULARISATION"]
    
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    for model in models:
        mnist_model_path = model + "_PERTURBED_MNIST"
        visualise_embeddings(mnist_model_path, test_dataset, 1, 10, 28)

def visualise_chestxray():
    models = ["BASELINE", "GROUP_DRO_age", "OVERSAMPLING_age_0", "AUGMENTATIONS_age_0", "COUNTERFACTUALS_age_0", "COUNTERFACTUALS_DRO_age_0"]
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)
    train_dataset = ChestXRay(mode="train", transform=transforms_list)

    for model in models:
        chestxray_model_path = model + "_CHESTXRAY"
        # visualise_embeddings(chestxray_model_path, test_dataset, 1, 2, 192)

        # visualise embeddings on train set to compare originals and cfs
        visualise_embeddings(chestxray_model_path, train_dataset, 1, 2, 192)

# visualise_perturbed_mnist()
visualise_chestxray()
