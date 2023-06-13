import matplotlib.pyplot as plt
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F

from utils.params import *
from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from datasets.coloredMNIST import ColoredMNIST
# from dscmchest.generate_counterfactuals import generate_cf
from dscm.generate_colored_counterfactuals import generate_colored_counterfactual

def plot_images():
    # transforms_list = transforms.Compose([transforms.ToTensor()])
    # test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=1.0)

    imgs = []
    labs = []
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)
    # transforms_list = transforms.Compose([transforms.ToTensor()])
    # test_dataset = ColoredMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=0.01)
    w = 15
    h = 7
    fig = plt.figure(figsize=(w, h))
    columns = 6
    rows = 2
    ax = []
    digits = list(range(10))
    for i in range(1, columns*rows +1):
        idx = random.randint(0, len(test_dataset)-1)
        img, _, label = test_dataset[idx]
        # while label != digits[i-1]:
        #     idx = random.randint(0, len(test_dataset)-1)
        #     img, _, label = test_dataset[idx]

        # imgs.append(img[0])
        # labs.append(label)
        ax.append(fig.add_subplot(rows, columns, i))
        lab = "No finding" if label.item()==0 else "Pleural Effusion"
        ax[-1].set_title(str(lab))
        # t = transforms.ToPILImage()
        # img = t(img)
        plt.axis('off')
        plt.imshow(img[0], cmap='gray')
    # plt.show()
    # plt.savefig("visualised-mnist.png")
    # np.save("imgs.npy", np.array(imgs))
    # np.save("labs.txt", np.array(labs))
    plt.savefig("visualised_chestxray.jpg")

def plot_colored_counterfactual():
    cols = [0,1,2,3,4,5]
    transforms_list = transforms.Compose([transforms.ToTensor()])
    # dataset = ColoredMNIST(train=False, transform=transforms_list)
    dataset = ColoredMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=0.01)
    w = 15
    h = 7
    fig = plt.figure(figsize=(w, h))
    columns = 7
    rows = 3
    ax = []

    def plot_colored_cf(obs, col, idx):
        img_cf, metrics_cf, _ = generate_colored_counterfactual(obs=obs, color=col)
        ax.append(fig.add_subplot(rows, columns, idx))
        ax[-1].set_title("do_c={}".format(col))
        plt.axis('off')
        # t = transforms.ToPILImage()
        # img = t(img_cf)
        plt.imshow(img_cf)

    for i in range(1,h+1):
        idx = random.randint(0, len(dataset)-1)
        img, metrics, label = dataset[idx]
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title("Original")
        plt.axis('off')
        plt.imshow(img[0])

        img = img.float() * 255.0
        img = transforms.Pad(padding=2)(img).type(torch.ByteTensor)
        obs = {
            'x': img,
            'color': F.one_hot(torch.tensor(metrics['color']).long(), num_classes=10),
            'digit': F.one_hot(torch.tensor(label).long(), num_classes=10)}
        
        for j in range(1,7):
            plot_colored_cf(obs, j-1, i+j)

    plt.savefig("cf_interventions_colored_mnist.jpg")


def plot_counterfactuals():
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    w = 15
    h = 3
    fig = plt.figure(figsize=(w, h))
    columns = 4
    rows = 2
    ax = []
    for i in range(h):
        idx = random.randint(0, len(test_dataset)-1)
        x, metrics, label = test_dataset[idx]
        obs = {'x':x, 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race'], 'finding':label}
        print(metrics['race'].shape) 
        ax.append(fig.add_subplot(rows, columns, i+1))
        ax[-1].set_title("Original")
        plt.imshow(x[0].numpy())

        age = (metrics['age'].item()//20)%5
        ages = [0,1,2,3,4]
        ages.remove(age)
        do_a = random.choice(ages)
        x_cf = generate_cf(obs=obs, do_a=do_a, do_f=None, do_r=None, do_s=None)
        ax.append(fig.add_subplot(rows, columns, i+2))
        ax[-1].set_title("do_a="+str(do_a))
        plt.imshow(x_cf[0])

        do_s = 0 if metrics['race'].item()==1 else 1
        x_cf = generate_cf(obs=obs, do_a=None, do_f=None, do_r=None, do_s=do_s)
        ax.append(fig.add_subplot(rows, columns, i+3))
        ax[-1].set_title("do_s="+str(do_s))
        plt.imshow(x_cf[0])

        race = metrics['race'].item()
        do_r = 0 if race==1 else (1 if race==2 else 2)
        x_cf = generate_cf(obs=obs, do_a=None, do_f=None, do_r=do_r, do_s=None)
        ax.append(fig.add_subplot(rows, columns, i+4))
        ax[-1].set_title("do_r="+str(do_r))
        plt.imshow(x_cf[0])

        x_cf = generate_cf(obs=obs, do_a=None, do_f=(1-label.item()), do_r=None, do_s=None)
        ax.append(fig.add_subplot(rows, columns, i+5))
        ax[-1].set_title("do_f="+str(1-label.item()))
        plt.imshow(x_cf[0])

    plt.savefig("cf_interventions_chestxray.png")

# plot_counterfactuals()
plot_images()
# plot_colored_counterfactual()
