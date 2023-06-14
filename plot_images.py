import matplotlib.pyplot as plt
from torchvision import transforms
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F

from utils.params import *
from utils.cf_utils import *
from datasets.perturbedMNIST import PerturbedMNIST
from datasets.chestXRay import ChestXRay
from datasets.coloredMNIST import ColoredMNIST
from dscmchest.generate_counterfactuals import generate_cf
# from dscm.generate_colored_counterfactuals import generate_colored_counterfactual

def plot_images():
    transforms_list = transforms.Compose([transforms.ToTensor()])
    test_dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=0.01)

    imgs = []
    labs = []
    # transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    # test_dataset = ChestXRay(mode="test", transform=transforms_list)
    # transforms_list = transforms.Compose([transforms.ToTensor()])
    # test_dataset = ColoredMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=0.01)
    w = 15
    h = 7
    fig = plt.figure(figsize=(w, h))
    columns = 6
    rows = 2
    ax = []
    digits = list(range(10))
    for i in range(1, columns*rows+1, 1):
        idx = random.randint(0, len(test_dataset)-1)
        img, metrics, label = test_dataset[idx]
        t = transforms.ToPILImage()
        img = t(img)
        ax.append(fig.add_subplot(rows, columns, i))
        ax[-1].set_title("label: {}".format(str(label)))
        plt.axis('off')
        plt.imshow(img, cmap='gray')
        # img = transforms.Pad(padding=2)(img).type(torch.ByteTensor)
        # obs = {
        #     'x': img,
        #     'color': F.one_hot(torch.tensor(metrics['color']).long(), num_classes=10),
        #     'digit': F.one_hot(torch.tensor(label).long(), num_classes=10)}
        # while label != digits[i-1]:
        #     idx = random.randint(0, len(test_dataset)-1)
        #     img, _, label = test_dataset[idx]

        # imgs.append(img[0])
        # labs.append(label)
        # ax.append(fig.add_subplot(rows, columns, i+1))
        # lab = "No finding" if label.item()==0 else "Pleural Effusion"
        # ax[-1].set_title(str(lab))
        # img_cf, metrics_cf, _ = generate_colored_counterfactual(obs=obs, color=1)
        # t = transforms.ToPILImage()
        # img = t(img_cf)
        # plt.axis('off')
        # plt.imshow(img_cf)
    # plt.show()
    # plt.savefig("visualised-mnist.png")
    # np.save("imgs.npy", np.array(imgs))
    # np.save("labs.txt", np.array(labs))
    plt.savefig("visualised_perturbed_mnist.jpg")

def plot_morpho_counterfactuals():
    transforms_list = transforms.Compose([transforms.ToTensor()])
    dataset = PerturbedMNIST(train=False, transform=transforms_list, bias_conflicting_percentage=0.01)
    w = 15
    h = 7
    fig = plt.figure(figsize=(w, h))
    columns = 11
    rows = 3
    ax = []

    def plot_morpho_cf(img_cf, idx):
        ax.append(fig.add_subplot(rows, columns, idx))
        # ax[-1].set_title("do t={}".format(round(metrics_cf[0], 2)))
        plt.axis('off')
        # t = transforms.ToPILImage()
        # img = t(img_cf)
        plt.imshow(img_cf, cmap='gray')

    digit = 0
    for i in range(1, columns*rows+1, columns):
        idx = random.randint(0, len(dataset)-1)
        img, metrics, label = dataset[idx]
        digit += 1
        ax.append(fig.add_subplot(rows, columns, i))
        # ax[-1].set_title("Original")
        t = transforms.ToPILImage()
        im = t(img)
        plt.axis('off')
        plt.imshow(im, cmap='gray')

        img = img.float() * 255.0
        img = transforms.Pad(padding=2)(img).type(torch.ByteTensor)
        
        for j in range(1, columns):
            img_cf = get_cf_for_mnist(img, metrics['thickness'], metrics['intensity', label])
            plot_morpho_cf(img_cf, i+j)

    plt.savefig("cf_interventions_morpho_mnist.jpg")

def plot_colored_counterfactuals():
    from dscm.generate_colored_counterfactuals import generate_colored_counterfactual
    cols = [0,1,2,3,4,5,6,7,8,9]
    transforms_list = transforms.Compose([transforms.ToTensor()])
    # dataset = ColoredMNIST(train=False, transform=transforms_list)
    dataset = ColoredMNIST(train=True, transform=transforms_list, bias_conflicting_percentage=0.01)
    w = 15
    h = 7
    fig = plt.figure(figsize=(w, h))
    columns = 11
    rows = 4
    ax = []

    digits = [1,3,5,7]

    def plot_colored_cf(img_cf, idx):
        ax.append(fig.add_subplot(rows, columns, idx))
        # ax[-1].set_title("do c={}".format(col))
        plt.axis('off')
        # t = transforms.ToPILImage()
        # img = t(img_cf)
        plt.imshow(img_cf)

    digit = 0
    for i in range(1, columns*rows+1, columns):
        idx = random.randint(0, len(dataset)-1)
        img, metrics, label = dataset[idx]
        while label != digits[digit]:
            idx = random.randint(0, len(dataset)-1)
            img, metrics, label = dataset[idx]
        digit += 1
        ax.append(fig.add_subplot(rows, columns, i))
        # ax[-1].set_title("Original")
        t = transforms.ToPILImage()
        im = t(img)
        plt.axis('off')
        plt.imshow(im)

        img = img.float() * 255.0
        img = transforms.Pad(padding=2)(img).type(torch.ByteTensor)
        
        for j in range(1,columns):
            img_cf = get_cf_for_colored_mnist(img, metrics['color'], label)
            plot_colored_cf(img_cf, i+j)

    plt.savefig("visualised_colored_mnist.jpg")

def plot_chest_counterfactuals():
    transforms_list = transforms.Compose([transforms.Resize((192,192)),])
    test_dataset = ChestXRay(mode="test", transform=transforms_list)

    w = 15
    h = 7
    fig = plt.figure(figsize=(w, h))
    columns = 5
    rows = 2
    ax = []
    for i in range(1, columns*rows+1, columns):
        idx = random.randint(0, len(test_dataset)-1)
        x, metrics, label = test_dataset[idx]
        obs = {'x':x, 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race'], 'finding':label}
        ax.append(fig.add_subplot(rows, columns, i))
        # ax[-1].set_title("Original")
        plt.axis('off')
        plt.imshow(x[0].numpy(), cmap='gray')

        age = (metrics['age'].item()//20)%5
        ages = [0,1,2,3,4]
        ages.remove(age)
        do_a = random.choice(ages)
        print(do_a)
        x_cf = generate_cf(obs=obs, do_a=do_a, do_f=None, do_r=None, do_s=None)
        ax.append(fig.add_subplot(rows, columns, i+1))
        # ax[-1].set_title("do_a="+str(do_a))
        plt.axis('off')
        plt.imshow(x_cf[0], cmap='gray')

        obs = {'x':x, 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race'], 'finding':label}
        do_s = 0 if metrics['race'].item()==1 else 1
        print(do_s)
        x_cf = generate_cf(obs=obs, do_a=None, do_f=None, do_r=None, do_s=do_s)
        ax.append(fig.add_subplot(rows, columns, i+2))
        # ax[-1].set_title("do_s="+str(do_s))
        plt.axis('off')
        plt.imshow(x_cf[0], cmap='gray')

        obs = {'x':x, 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race'], 'finding':label}
        race = metrics['race'].item()
        do_r = 0 if race==1 else (1 if race==2 else 2)
        print(do_r)
        x_cf = generate_cf(obs=obs, do_a=None, do_f=None, do_r=do_r, do_s=None)
        ax.append(fig.add_subplot(rows, columns, i+3))
        # ax[-1].set_title("do_r="+str(do_r))
        plt.axis('off')
        plt.imshow(x_cf[0], cmap='gray')

        obs = {'x':x, 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race'], 'finding':label}
        print((1-label.item()))
        x_cf = generate_cf(obs=obs, do_a=None, do_f=(1-label.item()), do_r=None, do_s=None)
        ax.append(fig.add_subplot(rows, columns, i+4))
        # ax[-1].set_title("do_f="+str(1-label.item()))
        plt.axis('off')
        plt.imshow(x_cf[0], cmap='gray')

    plt.savefig("cf_interventions_chestxray.jpg")

# plot_images()
# plot_colored_counterfactuals()
# plot_morpho_counterfactuals()
plot_chest_counterfactuals()
