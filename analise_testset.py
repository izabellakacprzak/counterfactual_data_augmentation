#from datasets.chestXRay import ChestXRay
from utils.params import *
from utils.utils import preprocess_age

from torch.utils.data import DataLoader
from torchvision import transforms

def analise_dataset(data):
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    counts_positive = {}
    counts_negative = {}
    for (image, metrics, label) in dataloader:
        if label == 0:
            age = "age: "+str(preprocess_age(metrics['age'].item()))
            counts_negative[age] = (0 if not age in counts_negative else counts_negative[age]) + 1
            race = "race: "+str(metrics['race'].item())
            counts_negative[race] = (0 if not race in counts_negative else counts_negative[race]) + 1
            sex = "sex: "+str(metrics['sex'].item())
            counts_negative[sex] = (0 if not sex in counts_negative else counts_negative[sex]) + 1
        else:
            age = "age: "+str(preprocess_age(metrics['age'].item()))
            counts_positive[age] = (0 if not age in counts_positive else counts_positive[age]) + 1
            race = "race: "+str(metrics['race'].item())
            counts_positive[race] = (0 if not race in counts_positive else counts_positive[race]) + 1
            sex = "sex: "+str(metrics['sex'].item())
            counts_positive[sex] = (0 if not sex in counts_positive else counts_positive[sex]) + 1
    
    ks = list(counts_positive.keys())
    ks.sort()
    counts_positive = {i: counts_positive[i] for i in ks}
    ks = list(counts_negative.keys())
    ks.sort()
    counts_negative = {i: counts_negative[i] for i in ks}
    print(counts_positive)
    print()
    print(counts_negative)
