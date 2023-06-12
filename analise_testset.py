#from datasets.chestXRay import ChestXRay
from utils.params import *
from utils.utils import preprocess_age

from torch.utils.data import DataLoader
from torchvision import transforms

def analise_dataset(data, num_classes=3):
    dataloader = DataLoader(data, batch_size=1, shuffle=False)
    counts = [{} for _ in range(num_classes)]
    for (image, metrics, label) in dataloader:
        age = "age: "+str(preprocess_age(metrics['age'].item()))
        counts[label][age] = (0 if not age in counts[label] else counts[label][age]) + 1
        race = "race: "+str(metrics['race'].item())
        counts[label][race] = (0 if not race in counts[label] else counts[label][race]) + 1
        sex = "sex: "+str(metrics['sex'].item())
        counts[label][sex] = (0 if not sex in counts[label] else counts[label][sex]) + 1
    
    for idx, d in enumerate(counts):
        ks = list(d.keys())
        ks.sort()
        counts[idx] = {i: d[i] for i in ks}
        print(counts[idx])
        print()

