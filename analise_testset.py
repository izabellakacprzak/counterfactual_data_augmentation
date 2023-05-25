from datasets.chestXRay import ChestXRay
from utils.params import *
from utils.utils import preprocess_age

from torch.utils.data import DataLoader
from torchvision import transforms

transforms_list = transforms.Compose([transforms.Resize((192,192)),])
test_dataset = ChestXRay(mode="test", transform=transforms_list)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

counts_positive = {}
counts_negative = {}
for (image, metrics, label) in test_loader:
    if label == 0:
        age = preprocess_age(metrics['age'].item())
        counts_negative[age] = (0 if age in counts_negative else counts_negative[age]) + 1
        race = metrics['race']
        counts_negative[race] = (0 if race in counts_negative else counts_negative[race]) + 1
        sex = metrics['sex']
        counts_negative[sex] = (0 if sex in counts_negative else counts_negative[sex]) + 1
    else:
        age = preprocess_age(metrics['age'].item())
        counts_positive[age] = (0 if age in counts_positive else counts_positive[age]) + 1
        race = metrics['race']
        counts_positive[race] = (0 if race in counts_positive else counts_positive[race]) + 1
        sex = metrics['sex']
        counts_positive[sex] = (0 if sex in counts_positive else counts_positive[sex]) + 1

print(counts_positive)
print()
print(counts_negative)