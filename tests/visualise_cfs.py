from torchvision import transforms
import random
import numpy as np

from utils.params import *
from datasets.chestXRay import ChestXRay
from dscmchest.generate_counterfactuals import generate_cf


originals = []
cfs = []
transforms_list = transforms.Compose([transforms.Resize((192,192)),])
test_dataset = ChestXRay(mode="test", transform=transforms_list)
for i in range(20):
    idx = random.randint(0, len(test_dataset)-1)
    x, metrics, label = test_dataset[idx]
    obs = {'x':x, 'sex':metrics['sex'], 'age':metrics['age'], 'race':metrics['race'], 'finding':label}
    do_a, do_r, do_s = None, None, 1
    x_cf, cf_metrics = generate_cf(obs, do_a, do_r, do_s)
    if len(cf_metrics) != 0:
        originals.append(x[0].numpy())
        cfs.append(x_cf[0])
    
np.save("original_imgs.npy", np.array(originals))
np.save("cf_imgs.npy", np.array(cfs))
