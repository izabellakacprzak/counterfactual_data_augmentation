import torch
from torchvision import transforms
import torchvision.transforms as TF
import torch.nn.functional as F

import sys
sys.path.append("..")

from utils.params import *

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

def get_cf_for_mnist(img, thickness, intensity, label):
    from dscm.generate_counterfactuals import generate_counterfactual_for_x
    img = img.float() * 254
    img = TF.Pad(padding=2)(img).type(torch.ByteTensor).unsqueeze(0)
    x_cf = generate_counterfactual_for_x(img, thickness, intensity, label)
    return torch.from_numpy(x_cf).unsqueeze(0).float()

def get_cf_for_chestxray(img, metrics, label, do_a, do_f, do_r, do_s):
    from dscmchest.generate_counterfactuals import generate_cf
    obs = {'x': img,
           'age': metrics['age'],
           'race': metrics['race'],
           'sex': metrics['sex'],
           'finding': label}
    cf = generate_cf(obs, do_a=do_a, do_f=do_f, do_r=do_r, do_s=do_s)
    return cf

def get_cf_for_colored_mnist(img, color, label, new_col):
    from dscm.generate_colored_counterfactuals import generate_colored_counterfactual
    img = img.float().cpu() * 255.0
    img = transforms.Pad(padding=2)(img).type(torch.ByteTensor)
    obs = {
        'x': img,
        'color': F.one_hot(torch.tensor(color).long(), num_classes=10),
        'digit': F.one_hot(torch.tensor(label).long().cpu(), num_classes=10)
        }
    
    img_cf, _, _ = generate_colored_counterfactual(obs=obs, do_c=new_col)
    
    return img_cf