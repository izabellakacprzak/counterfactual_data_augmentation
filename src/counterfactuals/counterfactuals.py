import os

import random
import pyro
import torch
import torch.nn.functional as F
import gdown

url='https://drive.google.com/uc?id=1BUj7rMGYW0WRkrmzkAiVeZiG8prv3lRZ'
output = ''
gdown.download(url, output, quiet=False)
os.system('mv /content/dscm_class_cond/* /content/')
os.system('rmdir /content/dscm_class_cond')

from class_conditional_vi_sem import ClassConditionalVISEM

def get_dscm():
    checkpoint = torch.load(os.path.join(os.getcwd(), 'dscm_checkpoint.pt'))
    model = ClassConditionalVISEM(latent_dim=16,
                                hidden_dim=128,
                                auxiliary_model=True)
    pyro.clear_param_store()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda().eval()

    return model

def get_counterfactual(model, factual, factual_label, num_counterfactuals=1):
    num_particles = 10
    A = torch.tensor([factual_label for _ in range(num_counterfactuals)])
    y_ = F.one_hot(A, num_classes = 10).cuda()
    # y_ = F.one_hot(torch.arange(10), num_classes=10).cuda()
    # do = {'y': y_}
    T = []
    if factual_label in [0, 7, 8]:
        T = [[1] for _ in range(num_counterfactuals)]
    elif factual_label in [1, 3, 6, 9]:
        T = [[5] for _ in range(num_counterfactuals)]

    t_ = torch.tensor(T).cuda()
    do = {'y': y_, 'thickness': t_}
    # do = {'y': y_, 'intensity': torch.tensor([[255.]]).repeat(10,1).cuda()}
    # do = {'y': y_,
    #       'thickness': torch.tensor([[1.]]).repeat(10,1).cuda(),
    #       'intensity': torch.tensor([[155.]]).repeat(10,1).cuda()}

    counterfactuals = []
    for jj in range(10):
        with torch.no_grad():
            counterfactual = model.counterfactual(obs=factual,
                                              condition=do,
                                              num_particles=num_particles)
        counterfactuals.append(counterfactual)

    return counterfactuals

def generate_counterfactuals(train_data):
    model = get_dscm()

    new_train_set = []
    for (im, label) in train_data:
        counterfactuals = get_counterfactual(model, im, label, 2)
        new_train_set = new_train_set + list(zip(counterfactuals, [label for _ in range(2)]))

    return random.shuffle(new_train_set + train_data)