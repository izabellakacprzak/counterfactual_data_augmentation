import torch
from torchvision import datasets
import pandas as pd

from params import *
from utils.utils import *
from tqdm import tqdm
from skimage.io import imread
import torch.nn.functional as F

class ChestXRay(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None, method=AugmentationMethod.NONE):
    super(ChestXRay, self).__init__('files', transform=transform, target_transform=target_transform)
    
    csv_file = "/homes/iek19/Documents/FYP/mimic_meta/mimic.sample." + ("train" if train else "test") + ".csv"
    if train:
        self.data = pd.read_csv(csv_file).head(70000)
    else:
        self.data = pd.read_csv(csv_file).head(10000)
    self.transform = transform
    self.labels = [
        'No Finding',
        'Enlarged Cardiomediastinum',
        'Cardiomegaly',
        'Lung Opacity',
        'Lung Lesion',
        'Edema',
        'Consolidation',
        'Pneumonia',
        'Atelectasis',
        'Pneumothorax',
        'Pleural Effusion',
        'Pleural Other',
        'Fracture',
        'Support Devices'
    ]

    self.samples = {
        'age': [],
        'sex': [],
        'finding': [],
        'x': [],
        'race': [],
    }
    for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
        img_path = os.path.join("/vol/biomedic3/bglocker/mimic-cxr-jpg-224/data/", self.data.loc[idx, 'path_preproc'])

        disease = np.zeros(len(self.labels)-1, dtype=int)
        for i in range(1, len(self.labels)):
            disease[i-1] = np.array(self.data.loc[idx,
                                    self.labels[i]] == 1)

        finding = 0 if disease.sum() == 0 else 1

        self.samples['x'].append(imread(img_path).astype(np.float32)[None, ...])
        self.samples['finding'].append(finding)
        self.samples['age'].append(self.data.loc[idx, 'age'])
        self.samples['race'].append(self.data.loc[idx, 'race_label'])
        self.samples['sex'].append(self.data.loc[idx, 'sex_label'])

  def debias(self, method):
    self.samples = debias_chestxray(self, method)
     
  def __getitem__(self, index):
    sample = {k: v[index] for k, v in self.samples.items()}

    metrics = {k: v for k, v in sample.items() if (k != 'x' and k != 'finding')}
    target = sample['finding']

    if self.transform:
        image = self.transform(sample['x'])

    if self.target_transform is not None:
      target = self.target_transform(target)

    return image, metrics, target

  def __len__(self):
    return len(self.samples)
