import torch
from torchvision import datasets
import pandas as pd

from params import *
from utils.utils import *
from tqdm import tqdm
from skimage.io import imread
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def norm(batch):
    for k, v in batch.items():
        if k == 'x':
            batch['x'] = (batch['x'].float() - 127.5) / 127.5  # [-1,1]
        elif k in ['age']:
            batch[k] = batch[k].float().unsqueeze(-1)
            batch[k] = batch[k] / 100.
            batch[k] = batch[k] * 2 - 1  # [-1,1]
        elif k in ['race']:
            batch[k] = F.one_hot(batch[k], num_classes=3).squeeze().float()
        elif k in ['finding']:
            batch[k] = batch[k].unsqueeze(-1).float()
        else:
            batch[k] = batch[k].float().unsqueeze(-1)
    return batch

class ChestXRay(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None, bias_conflicting_percentage=1, method=AugmentationMethod.NONE):
    super(ChestXRay, self).__init__('files', transform=transform, target_transform=target_transform)
    metrics_df = pd.read_csv("csv_file")

    train_data, test_data = train_test_split(metrics_df, test_size=0.2)
    if train:
       self.data = train_data
    else:
       self.data = test_data

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
        img_path = os.path.join("root", self.data.loc[idx, 'path_preproc'])

        disease = np.zeros(len(self.labels)-1, dtype=int)
        for i in range(1, len(self.labels)):
            disease[i-1] = np.array(self.data.loc[idx,
                                    self.labels[i]] == 1)

        finding = 0 if disease.sum() == 0 else 1

        self.samples['x'].append(img_path)
        self.samples['finding'].append(finding)
        self.samples['age'].append(self.data.loc[idx, 'age'])
        self.samples['race'].append(self.data.loc[idx, 'race_label'])
        self.samples['sex'].append(self.data.loc[idx, 'sex_label'])
      
    if method != AugmentationMethod.NONE and method != AugmentationMethod.CF_REGULARISATION:
        self.data, self.metrics = debias_chestxray(train_data=self.samples, method=method)

  def __getitem__(self, index):
    sample = {k: v[index] for k, v in self.samples.items()}
    sample = norm(sample)

    image = imread(sample['x']).astype(np.float32)[None, ...]
    metrics = {k: torch.tensor(v) for k, v in sample if (k != 'x' and k != 'finding')}
    target = sample['finding']

    if self.transform:
        image = self.transform(image)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return image, metrics, target

  def __len__(self):
    return len(self.data)