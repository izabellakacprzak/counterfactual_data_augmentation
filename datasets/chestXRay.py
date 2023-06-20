import torch
from torchvision import datasets
import pandas as pd

from utils.params import *
from utils.utils import *
from tqdm import tqdm
from skimage.io import imread
import torch.nn.functional as F

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
    def __init__(self, mode="train", transform=None, target_transform=None, method=DebiasingMethod.NONE):
        super(ChestXRay, self).__init__('files', transform=transform, target_transform=target_transform)

        csv_file = "{}{}.csv".format(MIMIC_METRICS, mode)
        self.data = pd.read_csv(csv_file)

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
            'augmented': []
        }

        # group counts for group DRO loss calculation
        self.group_counts = {}

        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = os.path.join(MIMIC_DATA, self.data.loc[idx, 'path_preproc'])

            disease = np.zeros(len(self.labels)-1, dtype=int)
            for i in range(1, len(self.labels)):
                disease[i-1] = np.array(self.data.loc[idx, self.labels[i]] == 1)
    
            # only consider patients with Pleural Effusion and healthy patients
            if disease.sum() == 0:
                finding = 0
            elif self.data.loc[idx, 'Pleural Effusion'] == 1:
                finding = 1
            else:
                continue

            self.samples['x'].append(imread(img_path).astype(np.float32)[None, ...])
            self.samples['finding'].append(finding)
            age = self.data.loc[idx, 'age']
            self.samples['age'].append(age)
            race = self.data.loc[idx, 'race_label']
            self.samples['race'].append(race)
            sex = self.data.loc[idx, 'sex_label']
            self.samples['sex'].append(sex)
            self.samples['augmented'].append(0)
    
            # groups for group DRO loss calculation
            age = (age//20)%5
            group_idx = race
            self.group_counts[group_idx] = (0 if group_idx not in self.group_counts else self.group_counts[group_idx]) + 1

        # debias data if data-driven debiasing method used
        if not method in [DebiasingMethod.NONE, DebiasingMethod.CF_REGULARISATION, DebiasingMethod.MIXUP]:
            self._debias(method)
        
        # print data counts for analysis
        self._analise_dataset()

    def _debias(self, method):
        new_samples = debias_chestxray(self, method)
        self.samples['x'] += new_samples['x']
        self.samples['finding'] += new_samples['finding']
        self.samples['age'] += new_samples['age']
        self.samples['race'] += new_samples['race']
        self.samples['sex'] += new_samples['sex']
        self.samples['augmented'] += new_samples['augmented']

    def _analise_dataset(self, num_classes=3):
        dataloader = DataLoader(self, batch_size=1, shuffle=False)
        counts = [{} for _ in range(num_classes)]
        for (_, metrics, label) in dataloader:
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

    def __getitem__(self, idx):
        sample = {k: v[idx] for k, v in self.samples.items()}
        group_idx = torch.tensor(sample['race'])
        for k, v in sample.items():
            sample[k] = torch.tensor(v)

        if self.transform:
            sample['x'] = self.transform(sample['x'])

        metrics = {'sex':sample['sex'], 'age':sample['age'], 'race':sample['race'],
                'augmented':sample['augmented'], 'group_idx':group_idx, 'finding':sample['finding']}
        return sample['x'], metrics, sample['finding']
        # return sample['x'], metrics, metrics['race']

    def __len__(self):
        return len(self.samples['x'])
