import torch
from torchvision import datasets

from utils.params import *
from utils.utils import *
from utils.colors import prepare_colored_mnist

class ColoredMNIST(datasets.VisionDataset):
  def __init__(self, train=True, transform=None, target_transform=None, bias_conflicting_percentage=1, method=DebiasingMethod.NONE):
    super(ColoredMNIST, self).__init__('files', transform=transform,
                                target_transform=target_transform)

    prepare_colored_mnist(datasets.mnist.MNIST('files', train=True, download=True), datasets.mnist.MNIST(self.root, train=False, download=True), bias_conflicting_percentage)
    if train:
      self.data_label_tuples = torch.load(TRAIN_COLORED_DATA+"_"+str(bias_conflicting_percentage).replace(".", "_")+".pt")
      self.metrics = pd.read_csv(TRAIN_COLORED_METRICS+"_"+str(bias_conflicting_percentage).replace(".", "_")+".csv", index_col='index').to_dict('records')
    
    else:
      self.data_label_tuples = torch.load(TEST_COLORED_DATA)
      self.metrics = pd.read_csv(TEST_COLORED_METRICS, index_col='index').to_dict('records')

    if not method in [DebiasingMethod.NONE, DebiasingMethod.CF_REGULARISATION, DebiasingMethod.MIXUP]:
      self.data_label_tuples, self.metrics = debias_colored_mnist(train_data=self.data_label_tuples, train_metrics=self.metrics, method=method)

    self.group_counts = {}
    for m in self.metrics:
      c = m['color']
      self.group_counts[c] = (0 if c not in self.group_counts else self.group_counts[c]) + 1


  def __getitem__(self, index):
    img, target = self.data_label_tuples[index]
    metrics = {k: torch.tensor(float(self.metrics[index][k])) for k in ['color', 'bias_aligned']}
    metrics['group_idx'] = torch.tensor(float(self.metrics[index]['color']))

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, metrics, target

  def __len__(self):
    return len(self.data_label_tuples)