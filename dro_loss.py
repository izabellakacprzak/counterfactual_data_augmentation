import torch
from utils.params import *

device = torch.device(GPU if torch.cuda.is_available() else "cpu")

# adapted from: https://github.com/kohpangwei/group_DRO.git
class DROLoss:
    def __init__(self, criterion, n_groups, group_counts, adj=None, min_var_weight=0, step_size=0.01, normalize_loss=False):
        self.criterion = criterion
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss

        self.n_groups = n_groups
        self.group_counts = group_counts

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().to(device)
        else:
            self.adj = torch.zeros(self.n_groups).float().to(device)

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).to(device)/self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).to(device)
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().to(device)

        #self.reset_stats()

    def loss(self, yhat, y, group_idx=None):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx)

        # compute overall loss
        actual_loss, _ = self.compute_robust_loss(group_loss)
        return actual_loss

    def compute_robust_loss(self, group_loss):
        adjusted_loss = group_loss
        if torch.all(self.adj>0):
            adjusted_loss += self.adj/torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(self.step_size*adjusted_loss.data)
        self.adv_probs = self.adv_probs/(self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().to(device)).float()
        group_count = group_map.sum(1)
        group_denom = group_count + (group_count==0).float() # avoid nans 
        group_loss = (group_map @ losses.view(-1))/group_denom
        return group_loss, group_count
