import torch 

GPU = "cuda:0"

LEARNING_RATE = 0.00005
EPOCHS = 15
LAMBDA = 1.0
BATCH_SIZE = 32
MSE = torch.nn.MSELoss()
TRAIN_PERTURBED_FILE = "data/train_mnist_perturbed.pt"

COUNTERFACTUALS_DATA = "../../DSCMv2/src/data/generated_counterfactuals.pt"
COUNTERFACTUALS_METRICS = "../../DSCMv2/src/data/generated_counterfactuals_metrics.csv"
CF_CHEST_DATA = "/vol/bitbucket/iek19/data/chestxray/generated_cfs_data_no_finding.npy"
CF_CHEST_METRICS = "/vol/bitbucket/iek19/data/chestxray/generated_cfs_metrics_no_finding.csv"

THIN_CLASSES = [0, 7, 8]
THICK_CLASSES = [1, 3, 6, 9]
OTHER_CLASSES = [2, 4, 5]
