import torch 

GPU = "cuda:0"

LEARNING_RATE = 0.0001
EPOCHS = 20
LAMBDA = 0.6
BATCH_SIZE = 32
MSE = torch.nn.MSELoss()
TRAIN_PERTURBED_FILE = "data/train_mnist_perturbed"

TRAIN_COLORED_DATA = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/colored/train_mnist_colored"
TEST_COLORED_DATA = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/colored/test_mnist_colored.pt"
TRAIN_COLORED_METRICS = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/colored/train_mnist_colored_metrics"
TEST_COLORED_METRICS = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/colored/test_mnist_colored_metrics.csv"

COUNTERFACTUALS_DATA = "/homes/iek19/Documents/FYP/DSCMv2/dscm/data/generated_counterfactuals.pt"
COUNTERFACTUALS_METRICS = "/homes/iek19/Documents/FYP/DSCMv2/dscm/data/generated_counterfactuals_metrics.csv"
COUNTERFACTUALS_COLORED_DATA = "/homes/iek19/Documents/FYP/DSCMv2/dscm/data/generated_counterfactuals_colored.pt"
COUNTERFACTUALS_COLORED_METRICS = "/homes/iek19/Documents/FYP/DSCMv2/dscm/data/generated_counterfactuals_colored_metrics.csv"
# CF_CHEST_DATA = "/vol/bitbucket/iek19/data/chestxray/generated_cfs_data_age_disease.npy"
# CF_CHEST_METRICS = "/vol/bitbucket/iek19/data/chestxray/generated_cfs_metrics_age_disease.csv"
CF_CHEST_DATA = "/vol/bitbucket/iek19/data/chestxray/generated_cfs_data_random.npy"
CF_CHEST_METRICS = "/vol/bitbucket/iek19/data/chestxray/generated_cfs_metrics_random.csv"

THIN_CLASSES = [0, 7, 8]
THICK_CLASSES = [1, 3, 6, 9]
OTHER_CLASSES = [2, 4, 5]
