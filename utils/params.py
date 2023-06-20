import torch 

GPU = "cuda:1"

LEARNING_RATE = 0.0001
EPOCHS = 20
LAMBDA = 0.6
BATCH_SIZE = 32
MSE = torch.nn.MSELoss()

# PERTURBED MNIST FILES AND SETTINGS #
TRAIN_PERTURBED_DATA = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/train_perturbed"
TEST_PERTURBED_DATA = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/test_perturbed.pt"
TRAIN_PERTURBED_METRICS = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/train_perturbed_mnist_metrics"
TEST_PERTURBED_METRICS = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/test_perturbed_mnist_metrics.csv"

COUNTERFACTUALS_DATA = "/homes/iek19/Documents/FYP/DSCMv2/dscm/data/generated_counterfactuals.pt"
COUNTERFACTUALS_METRICS = "/homes/iek19/Documents/FYP/DSCMv2/dscm/data/generated_counterfactuals_metrics.csv"

THIN_CLASSES = [0, 7, 8]
THICK_CLASSES = [1, 3, 6, 9]
OTHER_CLASSES = [2, 4, 5]

# COLORED MNIST FILES #
TRAIN_COLORED_DATA = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/colored/train_mnist_colored"
TEST_COLORED_DATA = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/colored/test_mnist_colored.pt"
TRAIN_COLORED_METRICS = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/colored/train_mnist_colored_metrics"
TEST_COLORED_METRICS = "/homes/iek19/Documents/FYP/counterfactual_data_augmentation/data/colored/test_mnist_colored_metrics.csv"

COUNTERFACTUALS_COLORED_DATA = "/homes/iek19/Documents/FYP/DSCMv2/dscm/data/generated_counterfactuals_colored.pt"
COUNTERFACTUALS_COLORED_METRICS = "/homes/iek19/Documents/FYP/DSCMv2/dscm/data/generated_counterfactuals_colored_metrics.csv"

# MIMIC CXR FILES #
MIMIC_DATA = "/vol/biomedic3/bglocker/mimic-cxr-jpg-224/data/"
MIMIC_METRICS = "/homes/iek19/Documents/FYP/mimic_meta/mimic.sample."

COUNTERFACTUALS_CHEST_DATA = "/vol/bitbucket/iek19/data/chestxray/generated_cfs_data_random.npy"
COUNTERFACTUALS_CHEST_METRICS = "/vol/bitbucket/iek19/data/chestxray/generated_cfs_metrics_random.csv"
