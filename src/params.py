import torch 

LEARNING_RATE = 0.001
EPOCHS = 15
LAMBDA = 1.0
IN_CHANNELS = 3
NUM_OF_CLASSES = 2
BATCH_SIZE = 128
LOSS_FN = torch.nn.CrossEntropyLoss()
MSE = torch.nn.MSELoss()
TRAIN_FILE = "data/train_colored"
TEST_FILE = "data/test_colored.pt"
MED_TRAIN_FILE = "data/train_med"
MED_TEST_FILE = "data/test_med.pt"
TRAIN_PERTURBED_FILE = "data/train_mnist_perturbed.pt"
CUT_PERCENTAGE = 0.01
UNDERSAMPLED_CLASSES = [7, 8]
COUNTERFACTUALS_DATA = "../../DSCMv2/src/data/generated_counterfactuals.pt"
COUNTERFACTUALS_METRICS = "../../DSCMv2/src/data/generated_counterfactuals_metrics.csv"

THIN_CLASSES = [0, 7, 8]
THICK_CLASSES = [1, 3, 6, 9]
OTHER_CLASSES = [2, 4, 5]