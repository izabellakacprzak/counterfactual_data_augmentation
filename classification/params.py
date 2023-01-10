import torch 

LEARNING_RATE = 0.005
EPOCHS = 10
IN_CHANNELS = 3
NUM_OF_CLASSES = 10
BATCH_SIZE = 64
LOSS_FN = torch.nn.CrossEntropyLoss()
TRAIN_FILE = "models/train_colored"
TEST_FILE = "models/test_colored.pt"
MED_TRAIN_FILE = "models/train_med"
MED_TEST_FILE = "models/test_med.pt"
TRAIN_PERTURBED_FILE = "models/train_mnist_perturbed.pt"
CUT_PERCENTAGE = 0.05
UNDERSAMPLED_CLASSES = [7, 8]