import torch 

LEARNING_RATE = 0.001
EPOCHS = 9
IN_CHANNELS = 3
NUM_OF_CLASSES = 10
BATCH_SIZE = 128
LOSS_FN = torch.nn.CrossEntropyLoss()
TRAIN_FILE = "train_med.pt"
TEST_FILE = "test_med.pt"