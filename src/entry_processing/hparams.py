ON_COLAB = False

if ON_COLAB:
    DEVICE = "cuda"
    TRAIN_DIR = "/content/datasets/entry_dataset/train/"
    TEST_DIR = "/content/datasets/entry_dataset/test/"
    VAL_DIR = "/content/datasets/entry_dataset/val/"

    MODEL_DIR = "/content/models/entry_classificator/"
    LOG_DIR = "/content/models_log/entry_classificator/"
    WORKERS = 2

else:
    DEVICE = "cuda"
    TRAIN_DIR = "../../../datasets/entry_dataset/train/"
    TEST_DIR = "../../../datasets/entry_dataset/test/"
    VAL_DIR = "../../../datasets/entry_dataset/val/"

    MODEL_DIR = "../../models/entry_classificator/"
    LOG_DIR = "../../../models_log/entry_classificator/"
    WORKERS = 4


SEED = 42
# Hyperparameters
ENTRY_BATCH_SIZE = 32
ENTRY_LEARNING_RATE = 1e-4
ENTRY_EPOCHS = 50
ENTRY_IMAGE_SIZE = 256
NUM_CLASSES = 2

# Training
ENTRY_PATIENCE = 10
ENTRY_SCHEDULER_PATIENCE = 3
ENTRY_WEIGHT_DECAY = 1e-4
