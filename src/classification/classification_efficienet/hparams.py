NUM_CLASSES = 5

DEVICE = "cuda"
MODEL_PATH = r"/home/krzeslaav/Projects/bachlor/models/classification/model_f1_0.8568_epoch0.pth"
MODEL_DIR = r""


TRAIN_DIR = "/home/krzeslaav/Projects/datasets/multi_classification_categorized/train"
TEST_DIR = "/home/krzeslaav/Projects/datasets/multi_classification_categorized/test"

TRAIN_LOG_DIR = r"/home/krzeslaav/Projects/bachlor/models_log/classification_1"

WORKERS = 4

SCHEDULER_PATIENCE = 5
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 2
IMAGE_SIZE = 224
LEARNING_RATE = 1.5e-3
EPOCHS = 2
PATIENCE = 10
