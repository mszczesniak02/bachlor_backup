import os

ON_COLAB = False

try:
    import google.colab
    DATA_DIR = "/content/datasets"
    TRAIN_DIR = os.path.join(DATA_DIR, "classification_width/train_img")
    TEST_DIR = os.path.join(DATA_DIR, "classification_width/test_img")
    DEVICE = "cuda"
    WORKERS = 0
    ON_COLAB = True

    ENET_MODEL_TRAIN_DIR = "/content/models/classification/efficienet/"
    ENET_MODEL_TRAIN_LOG_DIR = "/content/models_log/classification/efficienet/"

    CONVNEXT_MODEL_TRAIN_DIR = "/content/models/classification/convnext/"
    CONVNEXT_MODEL_TRAIN_LOG_DIR = "/content/models_log/classification/convnext/"

except ImportError:
    DATA_DIR = "/home/krzeslaav/Projects/datasets"
    TRAIN_DIR = os.path.join(DATA_DIR, "classification_width/train_img")
    TEST_DIR = os.path.join(DATA_DIR, "classification_width/test_img")
    DEVICE = "cuda"
    WORKERS = 4
    ON_COLAB = False

    ENET_MODEL_TRAIN_DIR = "../../../models/classification/efficienet/"
    ENET_MODEL_TRAIN_LOG_DIR = "../../../models_log/classification/efficienet/"

    CONVNEXT_MODEL_TRAIN_DIR = "../../../models/classification/convnext/"
    CONVNEXT_MODEL_TRAIN_LOG_DIR = "../../../models_log/classification/convnext/"


# Common
DEFAULT_IMAGE_SIZE = 256
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EPOCHS = 50
PATIENCE = 18
NUM_CLASSES = 4
SEED = 42

# EfficientNet
ENET_IMAGE_SIZE = 512
ENET_BATCH_SIZE = 28
ENET_LEARNING_RATE = 1e-4
ENET_WEIGHT_DECAY = 1e-5
ENET_EPOCHS = 15
ENET_SCHEDULER_PATIENCE = 5

# ConvNext
CONVNEXT_IMAGE_SIZE = 512
CONVNEXT_BATCH_SIZE = 16
CONVNEXT_LEARNING_RATE = 2e-5
CONVNEXT_WEIGHT_DECAY = 1e-4
CONVNEXT_EPOCHS = 15
CONVNEXT_SCHEDULER_PATIENCE = 5
MODEL_PATH = ENET_MODEL_TRAIN_DIR + "best_model.pth"


def main():
    print("nothing to do")


if __name__ == "__main__":
    main()
