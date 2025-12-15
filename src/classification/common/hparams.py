ON_COLAB = False

if ON_COLAB == True:
    TRAIN_DIR = "/content/datasets/classification/train_img"
    TEST_DIR = "/content/datasets/classification/test_img"
    DEVICE = "cuda"
    WORKERS = 0

    ENET_MODEL_TRAIN_DIR = "/content/models/classification/efficienet/"
    ENET_MODEL_TRAIN_LOG_DIR = "/content/models_log/classification/efficienet/"

    CONVNEXT_MODEL_TRAIN_DIR = "/content/models/classification/convnext/"
    CONVNEXT_MODEL_TRAIN_LOG_DIR = "/content/models_log/classification/convnext/"

else:
    TRAIN_DIR = "../../../../datasets/multi_class/train_balanced/"
    TEST_DIR = "../../../../datasets/multi_class/test_balanced/"
    DEVICE = "cuda"
    WORKERS = 4

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
PATIENCE = 10
NUM_CLASSES = 4
SEED = 42

# EfficientNet
ENET_IMAGE_SIZE = 256
ENET_BATCH_SIZE = 28
ENET_LEARNING_RATE = 1e-3
ENET_WEIGHT_DECAY = 1e-5
ENET_EPOCHS = 20
ENET_SCHEDULER_PATIENCE = 5

# ConvNext
CONVNEXT_IMAGE_SIZE = 256
CONVNEXT_BATCH_SIZE = 16
CONVNEXT_LEARNING_RATE = 1e-3
CONVNEXT_WEIGHT_DECAY = 1e-5
CONVNEXT_EPOCHS = 20
CONVNEXT_SCHEDULER_PATIENCE = 5
MODEL_PATH = ENET_MODEL_TRAIN_DIR + "best_model.pth"


def main():
    print("nothing to do")


if __name__ == "__main__":
    main()
