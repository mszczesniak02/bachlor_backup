ON_COLAB = False

if ON_COLAB:
    # if using 16 GB gpu :>
    MASK_TRAIN_PATH = r"/content/datasets/multi/train_lab"
    IMG_TRAIN_PATH = r"/content/datasets/multi/train_img"
    MASK_TEST_PATH = r"/content/datasets/multi/test_lab"
    IMG_TEST_PATH = r"/content/datasets/multi/test_img"
    DEVICE = "cuda"
    WORKERS = 2

    UNET_MODEL_TRAIN_DIR = r"/content/models/segmentation/unet/"
    UNET_MODEL_TRAIN_LOG_DIR = r"/content/models_log/segmentation/unet/"

    SEGFORMER_MODEL_TRAIN_DIR = r"/content/models/segmentation/segformer"
    SEGFORMER_MODEL_TRAIN_LOG_DIR = r"/content/models_log/segmentation/segformer/"

else:
    # using local 2GB laptop :|
    MASK_TRAIN_PATH = r"../../../../datasets/multi/train_lab"
    IMG_TRAIN_PATH = r"../../../../datasets/multi/train_img"
    MASK_TEST_PATH = r"../../../../datasets/multi/test_lab"
    IMG_TEST_PATH = r"../../../../datasets/multi/test_img"

    UNET_MODEL_TRAIN_DIR = r"../../../models/segmentation/unet/"
    UNET_MODEL_TRAIN_LOG_DIR = r"../../../models_log/segmentation/unet/"

    SEGFORMER_MODEL_TRAIN_DIR = r"../../../models/segmentation/segformer"
    SEGFORMER_MODEL_TRAIN_LOG_DIR = r"../../../models_log/segmentation/segformer/"

    DEVICE = "cpu"
    WORKERS = 4

# DEVICE = "cpu"
MODEL_INFERENCE_PATH = r"../../../models/segmentation/model_dice.pth"
MODEL_INFERENCE_DIR = r"../../../models/segmentation/"


# DEFAULT HYPER PARAMS
DEFAULT_BATCH_SIZE = 2  # 32
DEFAULT_LEARNING_RATE = .482e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_EPOCHS = 1  # 3


# HYPER PARAMS FOR UNET SEGMENTATION
UNET_BATCH_SIZE = 16  # 32
UNET_LEARNING_RATE = 0.00001
UNET_WEIGHT_DECAY = 1e-3
UNET_EPOCHS = 10  # 3

# HYPER PARAMS FOR SEGFORMER SEGMENTATION
SEGFORMER_BATCH_SIZE = 16  # 32
SEGFORMER_LEARNING_RATE = 0.00010000
SEGFORMER_WEIGHT_DECAY = 1e-5
SEGFORMER_EPOCHS = 100  # 3

# HELPER PARAMS FOR EASIER DETECTION
PIN_MEMORY = True
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
SEED = 42


def main():
    print("nothing to do.", end="")


if __name__ == "__main__":
    main()
