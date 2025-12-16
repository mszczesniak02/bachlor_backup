ON_COLAB = False


if ON_COLAB:
    # if using 16 GB gpu :>
    MASK_TRAIN_PATH = r"/content/datasets/multi/train_lab"
    IMG_TRAIN_PATH = r"/content/datasets/multi/train_img"
    MASK_TEST_PATH = r"/content/datasets/multi/test_lab"
    IMG_TEST_PATH = r"/content/datasets/multi/test_img"

    YOLO_DATASET_DIR = r"/content/datasets/yolo_seg_data"

    DEVICE = "cuda"
    WORKERS = 0

    UNET_MODEL_TRAIN_DIR = r"/content/models/segmentation/unet/"
    UNET_MODEL_TRAIN_LOG_DIR = r"/content/models_log/segmentation/unet/"

    SEGFORMER_MODEL_TRAIN_DIR = r"/content/models/segmentation/segformer"
    SEGFORMER_MODEL_TRAIN_LOG_DIR = r"/content/models_log/segmentation/segformer/"

else:
    # using local 2GB laptop :|
    MASK_TRAIN_PATH = r"../../../../datasets/DeepCrack/train_lab"
    IMG_TRAIN_PATH = r"../../../../datasets/DeepCrack/train_img"
    MASK_TEST_PATH = r"../../../../datasets/DeepCrack/test_lab"
    IMG_TEST_PATH = r"../../../../datasets/DeepCrack/test_img"

    YOLO_DATASET_DIR = r"../../../../datasets/yolo_seg_data"

    UNET_MODEL_TRAIN_DIR = r"../../../models/segmentation/unet/"
    UNET_MODEL_TRAIN_LOG_DIR = r"../../../models_log/segmentation/unet/"

    SEGFORMER_MODEL_TRAIN_DIR = r"../../../models/segmentation/segformer"
    SEGFORMER_MODEL_TRAIN_LOG_DIR = r"../../../models_log/segmentation/segformer/"

    DEVICE = "cuda"
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
UNET_EPOCHS = 100  # 3

# HYPER PARAMS FOR SEGFORMER SEGMENTATION
SEGFORMER_BATCH_SIZE = 16  # 32
SEGFORMER_LEARNING_RATE = 6e-5
SEGFORMER_WEIGHT_DECAY = 1e-5
SEGFORMER_EPOCHS = 100  # 3

# HYPER PARAMS FOR YOLO SEGMENTATION
YOLO_BATCH_SIZE = 16
YOLO_LEARNING_RATE = 1e-3
YOLO_EPOCHS = 100
YOLO_MODEL_SIZE = "m"  # n, s, m, l, x

# HELPER PARAMS FOR EASIER DETECTION
PIN_MEMORY = True
EARLY_STOPPING_PATIENCE = 18
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.1
SEED = 42


def main():
    print("nothing to do.", end="")


if __name__ == "__main__":
    main()
