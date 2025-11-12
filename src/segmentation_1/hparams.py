# training through Google Colab -> train using notebook rather then .py script
ON_COLAB = False

if ON_COLAB != True:
    # if using 16 GB gpu :>
    MASK_TRAIN_PATH = "../../assets/datasets/DeepCrack/train_lab"
    IMG_TRAIN_PATH = "../../assets/datasets/DeepCrack/train_img"
    MASK_TEST_PATH = "../../assets/datasets/DeepCrack/test_lab"
    IMG_TEST_PATH = "../../assets/datasets/DeepCrack/test_img"
    DEVICE = "cuda"
    WORKERS = 4
else:
    # using local 2GB laptop :|
    MASK_TRAIN_PATH = "/content/DeepCrack/train_lab"
    IMG_TRAIN_PATH = "/content/DeepCrack/train_img"
    MASK_TEST_PATH = "/content/DeepCrack/test_lab"
    IMG_TEST_PATH = "/content/DeepCrack/test_img"
    DEVICE = "cuda"
    WORKERS = 2

DEVICE = "cpu"
MODEL_PATH = "../../models/segmentation_1/model_dice.pth"
MODEL_DIR = "../../models/segmentation_1/"

# HYPER PARAMS FOR SEGMENTATION MODEL NR 1
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 40

# HELPER PARAMS FOR EASIER DETECTION
PIN_MEMORY = True
EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5


def main():
    print("nothing to do.", end="")


if __name__ == "__main__":
    main()
