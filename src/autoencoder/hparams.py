ON_COLAB = False

if ON_COLAB:
    DEVICE = "cuda"
    TRAIN_DIR = "/content/datasets/multi/train_img/"
    TEST_DIR = "/content/datasets/multi/test_img/"

    MODEL_DIR = "/content/models/autoencoder/"
    LOG_DIR = "/content/models_log/autoencoder/"
    WORKERS = 2

else:
    DEVICE = "cuda"
    TRAIN_DIR = "../../../datasets/multi/train_img/"
    TEST_DIR = "../../../datasets/multi/test_img/"

    MODEL_DIR = "../../../models/autoencoder/"
    LOG_DIR = "../../../models_log/autoencoder/"
    WORKERS = 4


SEED = 42
# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.00050000
EPOCHS = 100
IMAGE_SIZE = 256  # Smaller than 512 for efficiency
# Size of the bottleneck feature map channels (or flattened vector)
LATENT_DIM = 64

# Training
PATIENCE = 15
SCHEDULER_PATIENCE = 5
WEIGHT_DECAY = 1e-5
