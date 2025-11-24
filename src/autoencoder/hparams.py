ON_COLAB = False

if ON_COLAB:
    DEVICE = "cuda"
    TRAIN_DIR = "/content/datasets/multi/train_img/"
    TEST_DIR = "/content/datasets/multi/test_img/"

    MODEL_DIR = "/content/models/autoencoder/"
    LOG_DIR = "/content/models_log/autoencoder/"

else:
    DEVICE = "cuda"
    TRAIN_DIR = "../../../datasets/multi/train_img/"
    TEST_DIR = "../../../datasets/multi/test_img/"

    MODEL_DIR = "../../../models/autoencoder/"
    LOG_DIR = "../../../models_log/autoencoder/"


SEED = 42
# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 50
IMAGE_SIZE = 256  # Smaller than 512 for efficiency
# Size of the bottleneck feature map channels (or flattened vector)
LATENT_DIM = 64

# Training
PATIENCE = 10
SCHEDULER_PATIENCE = 5
WEIGHT_DECAY = 1e-5
