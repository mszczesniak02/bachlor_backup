import torch

# General
ON_COLAB = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Data Paths
# We will filter for "0_brak" in the dataloader, but the root dir is the same
TRAIN_DIR = "../../../datasets/multi/train_img/"
TEST_DIR = "../../../datasets/multi/test_img/"


# Model & Logging Paths
MODEL_DIR = "../../../models/autoencoder/"
LOG_DIR = "../../../models_log/autoencoder/"

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
