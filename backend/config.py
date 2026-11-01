import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models_saved"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Image and Class definitions
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
INPUT_DIM = IMAGE_WIDTH * IMAGE_HEIGHT
NUM_CLASSES = 10
DIGIT_LABELS = [str(i) for i in range(10)]

# Dataset defaults
DEFAULT_TRAIN_SIZE = 1200
DEFAULT_TEST_SIZE = 600
MAX_TRAIN_SIZE = 10000

# Training defaults
DEFAULT_EPOCHS = {
    "hebb": 5,
    "perceptron": 20,
    "adaline": 25,
}

DEFAULT_LEARNING_RATES = {
    "hebb": 0.01,
    "perceptron": 0.01,
    "adaline": 0.0005,  # Smaller LR for LMS stability with 784 inputs
}

PORT = 5000
HOST = "127.0.0.1"
