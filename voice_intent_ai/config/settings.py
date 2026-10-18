import os
from pathlib import Path

# Project directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, OUTPUT_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Audio parameters
AUDIO_DIR = RAW_DATA_DIR / "clips"  # Directory containing audio files
SAMPLE_RATE = 16000  # Hz
DURATION = 5  # seconds
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
MFCC_FEATURES = 40

# Model parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
DROPOUT_RATE = 0.3
HIDDEN_DIM = 256
NUM_CLASSES = 10  # Update based on your intent classes

# Training parameters
EARLY_STOPPING_PATIENCE = 5
LR_PATIENCE = 3
LR_FACTOR = 0.5

# File paths
TRAIN_TSV = RAW_DATA_DIR / "train.tsv"
TEST_TSV = RAW_DATA_DIR / "test.tsv"
AUDIO_DIR = RAW_DATA_DIR / "clips"

# Set random seed for reproducibility
RANDOM_SEED = 42
