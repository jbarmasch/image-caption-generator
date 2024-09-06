from pathlib import Path

# Define paths
BASE_PATH = Path(__file__).parent.resolve()
DATASET_PATH = BASE_PATH / "data" / "your_dataset_folder"
CHECKPOINT_PATH = BASE_PATH / "checkpoints"
OUTPUT_PATH = BASE_PATH / "output"
TEST_IMAGES_PATH = BASE_PATH / "data" / "test_images"

# Training parameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10

# Model parameters
CNN_MODEL_NAME = 'resnet50'
TRANSFORMER_MODEL_NAME = 'bert-base-uncased'
MAX_SEQ_LENGTH = 128