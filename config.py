# import torch
from pathlib import Path

# Moondream
ORIGINAL_MODEL = "vikhyatk/moondream2"
MOONDREAM_VERSION = "2024-07-23"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Fine-tuning
EPOCHS = 1
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
LR = 3e-6
ATTN_IMPL = None
IMG_TOKENS = 729
ANSWER_EOS = "<|endoftext|>"
OUTPUT_DIR = Path('./Training results/Weights/Moondream/Current')
TRAIN_LEN = 29000
VAL_LEN = 1014
TEST_LEN = 1000
ADAM_EPS = 1e-6

# Metrics
MAX_METRIC_ITER = 1000
METRICS_DIR = Path('./Training results/Metrics/Moondream')
TEMPERATURES = [None]

# Graphs
GRAPH_DIR = Path('./Training results/Graphs/Moondream')
