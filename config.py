import torch

ORIGINAL_MODEL = "vikhyatk/moondream2"
MOONDREAM_VERSION = "2024-07-23"
EPOCHS = 3
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 1
LR = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
ATTN_IMPL = None
IMG_TOKENS = 729
ANSWER_EOS = "<|endoftext|>"
OUTPUT_DIR = './Training results/Weights/Moondream/Current'
TRAIN_LEN = 29000
VAL_LEN = 1014
TEST_LEN = 1000
ADAM_EPS = 1e-6
