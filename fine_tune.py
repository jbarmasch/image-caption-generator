from models.moondream import MoondreamCaptioner
import torch
from einops import rearrange
from dataset import CustomDataset
import math
from bitsandbytes.optim import Adam8bit
from tqdm import tqdm
from tokenizers.pre_tokenizers import Whitespace

# Number of times to repeat the training dataset. Increasing this may cause the model to overfit or
# lose generalization due to catastrophic forgetting. Decreasing it may cause the model to underfit.
EPOCHS = 2

# Number of samples to process in each batch. Set this to the highest value that doesn't cause an
# out-of-memory error. Decrease it if you're running out of memory. Batch size 8 currently uses around
# 15 GB of GPU memory during fine-tuning.
BATCH_SIZE = 16

# Number of batches to process before updating the model. You can use this to simulate a higher batch
# size than your GPU can handle. Set this to 1 to disable gradient accumulation.
GRAD_ACCUM_STEPS = 1

# Learning rate for the Adam optimizer. Needs to be tuned on a case-by-case basis. As a general rule
# of thumb, increase it by 1.4 times each time you double the effective batch size.
#
# Source: https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
#
# Note that we linearly warm the learning rate up from 0.1 * LR to LR over the first 10% of the
# training run, and then decay it back to 0.1 * LR over the last 90% of the training run using a
# cosine schedule.
LR = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32 # CPU doesn't support float16
IMG_TOKENS = 729
ANSWER_EOS = "<|endoftext|>"
OUTPUT_DIR = './Training results/Weights/Moondream/Current'

captioner = MoondreamCaptioner(torch_device=DEVICE, dtype = DTYPE)
moondream = captioner._moondream_model
tokenizer = captioner._moondream_tokenizer
tokenizer.pre_tokenizer = Whitespace()
print("Model loaded")

def collate_fn(batch):
    images = [sample['image'] for sample in batch]
    preprocessed_images = [moondream.vision_encoder.preprocess(image) for image in images]
    images = torch.stack(preprocessed_images)
    images = rearrange(images, "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=14, p2=14)

    labels_acc = []
    tokens_acc = []

    for sample in batch:
        toks = [tokenizer.bos_token_id]
        labs = [-100] * (IMG_TOKENS + 1)

        for caption in sample['caption']:
            q_t = tokenizer(
                f"\n\nQuestion: {captioner.default_prompt}\n\nAnswer:",
                add_special_tokens=False
            ).input_ids
            toks.extend(q_t)
            labs.extend([-100] * len(q_t))

            a_t = tokenizer(
                f" {caption}{ANSWER_EOS}",
                add_special_tokens=False
            ).input_ids
            toks.extend(a_t)
            labs.extend(a_t)

        tokens_acc.append(toks)
        labels_acc.append(labs)

    max_len = -1
    for labels in labels_acc:
        max_len = max(max_len, len(labels))

    attn_mask_acc = []

    for i in range(len(batch)):
        len_i = len(labels_acc[i])
        pad_i = max_len - len_i

        labels_acc[i].extend([-100] * pad_i)
        tokens_acc[i].extend([tokenizer.eos_token_id] * pad_i)
        attn_mask_acc.append([1] * len_i + [0] * pad_i)

    return (
        images.to(dtype=DTYPE),
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )

def compute_loss(batch):
    images, tokens, labels, attn_mask = batch

    images = images.to(DEVICE)
    tokens = tokens.to(DEVICE)
    labels = labels.to(DEVICE)
    attn_mask = attn_mask.to(DEVICE)

    with torch.no_grad():
        img_embs = moondream.vision_encoder.encoder(images)
        img_embs = moondream.vision_encoder.projection(img_embs)

    tok_embs = moondream.text_model.get_input_embeddings()(tokens)
    inputs_embeds = torch.cat((tok_embs[:, 0:1, :], img_embs, tok_embs[:, 1:, :]), dim=1)

    outputs = moondream.text_model(
        inputs_embeds=inputs_embeds,
        labels=labels,
        attention_mask=attn_mask,
    )

    return outputs.loss

def lr_schedule(step, max_steps):
    x = step / max_steps
    if x < 0.1:
        return 0.1 * LR + 0.9 * LR * x / 0.1
    else:
        return 0.1 * LR + 0.9 * LR * (1 + math.cos(math.pi * (x - 0.1))) / 2

print("Loading data")
dataset = CustomDataset()
print("Dataset loaded")
train_dataset, val_dataset, test_dataset = dataset.get_datasets()
train_loader, val_loader, test_loader = dataset.get_dataloaders(BATCH_SIZE, collate_fn)
print("Loaders loaded")
train_len = 29000
val_len = 1014
test_len = 1000
print("Data loaded")


moondream.text_model.train()
moondream.text_model.transformer.gradient_checkpointing_enable()

total_steps = EPOCHS * train_len // GRAD_ACCUM_STEPS
optimizer = Adam8bit(
    [
        {"params": moondream.text_model.parameters()},
    ],
    lr=LR * 0.1,
    betas=(0.9, 0.95),
    eps=1e-6
)

def validate(val_loader):
    moondream.text_model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            val_loss = compute_loss(batch)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / val_len
    return avg_val_loss

best_val_loss = float('inf')
i = 0
for epoch in range(EPOCHS):
    moondream.text_model.train()  # Set model back to training mode
    epoch_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        i += 1

        loss = compute_loss(batch)
        loss.backward()

        if i % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

            lr = lr_schedule(i / GRAD_ACCUM_STEPS, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        epoch_loss = loss.item()
    
    print(f"\nEpoch {epoch + 1}/{EPOCHS}, Training Loss: {epoch_loss:.4f}")

    # Run validation after each epoch
    avg_val_loss = validate(val_loader)
    print(f"Validation Loss after epoch {epoch + 1}: {avg_val_loss}")

    # Save the model if validation loss improves
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        moondream.save_pretrained(OUTPUT_DIR + f"/best_model_epoch_{epoch + 1}")
        tokenizer.save_pretrained(OUTPUT_DIR + f"/tokenizer_epoch_{epoch + 1}")
        print(f"Best model saved at epoch {epoch + 1}")

moondream.save_pretrained(OUTPUT_DIR + "/model")
tokenizer.save_pretrained(OUTPUT_DIR + "/tokenizer")
print("Model saved")
