from tokenizers.pre_tokenizers import Whitespace
from models.moondream import MoondreamCaptioner
from dataset import CustomDataset
from config import *
from tqdm import tqdm
from utils import *
import torch

captioner = MoondreamCaptioner()
moondream = captioner._moondream_model
tokenizer = captioner._moondream_tokenizer
tokenizer.pre_tokenizer = Whitespace()

def collate_fn(batch):
    images = [sample['image'] for sample in batch]
    images = [moondream.vision_encoder.preprocess(image) for image in images]

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
        images,
        torch.stack([torch.tensor(t, dtype=torch.long) for t in tokens_acc]),
        torch.stack([torch.tensor(l, dtype=torch.long) for l in labels_acc]),
        torch.stack([torch.tensor(a, dtype=torch.bool) for a in attn_mask_acc]),
    )

dataset = CustomDataset()
train_dataset, val_dataset, test_dataset = dataset.get_datasets()
train_loader, val_loader, test_loader = dataset.get_dataloaders(BATCH_SIZE, collate_fn)

i = 1
metrics = {}
rouge_metrics = []
bleu_metrics = []
meteor_metrics = []
for batch in tqdm(test_loader, desc=f"Iter {i}/{MAX_METRIC_ITER}"):
    if i > MAX_METRIC_ITER:
        break
    image = batch['image']
    image_name = batch['filename']
    hypothesis = captioner.get_caption(image)
    references = batch['caption']
    rouge_metrics = rouge_metrics.append(captioner.get_rouge_metrics(hypothesis, references))
    # bleu_metrics = bleu_metrics + captioner.get_bleu_metrics(hypothesis, references)
    # meteor_metrics = meteor_metrics + captioner.get_meteor_metrics(hypothesis, references)
    
    # Create a dictionary to store the results of every metric for an image name as its key
    if image_name not in metrics:
        metrics[image_name] = {}
    metrics[image_name] = {
        "rouge": rouge_metrics,
        # "bleu": bleu_metrics,
        # "meteor": meteor_metrics
    }

    save_rouge_metrics(rouge_metrics=rouge_metrics)
    # save_bleu_metrics(bleu_metrics=bleu_metrics)
    # save_meteor_metrics(meteor_metrics=meteor_metrics)
    i += 1

save_metrics(metrics=metrics)

# captioner.compute_metrics(test_loader)
# captioner.save_metrics_to_file(captioner.metric_logs)
