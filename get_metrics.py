from tokenizers.pre_tokenizers import Whitespace
from models.moondream import MoondreamCaptioner
from dataset import CustomDataset
from config import *
from tqdm import tqdm
from utils import *

captioner = MoondreamCaptioner()
moondream = captioner._moondream_model
tokenizer = captioner._moondream_tokenizer
tokenizer.pre_tokenizer = Whitespace()

dataset = CustomDataset()
train_dataset, val_dataset, test_dataset = dataset.get_datasets()

i = 0
metrics = {}
rouge_metrics = []
bleu_metrics = []
meteor_metrics = []
for sample in tqdm(test_dataset, desc=f"Iterating over {MAX_METRIC_ITER} photos"):
    if i >= MAX_METRIC_ITER:
        break
    image = sample['image']
    image_name = sample['filename']
    moondream_caption = captioner.get_caption(image)
    hypothesis = moondream_caption
    references = sample['caption']

    rouge_metrics.append(captioner.get_rouge_metrics(hypothesis, references))
    bleu_metrics.append(captioner.get_bleu_metrics(hypothesis, references))
    meteor_metrics.append(captioner.get_meteor_metrics(hypothesis, references))

    if image_name not in metrics:
        metrics[image_name] = {}
    metrics[image_name] = {
        "rouge": rouge_metrics[i],
        "bleu": bleu_metrics[i],
        "meteor": meteor_metrics[i]
    }
    i += 1

save_single_metric(metric_name = 'rouge', metric=rouge_metrics)
save_single_metric(metric_name = 'bleu', metric=bleu_metrics)
save_single_metric(metric_name = 'meteor', metric=meteor_metrics)
save_metrics(metrics=metrics)
