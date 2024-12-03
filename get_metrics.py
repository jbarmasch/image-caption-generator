from tokenizers.pre_tokenizers import Whitespace
from models.moondream import MoondreamCaptioner
from dataset import CustomDataset
from config import *
from tqdm import tqdm
from utils import *
import time
import torch

dataset = CustomDataset()
train_dataset, val_dataset, test_dataset = dataset.get_datasets()


def original_metrics():
    captioner = MoondreamCaptioner()
    moondream = captioner._moondream_model
    tokenizer = captioner._moondream_tokenizer
    tokenizer.pre_tokenizer = Whitespace()
    i = 0
    metrics = {}
    rouge_metrics = []
    bleu_metrics = []
    meteor_metrics = []
    
    for sample in tqdm(test_dataset, desc=f"Iterating over 1000 photos for original model"):
        image = sample['image']
        image_name = sample['filename']
        start_time = time.time()
        moondream_caption = captioner.get_caption(image)
        end_time = time.time()
        time_taken = end_time - start_time
        hypothesis = moondream_caption
        references = sample['caption']
        rouge_metrics.append(captioner.get_rouge_metrics(hypothesis, references))
        bleu_metrics.append(captioner.get_bleu_metrics(hypothesis, references))
        meteor_metrics.append(captioner.get_meteor_metrics(hypothesis, references))
        if image_name not in metrics:
            metrics[image_name] = {}
        metrics[image_name] = {
            "caption": hypothesis,
            "time_taken": time_taken,
            "references": references,
            "rouge": rouge_metrics[i],
            "bleu": bleu_metrics[i],
            "meteor": meteor_metrics[i]
        }
        i += 1
    
    save_single_metric(metric_name = 'rouge', metric=rouge_metrics, temperature=0, epochs=0, batch_size=0, lr=0)
    save_single_metric(metric_name = 'bleu', metric=bleu_metrics, temperature=0, epochs=0, batch_size=0, lr=0)
    save_single_metric(metric_name = 'meteor', metric=meteor_metrics, temperature=0, epochs=0, batch_size=0, lr=0)
    save_metrics(metrics=metrics, temperature=0, epochs=0, batch_size=0, lr=0)
    rouge_metrics = []
    bleu_metrics = []
    meteor_metrics = []

def get_all_metrics(short = False, temperatures = [0]):
    parent_dir = Path('F:/Storage de fine tunings/1epochs_16batch_3e-7lr')
    for folder in parent_dir.iterdir():
        if folder.is_dir():
            torch.cuda.empty_cache()
            captioner = MoondreamCaptioner(model_path=parent_dir / Path(folder.name) / Path('model'), tokenizer_path=parent_dir / Path(folder.name) / Path('tokenizer'))
            moondream = captioner._moondream_model
            tokenizer = captioner._moondream_tokenizer
            tokenizer.pre_tokenizer = Whitespace()
            i = 0
            metrics = {}
            rouge_metrics = []
            bleu_metrics = []
            meteor_metrics = []
            for temperature in temperatures:
                for sample in tqdm(test_dataset, desc=f"Iterating over {'150' if short else '1000'} photos for model {folder.name} with temperature {temperature}"):
                    if (short and i>=150):
                        break
                    image = sample['image']
                    image_name = sample['filename']
                    start_time = time.time()
                    moondream_caption = captioner.get_caption(image)
                    end_time = time.time()
                    time_taken = end_time - start_time
                    hypothesis = moondream_caption
                    references = sample['caption']

                    rouge_metrics.append(captioner.get_rouge_metrics(hypothesis, references))
                    bleu_metrics.append(captioner.get_bleu_metrics(hypothesis, references))
                    meteor_metrics.append(captioner.get_meteor_metrics(hypothesis, references))

                    if image_name not in metrics:
                        metrics[image_name] = {}
                    metrics[image_name] = {
                        "caption": hypothesis,
                        "time_taken": time_taken,
                        "references": references,
                        "rouge": rouge_metrics[i],
                        "bleu": bleu_metrics[i],
                        "meteor": meteor_metrics[i]
                    }
                    i += 1
                i = 0
            
                parts = folder.name.split('_')

                # Extract values and remove the descriptive words
                epochs = int(parts[0].replace('epochs', ''))
                batch_size = int(parts[1].replace('batch', ''))
                lr = float(parts[2].replace('lr', ''))

                save_single_metric(metric_name = 'rouge', metric=rouge_metrics, epochs=epochs, batch_size=batch_size, lr=lr, temperature=temperature)
                save_single_metric(metric_name = 'bleu', metric=bleu_metrics, epochs=epochs, batch_size=batch_size, lr=lr, temperature=temperature)
                save_single_metric(metric_name = 'meteor', metric=meteor_metrics, epochs=epochs, batch_size=batch_size, lr=lr, temperature=temperature)
                save_metrics(metrics=metrics, epochs=epochs, batch_size=batch_size, lr=lr, temperature=temperature)
                rouge_metrics = []
                bleu_metrics = []
                meteor_metrics = []

if __name__ == "__main__":
    original_metrics()
    # get_all_metrics()