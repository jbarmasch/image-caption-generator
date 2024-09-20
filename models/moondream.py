import torch
import json
import os
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader

class MoondreamCaptioner:
    def __init__(self, torch_device = torch.device("cpu")):
        self._device = torch_device
        self._moondream_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-07-23")
        self._moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True, revision="2024-07-23"
        ).to(self._device)

        self.bleu_metric = evaluate.load("bleu")
        self.meteor_metric = evaluate.load("meteor")
        self.rouge_metric = evaluate.load("rouge")

        self._default_prompt = "Describe this image in a short yet informative sentence. It must be a concise caption, ignore the background."

        self.metric_logs = {
            "bleu": [],
            "meteor": [],
            "rougeL": []
        }

    @property
    def moondream_tokenizer(self):
        return self._moondream_tokenizer

    @property
    def moondream_model(self):
        return self._moondream_model

    @property
    def default_prompt(self):
        return self._default_prompt
    
    @property
    def device(self):
        return self._device

    @default_prompt.setter
    def default_prompt(self, prompt):
        self._default_prompt = prompt

    @device.setter
    def device(self, device):
        self._device = device

    def get_caption(self, image, prompt=None):
        if prompt is None:
            prompt = self._default_prompt
        enc_image = self._moondream_model.encode_image(image)
        return self._moondream_model.answer_question(enc_image, prompt, self._moondream_tokenizer)
    
    def fine_tune(self, dataset_name='flickr30k', output_dir='./Training results/Weights/Moondream', num_train_epochs=3, batch_size=4, save_steps=500):
        # Load the train and validation datasets with streaming enabled
        train_dataset = load_dataset(dataset_name, split='train', streaming=True)
        val_dataset = load_dataset(dataset_name, split='validation', streaming=True)

        def preprocess_function(examples):
            inputs = self._moondream_tokenizer(examples['caption'], truncation=True, padding='max_length', max_length=128)
            return inputs

        # Preprocess datasets in streaming mode
        processed_train_dataset = train_dataset.map(preprocess_function)
        processed_val_dataset = val_dataset.map(preprocess_function)

        # Create DataLoaders for batch processing
        train_dataloader = DataLoader(processed_train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(processed_val_dataset, batch_size=batch_size)

        optimizer = AdamW(self._moondream_model.parameters(), lr=5e-5)
        num_training_steps = num_train_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        self._moondream_model.train()

        for epoch in range(num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(self._device) for k, v in batch.items()}
                outputs = self._moondream_model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step % save_steps == 0:
                    self._moondream_model.eval()
                    self.compute_metrics(val_dataloader)
                    self._moondream_model.train()

            self._moondream_model.save_pretrained(output_dir)
            self._moondream_tokenizer.save_pretrained(output_dir)

        self.save_metrics_to_file(self.metric_logs, output_file="metrics_log.json")

    def compute_metrics(self, val_dataloader):
        decoded_preds = []
        decoded_labels = []

        for val_batch in val_dataloader:
            val_batch = {k: v.to(self._device) for k, v in val_batch.items()}
            with torch.no_grad():
                outputs = self._moondream_model(**val_batch)
                predictions = outputs.logits.argmax(dim=-1)

                decoded_preds.extend(self._moondream_tokenizer.batch_decode(predictions, skip_special_tokens=True))
                decoded_labels.extend(self._moondream_tokenizer.batch_decode(val_batch['input_ids'], skip_special_tokens=True))

        decoded_labels = [[label] for label in decoded_labels]

        bleu_result = self.bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_result = self.meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_result = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        self.metric_logs["bleu"].append(bleu_result["bleu"])
        self.metric_logs["meteor"].append(meteor_result["meteor"])
        self.metric_logs["rougeL"].append(rouge_result["rougeL"].mid.fmeasure)

    def save_metrics_to_file(self, metrics, output_file="metrics_log.json"):
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(metrics)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)