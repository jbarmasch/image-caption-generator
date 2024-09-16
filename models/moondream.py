import torch
import json
import os
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset


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
        dataset = load_dataset(dataset_name)

        def preprocess_function(examples):
            inputs = self._moondream_tokenizer(examples['caption'], truncation=True, padding='max_length', max_length=128)
            return inputs

        encoded_dataset = dataset.map(preprocess_function, batched=True)

        train_dataset = encoded_dataset['train'].remove_columns(['caption']).with_format('torch')
        val_dataset = encoded_dataset['validation'].remove_columns(['caption']).with_format('torch')

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="steps",
            eval_steps=500,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            logging_dir='./Training results/logs',
            logging_steps=10,
            save_steps=save_steps,
            save_total_limit=3,
            save_strategy="epoch",
            learning_rate=5e-5,
            weight_decay=0.01,
        )

        trainer = Trainer(
            model=self._moondream_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self._moondream_tokenizer,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        self._moondream_model.save_pretrained(output_dir)
        self._moondream_tokenizer.save_pretrained(output_dir)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self._moondream_tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self._moondream_tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_labels = [[label] for label in decoded_labels]

        bleu_result = self.bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
        meteor_result = self.meteor_metric.compute(predictions=decoded_preds, references=decoded_labels)
        rouge_result = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

        metrics = {
            "bleu": bleu_result["bleu"],
            "meteor": meteor_result["meteor"],
            "rougeL": rouge_result["rougeL"].mid.fmeasure
        }

        self.save_metrics_to_file(metrics)

        return metrics

    def save_metrics_to_file(self, metrics, output_file="metrics_log.json"):
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        data.append(metrics)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=4)