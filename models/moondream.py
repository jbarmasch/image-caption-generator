import torch
import json
import os
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

class MoondreamCaptioner:
    def __init__(self, torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), dtype = torch.float32):
        print("CUDA available: ", torch.cuda.is_available())
        self._device = torch_device
        print("Loading tokenizer...")
        self._moondream_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-05-20")
        print("Tokenizer loaded")
        print("Loading model...")
        print(f"Dtype: {dtype}")
        self._moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True, revision="2024-05-20", torch_dtype = dtype, attn_implementation="flash_attention_2"
        ).to(self._device)
        print("Model loaded")

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
    

    def fine_tune2(self, tokenizer, train_loader, val_loader, num_epochs=5, learning_rate=5e-5):
        device = self._device
        self._moondream_model.to(device)
    
        optimizer = torch.optim.AdamW(self._moondream_model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            self._moondream_model.train()
            total_loss = 0

            for images, captions in train_loader:
                images = images.to(device)
                captions = captions.to(device)

                optimizer.zero_grad()
                outputs = self._moondream_model.text_model(images, labels=captions)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

            # Validation
            self._moondream_model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for images, captions in val_loader:
                    images = images.to(device)
                    captions = captions.to(device)

                    outputs = self._moondream_model.text_model(images, labels=captions)
                    val_loss = outputs.loss
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

        return self._moondream_model

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