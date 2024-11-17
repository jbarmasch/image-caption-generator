import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import *
from rouge_metric import PyRouge
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor
from nltk import word_tokenize
from pathlib import Path

class MoondreamCaptioner:
    def __init__(self, torch_device = DEVICE, dtype = DTYPE, model_path = None, tokenizer_path = None):
        print("CUDA available: ", torch.cuda.is_available())
        self._device = torch_device
        print("Loading tokenizer...")
        if (tokenizer_path is None):
            self._moondream_tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL, revision=MOONDREAM_VERSION)
        else:
            print("Loading from file")
            self._moondream_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("Tokenizer loaded")
        print("Loading model...")
        print(f"Dtype: {dtype}")
        if (model_path is None):
            self._moondream_model = AutoModelForCausalLM.from_pretrained(ORIGINAL_MODEL, revision=MOONDREAM_VERSION, trust_remote_code=True, torch_dtype=dtype, device_map={"": self._device})
        else:
            print("Loading from file")
            self._moondream_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config= model_path / Path("config.json"),
                state_dict=None,
                trust_remote_code=True,
                torch_dtype=dtype,
                # ignore_mismatched_sizes=True
            ).to(torch.device(self._device))
        print("Model loaded")

        self._default_prompt = "Describe this image in a short yet informative sentence. It must be a concise caption, ignore the background."

        self.metric_logs = {
            "bleu": [],
            "meteor": [],
            "rouge": []
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

    def get_caption(self, image, prompt=None, temperature=None):
        if prompt is None:
            prompt = self._default_prompt
        enc_image = self._moondream_model.encode_image(image).to(torch.device(self._device))   
        if temperature is None:
            return self._moondream_model.answer_question(enc_image, prompt, self._moondream_tokenizer)
        else:
            return self._moondream_model.answer_question(enc_image, prompt, self._moondream_tokenizer, temperature=temperature, do_sample=True)
    
    def get_rouge_metrics(self, hypothesis, references):
        rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True, rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
        scores = rouge.evaluate([hypothesis], [references])
        self.metric_logs["rouge"].append(scores)
        return scores
    
    def get_bleu_metrics(self, hypothesis, references):
        references = [word_tokenize(reference) for reference in references]
        hypothesis = word_tokenize(hypothesis)
        score = sentence_bleu(references, hypothesis)
        self.metric_logs["bleu"].append(score)
        return score
    
    def get_meteor_metrics(self, hypothesis, references):
        references = [word_tokenize(reference) for reference in references]
        hypothesis = word_tokenize(hypothesis)
        meteor_score = round(meteor(references, hypothesis), 4)
        return meteor_score