import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MoondreamCaptioner:
    def __init__(self, device = torch.device("cpu")):
        self.device = device
        self.moondream_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-07-23")
        self.moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True, revision="2024-07-23"
        ).to(self.device)
        self.default_prompt = "Describe this image in a short yet informative sentence. It must be a concise caption, ignore the background."

    # Getter for moondream_tokenizer
    @property
    def moondream_tokenizer(self):
        return self.moondream_tokenizer

    # Getter for moondream_model
    @property
    def moondream_model(self):
        return self.moondream_model

    # Getter for default_prompt
    @property
    def default_prompt(self):
        return self.default_prompt
    
    # Getter for device
    @property
    def device(self):
        return self.device

    # Setter for default_prompt
    @default_prompt.setter
    def default_prompt(self, prompt):
        self.default_prompt = prompt

    # Setter for device
    @device.setter
    def device(self, device):
        self.device = device

    # Method to generate a caption
    def get_caption(self, image, prompt=None):
        if prompt is None:
            prompt = self.default_prompt
        enc_image = self.moondream_model.encode_image(image)
        return self.moondream_model.answer_question(enc_image, prompt, self.moondream_tokenizer)
