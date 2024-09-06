import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class MoondreamCaptioner:
    def __init__(self, torch_device = torch.device("cpu")):
        self._device = torch_device
        self._moondream_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-07-23")
        self._moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True, revision="2024-07-23"
        ).to(self._device)
        self._default_prompt = "Describe this image in a short yet informative sentence. It must be a concise caption, ignore the background."

    # Getter for moondream_tokenizer
    @property
    def moondream_tokenizer(self):
        return self._moondream_tokenizer

    # Getter for moondream_model
    @property
    def moondream_model(self):
        return self._moondream_model

    # Getter for default_prompt
    @property
    def default_prompt(self):
        return self._default_prompt
    
    # Getter for device
    @property
    def device(self):
        return self._device

    # Setter for default_prompt
    @default_prompt.setter
    def default_prompt(self, prompt):
        self._default_prompt = prompt

    # Setter for device
    @device.setter
    def device(self, device):
        self._device = device

    # Method to generate a caption
    def get_caption(self, image, prompt=None):
        if prompt is None:
            prompt = self._default_prompt
        enc_image = self._moondream_model.encode_image(image)
        return self._moondream_model.answer_question(enc_image, prompt, self._moondream_tokenizer)
