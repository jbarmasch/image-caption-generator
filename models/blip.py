import torch
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

class BLIPCaptioner:
    def __init__(self, torch_device = torch.device("cpu")):
        self._device = torch_device
        self._blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self._blip_model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b").to(self._device)
        self._default_prompt = "Question: Can you give a long, extremely detailed description of every aspect in the image? Answer:"

    # Getter for blip_processor
    @property
    def blip_processor(self):
        return self._blip_processor
    
    # Getter for blip_model
    @property
    def blip_model(self):
        return self._blip_model
    
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
    def get_caption(self, image, requested_prompt = None):
        if requested_prompt is None:
            requested_prompt = self._default_prompt
        inputs = self._blip_processor(images=image, prompt=requested_prompt, return_tensors="pt").to(self._device)
        outputs = self._blip_model.generate(**inputs).to(self._device)
        return self._blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
