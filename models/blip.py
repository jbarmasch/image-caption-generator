import torch
from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering

class BLIPCaptioner:
    def __init__(self, device = torch.device("cpu")):
        self.device = device
        self.blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b").to(self.device)
        self.default_prompt = "Question: Can you give a long, extremely detailed description of every aspect in the image? Answer:"

    # Getter for blip_processor
    @property
    def blip_processor(self):
        return self.blip_processor
    
    # Getter for blip_model
    @property
    def blip_model(self):
        return self.blip_model
    
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
    def get_caption(self, image, requested_prompt = None):
        if requested_prompt is None:
            requested_prompt = self.default_prompt
        inputs = self.blip_processor(image, prompt = requested_prompt, return_tensors="pt")
        outputs = self.blip_model(**inputs)
        return self.blip_processor.decode(outputs["output"], skip_special_tokens=True)
