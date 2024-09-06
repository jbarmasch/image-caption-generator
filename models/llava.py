import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": "What can you see in the image?"},
                {"type": "image"},
            ],
        },  
    ]


class llavaCaptioner:
    def __init__(self, torch_device = torch.device("cpu")):
        self._device = torch_device
        self._llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        self._llava_model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf", 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(self._device)
        self._default_prompt = self._llava_processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    # Getter for llava_processor
    @property
    def llava_processor(self):
        return self._llava_processor
    
    # Getter for llava_model
    @property
    def llava_model(self):
        return self._llava_model
    
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
    def get_caption(self, image, prompt = None):
        if prompt is None:
            prompt = self._default_prompt
        inputs = self._llava_processor(images=image, text=prompt, return_tensors='pt').to(self._device, torch.float16)
        output = self._llava_model.generate(**inputs, max_new_tokens=200, do_sample=False)
        return output
