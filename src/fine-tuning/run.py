import torch
import time
from pathlib import Path
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration, AutoModelForVisualQuestionAnswering, AutoTokenizer, LlavaForConditionalGeneration
from PIL import Image

# Check for GPU availability
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("device: " + str(device))
#torch.cuda.empty_cache()

# Define paths
image_groups = Path('./data/images/groups')  # Path for grouped images
image_dir = Path('./data/images/processed')  # Path for images

# Grab all image files with common image extensions
image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.jpeg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.bmp')) + list(image_dir.glob('*.gif'))

if not image_paths:
    raise FileNotFoundError(f"No image files found in {image_dir}")

print("Images loaded")

import numpy as np

def compute_statistics(execution_times):
    """
    Computes various statistics from an array of execution times.
    
    Parameters:
        execution_times (list or numpy array): Array of execution times.
        
    Returns:
        dict: Dictionary containing the computed statistics.
    """
    # Convert to a numpy array if it's not already
    execution_times = np.array(execution_times)
    
    # Calculate the statistics
    stats = {
        "average": np.mean(execution_times),
        "median": np.median(execution_times),
        "standard_deviation": np.std(execution_times),
        "variance": np.var(execution_times),
        "min": np.min(execution_times),
        "max": np.max(execution_times),
        "count": len(execution_times),
        "percentile_25": np.percentile(execution_times, 25),
        "percentile_75": np.percentile(execution_times, 75),
    }
    
    return stats

# Load LLava model and processor
#llava_processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
#llava_model = LlavaForConditionalGeneration.from_pretrained(
#    "llava-hf/llava-1.5-7b-hf", 
#    torch_dtype=torch.float16, 
#    low_cpu_mem_usage=True, 
#)

# Load BLIP-2 model
torch.cuda.empty_cache()
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-2.7b")

print("BLIP loaded")
# Load Moondream
moondream_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-07-23")

moondream_model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2", trust_remote_code=True, revision="2024-07-23"
    )


print("Moondream loaded")
# Function to generate captions using LLava
#def llava_caption(image):
    # Define a chat histiry and use apply_chat_template to get correctly formatted prompt
    # Each value in "content" has to be a list of dicts with types ("text", "image") 
#    conversation = [
#        {

#        "role": "user",
#        "content": [
#            {"type": "text", "text": "What can you see in the image?"},
#            {"type": "image"},
#        ],
#        },  
#    ]
#    prompt = llava_processor.apply_chat_template(conversation, add_generation_prompt=True)

#    inputs = llava_processor(images=image, text=prompt, return_tensors='pt').to(0, torch.float16)

#    output = llava_model.generate(**inputs, max_new_tokens=200, do_sample=False)
#    return output

# Function to generate captions using BLIP-2
def blip_caption(image):
    question = "Question: Can you give a long, extremely detailed description of every aspect in the image? Answer:"
    inputs = blip_processor(images=image, prompt=question, return_tensors="pt").to(device)
    outputs = blip_model.generate(**inputs).to(device)
    return blip_processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

def moondream_caption(image):
    enc_image = moondream_model.encode_image(image)
    return moondream_model.answer_question(enc_image, "Describe this image in a short yet informative sentence. It must be a concise caption, ignore the background", moondream_tokenizer) #

blip_times = []
moondream_times = []
for folder in image_groups.iterdir():
    if folder.is_dir():
        print(folder)
        for image in folder.iterdir():
            print(image)
            image = Image.open(image).convert("RGB")
            #start_llava = time.time()
            # llava_result = llava_caption(image)
            #end_llava = time.time()
            #total_llava = end_llava - start_llava
            start_blip = time.time()
            blip_result = blip_caption(image)
            end_blip = time.time()
            total_blip = end_blip - start_blip
            blip_times.append(total_blip)
            start_moon = time.time()
            moondream_result = moondream_caption(image)
            end_moon = time.time()
            total_moon = end_moon - start_moon
            moondream_times.append(total_moon)
            print("="*50)
            # Print results
            print(f"Image: {image}")
            #print(f"LLava Caption: {llava_result}")
            #print("Caption obtained in: " + str(total_llava))
            #print("-"*50)
            print(f"BLIP-2 Caption: {blip_result}")
            print("Caption obtained in: " + str(total_blip))
            print("-"*50)
            print(f"Moondream Caption: {moondream_result}")
            print("Caption obtained in: " + str(total_moon))
            print("="*50)

blip_stats = compute_statistics(blip_times)
moondream_stats = compute_statistics(moondream_times)
print("BLIP-2 Statistics:")
print(blip_stats)
print("Moondream Statistics:")
print(moondream_stats)
exit()

# Iterate over all images and compare captions
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    
    # Generate captions
    start_llava = time.time()
    llava_result = llava_caption(image)
    end_llava = time.time()
    total_llava = end_llava - start_llava
    start_blip = time.time()
    blip_result = blip_caption(image)
    end_blip = time.time()
    total_blip = end_blip - start_blip
    start_moon = time.time()
    moondream_result = moondream_caption(image)
    end_moon = time.time()
    total_moon = end_moon - start_moon
    print("="*50)
    # Print results
    print(f"Image: {image_path}")
    print(f"LLava Caption: {llava_result}")
    print("Caption obtained in: " + str(total_llava))
    print("-"*50)
    print(f"BLIP-2 Caption: {blip_result}")
    print("Caption obtained in: " + str(total_blip))
    print("-"*50)
    print(f"Moondream Caption: {moondream_result}")
    print("Caption obtained in: " + str(total_moon))
    print("="*50)