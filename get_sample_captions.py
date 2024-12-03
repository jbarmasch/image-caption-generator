import torch
import time
from pathlib import Path
from PIL import Image
# from models.llava import llavaCaptioner
from models.blip import BLIPCaptioner
from models.moondream import MoondreamCaptioner
from utils import compute_statistics

def generate_captions(image_paths, captioners, get_statistics = False):
    # Create a dictionary to store the results
    results = {}
    
    # Iterate over all images and compare captions
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        print("="*50)
        print(f"Image: {image_path}")
        print()
        
        # Generate captions
        for captioner in captioners:
            start_time = time.time()
            caption = captioner.get_caption(image, temperature=1)
            end_time = time.time()
            total_time = end_time - start_time
            
            # Store the results
            if captioner not in results:
                results[captioner] = []
            results[captioner].append((image_path, caption, total_time))

            # Print results
            print(f"Results for {captioner.__class__.__name__}:")
            print(f"Caption: {caption}")
            print(f"Time taken: {total_time}")
            print()
        
        print("="*50)
    
    # Compute statistics if needed
    if get_statistics:
        for captioner, captions in results.items():
            times = [time for _, _, time in captions]
            stats = compute_statistics(times)
            print(f"Statistics for {captioner.__class__.__name__}:")
            print(stats)
            print("="*50)

if __name__ == "__main__":
    # torch_device = torch.device("cpu") # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the image paths
    # src_directory = Path("F:\\Datasets\\archive\\flickr30k_images\\flickr30k_images\\flickr30k_images")
    # dest_directory = Path("./data/images/groups")
    # group_pictures(src_directory, dest_directory)
    image_dir = Path("data/images/processed")
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpeg"))
    
    # Initialize the captioners
    # llava = llavaCaptioner()
    # blip = BLIPCaptioner(torch_device=torch_device)
    # print("BLIP model loaded")
    moondream = MoondreamCaptioner(model_path=Path("F:/Storage de fine tunings/BUENOS/1epochs_8batch_3e-5lr/model"), tokenizer_path=Path("F:/Storage de fine tunings/BUENOS/1epochs_8batch_3e-5lr/tokenizer"))
    print("Moondream model loaded")
    
    # Generate captions and compare performance
    generate_captions(image_paths, [moondream], get_statistics=False)