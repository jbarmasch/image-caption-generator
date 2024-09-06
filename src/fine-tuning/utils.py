import os
import shutil
from pathlib import Path

def group_pictures(src_dir, dest_dir, group_size=5, num_groups=30):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)

    # Get all picture files from the source directory
    pictures = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    # Shuffle the list to randomize the selection if needed
    # import random
    # random.shuffle(pictures)

    # Check if there are enough pictures to create the desired number of groups
    total_needed = group_size * num_groups
    if len(pictures) < total_needed:
        raise ValueError(f"Not enough pictures to create {num_groups} groups of {group_size} pictures each. Found only {len(pictures)} pictures.")

    for i in range(num_groups):
        # Create a directory for the current group
        group_dir = os.path.join(dest_dir, f"group_{i+1}")
        os.makedirs(group_dir, exist_ok=True)
        
        # Copy the pictures for the current group
        group_pictures = pictures[i * group_size:(i + 1) * group_size]
        for pic in group_pictures:
            shutil.copy(os.path.join(src_dir, pic), os.path.join(group_dir, pic))

    print(f"Successfully created {num_groups} groups of {group_size} pictures each in {dest_dir}")

# Example usage
src_directory = Path("F:\\Datasets\\archive\\flickr30k_images\\flickr30k_images\\flickr30k_images")
dest_directory = Path("./data/images/groups")
group_pictures(src_directory, dest_directory)