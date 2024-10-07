import os
import shutil
from config import *
from pathlib import Path
import numpy as np

def save_metrics(metrics, output_file=METRICS_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    """
    Save metrics to a JSON file.
    
    Parameters:
        metrics (dict): Dictionary containing metrics.
        output_file (str): Path to the output file.
    """
    output_file = output_file / Path(f"metrics_{epochs}epochs_{batch_size}batchsize_{lr}lr.json")
    import json
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

def save_single_metric(metric_name, metric, output_file=METRICS_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR):
    """
    Save ROUGE metrics to a JSON file.
    
    Parameters:
        rouge_metrics (dict): Dictionary containing ROUGE metrics.
        output_file (str): Path to the output file.
    """
    output_file = output_file / Path(f"/{metric_name}/{metric_name}_metrics_{epochs}epochs_{batch_size}batchsize_{lr}lr.json")
    import json
    with open(output_file, "w") as f:
        json.dump(metric, f, indent=4)



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
