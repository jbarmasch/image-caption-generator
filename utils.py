import os
import shutil
from config import *
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches 
import json

def load_metrics(path, metric):
    with open(path) as f:
        if (metric == 'Rouge'):
            return extract_rouge_values_from_list(path)
        else:
            return json.load(f)

def extract_rouge_values_from_list(json_path, rouge_variant="rouge-l", metric="f"):
    # Load the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Extract the specified metric values for the given rouge variant
    values = []
    for item in data:
        if rouge_variant in item:
            metric_value = item[rouge_variant].get(metric)
            if metric_value is not None:
                values.append(metric_value)
    
    return values

def plot_stats(x_label, y_label, stats_dict, temp, reference_mean=None, reference_stddev=None):
    # Extract x values, means, and standard deviations from the dictionary
    x_values = list(stats_dict.keys())
    means = [stats_dict[x]["average"] for x in x_values]
    std_devs = [stats_dict[x]["standard_deviation"] for x in x_values]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the means with error bars representing standard deviation
    errorbar_handle = plt.errorbar(x_values, means, yerr=std_devs, fmt='o', capsize=5, label='Mean ± Standard Deviation', color='b')

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')

    # Set Y-axis limits from 0 to 1
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.1, 0.1))

    for i, (x, mean) in enumerate(zip(x_values, means)):
        plt.text(x, mean, f'{mean:.3f}', fontsize=9, ha='right', va='bottom', color='blue')

    # If reference_mean and reference_stddev are provided, plot the reference line and shaded area
    if reference_mean is not None and reference_stddev is not None:
        # Draw the reference mean as a red dotted line
        mean_line_handle = plt.axhline(y=reference_mean, color='red', linestyle='--', label='Original Model\'s Mean')

        # Draw lines for one standard deviation above and below the reference mean
        plt.axhline(y=reference_mean + reference_stddev, color='red', linestyle=':', 
                    linewidth=1.5, label='Original Model\'s Std Deviation')
        plt.axhline(y=reference_mean - reference_stddev, color='red', linestyle=':', 
                    linewidth=1.5)
        
        plt.annotate(
            f'{reference_mean:.3f}', 
            xy=(0, reference_mean), 
            xytext=(-25, 0),  # Position text outside the plot area (left)
            textcoords='offset points', 
            color='red', 
            va='center', 
            ha='right', 
            fontsize=10
        )

    # Add grid for better readability
    plt.grid(True)

    plt.legend()

    # Save the plot to the specified directory
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)

    save_path = os.path.join(GRAPH_DIR, f'{y_label}_vs_{x_label}_{temp}temperature.png')
    plt.savefig(save_path)

    # Optionally, you can comment out plt.show() in headless environments
    # plt.show()

    print(f'Graph saved at {save_path}')

def plot_time_stats(x_label, y_label, stats_dict, temp, reference_mean=None, reference_stddev=None):
    # Extract x values, means, and standard deviations from the dictionary
    x_values = list(stats_dict.keys())
    means = [stats_dict[x]["average"] for x in x_values]
    std_devs = [stats_dict[x]["standard_deviation"] for x in x_values]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the means with error bars representing standard deviation
    errorbar_handle = plt.errorbar(x_values, means, yerr=std_devs, fmt='o', capsize=5, label='Mean ± Standard Deviation', color='b')

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')

    # Set Y-axis limits from 0 to 30
    plt.ylim(0, 60)
    plt.yticks(np.arange(0, 61, 5))

    for i, (x, mean) in enumerate(zip(x_values, means)):
        plt.text(x, mean, f'{mean:.3f}', fontsize=9, ha='right', va='bottom', color='blue')

    # If reference_mean and reference_stddev are provided, plot the reference line and shaded area
    if reference_mean is not None and reference_stddev is not None:
        # Draw the reference mean as a red dotted line
        mean_line_handle = plt.axhline(y=reference_mean, color='red', linestyle='--', label='Original Model\'s Mean')

        # Draw lines for one standard deviation above and below the reference mean
        plt.axhline(y=reference_mean + reference_stddev, color='red', linestyle=':', 
                    linewidth=1.5, label='Original Model\'s Std Deviation')
        plt.axhline(y=reference_mean - reference_stddev, color='red', linestyle=':', 
                    linewidth=1.5)
        
        plt.annotate(
            f'{reference_mean:.3f}', 
            xy=(0, reference_mean), 
            xytext=(-25, 0),  # Position text outside the plot area (left)
            textcoords='offset points', 
            color='red', 
            va='center', 
            ha='right', 
            fontsize=10
        )

    # Add grid for better readability
    plt.grid(True)

    plt.legend()

    # Save the plot to the specified directory
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)

    save_path = os.path.join(GRAPH_DIR, f'{y_label}_vs_{x_label}_{temp}temperature.png')
    plt.savefig(save_path)

    # Optionally, you can comment out plt.show() in headless environments
    # plt.show()

    print(f'Graph saved at {save_path}')

def plot_final_stats(x_label, y_label, stats_dict, temp, reference_mean=None, reference_stddev=None):
    # Extract x values, means, and standard deviations from the dictionary
    x_values = list(stats_dict.keys())
    means = [stats_dict[x]["average"] for x in x_values]
    std_devs = [stats_dict[x]["standard_deviation"] for x in x_values]

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the means with error bars representing standard deviation
    errorbar_handle = plt.errorbar(x_values, means, yerr=std_devs, fmt='o', capsize=5, label='Mean ± Standard Deviation', color='b')

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} vs {x_label}')

    # Set Y-axis limits from 0 to 30
    plt.ylim(0, 60)
    plt.yticks(np.arange(0, 61, 5))

    for i, (x, mean) in enumerate(zip(x_values, means)):
        plt.text(x, mean, f'{mean:.3f}', fontsize=9, ha='right', va='bottom', color='blue')

    # If reference_mean and reference_stddev are provided, plot the reference line and shaded area
    if reference_mean is not None and reference_stddev is not None:
        # Draw the reference mean as a red dotted line
        mean_line_handle = plt.axhline(y=reference_mean, color='red', linestyle='--', label='Original Model\'s Mean')

        # Draw lines for one standard deviation above and below the reference mean
        plt.axhline(y=reference_mean + reference_stddev, color='red', linestyle=':', 
                    linewidth=1.5, label='Original Model\'s Std Deviation')
        plt.axhline(y=reference_mean - reference_stddev, color='red', linestyle=':', 
                    linewidth=1.5)
        
        plt.annotate(
            f'{reference_mean:.3f}', 
            xy=(0, reference_mean), 
            xytext=(-25, 0),  # Position text outside the plot area (left)
            textcoords='offset points', 
            color='red', 
            va='center', 
            ha='right', 
            fontsize=10
        )

    # Add grid for better readability
    plt.grid(True)

    plt.legend()

    # Save the plot to the specified directory
    if not os.path.exists(GRAPH_DIR):
        os.makedirs(GRAPH_DIR)

    save_path = os.path.join(GRAPH_DIR, f'{y_label}_vs_{x_label}_{temp}temperature.png')
    plt.savefig(save_path)

    # Optionally, you can comment out plt.show() in headless environments
    # plt.show()

    print(f'Graph saved at {save_path}')


def save_metrics(metrics, output_file=METRICS_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, temperature=None):
    """
    Save metrics to a JSON file.
    
    Parameters:
        metrics (dict): Dictionary containing metrics.
        output_file (str): Path to the output file.
    """
    output_file = output_file / Path(f"metrics_{epochs}epochs_{batch_size}batchsize_{lr}lr_{temperature if temperature != None else '0'}temperature.json")
    import json
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)

def save_single_metric(metric_name, metric, output_file=METRICS_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, temperature=None):
    """
    Save ROUGE metrics to a JSON file.
    
    Parameters:
        rouge_metrics (dict): Dictionary containing ROUGE metrics.
        output_file (str): Path to the output file.
    """
    output_file = output_file / Path(f"{metric_name}/{metric_name}_metrics_{epochs}epochs_{batch_size}batchsize_{lr}lr_{temperature if temperature != None else '0'}temperature.json")
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
