import json
from utils import compute_statistics, plot_time_stats, plot_stats, plot_final_stats, load_metrics
from config import *
from pathlib import Path

METRICS = ['Bleu', 'Meteor', 'Rouge']
VARIABLE = {
    'EPOCHS' : [1, 2, 3],
    'BATCH_SIZE' : [2, 4, 8, 16, 32],
    'LR' : [3e-7, 3e-6, 3e-5, 3e-4],
}

def hyperparameter_graphs(temperatures = [0]):
    for metric in METRICS:
        for temperature in temperatures:
            for var in VARIABLE:
                stats = {}
                for value in VARIABLE[var]:
                    path = METRICS_DIR / Path(f'{metric.lower()}')
                    if var == 'EPOCHS':
                        path = path / Path(f'{metric.lower()}_metrics_{value}epochs_{BATCH_SIZE}batchsize_{LR}lr_{temperature}temperature.json')
                        data = load_metrics(path, metric)
                    elif var == 'BATCH_SIZE':
                        path = path / Path(f'{metric.lower()}_metrics_{EPOCHS}epochs_{value}batchsize_{LR}lr_{temperature}temperature.json')
                        data = load_metrics(path, metric)
                    elif var == 'LR':
                        path = path / Path(f'{metric.lower()}_metrics_{EPOCHS}epochs_{BATCH_SIZE}batchsize_{value}lr_{temperature}temperature.json')
                        data = load_metrics(path, metric)
                    stats[f'{value}'] = compute_statistics(data)
                ori_stats = compute_statistics(load_metrics(METRICS_DIR / Path(f'{metric.lower()}/original_{temperature}temperature.json'), metric))
                plot_stats(var, metric, stats, temperature, ori_stats['average'], ori_stats['standard_deviation'])

def time_graphs(temperatures = [0]):
    for temperature in temperatures:
        for var in VARIABLE:
            stats = {}
            for value in VARIABLE[var]:
                execution_times = []
                path = METRICS_DIR
                if var == 'EPOCHS':
                    path = path / Path(f'metrics_{value}epochs_{BATCH_SIZE}batchsize_{LR}lr_{temperature}temperature.json')
                    with open(path, 'r') as f:
                        data = json.load(f)
                        for key in data:
                            execution_times.append(data[key]['time_taken'])
                elif var == 'BATCH_SIZE':
                    path = path / Path(f'metrics_{EPOCHS}epochs_{value}batchsize_{LR}lr_{temperature}temperature.json')
                    with open(path, 'r') as f:
                        data = json.load(f)
                        for key in data:
                            execution_times.append(data[key]['time_taken'])
                elif var == 'LR':
                    path = path / Path(f'metrics_{EPOCHS}epochs_{BATCH_SIZE}batchsize_{value}lr_{temperature}temperature.json')
                    with open(path, 'r') as f:
                        data = json.load(f)
                        for key in data:
                            execution_times.append(data[key]['time_taken'])
                stats[f'{value}'] = compute_statistics(execution_times)
            ori_execution_times = []
            with open(METRICS_DIR / Path(f'original_{temperature}temperature.json'), 'r') as f:
                ori_data = json.load(f)
                for key in ori_data:
                    ori_execution_times.append(ori_data[key]['time_taken'])
            ori_stats = compute_statistics(ori_execution_times)
            plot_time_stats(var, 'Captioning time', stats, temperature, ori_stats['average'], ori_stats['standard_deviation'])

def final_comparison(temperatures = [0]):
    for metric in METRICS:
        for temperature in temperatures:
            stats = {}
            for var in VARIABLE:
                path = METRICS_DIR / Path(f'{metric.lower()}')
                stats['candidate'] = compute_statistics(load_metrics(path / Path(f'{metric.lower()}_metrics_2epochs_8batchsize_3e-06lr_0temperature.json'), metric))
                if var == 'EPOCHS':
                    path = path / Path(f'{metric.lower()}_metrics_2epochs_{BATCH_SIZE}batchsize_{LR}lr_{temperature}temperature.json')
                    data = load_metrics(path, metric)
                elif var == 'BATCH_SIZE':
                    path = path / Path(f'{metric.lower()}_metrics_{EPOCHS}epochs_8batchsize_{LR}lr_{temperature}temperature.json')
                    data = load_metrics(path, metric)
                elif var == 'LR':
                    path = path / Path(f'{metric.lower()}_metrics_{EPOCHS}epochs_{BATCH_SIZE}batchsize_3e-06lr_{temperature}temperature.json')
                    data = load_metrics(path, metric)
                stats[f'best_{var}'] = compute_statistics(data)
                ori_stats = compute_statistics(load_metrics(METRICS_DIR / Path(f'{metric.lower()}/original_{temperature}temperature.json'), metric))
            plot_stats('Models', metric, stats, temperature, ori_stats['average'], ori_stats['standard_deviation'])

def final_time(temperatures = [0]):
    for temperature in temperatures:
        stats = {}
        execution_times = []
        with open(METRICS_DIR / Path(f'metrics_2epochs_8batchsize_3e-06lr_0temperature.json'), 'r') as f:
                data = json.load(f)
                for key in data:
                    execution_times.append(data[key]['time_taken'])
        stats['candidate'] = compute_statistics(execution_times)
        for var in VARIABLE:
            path = METRICS_DIR
            execution_times = []
            if var == 'EPOCHS':
                path = path / Path(f'metrics_2epochs_{BATCH_SIZE}batchsize_{LR}lr_{temperature}temperature.json')
            elif var == 'BATCH_SIZE':
                path = path / Path(f'metrics_{EPOCHS}epochs_8batchsize_{LR}lr_{temperature}temperature.json')
            elif var == 'LR':
                path = path / Path(f'metrics_{EPOCHS}epochs_{BATCH_SIZE}batchsize_3e-06lr_{temperature}temperature.json')
            with open(path, 'r') as f:
                data = json.load(f)
                for key in data:
                    execution_times.append(data[key]['time_taken'])
            stats[f'best_{var}'] = compute_statistics(execution_times)
            ori_execution_times = []
            with open(METRICS_DIR / Path(f'original_{temperature}temperature.json'), 'r') as f:
                ori_data = json.load(f)
                for key in ori_data:
                    ori_execution_times.append(ori_data[key]['time_taken'])
            ori_stats = compute_statistics(ori_execution_times)
        plot_time_stats('Models', 'Captioning time', stats, temperature, ori_stats['average'], ori_stats['standard_deviation'])

if __name__ == "__main__":
    # hyperparameter_graphs()
    # time_graphs()
    # final_comparison()
    final_time()