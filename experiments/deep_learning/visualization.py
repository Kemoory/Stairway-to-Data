import matplotlib.pyplot as plt
import pandas as pd
import os
from config_dl import RESULTS_DIR
import numpy as np

def plot_metrics(history, model_type, output_dir):
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Evolution - {model_type}')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_type}_loss_curve.png'))
    plt.close()

def save_results(results, output_dir=RESULTS_DIR):
    df = pd.DataFrame([{
        'Model': res['model'],
        'Avg Val Loss': np.mean([r['best_val_loss'] for r in res['results']]),
        'Total Time (s)': res['total_time'],
        'Time per Fold (s)': res['avg_time']
    } for res in results])
    
    # Comparaison des performances
    plt.figure(figsize=(10, 6))
    ax = df.plot(x='Model', y=['Avg Val Loss', 'Total Time (s)'], kind='bar', secondary_y='Total Time (s)')
    ax.set_ylabel('Loss')
    ax.right_ax.set_ylabel('Time (seconds)')
    plt.title('Comparaison des Modèles')
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.close()
    
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)