import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import CUSTOM_CMAP, RESULTS_DIR
import json

def save_fold_comparison(results, model_type, output_dir):
    """Create fold comparison visualization."""
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(3, 1, 1)
    df[['train_rmse', 'test_rmse']].plot(kind='bar', ax=plt.gca())
    plt.title(f'RMSE by Fold - {model_type.capitalize()}')
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(3, 1, 2)
    df[['train_mae', 'test_mae']].plot(kind='bar', ax=plt.gca())
    plt.title(f'MAE by Fold - {model_type.capitalize()}')
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.subplot(3, 1, 3)
    df[['train_r2', 'test_r2']].plot(kind='bar', ax=plt.gca())
    plt.title(f'R² by Fold - {model_type.capitalize()}')
    plt.ylabel('R²')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_fold_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_prediction_analysis(true, pred, model_type, output_dir):
    """Create comprehensive prediction analysis."""
    errors = np.array(pred) - np.array(true)
    
    plt.figure(figsize=(14, 10))
    
    # Scatter plot with error coloring
    plt.subplot(2, 2, 1)
    sc = plt.scatter(true, pred, c=np.abs(errors), cmap=CUSTOM_CMAP, alpha=0.6)
    plt.colorbar(sc, label='Absolute Error')
    min_val = min(min(true), min(pred))
    max_val = max(max(true), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'b--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - {model_type.capitalize()}')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(2, 2, 2)
    sns.histplot(errors, kde=True, color='orange', bins=20)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Residual plot
    plt.subplot(2, 2, 3)
    plt.scatter(true, errors, c=np.abs(errors), cmap=CUSTOM_CMAP, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Actual Value')
    plt.ylabel('Residual')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    # Error by magnitude
    plt.subplot(2, 2, 4)
    error_df = pd.DataFrame({'true': true, 'error': errors})
    error_df['true_bin'] = pd.cut(error_df['true'], bins=10)
    sns.boxplot(x='true_bin', y='error', data=error_df)
    plt.xticks(rotation=45)
    plt.xlabel('Actual Value Bins')
    plt.ylabel('Error')
    plt.title('Error Distribution by Value Range')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_prediction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def combine_and_visualize_results(output_dir=RESULTS_DIR):
    """Combine results from all models and create comprehensive visualizations."""
    model_files = [f for f in os.listdir(output_dir) if f.endswith('_results.json')]
    
    if not model_files:
        print("No model result files found.")
        return None

    combined_data = []

    for file in model_files:
        model_name = file.replace('_results.json', '')
        file_path = os.path.join(output_dir, file)

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            combined_data.append(process_evaluation(
                                item.get('image_name', 'unknown'),
                                model_name,
                                item
                            ))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    if not combined_data:
        print("No valid data found in result files.")
        return None

    df = pd.DataFrame(combined_data)

    # Adapt data structure to new features (from feature_extraction)
    if 'features' in df.columns:
        feature_columns = df['features'].apply(pd.Series)
        df = pd.concat([df, feature_columns], axis=1).drop(columns='features')

    # Calculations
    df['error'] = abs(df['prediction'] - df['ground_truth'])
    df['relative_error'] = (df['error'] / df['ground_truth'].clip(lower=1e-10)) * 100
    df['squared_error'] = df['error']**2

    combined_json_path = os.path.join(output_dir, 'combined_results.json')
    df.to_json(combined_json_path, orient='records', indent=4)

    create_model_comparison_plots(df, output_dir)
    create_error_analysis_plots(df, output_dir)
    create_per_image_analysis(df, output_dir)

    return df

def process_evaluation(image_name, model_name, eval_dict):
    return {
        'image': image_name,
        'model': model_name,
        'ground_truth': float(eval_dict.get('ground_truth', 0)),
        'prediction': float(eval_dict.get('prediction', 0)),
        'features': eval_dict.get('features', [])  # Adapted for new feature structure
    }

def create_model_comparison_plots(df, output_dir):
    """Create comparison plots between different models."""
    plt.figure(figsize=(14, 10))
    
    # Calculate model statistics
    model_stats = df.groupby('model').agg({
        'error': ['mean', 'median', 'std'],
        'relative_error': ['mean', 'median'],
        'squared_error': ['mean']
    })
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    model_stats['rmse'] = np.sqrt(model_stats['squared_error_mean'])
    
    # Plot 1: Error metrics comparison
    plt.subplot(2, 2, 1)
    model_stats[['error_mean', 'error_median', 'rmse']].plot(kind='bar', ax=plt.gca())
    plt.title('Model Error Metrics Comparison')
    plt.ylabel('Error Value')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Plot 2: Error distribution
    plt.subplot(2, 2, 2)
    sns.boxplot(x='model', y='error', data=df)
    plt.title('Error Distribution by Model')
    plt.ylabel('Absolute Error')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

def create_error_analysis_plots(df, output_dir):
    """Create detailed error analysis plots."""
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Error vs Ground Truth
    plt.subplot(2, 2, 1)
    sns.scatterplot(x='ground_truth', y='error', hue='model', data=df)
    plt.title('Error vs Actual Stair Count')
    plt.xlabel('Actual Stair Count')
    plt.ylabel('Prediction Error')
    
    # Plot 2: Error distribution
    plt.subplot(2, 2, 2)
    sns.histplot(data=df, x='error', hue='model', kde=True, bins=20)
    plt.title('Error Distribution')
    plt.xlabel('Absolute Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'), dpi=300)
    plt.close()

def create_per_image_analysis(df, output_dir):
    """Create visualizations for each individual image."""
    image_dir = os.path.join(output_dir, 'per_image_analysis')
    os.makedirs(image_dir, exist_ok=True)
    
    for image_name, group in df.groupby('image'):
        plt.figure(figsize=(12, 6))
        
        # Prediction comparison
        plt.subplot(1, 2, 1)
        sns.barplot(x='model', y='prediction', data=group)
        plt.axhline(y=group['ground_truth'].iloc[0], color='r', linestyle='--')
        plt.title(f'Predictions for {image_name[:20]}...')
        plt.ylabel('Stair Count')
        
        # Error comparison
        plt.subplot(1, 2, 2)
        sns.barplot(x='model', y='error', data=group)
        plt.title('Prediction Errors')
        plt.ylabel('Absolute Error')
        
        plt.tight_layout()
        safe_name = "".join(c for c in image_name if c.isalnum() or c in ('_', '.')).rstrip()
        plt.savefig(os.path.join(image_dir, f'{safe_name[:50]}.png'), dpi=150)
        plt.close()

def plot_training_comparison(results_dir=RESULTS_DIR):
    model_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]
    
    times = []
    metrics = []
    
    for file in model_files:
        with open(os.path.join(results_dir, file)) as f:
            data = json.load(f)
            model_name = file.split('_')[0]
            times.append({
                'model': model_name,
                'train_time': data['total_time'],
                'inference_time': np.mean([v['inference_time'] for k,v in data.items() if 'fold' in k])
            })
            metrics.append({
                'model': model_name,
                'mae': np.mean([v['metrics']['mae'] for k,v in data.items() if 'fold' in k]),
                'rmse': np.mean([v['metrics']['rmse'] for k,v in data.items() if 'fold' in k])
            })
    
    # Visualisation temps
    plt.figure(figsize=(12,6))
    pd.DataFrame(times).plot(x='model', kind='bar', secondary_y='inference_time')
    plt.title("Comparaison des temps d'exécution")
    plt.savefig(os.path.join(results_dir, 'time_comparison.png'))
    
    # Visualisation métriques
    plt.figure(figsize=(12,6))
    pd.DataFrame(metrics).plot(x='model', kind='bar')
    plt.title("Comparaison des performances")
    plt.savefig(os.path.join(results_dir, 'metrics_comparison.png'))