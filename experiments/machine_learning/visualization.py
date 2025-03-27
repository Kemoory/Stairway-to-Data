import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from config import CUSTOM_CMAP, RESULTS_DIR
import json
from scipy import stats
import seaborn as sns

def save_fold_comparison(results, model_type, output_dir):
    """Create fold comparison visualization with additional metrics."""
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(15, 18))  # Increased figure size
    
    # RMSE
    plt.subplot(4, 2, 1)
    df[['train_rmse', 'test_rmse']].plot(kind='bar', ax=plt.gca())
    plt.title(f'RMSE by Fold - {model_type.capitalize()}')
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAE
    plt.subplot(4, 2, 2)
    df[['train_mae', 'test_mae']].plot(kind='bar', ax=plt.gca())
    plt.title(f'MAE by Fold - {model_type.capitalize()}')
    plt.ylabel('MAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # R²
    plt.subplot(4, 2, 3)
    df[['train_r2', 'test_r2']].plot(kind='bar', ax=plt.gca())
    plt.title(f'R² by Fold - {model_type.capitalize()}')
    plt.ylabel('R²')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # MAPE
    plt.subplot(4, 2, 4)
    df[['train_mape', 'test_mape']].plot(kind='bar', ax=plt.gca())
    plt.title(f'MAPE by Fold - {model_type.capitalize()}')
    plt.ylabel('MAPE (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # MedAE
    plt.subplot(4, 2, 6)
    df[['train_medae', 'test_medae']].plot(kind='bar', ax=plt.gca())
    plt.title(f'Median Absolute Error by Fold - {model_type.capitalize()}')
    plt.ylabel('MedAE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Performance Ratio (Test/Train)
    plt.subplot(4, 2, 7)
    df['performance_ratio'] = df['test_rmse'] / df['train_rmse']
    df['performance_ratio'].plot(kind='bar', ax=plt.gca(), color='purple')
    plt.title(f'Performance Ratio (Test/Train RMSE) - {model_type.capitalize()}')
    plt.ylabel('Ratio')
    plt.axhline(1, color='red', linestyle='--')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_fold_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_prediction_analysis(true, pred, model_type, output_dir):
    """Focus on prediction visualization and call enhanced error analysis."""
    # Basic prediction visualization
    errors = np.array(pred) - np.array(true)
    
    plt.figure(figsize=(14, 6))
    
    # Actual vs Predicted with error coloring
    plt.subplot(1, 2, 1)
    sc = plt.scatter(true, pred, c=np.abs(errors), cmap=CUSTOM_CMAP, alpha=0.7)
    plt.colorbar(sc, label='Absolute Error')
    min_val = min(min(true), min(pred))
    max_val = max(max(true), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'b--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted - {model_type}')
    plt.grid(True, alpha=0.3)
    
    # Residual plot with trend line
    plt.subplot(1, 2, 2)
    plt.scatter(pred, errors, c=np.abs(errors), cmap=CUSTOM_CMAP, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    
    # Add trend line
    z = np.polyfit(pred, errors, 1)
    p = np.poly1d(z)
    plt.plot(pred, p(pred), "r--")
    
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual')
    plt.title('Residuals vs Predicted Values')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_prediction_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Call enhanced error analysis
    enhanced_error_analysis(true, pred, model_type, output_dir)

def combine_and_visualize_results(output_dir=RESULTS_DIR):
    """Combine results from all models and create comprehensive visualizations."""
    # Find all result files
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
                
                # Handle different JSON formats
                if isinstance(data, dict):
                    for image_name, evaluations in data.items():
                        # Case 1: evaluations is a list of dicts
                        if isinstance(evaluations, list):
                            for eval_item in evaluations:
                                if isinstance(eval_item, dict):
                                    combined_data.append(process_evaluation(image_name, model_name, eval_item))
                        # Case 2: evaluations is a single dict
                        elif isinstance(evaluations, dict):
                            combined_data.append(process_evaluation(image_name, model_name, evaluations))
                        # Case 3: evaluations is a direct value (unlikely but possible)
                        else:
                            print(f"Unexpected format in {file} for image {image_name}")
                # Handle case where JSON is a list directly
                elif isinstance(data, list):
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
    
    # Create DataFrame
    df = pd.DataFrame(combined_data)
    
    # Calculate metrics
    df['error'] = abs(df['prediction'] - df['ground_truth'])
    df['relative_error'] = (df['error'] / df['ground_truth'].clip(lower=1e-10)) * 100
    df['squared_error'] = df['error']**2
    
    # Save combined results
    combined_json_path = os.path.join(output_dir, 'combined_results.json')
    df.to_json(combined_json_path, orient='records', indent=4)
    
    # Create visualizations
    try:
        create_model_comparison_plots(df, output_dir)
        create_error_analysis_plots(df, output_dir)
        create_per_image_analysis(df, output_dir)
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    return df

def enhanced_error_analysis(true, pred, model_type, output_dir):
    """Comprehensive error analysis with advanced visualizations."""
    errors = np.array(pred) - np.array(true)
    abs_errors = np.abs(errors)
    relative_errors = abs_errors / (np.array(true) + 1e-10)  # Avoid division by zero
    
    # Calculate error statistics
    error_stats = {
        'MAE': np.mean(abs_errors),
        'RMSE': np.sqrt(np.mean(errors**2)),
        'MedAE': np.median(abs_errors),
        'MAPE': np.mean(relative_errors) * 100,
        'Underestimation': np.mean(errors < 0) * 100,
        'Overestimation': np.mean(errors > 0) * 100,
        'ExactMatch': np.mean(errors == 0) * 100,
        'Within1Step': np.mean(abs_errors <= 1) * 100,
        'Within2Steps': np.mean(abs_errors <= 2) * 100,
        'LargeErrors': np.mean(abs_errors > 3) * 100
    }
    
    plt.figure(figsize=(20, 20))
    plt.suptitle(f'Advanced Error Analysis - {model_type}', y=1.02, fontsize=16)
    
    # 1. Error Distribution with Kernel Density
    plt.subplot(3, 3, 1)
    sns.histplot(errors, kde=True, bins=20, color='skyblue')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Prediction Error (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution with Density')
    plt.grid(True, alpha=0.3)
    
    # 2. Quantile-Quantile Plot
    plt.subplot(3, 3, 2)
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Errors')
    plt.grid(True, alpha=0.3)
    
    # 3. Error Magnitude vs Actual Value
    plt.subplot(3, 3, 3)
    plt.scatter(true, abs_errors, c=abs_errors, cmap=CUSTOM_CMAP, alpha=0.6)
    plt.colorbar(label='Absolute Error')
    plt.xlabel('Actual Value')
    plt.ylabel('Absolute Error')
    plt.title('Error Magnitude vs Actual Value')
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative Error Distribution
    plt.subplot(3, 3, 4)
    sorted_errors = np.sort(abs_errors)
    cum_dist = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    plt.plot(sorted_errors, cum_dist, linewidth=2)
    plt.fill_between(sorted_errors, 0, cum_dist, alpha=0.2)
    plt.xlabel('Absolute Error Threshold')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for key percentiles
    for p in [0.5, 0.75, 0.9]:
        threshold = np.percentile(sorted_errors, p*100)
        plt.axvline(threshold, color='red', linestyle='--', alpha=0.5)
        plt.text(threshold, p, f'{p:.0%}', ha='right', va='bottom')
    
    # 5. Error Type Breakdown
    plt.subplot(3, 3, 5)
    error_types = {
        'Underestimation': error_stats['Underestimation'],
        'Overestimation': error_stats['Overestimation'],
        'Exact Match': error_stats['ExactMatch']
    }
    plt.pie(error_types.values(), labels=error_types.keys(), 
            autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Error Type Distribution')
    
    # 6. Error Tolerance Analysis
    plt.subplot(3, 3, 6)
    tolerance_levels = {
        'Exact Match': error_stats['ExactMatch'],
        '±1 Step': error_stats['Within1Step'] - error_stats['ExactMatch'],
        '±2 Steps': error_stats['Within2Steps'] - error_stats['Within1Step'],
        'Large Errors': error_stats['LargeErrors']
    }
    plt.bar(tolerance_levels.keys(), tolerance_levels.values(), 
            color=['#2ca02c', '#98df8a', '#d62728', '#ff9896'])
    plt.ylabel('Percentage of Predictions')
    plt.title('Error Tolerance Analysis')
    
    # 7. Error by Value Range (Boxplot)
    plt.subplot(3, 3, 7)
    error_df = pd.DataFrame({'Actual': true, 'Error': errors})
    error_df['ValueRange'] = pd.cut(error_df['Actual'], bins=5)
    sns.boxplot(x='ValueRange', y='Error', data=error_df)
    plt.xticks(rotation=45)
    plt.xlabel('Actual Value Range')
    plt.ylabel('Prediction Error')
    plt.title('Error Distribution by Value Range')
    
    # 8. Metrics Summary Table
    plt.subplot(3, 3, 8)
    metrics_table = [
        ["MAE", f"{error_stats['MAE']:.2f}"],
        ["RMSE", f"{error_stats['RMSE']:.2f}"],
        ["MedAE", f"{error_stats['MedAE']:.2f}"],
        ["MAPE", f"{error_stats['MAPE']:.2f}%"],
        ["±1 Step Accuracy", f"{error_stats['Within1Step']:.1f}%"],
        ["±2 Steps Accuracy", f"{error_stats['Within2Steps']:.1f}%"]
    ]
    plt.table(cellText=metrics_table, 
              colLabels=["Metric", "Value"], 
              loc='center', 
              cellLoc='center',
              colWidths=[0.4, 0.4])
    plt.axis('off')
    plt.title('Key Performance Metrics')
    
    # 9. Error Autocorrelation Plot
    plt.subplot(3, 3, 9)
    pd.plotting.autocorrelation_plot(errors)
    plt.xlim(0, min(20, len(errors)//2))
    plt.title('Error Autocorrelation')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_advanced_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save error statistics to JSON
    stats_path = os.path.join(output_dir, f"{model_type}_error_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(error_stats, f, indent=4)

def process_evaluation(image_name, model_name, eval_dict):
    """Process a single evaluation record into standardized format."""
    return {
        'image': image_name,
        'model': model_name,
        'ground_truth': float(eval_dict.get('ground_truth', 0)),
        'prediction': float(eval_dict.get('prediction', 0))
    }

def create_model_comparison_plots(df, output_dir):
    """Create comparison plots between different models with enhanced metrics."""
    plt.figure(figsize=(18, 12))
    
    # Calculate comprehensive model statistics
    model_stats = df.groupby('model').agg({
        'error': ['mean', 'median', 'std', 'max', 'min'],
        'relative_error': ['mean', 'median', 'std'],
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
    plt.legend(['MAE', 'MedAE', 'RMSE'])
    
    # Plot 2: Relative error comparison
    plt.subplot(2, 2, 2)
    model_stats[['relative_error_mean', 'relative_error_median']].plot(kind='bar', ax=plt.gca())
    plt.title('Relative Error Comparison')
    plt.ylabel('Relative Error (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(['Mean Relative Error', 'Median Relative Error'])
    
    # Plot 3: Error distribution
    plt.subplot(2, 2, 3)
    sns.boxplot(x='model', y='error', data=df)
    plt.title('Error Distribution by Model')
    plt.ylabel('Absolute Error')
    plt.xticks(rotation=45)
    
    # Plot 4: Error consistency (std dev)
    plt.subplot(2, 2, 4)
    model_stats[['error_std', 'relative_error_std']].plot(kind='bar', ax=plt.gca())
    plt.title('Error Consistency (Standard Deviation)')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(['Absolute Error Std', 'Relative Error Std'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

def create_error_analysis_plots(df, output_dir):
    """Create comprehensive error analysis plots for model comparison."""
    plt.figure(figsize=(20, 15))
    plt.suptitle('Advanced Model Comparison and Error Analysis', fontsize=16)
    
    # 1. Performance Metrics Comparison
    plt.subplot(2, 3, 1)
    metrics_summary = df.groupby('model').agg({
        'error': ['mean', 'median', 'max'],
        'squared_error': lambda x: np.sqrt(np.mean(x))  # RMSE
    }).reset_index()
    
    metrics_summary.columns = ['model', 'MAE', 'MedAE', 'Max Error', 'RMSE']
    metrics_to_plot = ['MAE', 'MedAE', 'Max Error', 'RMSE']
    
    sns.barplot(x='model', y='value', hue='model', 
                data=pd.melt(metrics_summary, id_vars=['model'], value_vars=metrics_to_plot), 
                ci=None)
    plt.title('Error Metrics by Model')
    plt.xticks(rotation=45)
    plt.ylabel('Error Value')
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Relative Error Distribution
    plt.subplot(2, 3, 2)
    df['relative_error'] = np.abs(df['prediction'] - df['ground_truth']) / df['ground_truth'] * 100
    sns.boxplot(x='model', y='relative_error', data=df)
    plt.title('Relative Error Distribution')
    plt.xticks(rotation=45)
    plt.ylabel('Relative Error (%)')
    
    # 3. Prediction Accuracy Analysis
    plt.subplot(2, 3, 3)
    accuracy_metrics = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        errors = np.abs(model_data['prediction'] - model_data['ground_truth'])
        accuracy_metrics.append({
            'Model': model,
            'Within 1 Step (%)': np.mean(errors <= 1) * 100,
            'Within 2 Steps (%)': np.mean(errors <= 2) * 100,
            'Exact Match (%)': np.mean(errors == 0) * 100
        })
    
    accuracy_df = pd.DataFrame(accuracy_metrics).set_index('Model')
    accuracy_df.plot(kind='bar', rot=45, ax=plt.gca())
    plt.title('Prediction Accuracy')
    plt.ylabel('Percentage')
    plt.legend(title='Accuracy Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Error vs Ground Truth Scatter
    plt.subplot(2, 3, 4)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['ground_truth'], model_data['prediction'] - model_data['ground_truth'], 
                    label=model, alpha=0.6)
    
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Error Distribution by Ground Truth')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction Error')
    plt.legend()
    
    # 5. Model Performance Radar Chart
    plt.subplot(2, 3, 5, polar=True)
    performance_metrics = []
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        errors = np.abs(model_data['prediction'] - model_data['ground_truth'])
        
        performance_metrics.append({
            'Model': model,
            'MAE': np.mean(errors),
            'RMSE': np.sqrt(np.mean(errors**2)),
            'Within 1 Step': np.mean(errors <= 1),
            'Within 2 Steps': np.mean(errors <= 2),
            'Exact Match': np.mean(errors == 0)
        })
    
    performance_df = pd.DataFrame(performance_metrics)
    performance_df = performance_df.set_index('Model')
    
    # Normalize metrics for radar chart
    categories = ['MAE', 'RMSE', 'Within 1 Step', 'Within 2 Steps', 'Exact Match']
    normalized_df = performance_df.copy()
    
    for col in categories:
        normalized_df[col] = (performance_df[col] - performance_df[col].min()) / (performance_df[col].max() - performance_df[col].min())
    
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    
    for i, model in enumerate(normalized_df.index):
        values = normalized_df.loc[model].values.flatten().tolist()
        values += values[:1]
        plt.polar(angles, values, linewidth=1, linestyle='solid', label=model)
        plt.fill(angles, values, alpha=0.1)
    
    plt.xticks(angles[:-1], categories)
    plt.title('Comprehensive Model Performance')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 6. Best Model Selection Criteria
    plt.subplot(2, 3, 6)
    criteria_summary = performance_df.copy()
    criteria_summary['Overall Score'] = (
        criteria_summary['Within 1 Step'] * 0.4 +
        criteria_summary['Within 2 Steps'] * 0.3 +
        (1 / criteria_summary['MAE']) * 0.2 +
        (1 / criteria_summary['RMSE']) * 0.1
    )
    criteria_summary = criteria_summary.sort_values('Overall Score', ascending=False)
    
    plt.barh(criteria_summary.index, criteria_summary['Overall Score'])
    plt.title('Model Ranking by Composite Performance Score')
    plt.xlabel('Score (Higher is Better)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print out best model details
    print("Model Performance Summary:")
    print(performance_df)
    print("\nBest Model Ranking:")
    print(criteria_summary['Overall Score'])

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