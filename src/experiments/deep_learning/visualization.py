import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from config_dl import RESULTS_DIR, CUSTOM_CMAP
import numpy as np
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import acf
from scipy.stats import norm
from matplotlib.colors import Normalize

def plot_metrics(history, model_type, fold, output_dir):
    # Loss curve
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(f'Loss Evolution - {model_type} (Fold {fold})')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_loss_curve.png'))
    plt.close()

def plot_prediction_vs_actual(y_true, y_pred, model_type, fold, output_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Ligne idéale (y = x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Predictions vs Actual Values - {model_type} (Fold {fold})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_pred_vs_actual.png'))
    plt.close()

def plot_error_distribution(y_true, y_pred, model_type, fold, output_dir):
    errors = y_pred - y_true
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Error Distribution - {model_type} (Fold {fold})')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_error_dist.png'))
    plt.close()

def plot_error_metrics(metrics_dict, model_type, output_dir):
    # Extraire les métriques principales pour la visualisation
    metrics_to_plot = {
        'RMSE': metrics_dict['avg_test_rmse'],
        'MAE': metrics_dict['avg_test_mae'],
        'R²': metrics_dict['avg_test_r2'],
        'MedAE': metrics_dict['avg_test_medae']
    }
    
    # Créer un graphique en barres pour les métriques de performance
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    plt.title(f'Performance Metrics - {model_type}')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, f'{model_type}_performance_metrics.png'))
    plt.close()
    
    # Graphique en barres pour les taux d'erreur
    error_rates = {
        'Underestimation': metrics_dict['avg_underestimation_rate'],
        'Overestimation': metrics_dict['avg_overestimation_rate'],
        'Exact Match': metrics_dict['avg_exact_match_rate'],
        'Within 1 Step': metrics_dict['avg_within_1_step'],
        'Within 2 Steps': metrics_dict['avg_within_2_steps'],
        'Large Errors': metrics_dict['avg_large_errors']
    }
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(error_rates.keys(), error_rates.values(), color=plt.cm.viridis(np.linspace(0, 1, len(error_rates))))
    plt.title(f'Error Analysis - {model_type}')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, f'{model_type}_error_analysis.png'))
    plt.close()

def create_metrics_heatmap(results, output_dir):
    """Crée une heatmap comparative des métriques pour tous les modèles."""
    metrics_df = pd.DataFrame()
    
    for res in results:
        model_name = res['model']
        metrics = res['avg_metrics']
        metrics['model'] = model_name
        
        # Ajouter à notre dataframe
        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
    
    # Sélectionner uniquement les métriques principales pour la heatmap
    key_metrics = ['model', 'avg_test_rmse', 'avg_test_mae', 'avg_test_r2', 'avg_test_medae',
                  'avg_within_1_step', 'avg_within_2_steps', 'avg_large_errors']
    
    # Créer un pivot pour la heatmap
    plot_df = metrics_df[key_metrics].set_index('model')
    
    # Renommer les colonnes pour plus de lisibilité
    plot_df.columns = ['RMSE', 'MAE', 'R²', 'MedAE', 'Within 1 Step (%)', 'Within 2 Steps (%)', 'Large Errors (%)']
    
    plt.figure(figsize=(14, len(results) * 1.5 + 2))
    sns.heatmap(plot_df, annot=True, cmap='YlGnBu', fmt='.3f', cbar_kws={'label': 'Value'})
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_metrics_heatmap.png'))
    plt.close()
    
    # Sauvegarder en CSV pour référence
    metrics_df.to_csv(os.path.join(output_dir, 'all_models_metrics.csv'), index=False)

def save_results(results, output_dir=RESULTS_DIR):
    # Créer un dossier pour les visualisations détaillées
    details_dir = os.path.join(output_dir, 'detailed_plots')
    os.makedirs(details_dir, exist_ok=True)
    
    # Pour chaque modèle, générer des visualisations détaillées pour chaque fold
    for res in results:
        model_type = res['model']
        model_dir = os.path.join(details_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        for fold_result in res['results']:
            fold = fold_result['metrics']['fold']
            
            # Get predictions and actual values
            y_true = np.array(fold_result['metrics'].get('test_actual', []))
            y_pred = np.array(fold_result['metrics'].get('test_pred', []))
            
        plot_qq(y_true, y_pred, model_type, fold, model_dir)
        plot_cumulative_error_distribution(y_true, y_pred, model_type, fold, model_dir)
        plot_error_vs_actual(y_true, y_pred, model_type, fold, model_dir)
        plot_density_distribution(y_true, y_pred, model_type, fold, model_dir)
        plot_error_by_value_range(y_true, y_pred, model_type, fold, model_dir)
        plot_error_autocorrelation(y_true, y_pred, model_type, fold, model_dir)

    # Tableau récapitulatif des performances
    df = pd.DataFrame([{
        'Model': res['model'],
        'Avg Test RMSE': res['avg_metrics']['avg_test_rmse'],
        'Avg Test MAE': res['avg_metrics']['avg_test_mae'],
        'Avg Test R²': res['avg_metrics']['avg_test_r2'],
        'Within 1 Step (%)': res['avg_metrics']['avg_within_1_step'],
        'Total Time (s)': res['total_time']
    } for res in results])
    
    # Comparaison des performances principales
    plt.figure(figsize=(12, 7))
    ax = df.plot(x='Model', y=['Avg Test RMSE', 'Avg Test MAE'], kind='bar', secondary_y='Total Time (s)')
    ax.set_ylabel('Error Value')
    # Create secondary y-axis explicitly
    ax2 = ax.twinx()
    ax2.plot(ax.get_xticks(), df['Total Time (s)'], color='red', marker='o', linestyle='--', label='Total Time')
    ax2.set_ylabel('Time (seconds)')

    plt.title('Comparaison des Modèles - Métriques Principales')
    plt.savefig(os.path.join(output_dir, 'model_comparison_metrics.png'))
    plt.close()
    
    # Graphique pour le score R²
    plt.figure(figsize=(10, 6))
    plt.bar(df['Model'], df['Avg Test R²'], color='green', alpha=0.7)
    plt.title('Comparaison des Modèles - Score R²')
    plt.ylabel('R² Score')
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(os.path.join(output_dir, 'model_comparison_r2.png'))
    plt.close()
    
    # Créer une heatmap comparative des métriques
    create_metrics_heatmap(results, output_dir)
    
    # Sauvegarder le résumé en CSV
    df.to_csv(os.path.join(output_dir, 'results_summary.csv'), index=False)

def plot_qq(y_true, y_pred, model_type, fold, output_dir):
    errors = y_pred - y_true
    plt.figure(figsize=(8, 6))
    qqplot(errors, line='s', fit=True)
    plt.title(f'Q-Q Plot - {model_type} (Fold {fold})')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_qq_plot.png'))
    plt.close()

def plot_cumulative_error_distribution(y_true, y_pred, model_type, fold, output_dir):
    errors = np.abs(y_pred - y_true)
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_errors, cdf, marker='.', linestyle='none')
    plt.title(f'Cumulative Error Distribution - {model_type} (Fold {fold})')
    plt.xlabel('Absolute Error')
    plt.ylabel('Cumulative Probability')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_cumulative_error.png'))
    plt.close()

def plot_error_vs_actual(y_true, y_pred, model_type, fold, output_dir):
    errors = y_pred - y_true
    
    # Skip plot if there's no data
    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"Warning: Empty data for {model_type} fold {fold} - skipping error vs actual plot")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, np.abs(errors), alpha=0.5)
    plt.title(f'Error Magnitude vs Actual Values - {model_type} (Fold {fold})')
    plt.xlabel('Actual Values')
    plt.ylabel('Absolute Error')
    plt.grid(True)
    
    # Add trend line only if sufficient data exists
    if len(y_true) > 1:
        try:
            z = np.polyfit(y_true, np.abs(errors), 1)
            p = np.poly1d(z)
            plt.plot(np.sort(y_true), p(np.sort(y_true)), "r--", label='Trend')
            plt.legend()
        except Exception as e:
            print(f"Could not plot trend line for {model_type} fold {fold}: {str(e)}")
    
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_error_vs_actual.png'))
    plt.close()

def plot_density_distribution(y_true, y_pred, model_type, fold, output_dir):
    errors = y_pred - y_true
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True, bins=30, stat='density')
    plt.title(f'Error Density Distribution - {model_type} (Fold {fold})')
    plt.xlabel('Prediction Error')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_density_dist.png'))
    plt.close()

def plot_error_by_value_range(y_true, y_pred, model_type, fold, output_dir):
    errors = y_pred - y_true
    bins = np.linspace(min(y_true), max(y_true), 6)
    bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(len(bins)-1)]
    
    df = pd.DataFrame({'Actual': y_true, 'Error': errors})
    df['Value Range'] = pd.cut(df['Actual'], bins=bins, labels=bin_labels)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Value Range', y='Error', data=df)
    plt.title(f'Error Distribution by Value Range - {model_type} (Fold {fold})')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_error_by_range.png'))
    plt.close()

def plot_error_autocorrelation(y_true, y_pred, model_type, fold, output_dir):
    errors = y_pred - y_true
    nlags = min(20, len(errors)//2)
    confint = 1.96 / np.sqrt(len(errors))
    
    plt.figure(figsize=(10, 6))
    acf_values = acf(errors, nlags=nlags)
    plt.stem(acf_values)
    plt.axhspan(-confint, confint, alpha=0.2, color='blue')
    plt.title(f'Error Autocorrelation - {model_type} (Fold {fold})')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.ylim(-0.5, 0.5)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'{model_type}_fold{fold}_error_autocorr.png'))
    plt.close()
    
def combine_and_visualize_results():
    """Combine les résultats de tous les modèles et génère des visualisations comparatives."""
    # Charger les résultats détaillés de chaque modèle
    results = []
    for model_file in os.listdir(RESULTS_DIR):
        if model_file.endswith('_detailed_metrics.json'):
            with open(os.path.join(RESULTS_DIR, model_file), 'r') as f:
                model_results = json.load(f)
                results.append(model_results)
    
    # Sauvegarder et visualiser les résultats combinés
    if results:
        save_results(results)
    else:
        print("Aucun résultat détaillé trouvé. Exécutez l'entraînement des modèles d'abord.")