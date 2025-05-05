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
    """Creer une visualisation de comparaison des folds avec des metriques supplementaires."""
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(15, 18))
    
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
    """Se concentre sur la visualisation des predictions et appelle l'analyse d'erreur amelioree."""
    # Visualisation basique des predictions
    errors = np.array(pred) - np.array(true)
    
    plt.figure(figsize=(14, 6))
    
    # Reel vs Predicted avec coloration des erreurs
    plt.subplot(1, 2, 1)
    sc = plt.scatter(true, pred, c=np.abs(errors), cmap=CUSTOM_CMAP, alpha=0.7)
    plt.colorbar(sc, label='Erreur Absolue')
    min_val = min(min(true), min(pred))
    max_val = max(max(true), max(pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'b--')
    plt.xlabel('Reel')
    plt.ylabel('Predicted')
    plt.title(f'Reel vs Predicted - {model_type}')
    plt.grid(True, alpha=0.3)
    
    # Graphique des residus avec ligne de tendance
    plt.subplot(1, 2, 2)
    plt.scatter(pred, errors, c=np.abs(errors), cmap=CUSTOM_CMAP, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    
    # Ajouter une ligne de tendance
    z = np.polyfit(pred, errors, 1)
    p = np.poly1d(z)
    plt.plot(pred, p(pred), "r--")
    
    plt.xlabel('Valeur Predite')
    plt.ylabel('Residu')
    plt.title('Residus vs Valeurs Predites')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_prediction_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Appeler l'analyse d'erreur amelioree
    enhanced_error_analysis(true, pred, model_type, output_dir)

def combine_and_visualize_results(output_dir=RESULTS_DIR):
    """Combiner les resultats de tous les modeles et creer des visualisations completes."""
    # Trouver tous les fichiers de resultats
    model_files = [f for f in os.listdir(output_dir) if f.endswith('_results.json')]
    
    if not model_files:
        print("Aucun fichier de resultat de modele trouve.")
        return None
    
    combined_data = []
    
    for file in model_files:
        model_name = file.replace('_results.json', '')
        file_path = os.path.join(output_dir, file)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Gerer les differents formats JSON
                if isinstance(data, dict):
                    for image_name, evaluations in data.items():
                        # Cas 1: evaluations est une liste de dictionnaires
                        if isinstance(evaluations, list):
                            for eval_item in evaluations:
                                if isinstance(eval_item, dict):
                                    combined_data.append(process_evaluation(image_name, model_name, eval_item))
                        # Cas 2: evaluations est un seul dictionnaire
                        elif isinstance(evaluations, dict):
                            combined_data.append(process_evaluation(image_name, model_name, evaluations))
                        # Cas 3: evaluations est une valeur directe (peu probable mais possible)
                        else:
                            print(f"Format inattendu dans {file} pour l'image {image_name}")
                # Gerer le cas ou JSON est une liste directement
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            combined_data.append(process_evaluation(
                                item.get('image_name', 'unknown'),
                                model_name,
                                item
                            ))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Erreur lors du traitement de {file}: {str(e)}")
            continue
    
    if not combined_data:
        print("Aucune donnee valide trouvee dans les fichiers de resultats.")
        return None
    
    # Creer un DataFrame
    df = pd.DataFrame(combined_data)
    
    # Calculer les metriques
    df['error'] = abs(df['prediction'] - df['ground_truth'])
    df['relative_error'] = (df['error'] / df['ground_truth'].clip(lower=1e-10)) * 100
    df['squared_error'] = df['error']**2
    
    # Sauvegarder les resultats combines
    combined_json_path = os.path.join(output_dir, 'combined_results.json')
    df.to_json(combined_json_path, orient='records', indent=4)
    
    # Creer des visualisations
    try:
        create_model_comparison_plots(df, output_dir)
        create_error_analysis_plots(df, output_dir)
        create_per_image_analysis(df, output_dir)
    except Exception as e:
        print(f"Erreur lors de la generation des visualisations: {str(e)}")
    
    return df

def enhanced_error_analysis(true, pred, model_type, output_dir):
    """Analyse d'erreur approfondie avec visualisations avancees"""
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
    plt.suptitle(f'Analyse d erreur approfondie - {model_type}', y=1.02, fontsize=16)
    
    # 1. Error Distribution with Kernel Density
    plt.subplot(3, 3, 1)
    sns.histplot(errors, kde=True, bins=20, color='skyblue')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Erreur de prediction (Predite - Reel)')
    plt.ylabel('Frequence')
    plt.title('Distribution des erreurs avec densite')
    plt.grid(True, alpha=0.3)
    
    # 2. Quantile-Quantile Plot
    plt.subplot(3, 3, 2)
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Diagramme Q-Q des erreurs')
    plt.grid(True, alpha=0.3)
    
    # 3. Error Magnitude vs Actual Value
    plt.subplot(3, 3, 3)
    plt.scatter(true, abs_errors, c=abs_errors, cmap=CUSTOM_CMAP, alpha=0.6)
    plt.colorbar(label='Erreur absolue')
    plt.xlabel('Valeur reelle')
    plt.ylabel('Erreur absolue')
    plt.title('Erreur en fonction de la valeur reelle')
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative Error Distribution
    plt.subplot(3, 3, 4)
    sorted_errors = np.sort(abs_errors)
    cum_dist = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
    plt.plot(sorted_errors, cum_dist, linewidth=2)
    plt.fill_between(sorted_errors, 0, cum_dist, alpha=0.2)
    plt.xlabel('Seuil d erreur absolue')
    plt.ylabel('Proportion cumulative')
    plt.title('Distribution cumulative des erreurs')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for key percentiles
    for p in [0.5, 0.75, 0.9]:
        threshold = np.percentile(sorted_errors, p*100)
        plt.axvline(threshold, color='red', linestyle='--', alpha=0.5)
        plt.text(threshold, p, f'{p:.0%}', ha='right', va='bottom')
    
    # 5. Error Type Breakdown
    plt.subplot(3, 3, 5)
    error_types = {
        'Sous estimation': error_stats['Underestimation'],
        'Sur estimation': error_stats['Overestimation'],
        'Concordance exacte': error_stats['ExactMatch']
    }
    plt.pie(error_types.values(), labels=error_types.keys(), 
            autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Type d erreur')
    
    # 6. Error Tolerance Analysis
    plt.subplot(3, 3, 6)
    tolerance_levels = {
        'Concordance exacte': error_stats['ExactMatch'],
        '±1 marche': error_stats['Within1Step'] - error_stats['ExactMatch'],
        '±2 marches': error_stats['Within2Steps'] - error_stats['Within1Step'],
        'Erreurs importantes': error_stats['LargeErrors']
    }
    plt.bar(tolerance_levels.keys(), tolerance_levels.values(), 
            color=['#2ca02c', '#98df8a', '#d62728', '#ff9896'])
    plt.ylabel('Pourcentage de predictions')
    plt.title('Analyse de tolerance')
    
    # 7. Error by Value Range (Boxplot)
    plt.subplot(3, 3, 7)
    error_df = pd.DataFrame({'Actual': true, 'Error': errors})
    error_df['ValueRange'] = pd.cut(error_df['Actual'], bins=5)
    sns.boxplot(x='ValueRange', y='Error', data=error_df)
    plt.xticks(rotation=45)
    plt.xlabel('Intervalle de valeur reelle')
    plt.ylabel('Erreur de prediction')
    plt.title('Distribution des erreurs par intervalle de valeur reelle')
    
    # 8. Metrics Summary Table
    plt.subplot(3, 3, 8)
    metrics_table = [
        ["MAE", f"{error_stats['MAE']:.2f}"],
        ["RMSE", f"{error_stats['RMSE']:.2f}"],
        ["MedAE", f"{error_stats['MedAE']:.2f}"],
        ["MAPE", f"{error_stats['MAPE']:.2f}%"],
        ["±1 mache de precision", f"{error_stats['Within1Step']:.1f}%"],
        ["±2 marches de precision", f"{error_stats['Within2Steps']:.1f}%"]
    ]
    plt.table(cellText=metrics_table, 
              colLabels=["Metrique", "Valeur"], 
              loc='center', 
              cellLoc='center',
              colWidths=[0.4, 0.4])
    plt.axis('off')
    plt.title('Metriques de performance')
    
    # 9. Error Autocorrelation Plot
    plt.subplot(3, 3, 9)
    pd.plotting.autocorrelation_plot(errors)
    plt.xlim(0, min(20, len(errors)//2))
    plt.title('Autocorrelation des erreurs')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_type}_error_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save error statistics to JSON
    stats_path = os.path.join(output_dir, f"{model_type}_error_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(error_stats, f, indent=4)

def process_evaluation(image_name, model_name, eval_dict):
    """Traiter un seul enregistrement d'evaluation dans un format standardise."""
    return {
        'image': image_name,
        'model': model_name,
        'ground_truth': float(eval_dict.get('ground_truth', 0)),
        'prediction': float(eval_dict.get('prediction', 0))
    }

def create_model_comparison_plots(df, output_dir):
    """Creer des graphiques de comparaison entre differents modeles avec des metriques ameliorees."""
    plt.figure(figsize=(18, 12))
    
    # Calculer des statistiques completes des modeles
    model_stats = df.groupby('model').agg({
        'error': ['mean', 'median', 'std', 'max', 'min'],
        'relative_error': ['mean', 'median', 'std'],
        'squared_error': ['mean']
    })
    model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns.values]
    model_stats['rmse'] = np.sqrt(model_stats['squared_error_mean'])
    
    # Graphique 1: Comparaison des metriques d'erreur
    plt.subplot(2, 2, 1)
    model_stats[['error_mean', 'error_median', 'rmse']].plot(kind='bar', ax=plt.gca())
    plt.title('Comparaison des Metriques d Erreur des Modeles')
    plt.ylabel('Valeur d Erreur')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(['MAE', 'MedAE', 'RMSE'])
    
    # Graphique 2: Comparaison des erreurs relatives
    plt.subplot(2, 2, 2)
    model_stats[['relative_error_mean', 'relative_error_median']].plot(kind='bar', ax=plt.gca())
    plt.title('Comparaison des Erreurs Relatives')
    plt.ylabel('Erreur Relative (%)')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(['Erreur Relative Moyenne', 'Erreur Relative Medianne'])
    
    # Graphique 3: Distribution des erreurs
    plt.subplot(2, 2, 3)
    sns.boxplot(x='model', y='error', data=df)
    plt.title('Distribution des Erreurs par Modele')
    plt.ylabel('Erreur Absolue')
    plt.xticks(rotation=45)
    
    # Graphique 4: Constance des erreurs (ecart-type)
    plt.subplot(2, 2, 4)
    model_stats[['error_std', 'relative_error_std']].plot(kind='bar', ax=plt.gca())
    plt.title('Constance des Erreurs (Ecart-type)')
    plt.ylabel('Ecart-type')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(['Ecart-type Erreur Absolue', 'Ecart-type Erreur Relative'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()

def create_error_analysis_plots(df, output_dir):
    """Creer des graphiques d'analyse d'erreur complets pour la comparaison des modeles."""
    plt.figure(figsize=(20, 15))
    plt.suptitle('Comparaison Avancee des Modeles et Analyse d Erreur', fontsize=16)
    
    # 1. Comparaison des Metriques de Performance
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
    plt.title('Metriques d Erreur par Modele')
    plt.xticks(rotation=45)
    plt.ylabel('Valeur d Erreur')
    plt.legend(title='Metrique', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Distribution des Erreurs Relatives
    plt.subplot(2, 3, 2)
    df['relative_error'] = np.abs(df['prediction'] - df['ground_truth']) / df['ground_truth'] * 100
    sns.boxplot(x='model', y='relative_error', data=df)
    plt.title('Distribution des Erreurs Relatives')
    plt.xticks(rotation=45)
    plt.ylabel('Erreur Relative (%)')
    
    # 3. Analyse de Precision des Predictions
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
    plt.title('Precision des Predictions')
    plt.ylabel('Pourcentage')
    plt.legend(title='Metrique de Precision', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 4. Erreur vs Ground Truth Scatter
    plt.subplot(2, 3, 4)
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.scatter(model_data['ground_truth'], model_data['prediction'] - model_data['ground_truth'], 
                    label=model, alpha=0.6)
    
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Distribution des Erreurs par Ground Truth')
    plt.xlabel('Ground Truth')
    plt.ylabel('Erreur de Prediction')
    plt.legend()
    
    # 5. Graphique Radar de Performance des Modeles
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
    
    # Normaliser les metriques pour le graphique radar
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
    plt.title('Performance Complete des Modeles')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # 6. Critere de Selection du Meilleur Modele
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
    plt.title('Classement des Modeles par Score Composite de Performance')
    plt.xlabel('Score (Plus haut est meilleur)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Afficher les details du meilleur modele
    print("Resume de Performance des Modeles:")
    print(performance_df)
    print("\nClassement des Meilleurs Modeles:")
    print(criteria_summary['Overall Score'])

def create_per_image_analysis(df, output_dir):
    """Creer des visualisations pour chaque image individuelle."""
    image_dir = os.path.join(output_dir, 'per_image_analysis')
    os.makedirs(image_dir, exist_ok=True)
    
    for image_name, group in df.groupby('image'):
        plt.figure(figsize=(12, 6))
        
        # Comparaison des predictions
        plt.subplot(1, 2, 1)
        sns.barplot(x='model', y='prediction', data=group)
        plt.axhline(y=group['ground_truth'].iloc[0], color='r', linestyle='--')
        plt.title(f'Predictions pour {image_name[:20]}...')
        plt.ylabel('Nombre de Marches')
        
        # Comparaison des erreurs
        plt.subplot(1, 2, 2)
        sns.barplot(x='model', y='error', data=group)
        plt.title('Erreurs de Prediction')
        plt.ylabel('Erreur Absolue')
        
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
