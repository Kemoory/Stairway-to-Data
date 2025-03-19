import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from matplotlib.gridspec import GridSpec

def load_and_prepare_data(json_file_path):
    """Charge les données d'évaluation depuis un fichier JSON et les convertit en DataFrame.
    (JSON, c'est bon, mangez-en !)"""
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Aplatir la structure imbriquée (on passe du mille-feuille au pancake)
    rows = []
    for image_name, evaluations in data.items():
        for evaluation in evaluations:
            row = {
                'image_name': image_name,
                **evaluation,
                'error': abs(evaluation['prediction'] - evaluation['ground_truth']),
                'relative_error': abs(evaluation['prediction'] - evaluation['ground_truth']) / 
                                 max(evaluation['ground_truth'], 1e-10) * 100  # Éviter la division par zéro, ce n'est pas très mathématique
            }
            rows.append(row)
    
    return pd.DataFrame(rows)

def visualize_image_evaluations(df, output_dir='results/visualisation/algorithm/evaluation_visualizations'):
    """Créer des visualisations pour chaque image (parce qu'une image vaut mille graphiques)"""
    # Créer le répertoire de sortie s'il n'existe pas (parce que mieux vaut prévenir que guérir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtenir les images uniques (on trie les doublons comme des chaussettes)
    unique_images = df['image_name'].unique()
    
    for image_name in unique_images:
        image_df = df[df['image_name'] == image_name]
        
        # Créer une figure avec des sous-graphiques (parce qu'un graphique seul, c'est triste)
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig)
        
        # Comparaison des prédictions avec la vérité terrain (ou vérité vraie, comme vous voulez)
        ax1 = fig.add_subplot(gs[0, :2])
        pivot_df = image_df.pivot(index='preprocessing', columns='model', values='prediction')
        pivot_df.plot(kind='bar', ax=ax1)
        ax1.axhline(y=image_df['ground_truth'].iloc[0], color='r', linestyle='-', linewidth=2)
        ax1.text(0, image_df['ground_truth'].iloc[0]*1.05, f"Vérité Terrain : {image_df['ground_truth'].iloc[0]}", 
                color='r', fontweight='bold')
        ax1.set_title(f'Prédictions vs Vérité Terrain pour {image_name}')
        ax1.set_ylabel('Valeur')
        ax1.tick_params(axis='x', rotation=45)
        
        # Erreur absolue par modèle et méthode de prétraitement (parce que l'erreur, ça pique)
        ax2 = fig.add_subplot(gs[0, 2:])
        sns.heatmap(image_df.pivot(index='preprocessing', columns='model', values='error'),
                   annot=True, cmap='YlOrRd', ax=ax2)
        ax2.set_title('Erreur Absolue')
        
        # Erreur relative par modèle et prétraitement (en pourcentage, pour faire plus chic)
        ax3 = fig.add_subplot(gs[1, :2])
        sns.heatmap(image_df.pivot(index='preprocessing', columns='model', values='relative_error'),
                   annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3)
        ax3.set_title('Erreur Relative (%)')
        
        # Meilleures combinaisons (parce qu'on aime les gagnants)
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Trier par erreur (on met les mauvais élèves au fond)
        best_df = image_df.sort_values('error')
        # Afficher les 5 meilleurs ou tous s'il y en a moins de 5
        n_best = min(5, len(best_df))
        best_combinations = best_df.iloc[:n_best][['preprocessing', 'model', 'prediction', 'ground_truth', 'error']]
        
        # Créer une table (parce qu'un tableau, c'est toujours classe)
        cell_text = []
        for _, row in best_combinations.iterrows():
            cell_text.append([
                row['preprocessing'], 
                row['model'], 
                f"{row['prediction']:.2f}", 
                f"{row['ground_truth']:.2f}", 
                f"{row['error']:.2f}"
            ])
        
        ax4.axis('off')
        table = ax4.table(
            cellText=cell_text,
            colLabels=['Prétraitement', 'Modèle', 'Préd', 'Vérité', 'Erreur'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax4.set_title('Meilleures Combinaisons')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{image_name.split('.')[0]}_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_visualization(df, output_dir='results/visualisation/algorithm/evaluation_visualizations'):
    """Créer des visualisations récapitulatives globales pour toutes les images (parce que l'union fait la force)"""
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculer les statistiques agrégées (parce que les moyennes, c'est la vie)
    model_preprocessing_stats = df.groupby(['model', 'preprocessing']).agg({
        'error': ['mean', 'median', 'std'],
        'relative_error': ['mean', 'median', 'std']
    }).reset_index()
    
    # Aplatir les noms de colonnes (on passe du mille-feuille au pancake, encore)
    model_preprocessing_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in model_preprocessing_stats.columns.values]
    
    # Créer une figure pour les visualisations récapitulatives
    fig = plt.figure(figsize=(18, 15))
    gs = GridSpec(3, 2, figure=fig)
    
    # Erreur moyenne par modèle et prétraitement
    ax1 = fig.add_subplot(gs[0, 0])
    pivoted = df.pivot_table(values='error', index='preprocessing', columns='model', aggfunc='mean')
    sns.heatmap(pivoted, annot=True, cmap='YlOrRd', ax=ax1)
    ax1.set_title('Erreur Absolue Moyenne par Modèle et Prétraitement')
    
    # Erreur relative moyenne par modèle et prétraitement
    ax2 = fig.add_subplot(gs[0, 1])
    pivoted = df.pivot_table(values='relative_error', index='preprocessing', columns='model', aggfunc='mean')
    sns.heatmap(pivoted, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax2)
    ax2.set_title('Erreur Relative Moyenne (%) par Modèle et Prétraitement')
    
    # Boîte à moustaches des erreurs par modèle
    ax3 = fig.add_subplot(gs[1, 0])
    sns.boxplot(x='model', y='error', data=df, ax=ax3)
    ax3.set_title('Distribution des Erreurs Absolues par Modèle')
    ax3.tick_params(axis='x', rotation=45)
    
    # Boîte à moustaches des erreurs par prétraitement
    ax4 = fig.add_subplot(gs[1, 1])
    sns.boxplot(x='preprocessing', y='error', data=df, ax=ax4)
    ax4.set_title('Distribution des Erreurs Absolues par Prétraitement')
    ax4.tick_params(axis='x', rotation=45)
    
    # Meilleures combinaisons globales
    ax5 = fig.add_subplot(gs[2, :])
    # Regrouper par modèle et prétraitement et calculer les erreurs moyennes
    best_combinations = df.groupby(['model', 'preprocessing']).agg({
        'error': 'mean',
        'relative_error': 'mean'
    }).reset_index().sort_values('error')
    
    # Top 10 des combinaisons
    top_10 = best_combinations.head(10)
    
    y_pos = np.arange(len(top_10))
    labels = [f"{row['preprocessing']} + {row['model']}" for _, row in top_10.iterrows()]
    
    # Créer un graphique en barres horizontales
    bars = ax5.barh(y_pos, top_10['error'], align='center')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(labels)
    ax5.invert_yaxis()  # Les étiquettes lisent de haut en bas
    ax5.set_xlabel('Erreur Absolue Moyenne')
    ax5.set_title('Top 10 des Combinaisons Modèle-Prétraitement par Erreur Moyenne')
    
    # Ajouter les valeurs d'erreur en texte
    for i, v in enumerate(top_10['error']):
        ax5.text(v + 0.1, i, f"{v:.2f}", va='center')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

def run_visualization(json_file_path='results/visualisation/algorithm/image_results.json', output_dir='results/visualisation/algorithm/evaluation_visualizations'):
    """Exécuter le pipeline complet de visualisation (parce que tout est mieux en pipeline)"""
    print("Chargement et préparation des données...")
    df = load_and_prepare_data(json_file_path)
    
    print(f"Création des visualisations individuelles pour {df['image_name'].nunique()} images...")
    visualize_image_evaluations(df, output_dir)
    
    print("Création des visualisations récapitulatives...")
    create_summary_visualization(df, output_dir)
    
    print(f"Visualisations terminées. Résultats sauvegardés dans {output_dir}/")

# Exemple d'utilisation
if __name__ == "__main__":
    run_visualization()