from data_loader import load_data
from training import cross_validate
from visualization import combine_and_visualize_results
from config_dl import MODEL_CONFIGS, RESULTS_DIR
import json
import os

def main():
    # Chargement des données
    image_paths, labels = load_data()
    
    # Liste des modèles à entraîner
    #model_types = ['resnet18', 'simple_cnn', 'vit', 'stairnet_depth']
    model_types = ['stairnet_depth']
    # Entraînement des modèles
    for model_type in model_types:
        if model_type in MODEL_CONFIGS:
            print(f"\n=== Entraînement du modèle {model_type.upper()} ===")
            result = cross_validate(image_paths, labels, model_type)
            
            # Informations sur les métriques moyennes
            print(f"\n--- Métriques moyennes pour {model_type} ---")
            for metric, value in result['avg_metrics'].items():
                print(f"{metric}: {value if value is not None else 'N/A'}")
        else:
            print(f"Configuration introuvable pour le modèle {model_type}")
    
    # Sauvegarde et visualisation
    combine_and_visualize_results()
    
    print(f"\nAnalyse terminée. Les résultats sont disponibles dans le dossier: {RESULTS_DIR}")

if __name__ == '__main__':
    main()