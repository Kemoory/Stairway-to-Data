from data_loader import load_data
from training import cross_validate
from visualization import save_results
from config_dl import MODEL_CONFIGS, RESULTS_DIR
import json
import os

def main():
    # Chargement des données
    image_paths, labels = load_data()
    
    # Entraînement des modèles
    all_results = []
    for model_type in MODEL_CONFIGS.keys():
        print(f"\n=== Entraînement du modèle {model_type.upper()} ===")
        result = cross_validate(image_paths, labels, model_type)
        all_results.append(result)
    
    # Sauvegarde et visualisation
    save_results(all_results)
    with open(os.path.join(RESULTS_DIR, 'full_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == '__main__':
    main()