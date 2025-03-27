from data_loader import load_data
from feature_extraction import prepare_dataset
from training import train_model
from visualization import combine_and_visualize_results
from config import MODEL_PARAMS

def main():
    # Load data
    image_paths, labels = load_data()
    
    # Prepare dataset
    features, labels, valid_paths = prepare_dataset(image_paths, labels)
    
    if features.size == 0 or labels.size == 0:
        print("No valid data to train models.")
        return
    
    # Liste étendue des modèles à entraîner
    model_types = list(MODEL_PARAMS.keys())
    
    # Ajout des modèles SVR avancés
    advanced_svr_kernels = ['advanced_svr_rbf', 'advanced_svr_polynomial', 'advanced_svr_sigmoid']
    model_types.extend(advanced_svr_kernels)
    
    # Train all models
    for model_type in model_types:
        try:
            print(f"\nTraining {model_type} model...")
            train_model(features, labels, valid_paths, model_type)
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
    
    # Combine and visualize results
    combine_and_visualize_results()

if __name__ == "__main__":
    main()