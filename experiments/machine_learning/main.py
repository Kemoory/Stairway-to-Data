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
    
    # Train all models
    for model_type in MODEL_PARAMS.keys():
        print(f"\nTraining {model_type} model...")
        train_model(features, labels, valid_paths, model_type)
    
    # Combine and visualize results
    combine_and_visualize_results()

if __name__ == "__main__":
    main()