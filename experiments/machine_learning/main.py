from data_loader import load_data
from feature_extraction import prepare_dataset
from training import train_model
from visualization import combine_and_visualize_results
from config import MODEL_PARAMS

def main():
    # Load data
    image_paths, labels = load_data()
    
    # Prepare dataset
    extracted_features = prepare_dataset(image_paths, labels)
    
    # Train all models
    for model_type in MODEL_PARAMS.keys():
        print(f"\nTraining {model_type} model...")
        for t in extracted_features:
            train_model(t[0], t[1], t[3], model_type)
    
    # Combine and visualize results
    combine_and_visualize_results()

if __name__ == "__main__":
    main()