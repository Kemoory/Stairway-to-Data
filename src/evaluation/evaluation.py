import json
import os
import cv2
import numpy as np
from .utils import calculate_mean_absolute_error, calculate_mean_squared_error, calculate_root_mean_squared_error, calculate_r2_score, calculate_relative_error
from .evaluation_ml import evaluate_ml_model

from src.preprocessing.gaussian import preprocess_gaussian
from src.preprocessing.median import preprocess_median
from src.preprocessing.split_and_merge import preprocess_splitAndMerge
from src.preprocessing.adaptive_thresholding import preprocess_adaptive_thresholding
from src.preprocessing.gradient_orientation import preprocess_gradient_orientation
from src.preprocessing.homomorphic_filter import preprocess_homomorphic_filter
from src.preprocessing.phase_congruency import preprocess_phase_congruency
from src.preprocessing.wavelet import preprocess_image_wavelet

from src.models.hough_line_seg import detect_steps_houghLineSeg
from src.models.hough_line_ext import detect_steps_houghLineExt
from src.models.ransac import detect_steps_RANSAC
from src.models.vanishing_line import detect_vanishing_lines
from src.models.intensity_profile import detect_steps_intensity_profile
from src.models.contour_hierarchy import detect_steps_contour_hierarchy
from src.models.edge_distance import detect_steps_edge_distance

def evaluate_all_combinations(image_paths, ground_truth):
    results = []
    image_results = {}
    
    # Compter le nombre total d'images
    total_images = len(image_paths)
    print(f"Nombre total d'images : {total_images}")
    
    # Définition des méthodes de prétraitement
    preprocessing_methods = {
        #'(None)': lambda img: img.copy(),
        'Gaussian Blur + Canny': preprocess_gaussian,
        'Median Blur + Canny': preprocess_median,
        #'Split and Merge': preprocess_splitAndMerge,
        #'Adaptive Thresholding': preprocess_adaptive_thresholding,
        #'Gradient Orientation': preprocess_gradient_orientation,
        #'Homomorphic Filter': preprocess_homomorphic_filter,
        #'Phase Congruency': preprocess_phase_congruency,
        #'Wavelet Transform': preprocess_image_wavelet,
    }
    
    # Définition des modèles
    models = {
        #'HoughLinesP (Segmented)': detect_steps_houghLineSeg,
        'HoughLinesP (Extended)': detect_steps_houghLineExt,
        #'Vanishing Lines': detect_vanishing_lines,
        #'RANSAC (WIP)': detect_steps_RANSAC,
        #'Intensity Profile': detect_steps_intensity_profile,
        #'Contour Hierarchy': detect_steps_contour_hierarchy,
        #'Edge Distance': detect_steps_edge_distance,
    }
    
    # Itérer sur toutes les combinaisons de prétraitement et de modèles
    for preprocess_name, preprocess_func in preprocessing_methods.items():
        for model_name, model_func in models.items():
            print(f"Évaluation de la combinaison : {preprocess_name} + {model_name}")
            
            preds = {}
            for img_path in image_paths:
                img = cv2.imread(img_path)
                img_name = os.path.basename(img_path)
                
                # Afficher l'image en cours de traitement
                print(f"Évaluation de l'image : {img_name}")
                
                # Prétraiter l'image
                processed = preprocess_func(img)
                
                # S'assurer que l'image est en niveaux de gris et au format uint8
                if len(processed.shape) > 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                if processed.dtype != np.uint8:
                    processed = cv2.convertScaleAbs(processed)
                
                # Appliquer le modèle
                count, _ = model_func(processed, img.copy())
                preds[img_name] = count
            
            # Évaluer les résultats
            mae_value = calculate_mean_absolute_error(preds, ground_truth)
            mse_value = calculate_mean_squared_error(preds, ground_truth)
            rmse_value = calculate_root_mean_squared_error(preds, ground_truth)
            r2_value = calculate_r2_score(preds, ground_truth)
            rel_error = calculate_relative_error(preds, ground_truth)
            
            # Sauvegarder les résultats
            results.append({
                'preprocessing': preprocess_name,
                'model': model_name,
                'MAE': mae_value,
                'MSE': mse_value,
                'RMSE': rmse_value,
                'R2_score': r2_value,
                'Relative Error': rel_error,
            })
            
            # Sauvegarder les résultats par image
            for img_name in preds.keys():
                if img_name in ground_truth:
                    if img_name not in image_results:
                        image_results[img_name] = []
                    image_results[img_name].append({
                        'preprocessing': preprocess_name,
                        'model': model_name,
                        'prediction': preds[img_name],
                        'ground_truth': ground_truth[img_name]
                    })
    
    # Évaluer les modèles d'apprentissage automatique
    ml_models = {
        'svr': 'src/models/svr_model.pkl',
        'random_forest': 'src/models/random_forest_model.pkl',
        'gradient_boosting_model': 'src/models/gradient_boosting_model.pkl',
        'mlp': 'src/models/mlp_model.pkl',
    }
    
    for model_name, model_path in ml_models.items():
        print(f"Évaluation du modèle {model_name}...")
        ml_results, ml_image_results = evaluate_ml_model(image_paths, ground_truth, model_path, model_name)
        
        # Sauvegarder les résultats ML
        results.append(ml_results)
        
        # Sauvegarder les résultats ML par image
        for img_name, result in ml_image_results.items():
            if img_name not in image_results:
                image_results[img_name] = []
            image_results[img_name].append(result)
    
    # Créer les répertoires s'ils n'existent pas
    os.makedirs('results/visualisation/machine_learning', exist_ok=True)
    os.makedirs('results/visualisation/algorithm', exist_ok=True)
    
    # Sauvegarder les résultats des algorithmes dans des fichiers JSON
    with open('results/visualisation/algorithm/algorithm_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    with open('results/visualisation/algorithm/algorithm_image_results.json', 'w') as f:
        json.dump(image_results, f, indent=4)
    
    # Sauvegarder les résultats ML dans des fichiers JSON
    with open('results/visualisation/machine_learning/ml_evaluation_results.json', 'w') as f:
        json.dump(ml_results, f, indent=4)
    
    with open('results/visualisation/machine_learning/ml_image_results.json', 'w') as f:
        json.dump(ml_image_results, f, indent=4)
    
    print("Évaluation terminée. Résultats sauvegardés dans :")
    print("- 'results/visualisation/algorithm/algorithm_evaluation_results.json' (résultats des algorithmes)")
    print("- 'results/visualisation/algorithm/algorithm_image_results.json' (résultats des images pour les algorithmes)")
    print("- 'results/visualisation/machine_learning/ml_evaluation_results.json' (résultats ML)")
    print("- 'results/visualisation/machine_learning/ml_image_results.json' (résultats des images pour les ML)")
    
    return results, image_results