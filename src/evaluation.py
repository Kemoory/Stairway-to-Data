import json
import os
import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error as mae, precision_score, recall_score

from src.preprocessing.gaussian import preprocess_gaussian
from src.preprocessing.median import preprocess_median
from src.preprocessing.splitAndMerge import preprocess_splitAndMerge
from src.preprocessing.adaptive_tresholding import preprocess_adaptive_thresholding
from src.preprocessing.gradient_orientation import preprocess_gradient_orientation
from src.preprocessing.homorphic_filter import preprocess_homomorphic_filter
from src.preprocessing.phase_congruency import preprocess_phase_congruency
from src.preprocessing.wavelet import preprocess_image_wavelet

from src.model.houghLineSeg import detect_steps_houghLineSeg
from src.model.houghLineExt import detect_steps_houghLineExt
from src.model.RANSAC import detect_steps_RANSAC
from src.model.vanishingLine import detect_vanishing_lines

def calculate_mean_absolute_error(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return mae(gt_values, pred_values)

def calculate_relative_error(preds, ground_truth):
    errors = []
    for img in preds.keys():
        if img in ground_truth:
            gt = ground_truth[img]
            pred = preds[img]
            if gt > 0:
                errors.append(abs(pred - gt) / gt)
    return sum(errors) / len(errors) if errors else 0

def calculate_precision_recall(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    precision = precision_score(gt_values, pred_values, average='macro', zero_division=0)
    recall = recall_score(gt_values, pred_values, average='macro', zero_division=0)
    return precision, recall

def evaluate_model(preds, ground_truth):
    error = calculate_mean_absolute_error(preds, ground_truth)
    rel_error = calculate_relative_error(preds, ground_truth)
    precision, recall = calculate_precision_recall(preds, ground_truth)
    return error, rel_error, precision, recall

# Fonction pour evaluer toutes les combinaisons de src.preprocessing et de modeles
def evaluate_all_combinations(image_paths, ground_truth):
    results = []
    
    # Definition des modeles et des preprocessing
    preprocessing_methods = {
        '(None)': lambda img: img.copy(),
        'Gaussian Blur + Canny': preprocess_gaussian,
        'Median Blur + Canny': preprocess_median,
        #'Split and Merge': preprocess_splitAndMerge, #Trop Lourd
        'Adaptive Thresholding': preprocess_adaptive_thresholding,
        'Gradient Orientation': preprocess_gradient_orientation,
        'Homomorphic Filter': preprocess_homomorphic_filter,
        'Phase Congruency': preprocess_phase_congruency,
        'Wavelet Transform': preprocess_image_wavelet,
    }
    
    models = {
        'HoughLinesP (Segmented)': detect_steps_houghLineSeg,
        'HoughLinesP (Extended)': detect_steps_houghLineExt,
        'Vanishing Lines': detect_vanishing_lines,
        'RANSAC (WIP)': detect_steps_RANSAC,
    }
    
    # Iteretion sur toutes les combinaisons de preprocessing et de modeles
    for preprocess_name, preprocess_func in preprocessing_methods.items():
        for model_name, model_func in models.items():
            print(f"Evaluating combination: {preprocess_name} + {model_name}")
            
            preds = {}
            for img_path in image_paths:
                img = cv2.imread(img_path)
                img_name = os.path.basename(img_path)
                
                # Print l'image en cours de traitement
                print(f"Evaluating image: {img_name}")
                
                # Preprocess image
                processed = preprocess_func(img)
                
                # S'assure que l'image est en niveaux de gris et en uint8
                if len(processed.shape) > 2:
                    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
                if processed.dtype != np.uint8:
                    processed = cv2.convertScaleAbs(processed)
                
                # Applique le modele
                count, _ = model_func(processed, img.copy())
                preds[img_name] = count
            
            # Evaluation des resultats
            error, rel_error, precision, recall = evaluate_model(preds, ground_truth)
            
            # Sauvegarde des resultats
            results.append({
                'src.preprocessing': preprocess_name,
                'model': model_name,
                'MAE': error,
                'Relative Error': rel_error,
                'Precision': precision,
                'Recall': recall,
            })
    
    # Balance les resultats dans un fichier JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Evaluation complete. Results saved to 'evaluation_results.json'.")
    return results