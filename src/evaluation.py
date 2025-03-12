import json
import os
import cv2
import numpy as np
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score

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

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def calculate_mean_absolute_error(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return mae(gt_values, pred_values)

def calculate_mean_squared_error(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return mse(gt_values, pred_values)

def calculate_root_mean_squared_error(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return np.sqrt(mse(gt_values, pred_values))

def calculate_r2_score(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return r2_score(gt_values, pred_values)

def calculate_relative_error(preds, ground_truth):
    errors = []
    for img in preds.keys():
        if img in ground_truth:
            gt = ground_truth[img]
            pred = preds[img]
            if gt > 0:
                errors.append(abs(pred - gt) / gt)
    return sum(errors) / len(errors) if errors else 0

def evaluate_model(preds, ground_truth):
    mae = calculate_mean_absolute_error(preds, ground_truth)
    mse = calculate_mean_squared_error(preds, ground_truth)
    rmse = calculate_root_mean_squared_error(preds, ground_truth)
    r2 = calculate_r2_score(preds, ground_truth)
    rel_error = calculate_relative_error(preds, ground_truth)
    return mae, mse, rmse, r2, rel_error

# Fonction pour evaluer toutes les combinaisons de src.preprocessing et de modeles
def evaluate_all_combinations(image_paths, ground_truth):
    results = []
    image_results = {}
    
    # Count the total number of images
    total_images = len(image_paths)
    print(f"Total number of images: {total_images}")
    
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
        'Intensity Profile': detect_steps_intensity_profile,
        'Contour Hierarchy': detect_steps_contour_hierarchy,
        'Edge Distance': detect_steps_edge_distance,
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
            mae, mse, rmse, r2, rel_error = evaluate_model(preds, ground_truth)
            
            # Sauvegarde des resultats
            results.append({
                'preprocessing': preprocess_name,
                'model': model_name,
                'MAE': mae, #plus c'est bas mieux c'est
                'MSE': mse, #plus c'est bas mieux c'est
                'RMSE': rmse, #plus c'est bas mieux c'est
                'R2_score': r2, #plus c'est haut mieux c'est
                'Relative Error': rel_error,
            })
            
            # Sauvegarde des resultats par image
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
    
    # Balance les resultats dans un fichier JSON
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
<<<<<<< HEAD

    print("Evaluation complete. Results saved to 'evaluation_results.json'.")

    # Charger les résultats du fichier JSON
    with open('evaluation_results.json', 'r') as f:
        results = json.load(f)

    # Convertir en DataFrame pour un affichage facile
    df = pd.DataFrame(results)

    # Configuration du style
    sns.set_theme(style="whitegrid")

    # Graphique en barres pour comparer les modèles
    plt.figure(figsize=(12, 6))
    metrics = ["MAE", "MSE", "RMSE", "R2_score", "Relative Error"]
    df_melted = df.melt(id_vars=["preprocessing", "model"], value_vars=metrics, var_name="Metric", value_name="Score")

    sns.barplot(data=df_melted, x="Metric", y="Score", hue="model")
    plt.title("Comparaison des modèles en fonction des métriques")
    plt.xticks(rotation=45)
    plt.legend(title="Modèle", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

    # Heatmap des erreurs MAE
    plt.figure(figsize=(10, 5))
    heatmap_data = df.pivot(index="model", columns="preprocessing", values="MAE")
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Heatmap des erreurs MAE pour chaque combinaison")
    plt.ylabel("Modèle")
    plt.xlabel("Prétraitement")
    plt.show()

    # Courbe des erreurs MAE
    plt.figure(figsize=(10, 5))
    for model in df["model"].unique():
        subset = df[df["model"] == model]
        plt.plot(subset["preprocessing"], subset["MAE"], marker="o", label=model)

    plt.xlabel("Prétraitement")
    plt.ylabel("MAE")
    plt.title("Comparaison des erreurs MAE en fonction du prétraitement")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()

    return results
=======
    
    # Balance les resultats par image dans un fichier JSON
    with open('image_results.json', 'w') as f:
        json.dump(image_results, f, indent=4)
    
    print("Evaluation complete. Results saved to 'evaluation_results.json' and 'image_results.json'.")
    return results, image_results
>>>>>>> f2399ca (Mess to sort)
