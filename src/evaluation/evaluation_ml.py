import json
import os
import cv2
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score

def extract_features(image):
    """
    Extraire des caracteristiques d'une image pour le comptage des marches.
    Retourne un vecteur de caracteristiques pour l'image donnee.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    edges = cv2.Canny(resized, 50, 150)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    horizontal_line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 20 or angle > 160:
                horizontal_line_count += 1
    
    win_size = (200, 200)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized)
    
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x_mean = np.mean(np.abs(sobelx))
    gradient_y_mean = np.mean(np.abs(sobely))
    
    custom_features = np.array([
        horizontal_line_count,
        edge_density,
        gradient_x_mean,
        gradient_y_mean
    ])
    
    hog_features_reduced = hog_features[::20].flatten()
    all_features = np.concatenate([custom_features, hog_features_reduced])
    
    return all_features

def evaluate_ml_model(image_paths, ground_truth, model_path, model_name):
        """
        Évaluer un modèle d'apprentissage automatique sur les images et les valeurs de vérité terrain données.

        Args:
            image_paths: Liste des chemins vers les images.
            ground_truth: Dictionnaire des valeurs de vérité terrain (nom_image: nombre_de_marches).
            model_path: Chemin vers le fichier .pkl contenant le modèle ML entraîné.
            model_name: Nom du modèle (par exemple, 'svr', 'random_forest').

        Returns:
            results: Dictionnaire contenant les métriques d'évaluation.
            image_results: Dictionnaire contenant les prédictions pour chaque image.
        """
        # Charger le modèle ML
        model = joblib.load(model_path)
        
        # Initialiser les dictionnaires pour stocker les résultats
        preds = {}
        image_results = {}
        
        # Itérer sur toutes les images
        for img_path in image_paths:
            img = cv2.imread(img_path)
            img_name = os.path.basename(img_path)
            
            # Extraire les caractéristiques
            features = extract_features(img)
            
            # S'assurer que la forme des caractéristiques correspond aux données d'entraînement
            if model.n_features_in_ > len(features):
                padded = np.zeros(model.n_features_in_)
                padded[:len(features)] = features
                features = padded
            elif model.n_features_in_ < len(features):
                features = features[:model.n_features_in_]
            
            preds[img_name] = int(round(model.predict(features.reshape(1, -1))[0]))
            
            # Sauvegarder les résultats pour l'image
            if img_name in ground_truth:
                image_results[img_name] = {
                    'model': model_name,
                    'prediction': preds[img_name],
                    'ground_truth': ground_truth[img_name]
                }
        
        # Calculer les métriques d'évaluation
        mae_value = mae([ground_truth[img] for img in preds.keys() if img in ground_truth],
                        [preds[img] for img in preds.keys() if img in ground_truth])
        mse_value = mse([ground_truth[img] for img in preds.keys() if img in ground_truth],
                        [preds[img] for img in preds.keys() if img in ground_truth])
        rmse_value = np.sqrt(mse_value)
        r2_value = r2_score([ground_truth[img] for img in preds.keys() if img in ground_truth],
                            [preds[img] for img in preds.keys() if img in ground_truth])
        rel_error = np.mean([abs(preds[img] - ground_truth[img]) / ground_truth[img] for img in preds.keys() if img in ground_truth and ground_truth[img] > 0])
        
        # Sauvegarder les résultats
        results = {
            'model': model_name,
            'MAE': mae_value,
            'MSE': mse_value,
            'RMSE': rmse_value,
            'R2_score': r2_value,
            'Relative Error': rel_error,
        }
        
        return results, image_results