import cv2
import numpy as np
import joblib
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
from src.models.edge_distance import detect_steps_edge_distance
from src.models.intensity_profile import detect_steps_intensity_profile
from src.models.contour_hierarchy import detect_steps_contour_hierarchy

def preprocess_image(image, method):
    """
    Prétraiter l'image en fonction de la méthode sélectionnée.

    Args:
        image: L'image d'entrée (tableau numpy).
        method: La méthode de prétraitement (chaîne de caractères).

    Returns:
        processed: L'image prétraitée (tableau numpy).
    """
    if method == 'Gaussian Blur + Canny':
        processed = preprocess_gaussian(image)
    elif method == 'Median Blur + Canny':
        processed = preprocess_median(image)
    elif method == 'Split and Merge':
        processed = preprocess_splitAndMerge(image)
    elif method == 'Adaptive Thresholding':
        processed = preprocess_adaptive_thresholding(image)
    elif method == 'Gradient Orientation':
        processed = preprocess_gradient_orientation(image)
    elif method == 'Homomorphic Filter':
        processed = preprocess_homomorphic_filter(image)
    elif method == 'Phase Congruency':
        processed = preprocess_phase_congruency(image)
    elif method == 'Wavelet Transform':
        processed = preprocess_image_wavelet(image)
    else:
        processed = image.copy()  # Pas de prétraitement

    # S'assurer que l'image est en niveaux de gris et au format uint8
    if len(processed.shape) > 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if processed.dtype != np.uint8:
        processed = cv2.convertScaleAbs(processed)

    return processed

def detect_steps(image, method, original_image, ground_truth=None):
    """
    Détecter les marches dans l'image en utilisant la méthode sélectionnée.

    Args:
        image: L'image prétraitée (tableau numpy).
        method: La méthode de détection des marches (chaîne de caractères).
        original_image: L'image originale (tableau numpy).

    Returns:
        count: Le nombre de marches détectées (entier).
        debug_image: L'image de débogage avec les marches détectées (tableau numpy).
    """
    if method == 'HoughLinesP (Segmented)':
        count, debug_image = detect_steps_houghLineSeg(image, original_image.copy())
    elif method == 'HoughLinesP (Extended)':
        count, debug_image = detect_steps_houghLineExt(image, original_image.copy())
    elif method == 'Vanishing Lines':
        count, debug_image = detect_vanishing_lines(image, original_image.copy())
    elif method == 'RANSAC':
        count, debug_image = detect_steps_RANSAC(image, original_image.copy())
    elif method == 'Edge Distance':
        count, debug_image = detect_steps_edge_distance(image, original_image.copy())
    elif method == 'Intensity Profile':
        count, debug_image = detect_steps_intensity_profile(image, original_image.copy())
    elif method == 'Contour Hierarchy':
        count, debug_image = detect_steps_contour_hierarchy(image, original_image.copy())
    elif method == 'Random Forest Regressor':  # Ajouter la prise en charge du modèle Random Forest
        model = load_ml_model("src/models/random_forest_model.pkl")
        if model:
            count = predict_with_ml(model, original_image)
            debug_image = original_image.copy()
        else:
            count, debug_image = 0, original_image.copy()
    elif method == 'MLP':  #
        model = load_ml_model("src/models/mlp_model.pkl")
        if model:
            count = predict_with_ml(model, original_image)
            debug_image = original_image.copy()
        else:
            count, debug_image = 0, original_image.copy()
    elif method == 'Gradient Boosting': 
        model = load_ml_model("src/models/gradient_boosting_model.pkl")
        if model:
            count = predict_with_ml(model, original_image)
            debug_image = original_image.copy()
        else:
            count, debug_image = 0, original_image.copy()
    elif method == 'Support Vector Regressor':  # Ajouter la prise en charge du modèle SVR
        model = load_ml_model("src/models/svr_model.pkl")
        if model:
            count = predict_with_ml(model, original_image)
            debug_image = original_image.copy()
        else:
            count, debug_image = 0, original_image.copy()
    else:
        count, debug_image = 0, original_image.copy()  # Par défaut : aucune marche détectée

    # Ajouter à la fin de la fonction :
    if ground_truth is not None:
        # Dessiner la vérité terrain sur l'image de débogage
        debug_image = draw_ground_truth(debug_image, count, ground_truth)
    
    return count, debug_image

# Charger le modèle ML depuis le fichier .pkl
def load_ml_model(model_path):
    """
    Charger le modèle d'apprentissage automatique depuis un fichier .pkl.

    Args:
        model_path: Chemin vers le fichier .pkl.

    Returns:
        model: Le modèle d'apprentissage automatique chargé.
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return None
    
def draw_ground_truth(image, prediction, ground_truth):
    # Créer un overlay élégant
    overlay = image.copy()
    height, width = image.shape[:2]
    
    # Positionner le cadre en bas à droite
    text = f"Prediction: {prediction}"
    gt_text = f"Ground Truth: {ground_truth}"
    
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    
    # Calculer la position
    x = width - 300
    y = height - 120
    box_height = 100 if ground_truth else 60
    
    # Dessiner le fond
    cv2.rectangle(overlay, (x, y), (x + 280, y + box_height), (30, 30, 30), -1)
    
    # Ajouter la transparence
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Ajouter le texte
    cv2.putText(image, text, (x + 10, y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
    
    cv2.putText(image, gt_text, (x + 10, y + 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
    
    # Ajouter une icône décorative
    cv2.rectangle(image, (x + 200, y + 10), (x + 230, y + 40), (100, 255, 100), 2)  # Carré vert
    cv2.line(image, (x + 240, y + 10), (x + 270, y + 40), (100, 200, 255), 2)  # Ligne bleue
    
    return image

# Prétraiter l'image pour le modèle ML
def preprocess_for_ml(image):
    """
    Prétraiter l'image pour correspondre au format d'entrée attendu par le modèle ML.

    Args:
        image: L'image d'entrée (tableau numpy).

    Returns:
        features: Les caractéristiques prétraitées (tableau numpy).
    """
    # Extraire les caractéristiques en utilisant la même logique que lors de l'entraînement
    features = extract_features(image)
    return features

# Faire des prédictions en utilisant le modèle ML
def predict_with_ml(model, image):
    """
    Faire des prédictions en utilisant le modèle ML.

    Args:
        model: Le modèle ML chargé.
        image: L'image d'entrée (tableau numpy).

    Returns:
        prediction: Le nombre prédit de marches (entier).
    """
    try:
        # Prétraiter l'image
        features = preprocess_for_ml(image)
        
        # S'assurer que la forme des caractéristiques correspond aux données d'entraînement
        if model.n_features_in_ > len(features):
            padded = np.zeros(model.n_features_in_)
            padded[:len(features)] = features
            features = padded
        elif model.n_features_in_ < len(features):
            features = features[:model.n_features_in_]
        
        # Faire une prédiction
        prediction = model.predict(features.reshape(1, -1))
        return int(round(prediction[0]))  # Arrondir à l'entier le plus proche
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return 0

def extract_features(image):
    """
    Extraire des caractéristiques d'une image pour le comptage des marches.
    Retourne un vecteur de caractéristiques pour l'image donnée.
    """
    # Convertir en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Redimensionner à une taille standard
    resized = cv2.resize(gray, (200, 200))
    
    # Détection des contours - utile pour trouver les bords des marches
    edges = cv2.Canny(resized, 50, 150)
    
    # Détection des lignes horizontales à l'aide de la transformée de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    # Compter les lignes horizontales (marches potentielles)
    horizontal_line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Considérer les lignes quasi-horizontales
            if angle < 20 or angle > 160:
                horizontal_line_count += 1
    
    # Extraire les caractéristiques de l'histogramme des gradients (HOG)
    win_size = (200, 200)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized)
    
    # Caractéristiques de densité des contours
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Caractéristiques des gradients horizontaux et verticaux
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x_mean = np.mean(np.abs(sobelx))
    gradient_y_mean = np.mean(np.abs(sobely))
    
    # Combiner les caractéristiques
    custom_features = np.array([
        horizontal_line_count,
        edge_density,
        gradient_x_mean,
        gradient_y_mean
    ])
    
    # Réduire la dimensionnalité des caractéristiques HOG (prendre un sous-ensemble)
    hog_features_reduced = hog_features[::20].flatten()
    
    # Combiner toutes les caractéristiques
    all_features = np.concatenate([custom_features, hog_features_reduced])
    
    return all_features