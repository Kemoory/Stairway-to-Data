import cv2
import numpy as np
import pywt
from scipy.signal import find_peaks, savgol_filter
from config import CUSTOM_CMAP

def wavelet_edge_detection(image):
    """Appliquer la transformée en ondelettes pour la détection des bords"""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Transformée en ondelettes à 2 niveaux
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    
    # Améliorer les détails horizontaux
    LH1 = coeffs[1][1] * 1.5  # Détails horizontaux niveau 1
    LH2 = coeffs[2][1] * 1.5  # Détails horizontaux niveau 2
    
    # Reconstituer les coefficients
    coeffs[1] = (coeffs[1][0], LH1, coeffs[1][2])
    coeffs[2] = (coeffs[2][0], LH2, coeffs[2][2])
    
    # Reconstituer l'image
    reconstructed = pywt.waverec2(coeffs, 'haar')
    reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(reconstructed)

def extract_hog_features(image):
    """Extraire les caractéristiques HOG de l'image"""
    win_size = (200, 200)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog.compute(image)

def extract_features(image_path):
    """Fonction principale d'extraction de caractéristiques avec détection de marches améliorée"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Redimensionner et convertir en niveaux de gris
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Traitement des ondelettes amélioré
    wavelet_img = wavelet_edge_detection(img)
    
    # Collecte des caractéristiques
    features = []
    
    # 1. Détection de marches améliorée par profil d'intensité
    step_count = intensity_profile_detection(img)
    features.append(step_count)
    
    # 2. Analyse de bordures supplémentaire
    edges = cv2.Canny(wavelet_img, 30, 150)  # Seuil ajusté
    edge_features = [
        np.sum(edges > 0),  # Total des pixels de bord
        cv2.countNonZero(edges),  # Pixels de bord non nuls
        np.mean(edges > 0)  # Densité de pixels de bord
    ]
    features.extend(edge_features)
    
    # 3. Caractéristiques basées sur le gradient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_features = [
        np.mean(np.abs(sobelx)), 
        np.mean(np.abs(sobely)),
        np.std(sobelx),  # Écart type ajouté
        np.std(sobely)   # Écart type ajouté
    ]
    features.extend(gradient_features)
    
    return np.array(features)

def intensity_profile_detection(image):
    """Détection améliorée de marches en utilisant un profil d'intensité"""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculer le profil d'intensité horizontal
    profile = np.mean(gray, axis=1)
    
    # Calculer la dérivée avec réduction de bruit
    derivative = np.diff(profile)
    smoothed_derivative = savgol_filter(derivative, window_length=11, polyorder=3)
    
    # Trouver les pics dans la dérivée pour détecter les bords de marches potentiels
    peaks, _ = find_peaks(np.abs(smoothed_derivative), height=np.std(smoothed_derivative) * 2, distance=10)
    
    return len(peaks)

def prepare_dataset(image_paths, labels):
    """Préparer le jeu de données avec les caractéristiques extraites"""
    features = []
    valid_labels = []
    valid_paths = []
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            valid_labels.append(label)
            valid_paths.append(path)
    
    if not features:
        return np.array([]), np.array([]), []
    
    # Standardiser la longueur des caractéristiques
    max_len = max(len(f) for f in features)
    standardized = [np.pad(f, (0, max_len - len(f))) if len(f) < max_len else f for f in features]
    
    return np.array(standardized), np.array(valid_labels), valid_paths

