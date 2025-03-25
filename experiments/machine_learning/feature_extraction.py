import cv2
import numpy as np
from config import CUSTOM_CMAP

def extract_features(image_path):
    """Extract features from an image for stair counting."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Conversion en niveaux de gris et redimensionnement
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (200, 200))
    
    # Détection des contours
    edges = cv2.Canny(resized, 50, 150)
    
    # Détection des lignes horizontales
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    horizontal_line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 20 or angle > 160:
                horizontal_line_count += 1
    
    # HOG features
    win_size = (200, 200)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized)
    
    # Densité des contours
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Gradients
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x_mean = np.mean(np.abs(sobelx))
    gradient_y_mean = np.mean(np.abs(sobely))
    
    # Features personnalisées
    custom_features = np.array([
        horizontal_line_count,
        edge_density,
        gradient_x_mean,
        gradient_y_mean
    ])
    
    # Réduction des dimensions HOG
    hog_features_reduced = hog_features[::20].flatten()
    all_features = np.concatenate([custom_features, hog_features_reduced])
    
    return all_features

def prepare_dataset(image_paths, labels):
    """Process all images and prepare feature vectors and labels."""
    features_list = []
    valid_labels = []
    valid_paths = []
    
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        
        features = extract_features(image_path)
        if features is not None:
            features_list.append(features)
            valid_labels.append(label)
            valid_paths.append(image_path)
    
    if not features_list:
        return np.array([]), np.array([]), []
    
    # Standardisation de la longueur des features
    max_length = max(len(f) for f in features_list)
    standardized_features = []
    for f in features_list:
        if len(f) < max_length:
            padded = np.zeros(max_length)
            padded[:len(f)] = f
            standardized_features.append(padded)
        else:
            standardized_features.append(f)
    
    return np.array(standardized_features), np.array(valid_labels), valid_paths