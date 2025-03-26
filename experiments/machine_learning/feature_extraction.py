import cv2
import numpy as np
import albumentations as A
from config import CUSTOM_CMAP

def extract_features(img):

    # Conversion en niveaux de gris et redimensionnement
    if img is None:
        raise ValueError("L'image est vide ou n'a pas été correctement chargée.")
    
    if len(img.shape) == 2:  # Image déjà en niveaux de gris
        gray = img
    elif len(img.shape) == 3 and img.shape[2] == 3:  # Image RGB classique
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Format d'image inattendu avec {img.shape} canaux.")    
    
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
    features_all= []
    valid_paths_all= []
    valid_labels_all= []
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        img = cv2.imread(image_path)
        img_list=augmentation(img)
        for img in img_list:
            features = extract_features(img)
            if features is not None:
                features_list.append(features)
                valid_labels.append(label)
                valid_paths.append(image_path)
            if not features_list:
                return np.array([]), np.array([]), []
            features_all.append(features_list)
            valid_paths_all.append(valid_paths)
            valid_labels_all.append(valid_labels)


    
    # Standardisation de la longueur des features
    extracted_features=[]
    for i in range(len(features_all)):
        max_length = max(len(f) for f in features_all[i])
        standardized_features = []
        for f in features_all[i]:
            if len(f) < max_length:
                padded = np.zeros(max_length)
                padded[:len(f)] = f
                standardized_features.append(padded)
            else:
                standardized_features.append(f)
        extracted_features.append((np.array(standardized_features), np.array(valid_labels_all[i]), valid_paths_all[i]))
    return extracted_features

def augmentation(img):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.CropAndPad(percent=(-0.1, 0), p=1.0),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Blur(blur_limit=3, p=0.5)
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.5, p=1.0),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.2, 0.2),
            rotate=(-25, 25),
            shear=(-8, 8),
            p=1.0
        ),
    ], p=1.0)  # p=1.0 garantit que toutes les augmentations sont toujours appliquées

    # Albumentations attend une image sous forme de tableau numpy
    img_aug = transform(image=img)['image']
    return img_aug
