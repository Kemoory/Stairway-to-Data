import cv2
import numpy as np

def preprocess_gradient_orientation(image):
    """
    Booste les bords horizontaux avec un filtrage basé sur l'orientation du gradient.
    Utile pour capter les structures bien droites
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcul des gradients avec Sobel (détecte les changements de luminosité)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Dérivée en X
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Dérivée en Y
    
    # Module et orientation du gradient (force et direction du changement)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi  # Converti en degrés
    
    # Masque les gradients quasi-horizontaux (±20° autour de l'horizontale)
    horizontal_mask = np.zeros_like(gray)
    horizontal_mask[np.where((orientation >= -20) & (orientation <= 20) | 
                             (orientation >= 160) | (orientation <= -160))] = 1
    
    # Applique le masque sur la magnitude du gradient
    horizontal_edges = (magnitude * horizontal_mask).astype(np.uint8)
    
    # Normalisation sur 8 bits (0-255)
    horizontal_edges = cv2.normalize(horizontal_edges, None, 0, 255, cv2.NORM_MINMAX)
    
    # Seuillage pour obtenir une image binaire
    _, binary = cv2.threshold(horizontal_edges, 50, 255, cv2.THRESH_BINARY)
    
    # Nettoyage avec morpho (comble les petits trous)
    kernel = np.ones((1, 15), np.uint8)  # Kernel allongé pour garder les lignes nettes
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned
