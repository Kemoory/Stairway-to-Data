import cv2
import numpy as np

def preprocess_adaptive_thresholding(image):
    """
    Prétraitement d'image avec seuillage adaptatif pour mieux gérer les variations de lumière.
    Utile quand l'éclairage est pas ouf, escaliers mal éclairés.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Filtre bilatéral : réduit le bruit sans flinguer les contours
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Seuillage adaptatif : meilleure gestion des contrastes
    thresh = cv2.adaptiveThreshold(
        bilateral, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Opérations morpho pour mettre en valeur les structures horizontales marches d'escalier)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Détection des contours
    edges = cv2.Canny(horizontal, 50, 150)
    
    return edges
