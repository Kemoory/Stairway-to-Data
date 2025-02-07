# src/preprocessing.py
import cv2

def preprocess_image(image):
    """Convertit en niveaux de gris et applique un flou gaussien"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges