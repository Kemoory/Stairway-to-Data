# src/preprocessing/median.py
import cv2
import numpy as np

def preprocess_median(image):
    """Methode de pre-traitement utilisant median"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 11)  # Use median blur instead of Gaussian
    edges = cv2.Canny(blurred, 75, 200)  # Different thresholds
    return edges