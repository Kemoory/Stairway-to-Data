# src/preprocessing.py
import cv2
import numpy as np

def preprocess_image(image):
    """Convertit en niveaux de gris et applique un flou gaussien"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def preprocess_image_alternative(image):
    """Methode de pre-traitement alternative"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 11)  # Use median blur instead of Gaussian
    edges = cv2.Canny(blurred, 75, 200)  # Different thresholds
    return edges

def split(image, threshold):
    """Divise l'image en regions basees sur la variance d'intensite"""
    h, w = image.shape[:2]
    regions = []
    stack = [(0, 0, w, h)]
    print("Debut du processus de division...")

    while stack:
        x, y, w, h = stack.pop()
        region = image[y:y+h, x:x+w]
        if region.size == 0:
            continue
        var = region.var()
        print(f"Traitement de la region   ({x}, {y}) avec taille ({w}, {h}) et variance {var:.2f}")
        if var > threshold:
            h2, w2 = h // 2, w // 2
            print(f"Division de la region   ({x}, {y}) en 4 sous-regions")
            stack.append((x, y, w2, h2))
            stack.append((x + w2, y, w - w2, h2))
            stack.append((x, y + h2, w2, h - h2))
            stack.append((x + w2, y + h2, w - w2, h - h2))
        else:
            print(f"La region   ({x}, {y}) repond aux criteres de variance. Ajout a la liste des regions finales.")
            regions.append((x, y, w, h))
    print("Fin du processus de division.")
    return regions

def merge(image, regions, threshold):
    """Fusionne les regions avec des intensites semblables"""
    merged_image = np.zeros_like(image)
    print("Debut du processus de fusion...")
    for i, (x, y, w, h) in enumerate(regions):
        region = image[y:y+h, x:x+w]
        mean = region.mean()
        print(f"Fusion de la region {i + 1}   ({x}, {y}) avec intensite  moyenne {mean:.2f}")
        merged_image[y:y+h, x:x+w] = mean
    print("Fin du processus de fusion.")
    return merged_image
