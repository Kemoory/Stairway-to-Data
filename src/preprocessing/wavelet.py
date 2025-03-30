import cv2
import numpy as np
import pywt  # Pour les ondelettes, parce que les vagues, c'est cool

def preprocess_image_wavelet(image):
    """Applique une transformée en ondelettes pour améliorer la détection des bords.
    
    Les transformées en ondelettes permettent de détecter des caractéristiques à différentes échelles
    et sont particulièrement efficaces pour détecter les bords horizontaux à différentes résolutions.
    """
    # Convertit en niveaux de gris si nécessaire
    if len(image.shape) > 2:  # Si l'image est en couleur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # On passe en gris
    else:
        gray = image.copy()  # Sinon, on garde l'image telle quelle
    
    # Applique la transformée en ondelettes (2 niveaux)
    coeffs = pywt.wavedec2(gray, 'haar', level=2)  # "haar" pour Haar, pas pour les cheveux
    
    # Extrait les coefficients de détails horizontaux des deux niveaux
    _, (LH1, _, _) = coeffs[0], coeffs[1]  # LH1 = détails horizontaux niveau 1
    _, (LH2, _, _) = coeffs[0], coeffs[2]  # LH2 = détails horizontaux niveau 2
    
    # Améliore les détails horizontaux (on booste un peu les bords horizontaux)
    LH1 = LH1 * 1.5  # On rend les bords plus visibles
    LH2 = LH2 * 1.5  # Pareil pour le niveau 2
    
    # Reconstruit les coefficients modifiés
    coeffs[1] = (coeffs[1][0], LH1, coeffs[1][2])  # On remplace LH1
    coeffs[2] = (coeffs[2][0], LH2, coeffs[2][2])  # On remplace LH2
    
    # Transformée inverse en ondelettes (on reconstruit l'image)
    reconstructed = pywt.waverec2(coeffs, 'haar')  # On remet tout ensemble
    
    # Normalise et reconvertit en uint8 (pour que l'image soit affichable)
    reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)  # On normalise
    reconstructed = np.uint8(reconstructed)  # On passe en entier 8 bits
    
    # Applique la détection de bords (Canny, parce que c'est le boss des bords)
    edges = cv2.Canny(reconstructed, 50, 150)  # Seuils bas et haut pour Canny
    
    return edges  # Retourne l'image avec les bords détectés