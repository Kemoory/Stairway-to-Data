import cv2
import numpy as np

def preprocess_homomorphic_filter(image):
    """
    Filtrage homomorphique pour booster les détails et homogénéiser la luminosité.
    Bien pratique pour les escaliers sous un éclairage galère.
    """
    # Passage en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Conversion en float + transformation logarithmique (compresse les grandes variations)
    gray_log = np.log1p(np.array(gray, dtype=np.float32))
    
    # Transformée de Fourier (FFT) pour passer en domaine fréquentiel
    gray_fft = np.fft.fft2(gray_log)
    gray_fft_shift = np.fft.fftshift(gray_fft)  # Recentre les basses fréquences
    
    # Création du filtre passe-haut (booste les détails, atténue les grosses variations)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2  # Centre de l'image
    
    mask = np.ones((rows, cols), np.uint8)  # Base tout à 1 (laisser passer les hautes fréquences)
    r = 10  # Rayon du filtre passe-bas à atténuer
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - crow)**2 + (y - ccol)**2 <= r*r
    mask[mask_area] = 0.5  # Réduit l'impact des basses fréquences (éclairage global)
    
    # Application du filtre
    fft_filtered = gray_fft_shift * mask
    
    # Retour en spatial avec transformée de Fourier inverse
    gray_filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered))
    gray_filtered = np.abs(gray_filtered)
    
    # Re-passage en domaine spatial avec l'exponentielle inverse
    gray_exp = np.expm1(gray_filtered)
    
    # Normalisation sur 8 bits (0-255)
    gray_out = cv2.normalize(gray_exp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Détection des contours avec Canny
    edges = cv2.Canny(gray_out, 50, 150)
    
    return edges