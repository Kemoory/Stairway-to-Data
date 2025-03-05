import cv2
import numpy as np

def preprocess_phase_congruency(image):
    """
    Utilise la congruence de phase pour détecter les caractéristiques indépendamment du contraste.
    C'est utile pour les escaliers avec des éclairages ou des contrastes variables.
    
    (implémentation simplifiée de la congruence de phase)
    Une implémentation complète utiliserait des bibliothèques comme pynformation ou similaires.
    """
    # Convertit l'image en niveaux de gris (parce que la couleur, c'est trop compliqué)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Utilise une série de filtres de Gabor pour approximer la congruence de phase
    orientations = [0]  # On se concentre sur les caractéristiques horizontales (les marches, c'est horizontal)
    scales = [1, 2, 4]  # Différentes échelles pour capturer les détails
    features = np.zeros_like(gray, dtype=np.float32)  # Initialise une image vide pour accumuler les résultats
    
    for theta in orientations:
        for scale in scales:
            # Crée un noyau de Gabor (un filtre magique pour détecter les motifs)
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=scale, theta=theta*np.pi/180,  # Taille, échelle, orientation
                lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F  # Paramètres du filtre
            )
            
            # Applique le filtre à l'image
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)  # Filtrage 2D, comme dans les films
            
            # Accumule la réponse du filtre (on additionne les résultats)
            features += np.abs(filtered)  # On prend la valeur absolue pour éviter les négatifs
    
    # Normalise les caractéristiques pour avoir des valeurs entre 0 et 255
    features = cv2.normalize(features, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # On normalise pour que l'image soit affichable
    
    # Applique un seuil pour obtenir une image binaire (noir et blanc)
    _, binary = cv2.threshold(features, 50, 255, cv2.THRESH_BINARY)  # Tout ce qui est >50 devient blanc, le reste noir
    
    # Améliore les caractéristiques horizontales (parce que les marches, c'est encore horizontal)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))  # Un noyau horizontal (15x1)
    enhanced = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)  # On nettoie l'image avec une opération morphologique
    
    return enhanced  # Retourne l'image améliorée