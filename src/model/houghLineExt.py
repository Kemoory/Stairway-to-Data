# src/model/detection.py
import cv2
import numpy as np

def detect_steps_houghLineExt(edges, original_image):
    """Détection alternative des marches en prolongeant les lignes horizontales.
    
    Cette fonction utilise HoughLinesP pour détecter les segments de ligne, filtre 
    ceux qui sont presque horizontaux, regroupe leurs coordonnées y (pour fusionner 
    les segments appartenant probablement à la même marche), et enfin, trace des 
    lignes horizontales sur toute la largeur de l'image pour chaque groupe détecté.
    """
    height, width = original_image.shape[:2]
    # Détecter les segments de ligne en utilisant la transformation de Hough probabiliste
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi/180,
        threshold=100,
        minLineLength=150,
        maxLineGap=30
    )
    
    horizontal_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculer l'angle de la ligne en degrés
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Vérifier que la ligne est presque horizontale (angle entre -5° et 5°)
            if -5 < angle < 5:
                # Utiliser la moyenne des coordonnées y comme position représentative de la ligne
                y_avg = (y1 + y2) // 2
                horizontal_lines.append(y_avg)
    
    # Regrouper les coordonnées y pour fusionner les détections proches en une seule ligne
    if horizontal_lines:
        horizontal_lines.sort()
        clusters = []
        current_cluster = [horizontal_lines[0]]
        
        for y in horizontal_lines[1:]:
            # Si la coordonnée y actuelle est proche de la précédente, l'ajouter au groupe
            if y - current_cluster[-1] <= 50:
                current_cluster.append(y)
            else:
                # Groupe terminé : calculer la moyenne des y-values pour obtenir une position unique
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [y]
        clusters.append(int(np.mean(current_cluster)))
        
        # Tracer des lignes horizontales sur toute la largeur de l'image pour chaque groupe détecté
        for y in clusters:
            cv2.line(original_image, (0, y), (width - 1, y), (0, 255, 0), 2)
        
        return len(clusters), original_image
    
    return 0, original_image