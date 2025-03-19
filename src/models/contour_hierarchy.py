import cv2
import numpy as np

def detect_steps_contour_hierarchy(edges, original_image):
    """
    Détecte les marches en analysant la hiérarchie et les relations des contours
    """
    # Trouver les contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrer les contours horizontaux
    contours_horizontaux = []
    for i, contour in enumerate(contours):
        # Vérifier si le contour est principalement horizontal
        x, y, w, h = cv2.boundingRect(contour)
        if w > h * 3:  # Largeur beaucoup plus grande que la hauteur
            contours_horizontaux.append((y + h//2, w))
    
    # Trier les contours par position verticale
    contours_horizontaux.sort(key=lambda x: x[0])
    
    # Grouper les contours
    marches = []
    if contours_horizontaux:
        groupe_actuel = [contours_horizontaux[0]]
        for contour in contours_horizontaux[1:]:
            if contour[0] - groupe_actuel[-1][0] > 50:  # Grand écart vertical
                # Utiliser le contour le plus proéminent du groupe (plus grande largeur)
                marches.append(max(groupe_actuel, key=lambda x: x[1])[0])
                groupe_actuel = [contour]
            else:
                groupe_actuel.append(contour)
        
        # Ajouter le dernier groupe
        marches.append(max(groupe_actuel, key=lambda x: x[1])[0])
        
        # Dessiner les marches détectées
        hauteur, largeur = original_image.shape[:2]
        for y in marches:
            cv2.line(original_image, (0, y), (largeur-1, y), (0, 255, 0), 2)
        
        return len(marches), original_image
    
    return 0, original_image