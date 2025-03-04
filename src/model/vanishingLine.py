import cv2
import numpy as np

def detect_vanishing_lines(processed, image):
    """
    Détecter les escaliers en utilisant la méthode du point de fuite.
    
    Args:
        image (numpy.ndarray): Image d'entrée
    
    Returns:
        tuple: (nombre d'escaliers, image de débogage avec les lignes détectées)
    """    
    # Détecter les lignes avec la Transformée de Hough Probabiliste
    lines = cv2.HoughLinesP(
        processed, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100, 
        minLineLength=50, 
        maxLineGap=10
    )
    
    if lines is None:
        return 0, image
    
    # Filtrer les lignes horizontales et quasi-horizontales (avec un petit angle)
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Considérer les lignes avec un angle proche de l'horizontale (entre -10 et 10 degrés)
        if abs(angle) <= 10:
            horizontal_lines.append((y1, y2, x1, x2))
    
    # Si pas assez de lignes horizontales trouvées, retourner 0
    if len(horizontal_lines) < 2:
        return 0, image
    
    # Trier les lignes par coordonnée y pour regrouper les lignes potentielles des escaliers
    horizontal_lines.sort(key=lambda x: (x[0] + x[1]) / 2)
    
    # Regrouper les lignes pour détecter des escaliers distincts
    stairs = []
    current_cluster = [horizontal_lines[0]]
    
    for line in horizontal_lines[1:]:
        # Vérifier si la nouvelle ligne est assez proche de la dernière ligne du groupe
        last_line = current_cluster[-1]
        y_distance = abs((line[0] + line[1]) / 2 - (last_line[0] + last_line[1]) / 2)
        
        if y_distance <= 50:  # Ajuster ce seuil selon les caractéristiques de votre image
            current_cluster.append(line)
        else:
            # Si la distance est trop grande, commencer un nouveau groupe
            if len(current_cluster) > 1:
                # Calculer la coordonnée y moyenne pour le groupe
                avg_y = np.mean([(l[0] + l[1]) / 2 for l in current_cluster])
                stairs.append(avg_y)
            
            current_cluster = [line]
    
    # Ajouter le dernier groupe s'il a plusieurs lignes
    if len(current_cluster) > 1:
        avg_y = np.mean([(l[0] + l[1]) / 2 for l in current_cluster])
        stairs.append(avg_y)
    
    # Dessiner les escaliers détectés sur l'image de débogage
    debug_image = image.copy()
    height, width = image.shape[:2]
    
    for y in stairs:
        cv2.line(debug_image, (0, int(y)), (width, int(y)), (0, 255, 0), 2)
    
    # Optionnel : Dessiner le point de fuite
    if stairs:
        # Estimation simple du point de fuite (moyenne des coordonnées x)
        x_points = []
        for line in horizontal_lines:
            x_points.extend([line[2], line[3]])
        
        vanishing_x = int(np.mean(x_points))
        vanishing_y = int(np.mean(stairs))
        cv2.circle(debug_image, (vanishing_x, vanishing_y), 10, (255, 0, 0), -1)
    
    return len(stairs), debug_image