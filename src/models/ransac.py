import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import math

def detect_steps_RANSAC(processed, image):
    '''
    Algorithme amélioré de détection d'escaliers basé sur RANSAC avec correction d'orientation,
    analyse des angles et filtrage des lignes parasites.
    '''
    # Obtenir les dimensions de l'image pour une utilisation ultérieure
    height, width = processed.shape[:2]
    
    # Étape 1 : Détecter l'orientation de l'image et corriger si nécessaire
    orientation_angle = detect_orientation(processed)
    if abs(orientation_angle) > 5:  # Ne pivoter que si l'angle est significatif
        processed = rotate_image(processed, -orientation_angle)
        image_rotated = rotate_image(image.copy(), -orientation_angle)
    else:
        image_rotated = image.copy()
    
    # Étape 2 : Détecter les lignes avec la Transformée de Hough
    lines = cv2.HoughLinesP(processed, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0, image
    
    lines = lines[:, 0, :]  # Remodeler pour un traitement ultérieur
    
    # Étape 3 : Filtrer les lignes par angle (les lignes horizontales sont probablement des escaliers)
    filtered_lines, line_angles = filter_lines_by_angle(lines)
    if len(filtered_lines) == 0:
        return 0, image
    
    # Étape 4 : Regrouper les lignes par position pour identifier les lignes d'escaliers
    clusters = cluster_lines_by_position(filtered_lines, line_angles, height)
    
    # Étape 5 : Compter les escaliers en fonction des groupes
    stair_count = len(clusters)
    
    # Étape 6 : Dessiner les escaliers détectés
    for cluster in clusters:
        color = (0, 255, 0)  # Couleur verte pour les escaliers
        for line_idx in cluster:
            x1, y1, x2, y2 = filtered_lines[line_idx]
            cv2.line(image_rotated, (x1, y1), (x2, y2), color, 2)
    
    # Si nous avons pivoté l'image, la remettre dans son orientation initiale
    if abs(orientation_angle) > 5:
        image_with_stairs = rotate_image(image_rotated, orientation_angle)
        # Recadrer à la taille d'origine si la rotation a modifié les dimensions
        h, w = image_with_stairs.shape[:2]
        y_offset = (h - height) // 2 if h > height else 0
        x_offset = (w - width) // 2 if w > width else 0
        image_with_stairs = image_with_stairs[y_offset:y_offset+height, x_offset:x_offset+width]
    else:
        image_with_stairs = image_rotated
    
    return stair_count, image_with_stairs

def detect_orientation(image):
    """
    Détecter l'orientation globale de l'image en utilisant la Transformée de Hough
    """
    lines = cv2.HoughLines(image, 1, np.pi/180, 150)
    if lines is None:
        return 0
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convertir theta en degrés et normaliser
        angle = np.degrees(theta) % 180
        # Rendre les angles relatifs à l'horizontale (0° ou 180°)
        if angle > 90:
            angle = angle - 180
        angles.append(angle)
    
    # Obtenir l'angle le plus fréquent
    angles = np.array(angles)
    hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
    dominant_angle = bins[np.argmax(hist)]
    
    return dominant_angle

def rotate_image(image, angle):
    """
    Pivoter une image selon l'angle spécifié
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Obtenir la matrice de rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculer les nouvelles dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Ajuster la matrice de rotation
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    
    # Effectuer la rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated

def filter_lines_by_angle(lines, horizontal_threshold=30):
    """
    Filtrer les lignes pour ne conserver que celles qui sont approximativement horizontales (escaliers potentiels)
    et supprimer les lignes parasites
    """
    filtered_lines = []
    line_angles = []
    
    for x1, y1, x2, y2 in lines:
        # Calculer l'angle de la ligne
        if x2 - x1 == 0:  # Ligne verticale
            angle = 90
        else:
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        
        # Conserver les lignes qui sont approximativement horizontales (escaliers)
        if angle < horizontal_threshold or angle > (180 - horizontal_threshold):
            # Filtrer les lignes très courtes (bruit)
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 30:  # Seuil minimum de longueur de ligne
                filtered_lines.append([x1, y1, x2, y2])
                line_angles.append(angle)
    
    return filtered_lines, line_angles

def cluster_lines_by_position(lines, angles, img_height, eps=30):
    """
    Regrouper les lignes par leur position verticale pour identifier les escaliers individuels
    """
    if not lines:
        return []
    
    # Extraire les positions verticales des lignes (moyenne de y1 et y2)
    positions = []
    for i, (x1, y1, x2, y2) in enumerate(lines):
        y_pos = (y1 + y2) / 2
        positions.append([y_pos, i])  # Stocker la position y et l'indice de la ligne originale
    
    # Regrouper les lignes par position en utilisant DBSCAN
    positions = np.array(positions)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(positions[:, 0].reshape(-1, 1))
    
    # Grouper les lignes par cluster
    clusters = {}
    for i, cluster_id in enumerate(clustering.labels_):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(int(positions[i, 1]))
    
    # Convertir en liste de clusters
    return [cluster for cluster in clusters.values()]