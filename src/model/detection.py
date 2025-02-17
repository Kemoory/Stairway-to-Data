# src/model/detection.py
import cv2
import numpy as np

def detect_steps_segments(edges, original_image):
    """Detecte les lignes horizontales et compte les marches"""
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    
    horizontal = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -10 < angle < 10:  #Lignes horizontales
                horizontal.append((y1 + y2) // 2)
                cv2.line(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    #Regroupement des lignes proches
    if horizontal:
        horizontal.sort()
        clusters = []
        current = horizontal[0]
        
        for y in horizontal[1:]:
            if y - current > 100:  #Seuil de regroupement
                clusters.append(current)
                current = y
        clusters.append(current)
        
        return len(clusters), original_image
    return 0, original_image

def detect_steps_extended(edges, original_image):
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

def fourier_transform(image):
    """Appliquer la transformée de Fourier et visualiser le spectre de magnitude."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return magnitude_spectrum

def detect_vanishing_point(image):
    """Détecter le point de fuite en utilisant les lignes de Hough."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return None

    # Trouver les points d'intersection de toutes les lignes
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            try:
                x, y = np.linalg.solve(A, b)
                intersections.append((int(x), int(y)))
            except np.linalg.LinAlgError:
                continue

    if not intersections:
        return None

    # Estimer le point de fuite comme la médiane de toutes les intersections
    vanishing_point = np.median(intersections, axis=0).astype(int)
    return vanishing_point

def detect_vanishing_lines(image):
    """Détecter l'escalier et compter les marches en utilisant le point de fuite et les lignes horizontales."""
    print("Début de la détection des lignes de fuite...")
    vanishing_point = detect_vanishing_point(image)
    if vanishing_point is None:
        print("Aucun point de fuite détecté. Sortie.")
        return image, 0

    # Dessiner le point de fuite
    vx, vy = vanishing_point
    cv2.circle(image, (vx, vy), 10, (0, 0, 255), -1)  # Dessiner le point de fuite en rouge
    print(f"Point de fuite dessiné à ({vx}, {vy})")

    # Détecter et compter les marches
    debug_img, count = detect_staircase_steps(image, vanishing_point)
    print("Détection des lignes de fuite terminée.")
    return debug_img, count

def detect_vanishing_point(image):
    """Détecter le point de fuite en utilisant les lignes de Hough."""
    print("Détection du point de fuite...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        print("Aucune ligne détectée. Le point de fuite ne peut pas être déterminé.")
        return None

    print(f"{len(lines)} lignes détectées. Recherche des points d'intersection...")
    intersections = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            rho1, theta1 = lines[i][0]
            rho2, theta2 = lines[j][0]
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            try:
                x, y = np.linalg.solve(A, b)
                intersections.append((int(x), int(y)))
                print(f"Intersection trouvée à ({x}, {y})")
            except np.linalg.LinAlgError:
                print("Aucune intersection trouvée pour cette paire de lignes.")
                continue

    if not intersections:
        print("Aucune intersection trouvée. Le point de fuite ne peut pas être déterminé.")
        return None

    # Estimer le point de fuite comme la médiane de toutes les intersections
    vanishing_point = np.median(intersections, axis=0).astype(int)
    print(f"Point de fuite estimé à ({vanishing_point[0]}, {vanishing_point[1]})")
    return vanishing_point

def detect_staircase_steps(image, vanishing_point):
    """Détecter les lignes horizontales et compter les marches dans la région de l'escalier."""
    print("Détection des marches d'escalier...")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print("Aucune ligne détectée. Aucune marche trouvée.")
        return image, 0

    print(f"{len(lines)} lignes détectées. Filtrage des lignes horizontales...")
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if -10 < angle < 10:  # Lignes horizontales
            horizontal_lines.append((x1, y1, x2, y2))
            print(f"Ligne horizontale détectée de ({x1}, {y1}) à ({x2}, {y2})")

    # Filtrer les lignes qui sont dans la région de l'escalier (en dessous du point de fuite)
    if vanishing_point is not None:
        vx, vy = vanishing_point
        print(f"Filtrage des lignes en dessous du point de fuite ({vx}, {vy})...")
        horizontal_lines = [line for line in horizontal_lines if line[1] > vy and line[3] > vy]
        print(f"{len(horizontal_lines)} lignes horizontales restent après filtrage.")

    # Dessiner les lignes horizontales et compter les marches
    for x1, y1, x2, y2 in horizontal_lines:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Dessiner en vert
        print(f"Dessin de la ligne de marche de ({x1}, {y1}) à ({x2}, {y2})")

    print(f"Nombre total de marches détectées : {len(horizontal_lines)}")
    return image, len(horizontal_lines)

