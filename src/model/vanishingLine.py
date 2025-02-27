# src/model/vanishingLine.py
import cv2
import numpy as np

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

