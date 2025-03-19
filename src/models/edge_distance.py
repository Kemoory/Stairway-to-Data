import cv2
import numpy as np

def detect_steps_edge_distance(edges, original_image):
    """
    Détecte les marches en mesurant les distances verticales entre les bords horizontaux
    """
    # Trouver les bords horizontaux (aka les lignes qui aiment se poser à plat)
    horizontal_edges = []
    height, width = edges.shape
    
    for y in range(height):
        # Compter les pixels de bord horizontal dans cette ligne
        edge_count = np.sum(edges[y, :] > 0)
        if edge_count > width * 0.3:  # Bord horizontal significatif
            horizontal_edges.append(y)
    
    # Grouper les bords proches (parce que l'union fait la force)
    steps = []
    if horizontal_edges:
        current_group = [horizontal_edges[0]]
        for edge in horizontal_edges[1:]:
            if edge - current_group[-1] > 50:  # Grand écart vertical
                steps.append(int(np.mean(current_group)))
                current_group = [edge]
            else:
                current_group.append(edge)
        
        steps.append(int(np.mean(current_group)))
        
        # Dessiner les marches détectées (parce que c'est plus joli avec des lignes vertes)
        for y in steps:
            cv2.line(original_image, (0, y), (width-1, y), (0, 255, 0), 2)
        
        return len(steps), original_image
    
    return 0, original_image