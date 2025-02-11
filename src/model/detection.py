# src/model/detection.py
import cv2
import numpy as np

def detect_steps(edges, original_image):
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