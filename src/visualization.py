# src/visualization.py
import cv2

def visualize_results(image, count, ground_truth=None):
    # Créer un fond semi-transparent pour le texte
    overlay = image.copy()
    y_start = 10
    box_height = 80 if ground_truth is not None else 50
    
    # Dessiner le rectangle de fond
    cv2.rectangle(overlay, (10, y_start), (400, y_start + box_height), (45, 52, 65), -1)
    
    # Ajouter la transparence
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Configuration du texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    color = (245, 245, 245)
    thickness = 2
    
    # Position du texte
    y = y_start + 40
    cv2.putText(image, f"Prediction: {count}", (20, y), 
                font, scale, color, thickness)
    
    if ground_truth is not None:
        cv2.putText(image, f"Ground Truth: {ground_truth}", (20, y + 40), 
                    font, scale, (152, 251, 152), thickness)
    
    # Ajouter un cadre décoratif
    cv2.rectangle(image, (10, y_start), (400, y_start + box_height), (92, 107, 132), 2)
    
    cv2.imshow("Resultat", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
