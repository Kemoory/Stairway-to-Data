import cv2
import numpy as np

def detect_steps_intensity_profile(image, original_image):
    """
    Détecte les marches en analysant les variations horizontales d'intensité
    Fonctionne mieux avec des images en niveaux de gris
    """
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculer le profil d'intensité horizontal (moyenne sur chaque ligne)
    horizontal_profile = np.mean(gray, axis=1)
    
    # Calculer la dérivée pour trouver les bords (ça va dériver !)
    derivative = np.diff(horizontal_profile)
    
    # Trouver les variations significatives d'intensité (les "marches")
    step_locations = np.where(np.abs(derivative) > np.std(derivative) * 2)[0]
    
    # Grouper les positions proches des marches (on évite les doublons)
    steps = []
    if len(step_locations) > 0:
        current_group = [step_locations[0]]
        for loc in step_locations[1:]:
            if loc - current_group[-1] > 50:  # Grand écart vertical
                steps.append(int(np.mean(current_group)))
                current_group = [loc]
            else:
                current_group.append(loc)
        
        steps.append(int(np.mean(current_group)))
        
        # Dessiner les marches détectées (on met les marches en lumière !)
        height, width = original_image.shape[:2]
        for y in steps:
            cv2.line(original_image, (0, y), (width-1, y), (0, 255, 0), 2)
        
        return len(steps), original_image
    
    return 0, original_image