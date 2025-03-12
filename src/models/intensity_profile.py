import cv2
import numpy as np

def detect_steps_intensity_profile(image, original_image):
    """
    Detects steps by analyzing horizontal intensity changes
    Works best with grayscale images
    """
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Compute horizontal intensity profile
    horizontal_profile = np.mean(gray, axis=1)
    
    # Compute first derivative to find edges
    derivative = np.diff(horizontal_profile)
    
    # Find significant changes in intensity
    step_locations = np.where(np.abs(derivative) > np.std(derivative) * 2)[0]
    
    # Group close step locations
    steps = []
    if len(step_locations) > 0:
        current_group = [step_locations[0]]
        for loc in step_locations[1:]:
            if loc - current_group[-1] > 50:  # Large vertical gap
                steps.append(int(np.mean(current_group)))
                current_group = [loc]
            else:
                current_group.append(loc)
        
        steps.append(int(np.mean(current_group)))
        
        # Draw detected steps
        height, width = original_image.shape[:2]
        for y in steps:
            cv2.line(original_image, (0, y), (width-1, y), (0, 255, 0), 2)
        
        return len(steps), original_image
    
    return 0, original_image