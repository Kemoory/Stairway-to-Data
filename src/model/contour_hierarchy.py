import cv2
import numpy as np

def detect_steps_contour_hierarchy(edges, original_image):
    """
    Detects steps by analyzing contour hierarchy and relationships
    """
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter horizontal contours
    horizontal_contours = []
    for i, contour in enumerate(contours):
        # Check if contour is primarily horizontal
        x, y, w, h = cv2.boundingRect(contour)
        if w > h * 3:  # Width much larger than height
            horizontal_contours.append((y + h//2, w))
    
    # Sort contours by vertical position
    horizontal_contours.sort(key=lambda x: x[0])
    
    # Group contours
    steps = []
    if horizontal_contours:
        current_group = [horizontal_contours[0]]
        for contour in horizontal_contours[1:]:
            if contour[0] - current_group[-1][0] > 50:  # Large vertical gap
                # Use the most prominent contour in the group (largest width)
                steps.append(max(current_group, key=lambda x: x[1])[0])
                current_group = [contour]
            else:
                current_group.append(contour)
        
        # Add last group
        steps.append(max(current_group, key=lambda x: x[1])[0])
        
        # Draw detected steps
        height, width = original_image.shape[:2]
        for y in steps:
            cv2.line(original_image, (0, y), (width-1, y), (0, 255, 0), 2)
        
        return len(steps), original_image
    
    return 0, original_image