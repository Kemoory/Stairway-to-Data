import cv2
import numpy as np

def detect_steps_edge_distance(edges, original_image):
    """
    Detects steps by measuring vertical distances between horizontal edges
    """
    # Find horizontal edges
    horizontal_edges = []
    height, width = edges.shape
    
    for y in range(height):
        # Count horizontal edge pixels in this row
        edge_count = np.sum(edges[y, :] > 0)
        if edge_count > width * 0.3:  # Significant horizontal edge
            horizontal_edges.append(y)
    
    # Group edges that are close together
    steps = []
    if horizontal_edges:
        current_group = [horizontal_edges[0]]
        for edge in horizontal_edges[1:]:
            if edge - current_group[-1] > 50:  # Large vertical gap
                steps.append(int(np.mean(current_group)))
                current_group = [edge]
            else:
                current_group.append(edge)
        
        steps.append(int(np.mean(current_group)))
        
        # Draw detected steps
        for y in steps:
            cv2.line(original_image, (0, y), (width-1, y), (0, 255, 0), 2)
        
        return len(steps), original_image
    
    return 0, original_image