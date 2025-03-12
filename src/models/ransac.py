import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.cluster import DBSCAN
import math

def detect_steps_RANSAC(processed, image):
    '''
    Enhanced stair detection algorithm based on RANSAC with orientation correction,
    angle analysis, and parasite line filtering.
    '''
    # Get image dimensions for later use
    height, width = processed.shape[:2]
    
    # Step 1: Detect image orientation and correct if needed
    orientation_angle = detect_orientation(processed)
    if abs(orientation_angle) > 5:  # Only rotate if angle is significant
        processed = rotate_image(processed, -orientation_angle)
        image_rotated = rotate_image(image.copy(), -orientation_angle)
    else:
        image_rotated = image.copy()
    
    # Step 2: Detect lines with Hough Transform
    lines = cv2.HoughLinesP(processed, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0, image
    
    lines = lines[:, 0, :]  # Reshape for further processing
    
    # Step 3: Filter lines by angle (horizontal lines are likely stairs)
    filtered_lines, line_angles = filter_lines_by_angle(lines)
    if len(filtered_lines) == 0:
        return 0, image
    
    # Step 4: Cluster lines by position to group stair lines
    clusters = cluster_lines_by_position(filtered_lines, line_angles, height)
    
    # Step 5: Count stairs based on clusters
    stair_count = len(clusters)
    
    # Step 6: Draw detected stairs
    for cluster in clusters:
        color = (0, 255, 0)  # Green color for stairs
        for line_idx in cluster:
            x1, y1, x2, y2 = filtered_lines[line_idx]
            cv2.line(image_rotated, (x1, y1), (x2, y2), color, 2)
    
    # If we rotated the image, rotate it back
    if abs(orientation_angle) > 5:
        image_with_stairs = rotate_image(image_rotated, orientation_angle)
        # Crop to original size if rotation changed dimensions
        h, w = image_with_stairs.shape[:2]
        y_offset = (h - height) // 2 if h > height else 0
        x_offset = (w - width) // 2 if w > width else 0
        image_with_stairs = image_with_stairs[y_offset:y_offset+height, x_offset:x_offset+width]
    else:
        image_with_stairs = image_rotated
    
    return stair_count, image_with_stairs

def detect_orientation(image):
    """
    Detect the overall orientation of the image using Hough Line Transform
    """
    lines = cv2.HoughLines(image, 1, np.pi/180, 150)
    if lines is None:
        return 0
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        # Convert theta to degrees and normalize
        angle = np.degrees(theta) % 180
        # Make angles relative to horizontal (0° or 180°)
        if angle > 90:
            angle = angle - 180
        angles.append(angle)
    
    # Get the most common angle
    angles = np.array(angles)
    hist, bins = np.histogram(angles, bins=36, range=(-90, 90))
    dominant_angle = bins[np.argmax(hist)]
    
    return dominant_angle

def rotate_image(image, angle):
    """
    Rotate an image by the specified angle
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)
    
    # Adjust rotation matrix
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    
    return rotated

def filter_lines_by_angle(lines, horizontal_threshold=30):
    """
    Filter lines to keep only those that are approximately horizontal (potential stairs)
    and remove parasite lines
    """
    filtered_lines = []
    line_angles = []
    
    for x1, y1, x2, y2 in lines:
        # Calculate line angle
        if x2 - x1 == 0:  # Vertical line
            angle = 90
        else:
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        
        # Keep lines that are approximately horizontal (stairs)
        if angle < horizontal_threshold or angle > (180 - horizontal_threshold):
            # Filter out very short lines (noise)
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if length > 30:  # Minimum line length threshold
                filtered_lines.append([x1, y1, x2, y2])
                line_angles.append(angle)
    
    return filtered_lines, line_angles

def cluster_lines_by_position(lines, angles, img_height, eps=30):
    """
    Cluster lines by their vertical position to identify individual stairs
    """
    if not lines:
        return []
    
    # Extract y-positions of lines (average of y1 and y2)
    positions = []
    for i, (x1, y1, x2, y2) in enumerate(lines):
        y_pos = (y1 + y2) / 2
        positions.append([y_pos, i])  # Store y-position and original line index
    
    # Cluster lines by position using DBSCAN
    positions = np.array(positions)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(positions[:, 0].reshape(-1, 1))
    
    # Group lines by cluster
    clusters = {}
    for i, cluster_id in enumerate(clustering.labels_):
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(int(positions[i, 1]))
    
    # Convert to list of clusters
    return [cluster for cluster in clusters.values()]