# src/model/fourier.py
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor

def detect_steps_RANSAC(processed, image):

    lines = cv2.HoughLinesP(processed, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is None:
        return 0, image

    lines = lines[:, 0, :]  # Reshape for RANSAC

    # Prepare data for RANSAC
    X = []
    y = []
    for x1, y1, x2, y2 in lines:
        X.append([x1, x2])
        y.append([y1, y2])

    X = np.array(X).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)

    # Apply RANSAC
    ransac = RANSACRegressor()
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_

    # Count the number of inliers (detected steps)
    step_count = np.sum(inlier_mask)

    # Draw the inlier lines
    for i, (x1, y1, x2, y2) in enumerate(lines):
        if inlier_mask[i]:
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return step_count, image