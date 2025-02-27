import cv2
import numpy as np

def preprocess_phase_congruency(image):
    """
    Use phase congruency to detect features irrespective of contrast.
    This is useful for stairs with varying lighting or contrast.
    
    Note: This is a simplified implementation of phase congruency.
    A full implementation would use the pynformation or similar libraries.
    
    Args:
        image: Input BGR image
        
    Returns:
        edges: Processed binary edge image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use a series of Gabor filters to approximate phase congruency
    orientations = [0]  # Focus on horizontal features
    scales = [1, 2, 4]
    features = np.zeros_like(gray, dtype=np.float32)
    
    for theta in orientations:
        for scale in scales:
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=scale, theta=theta*np.pi/180, 
                lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F
            )
            
            # Apply filter
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            
            # Accumulate feature response
            features += np.abs(filtered)
    
    # Normalize features
    features = cv2.normalize(features, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply threshold
    _, binary = cv2.threshold(features, 50, 255, cv2.THRESH_BINARY)
    
    # Enhance horizontal features
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    enhanced = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    
    return enhanced