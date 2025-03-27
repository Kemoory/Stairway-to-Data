import cv2
import numpy as np
import pywt
from scipy.signal import find_peaks, savgol_filter
from config import CUSTOM_CMAP

def wavelet_edge_detection(image):
    """Apply wavelet transform for edge detection"""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 2-level wavelet transform
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    
    # Enhance horizontal details
    LH1 = coeffs[1][1] * 1.5  # Horizontal details level 1
    LH2 = coeffs[2][1] * 1.5  # Horizontal details level 2
    
    # Reconstruct coefficients
    coeffs[1] = (coeffs[1][0], LH1, coeffs[1][2])
    coeffs[2] = (coeffs[2][0], LH2, coeffs[2][2])
    
    # Reconstruct image
    reconstructed = pywt.waverec2(coeffs, 'haar')
    reconstructed = cv2.normalize(reconstructed, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(reconstructed)

def extract_hog_features(image):
    """Extract HOG features from image"""
    win_size = (200, 200)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog.compute(image)

def extract_features(image_path):
    """Main feature extraction function with enhanced step detection"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Resize and convert to grayscale
    img = cv2.resize(img, (200, 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced wavelet processing
    wavelet_img = wavelet_edge_detection(img)
    
    # Feature collection
    features = []
    
    # 1. Improved Intensity Profile Step Detection
    step_count = intensity_profile_detection(img)
    features.append(step_count)
    
    # 2. Additional edge analysis
    edges = cv2.Canny(wavelet_img, 30, 150)  # Adjusted threshold
    edge_features = [
        np.sum(edges > 0),  # Total edge pixels
        cv2.countNonZero(edges),  # Non-zero edge pixels
        np.mean(edges > 0)  # Edge pixel density
    ]
    features.extend(edge_features)
    
    # 3. Gradient-based features
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_features = [
        np.mean(np.abs(sobelx)), 
        np.mean(np.abs(sobely)),
        np.std(sobelx),  # Added standard deviation
        np.std(sobely)   # Added standard deviation
    ]
    features.extend(gradient_features)
    
    return np.array(features)

def intensity_profile_detection(image):
    """Enhanced step detection using intensity profile"""
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate horizontal intensity profile
    profile = np.mean(gray, axis=1)
    
    # Compute derivative with noise reduction
    derivative = np.diff(profile)
    smoothed_derivative = savgol_filter(derivative, window_length=11, polyorder=3)
    
    # Find peaks in derivative to detect potential step edges
    peaks, _ = find_peaks(np.abs(smoothed_derivative), height=np.std(smoothed_derivative) * 2, distance=10)
    
    return len(peaks)

def prepare_dataset(image_paths, labels):
    """Prepare dataset with extracted features"""
    features = []
    valid_labels = []
    valid_paths = []
    
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            valid_labels.append(label)
            valid_paths.append(path)
    
    if not features:
        return np.array([]), np.array([]), []
    
    # Standardize feature length
    max_len = max(len(f) for f in features)
    standardized = [np.pad(f, (0, max_len - len(f))) if len(f) < max_len else f for f in features]
    
    return np.array(standardized), np.array(valid_labels), valid_paths