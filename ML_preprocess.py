import os
import numpy as np
import cv2
import joblib
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

def extract_features(image_path):
    """
    Extract features from an image for stair counting.
    Returns a feature vector for the given image.
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize to a standard size
    resized = cv2.resize(gray, (200, 200))
    
    # Edge detection - useful for finding stair edges
    edges = cv2.Canny(resized, 50, 150)
    
    # Horizontal line detection using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    # Count horizontal lines (potential stairs)
    horizontal_line_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            # Consider near-horizontal lines
            if angle < 20 or angle > 160:
                horizontal_line_count += 1
    
    # Extract histogram of gradients (HOG) features
    win_size = (200, 200)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    hog_features = hog.compute(resized)
    
    # Edge density features
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Horizontal and vertical gradient features
    sobelx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    gradient_x_mean = np.mean(np.abs(sobelx))
    gradient_y_mean = np.mean(np.abs(sobely))
    
    # Combine features
    custom_features = np.array([
        horizontal_line_count,
        edge_density,
        gradient_x_mean,
        gradient_y_mean
    ])
    
    # Reduce dimensionality of HOG features (take a subset)
    hog_features_reduced = hog_features[::20].flatten()
    
    # Combine all features
    all_features = np.concatenate([custom_features, hog_features_reduced])
    
    return all_features

def prepare_dataset(image_paths, labels):
    """
    Process all images and prepare feature vectors and labels.
    """
    features_list = []
    valid_labels = []
    
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        
        features = extract_features(image_path)
        if features is not None:
            features_list.append(features)
            valid_labels.append(label)
    
    if not features_list:
        return np.array([]), np.array([])
    
    # Standardize feature lengths
    max_length = max(len(f) for f in features_list)
    standardized_features = []
    for f in features_list:
        if len(f) < max_length:
            # Pad with zeros if necessary
            padded = np.zeros(max_length)
            padded[:len(f)] = f
            standardized_features.append(padded)
        else:
            standardized_features.append(f)
    
    return np.array(standardized_features), np.array(valid_labels)

def train_ml(features, labels):
    """
    Train a machine learning model to predict the number of stairs using k-fold cross-validation.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_test_preds = []
    all_test_labels = []
    
    for train_index, test_index in kf.split(features):
        print(f"Training fold {fold}")
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        print(f"Training with {X_train.shape[0]} samples, testing with {X_test.shape[0]} samples")
        
        # Train a Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"Fold {fold} - Training RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
        print(f"Fold {fold} - Testing RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")
        
        all_test_preds.extend(test_pred)
        all_test_labels.extend(y_test)
        
        fold += 1
    
    # Train final model on all data
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(features, labels)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(final_model.feature_importances_)), final_model.feature_importances_)
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()
    
    # Plot actual vs prediction
    plt.figure(figsize=(10, 6))
    plt.scatter(all_test_labels, all_test_preds, alpha=0.5)
    plt.plot([min(all_test_labels), max(all_test_labels)], [min(all_test_labels), max(all_test_labels)], color='red')
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.title('Actual vs Prediction')
    plt.show()
    
    return final_model

def predict_stairs(model, image_path):
    """
    Predict the number of stairs in a new image.
    """
    features = extract_features(image_path)
    if features is None:
        return None
    
    # Ensure feature shape matches training data
    if model.n_features_in_ > len(features):
        padded = np.zeros(model.n_features_in_)
        padded[:len(features)] = features
        features = padded
    elif model.n_features_in_ < len(features):
        features = features[:model.n_features_in_]
    
    # Make prediction
    prediction = model.predict(features.reshape(1, -1))[0]
    # Round to nearest integer (stairs are whole numbers)
    return round(prediction)

def main():
    # Load data from the database dump
    image_paths = []
    labels = []
    
    # Parse the database dump
    with open('data/stairsData_dump', 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith("COPY public.images_data"):
            break
    
    for line in lines[lines.index(line) + 1:]:
        if line.strip() == '\\.':
            break
        parts = line.strip().split('\t')
        if len(parts) == 4:
            image_path = parts[1]
            label = int(parts[2])
            
            # Debugging: Print the path being checked
            print(f"Checking path: {image_path}")
            
            # Ignore the first point in the file path
            if image_path.startswith('.'):
                image_path = image_path[2:]
            
            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(label)
            else:
                # Debugging: Print a message if the file is not found
                print(f"File not found: {image_path}")
    
    # Debugging: Print the collected image paths and labels
    print(f"Collected {len(image_paths)} image paths and {len(labels)} labels")
    
    features, labels = prepare_dataset(image_paths, labels)
    
    # Debugging: Print the shape of features and labels arrays
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    if features.size == 0 or labels.size == 0:
        print("No valid data to train the model.")
        return
    
    model = train_ml(features, labels)
    joblib.dump(model, "ml_model.pkl")  # Save the trained model
    
    print("Model training complete and saved to ml_model.pkl")
    
    # Example of how to use the model for prediction
    if image_paths:
        test_image = image_paths[0]  # Just for demonstration
        print(f"Example prediction for {test_image}:")
        stairs_count = predict_stairs(model, test_image)
        print(f"Predicted number of stairs: {stairs_count}")

if __name__ == "__main__":
    main()