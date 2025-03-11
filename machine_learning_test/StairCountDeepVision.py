import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.saving import register_keras_serializable

@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

def load_and_preprocess_data(image_paths, labels):
    """
    Load and preprocess images for CNN model
    """
    images = []
    valid_labels = []
    
    for i, (image_path, label) in enumerate(zip(image_paths, labels)):
        if i % 10 == 0:
            print(f"Processing image {i+1}/{len(image_paths)}")
        
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
            continue
        
        # Resize to standard size for CNN
        img = cv2.resize(img, (224, 224))
        
        # Convert to RGB (from BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to range [0, 1]
        img = img / 255.0
        
        images.append(img)
        valid_labels.append(label)
    
    return np.array(images), np.array(valid_labels)

def build_cnn_model(input_shape=(224, 224, 3)):
    """
    Build a CNN model for stair counting
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1)  # Single output for regression (stair count)
    ])
    
    # Compile model with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_cnn(image_paths, labels):
    """
    Train a CNN model to predict the number of stairs
    """
    # Load and preprocess data
    images, labels = load_and_preprocess_data(image_paths, labels)
    
    if len(images) == 0:
        print("No valid images to train the model.")
        return None
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training with {X_train.shape[0]} samples, validating with {X_val.shape[0]} samples")
    
    # Build the CNN model
    model = build_cnn_model()
    model.summary()
    
    # Setup callbacks for early stopping and model checkpointing
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint("staircount_cnn_model.h5", save_best_only=True)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # Load the best model
    model = load_model("staircount_cnn_model.h5", custom_objects={'mse': mse})
    
    # Evaluate the model
    train_pred = model.predict(X_train).flatten()
    val_pred = model.predict(X_val).flatten()
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)
    
    print(f"Training RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.title('Training and Validation MAE')
    
    plt.tight_layout()
    plt.show()
    
    # Plot actual vs prediction for validation set
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, val_pred, alpha=0.5)
    plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='red')
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.title('Actual vs Prediction (Validation Set)')
    plt.show()
    
    return model

def predict_stairs_cnn(model, image_path):
    """
    Predict the number of stairs in a new image using the CNN model
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    
    # Preprocess the image
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    
    # Make prediction
    prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
    
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
            
            # Remove the leading dot if present
            if image_path.startswith('.'):
                image_path = image_path[2:]
            
            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(label)
            else:
                print(f"File not found: {image_path}")
    
    print(f"Collected {len(image_paths)} image paths and {len(labels)} labels")
    
    # Train the CNN model
    model = train_cnn(image_paths, labels)
    
    if model is None:
        print("Model training failed.")
        return
    
    # Save the model
    model.save("staircount_cnn_model.h5")
    print("Model training complete and saved to staircount_cnn_model.h5")
    
    # Example of how to use the model for prediction
    if image_paths:
        test_image = image_paths[0]  # Just for demonstration
        print(f"Example prediction for {test_image}:")
        stairs_count = predict_stairs_cnn(model, test_image)
        print(f"Predicted number of stairs: {stairs_count}")

if __name__ == "__main__":
    # Set memory growth for GPU to prevent OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    main()