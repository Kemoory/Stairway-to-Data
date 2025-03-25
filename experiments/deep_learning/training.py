import os
import json
import time
import torch
import numpy as np

from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.nn import MSELoss
from visualization import save_results
from models import get_model
from config_dl import *
from data_loader import create_loaders

def train_model(model_type, train_loader, val_loader, device):
    model = get_model(model_type, MODEL_CONFIGS[model_type]).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = MSELoss()
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).flatten()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss.append(loss.item())
        
        # Validation
        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images.to(device)).flatten()
                loss = criterion(outputs, labels.to(device))
                epoch_val_loss.append(loss.item())
                
        avg_train = np.mean(epoch_train_loss)
        avg_val = np.mean(epoch_val_loss)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_type}_best.pth"))
            
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_loss
    }

def cross_validate(image_paths, labels, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    loaders = create_loaders(image_paths, labels, kfold)
    
    results = []
    total_time = 0
    
    for fold, (train_loader, val_loader) in enumerate(loaders):
        start_time = time.time()
        print(f"Entraînement fold {fold+1}/{CV_FOLDS} pour {model_type}")
        
        fold_result = train_model(model_type, train_loader, val_loader, device)
        fold_time = time.time() - start_time
        total_time += fold_time
        
        results.append({
            **fold_result,
            'fold_time': fold_time
        })
    
    return {
        'model': model_type,
        'results': results,
        'total_time': total_time,
        'avg_time': total_time / CV_FOLDS
    }