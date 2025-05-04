import os
import json
import time
import torch
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.optim import Adam
from torch.nn import MSELoss
from models import get_model
from config_dl import *
from data_loader import create_loaders

def train_model(model_type, train_loader, val_loader, device, fold):
    model = get_model(model_type, MODEL_CONFIGS[model_type]).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = MSELoss()
    
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Pour stocker les prédictions et valeurs réelles pour le calcul des métriques
    all_train_preds = []
    all_train_labels = []
    all_val_preds = []
    all_val_labels = []
    
    for epoch in range(EPOCHS):
        # Entraînement
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
            
            # Collecter les prédictions et labels pour la dernière époque
            if epoch == EPOCHS - 1:
                all_train_preds.extend(outputs.detach().cpu().numpy())
                all_train_labels.extend(labels.detach().cpu().numpy())
        
        # Validation
        model.eval()
        epoch_val_loss = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).flatten()
                loss = criterion(outputs, labels)
                epoch_val_loss.append(loss.item())
                
                # Collecter les prédictions et labels pour la dernière époque
                if epoch == EPOCHS - 1:
                    all_val_preds.extend(outputs.detach().cpu().numpy())
                    all_val_labels.extend(labels.detach().cpu().numpy())
                
        avg_train = np.mean(epoch_train_loss)
        avg_val = np.mean(epoch_val_loss)
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{model_type}_fold{fold}_best.pth"))
    
    # Convertir listes en arrays numpy
    y_train = np.array(all_train_labels)
    train_pred = np.array(all_train_preds)
    y_test = np.array(all_val_labels)
    test_pred = np.array(all_val_preds)
    
    # Calculer les métriques d'évaluation
    metrics = {
        'fold': fold,
        'test_actual': y_test.tolist(),
        'test_pred': test_pred.tolist(),
        'train_actual': y_train.tolist(),
        'train_pred': train_pred.tolist(),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, train_pred))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, test_pred))),
        'train_mae': float(mean_absolute_error(y_train, train_pred)),
        'test_mae': float(mean_absolute_error(y_test, test_pred)),
        'train_r2': float(r2_score(y_train, train_pred)),
        'test_r2': float(r2_score(y_test, test_pred)),
        'train_mape': float(np.mean(np.abs((y_train - train_pred) / (y_train + 1e-10))) * 100),  # Éviter division par zéro
        'test_mape': float(np.mean(np.abs((y_test - test_pred) / (y_test + 1e-10))) * 100),  # Éviter division par zéro
        'train_medae': float(np.median(np.abs(y_train - train_pred))),
        'test_medae': float(np.median(np.abs(y_test - test_pred))),
        # Analyse des erreurs
        'underestimation_rate': float(np.mean(test_pred < y_test) * 100),
        'overestimation_rate': float(np.mean(test_pred > y_test) * 100),
        'exact_match_rate': float(np.mean(np.abs(test_pred - y_test) < 0.5) * 100),  # Arrondi pour classification
        'within_1_step': float(np.mean(np.abs(test_pred - y_test) <= 1) * 100),
        'within_2_steps': float(np.mean(np.abs(test_pred - y_test) <= 2) * 100),
        'large_errors': float(np.mean(np.abs(test_pred - y_test) > 3) * 100)
    }
            
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_loss,
        'metrics': metrics
    }

def cross_validate(image_paths, labels, model_type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kfold = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    loaders = create_loaders(image_paths, labels, kfold)
    
    results = []
    total_time = 0
    all_metrics = []
    
    for fold, (train_loader, val_loader) in enumerate(loaders):
        start_time = time.time()
        print(f"Entraînement fold {fold+1}/{CV_FOLDS} pour {model_type}")
        
        fold_result = train_model(model_type, train_loader, val_loader, device, fold)
        fold_time = time.time() - start_time
        total_time += fold_time
        
        results.append({
            **fold_result,
            'fold_time': fold_time
        })
        
        # Ajouter les métriques de ce fold
        all_metrics.append(fold_result['metrics'])
    
    # Calculer les moyennes des métriques sur tous les folds
    avg_metrics = {}
    scalar_metrics = [k for k in all_metrics[0].keys() 
                    if k not in ['fold', 'test_actual', 'test_pred', 'train_actual', 'train_pred']]
    
    for key in scalar_metrics:
        try:
            # Convert to float to handle numpy types
            avg_metrics[f'avg_{key}'] = float(np.mean([
                float(m[key]) for m in all_metrics
            ]))
        except Exception as e:
            print(f"Error averaging {key}: {str(e)}")
            avg_metrics[f'avg_{key}'] = None
    
    # Ajouter les métriques à l'objet de résultat final
    final_result = {
        'model': model_type,
        'results': results,
        'total_time': total_time,
        'avg_time': total_time / CV_FOLDS,
        'fold_metrics': all_metrics,
        'avg_metrics': avg_metrics
    }
    
    # Sauvegarder les métriques détaillées pour ce modèle
    with open(os.path.join(RESULTS_DIR, f'{model_type}_detailed_metrics.json'), 'w') as f:
        json.dump(final_result, f, indent=4)
    
    return final_result