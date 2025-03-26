import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import MODEL_DIR, RESULTS_DIR, CV_FOLDS, RANDOM_STATE
from visualization import save_fold_comparison, save_prediction_analysis
from models import get_model  # <-- Ajoutez cette ligne pour importer get_model

def train_model(features, labels, image_paths, model_type, output_dir=RESULTS_DIR):
    """Train a model with k-fold cross-validation."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    all_test_preds = []
    all_test_labels = []
    all_test_paths = []
    
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    for fold, (train_index, test_index) in enumerate(kf.split(features), 1):
        print(f"Training fold {fold}")
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        test_paths = [image_paths[i] for i in test_index]
        
        model = get_model(model_type)  # <-- Maintenant get_model est défini
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        train_pred = np.round(train_pred).astype(int)
        test_pred = model.predict(X_test)
        test_pred = np.round(test_pred).astype(int)
        
        # Calculate metrics
        metrics = {
            # Evaluation de la performance
            'fold': fold,
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'test_r2': r2_score(y_test, test_pred),
            'train_mape': np.mean(np.abs((y_train - train_pred) / y_train)) * 100,
            'test_mape': np.mean(np.abs((y_test - test_pred) / y_test)) * 100,
            'train_medae': np.median(np.abs(y_train - train_pred)),
            'test_medae': np.median(np.abs(y_test - test_pred)),
            # Analyse des erreurs
            'underestimation_rate': np.mean(test_pred < y_test) * 100,
            'overestimation_rate': np.mean(test_pred > y_test) * 100,
            'exact_match_rate': np.mean(test_pred == y_test) * 100,
            'within_1_step': np.mean(np.abs(test_pred - y_test) <= 1) * 100,
            'within_2_steps': np.mean(np.abs(test_pred - y_test) <= 2) * 100,
            'large_errors': np.mean(np.abs(test_pred - y_test) > 3) * 100
        }
        all_results.append(metrics)
        
        all_test_preds.extend(test_pred.tolist())
        all_test_labels.extend(y_test.tolist())
        all_test_paths.extend(test_paths)
    
    # Save fold comparison
    save_fold_comparison(all_results, model_type, output_dir)
    
    # Save prediction analysis
    save_prediction_analysis(all_test_labels, all_test_preds, model_type, output_dir)
    
    # Train final model on all data
    final_model = get_model(model_type)
    final_model.fit(features, labels)
    
    # Save the model
    model_path = os.path.join(MODEL_DIR, f"{model_type}_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save results to JSON
    results_json = {}
    for path, actual, pred in zip(all_test_paths, all_test_labels, all_test_preds):
        image_name = os.path.basename(path)
        results_json[image_name] = {
            'model': model_type,
            'ground_truth': int(actual),
            'prediction': int(pred)
        }
    
    json_path = os.path.join(output_dir, f"{model_type}_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=4)
    
    return final_model, results_json