# src/evaluation.py
import json
from sklearn.metrics import accuracy_score, mean_absolute_error as mae

def load_ground_truth(file_path):
    with open(file_path) as f:
        data = json.load(f)
    ground_truth = {item["Images"]: item["Nombre de marches"] for item in data}
    return ground_truth

def calculate_accuracy(preds, ground_truth):
    return accuracy_score(list(ground_truth.values()), preds)

def calculate_mean_absolute_error(preds, ground_truth):
    return mae(list(ground_truth.values()), preds)

def evaluate_model(preds, file_path):
    ground_truth = load_ground_truth(file_path)
    acc = calculate_accuracy(preds, ground_truth)
    error = calculate_mean_absolute_error(preds, ground_truth)
    return acc, error

