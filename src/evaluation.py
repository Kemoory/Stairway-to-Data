import json
from sklearn.metrics import mean_absolute_error as mae, precision_score, recall_score, confusion_matrix

def calculate_mean_absolute_error(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return mae(gt_values, pred_values)

def calculate_relative_error(preds, ground_truth):
    errors = []
    for img in preds.keys():
        if img in ground_truth:
            gt = ground_truth[img]
            pred = preds[img]
            if gt > 0:
                errors.append(abs(pred - gt) / gt)
    return sum(errors) / len(errors) if errors else 0

def calculate_precision_recall(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    precision = precision_score(gt_values, pred_values, average='macro', zero_division=0)
    recall = recall_score(gt_values, pred_values, average='macro', zero_division=0)
    return precision, recall

def calculate_confusion_matrix(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return confusion_matrix(gt_values, pred_values)

def evaluate_model(preds, ground_truth):
    error = calculate_mean_absolute_error(preds, ground_truth)
    rel_error = calculate_relative_error(preds, ground_truth)
    precision, recall = calculate_precision_recall(preds, ground_truth)
    conf_matrix = calculate_confusion_matrix(preds, ground_truth)
    return error, rel_error, precision, recall, conf_matrix
