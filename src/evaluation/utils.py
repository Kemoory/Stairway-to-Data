import numpy as np
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse, r2_score

def calculate_mean_absolute_error(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return mae(gt_values, pred_values)

def calculate_mean_squared_error(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return mse(gt_values, pred_values)

def calculate_root_mean_squared_error(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return np.sqrt(mse(gt_values, pred_values))

def calculate_r2_score(preds, ground_truth):
    gt_values = [ground_truth[img] for img in preds.keys() if img in ground_truth]
    pred_values = [preds[img] for img in preds.keys() if img in ground_truth]
    return r2_score(gt_values, pred_values)

def calculate_relative_error(preds, ground_truth):
    errors = []
    for img in preds.keys():
        if img in ground_truth:
            gt = ground_truth[img]
            pred = preds[img]
            if gt > 0:
                errors.append(abs(pred - gt) / gt)
    return sum(errors) / len(errors) if errors else 0