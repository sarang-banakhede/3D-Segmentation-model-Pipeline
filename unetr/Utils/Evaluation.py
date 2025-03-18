import numpy as np
from sklearn.metrics import matthews_corrcoef

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    return (2.0 * intersection) / denominator if denominator > 0 else 1.0

def iou(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union if union > 0 else 1.0

def precision(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 1.0

def recall(y_true, y_pred):
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return tp / (tp + fn) if (tp + fn) > 0 else 1.0

def specificity(y_true, y_pred):
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    return tn / (tn + fp) if (tn + fp) > 0 else 1.0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 1.0

def volume_overlap_error(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return 1 - (intersection / union) if union > 0 else 0.0

def matthews_correlation_coefficient(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = (y_pred.flatten() > 0.5).astype(np.uint8)
    return matthews_corrcoef(y_true_flat, y_pred_flat) if len(np.unique(y_true_flat)) > 1 else 1.0

def evaluate(prediction, annotation, threshold=0.5):
    prediction = np.copy(prediction)
    prediction = (prediction > threshold).astype(np.uint8)
    return {
        "Dice Coefficient": dice_coefficient(annotation, prediction),
        "IoU": iou(annotation, prediction),
        "Precision": precision(annotation, prediction),
        "Recall": recall(annotation, prediction),
        "F1-Score": f1_score(annotation, prediction),
        "Specificity": specificity(annotation, prediction),
        "Volume Overlap Error": volume_overlap_error(annotation, prediction),
        "MCC": matthews_correlation_coefficient(annotation, prediction),
    }

if __name__ == "__main__":
    pass