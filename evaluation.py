import numpy as np


def predict_labels(probabilities, threshold=0.5):
    """Predict 1 when the probability is at least the threshold, else 0."""

    return (probabilities >= threshold).astype(int)


def confusion_counts(y_true, y_pred):
    """True positives, false positives, true negatives, false negatives."""

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return tp, fp, tn, fn


def compute_metrics(y_true, y_pred):
    """Accuracy, precision, recall and F1 score (positive class = 1)."""

    tp, fp, tn, fn = confusion_counts(y_true, y_pred)

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def best_threshold(y_true, probabilities):
    """Threshold between 0.01 and 0.99 that gives the highest F1 score."""

    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [
        compute_metrics(y_true, predict_labels(probabilities, t))["f1"]
        for t in thresholds
    ]
    best = int(np.argmax(f1_scores))

    return thresholds[best], f1_scores[best]