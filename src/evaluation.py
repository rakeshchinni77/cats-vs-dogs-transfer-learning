import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, test_gen):
    """
    Evaluate model on test data and compute classification metrics.
    """

    # Reset generator
    test_gen.reset()

    # Get predictions
    preds = model.predict(test_gen, verbose=1)
    y_pred = (preds > 0.5).astype(int).ravel()

    # True labels
    y_true = test_gen.classes

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

    return metrics, y_true, y_pred


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix
    """
    return confusion_matrix(y_true, y_pred)
