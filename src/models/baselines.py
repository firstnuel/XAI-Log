"""
Baseline Model for Log Anomaly Detection

Implements Isolation Forest for unsupervised anomaly detection.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class IsolationForestBaseline:
    """
    Isolation Forest for unsupervised anomaly detection.
    Does not use labels during training.
    """

    def __init__(self, contamination='auto', n_estimators=100, random_state=42, **kwargs):
        """
        Initialize Isolation Forest.

        Args:
            contamination: Expected proportion of outliers (default: 'auto')
            n_estimators: Number of trees (default: 100)
            random_state: Random seed (default: 42)
            **kwargs: Additional IsolationForest parameters
        """
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs
        )
        self.contamination = contamination
        self.n_estimators = n_estimators
        logger.info(f"Isolation Forest initialized: n_estimators={n_estimators}, "
                   f"contamination={contamination}")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit model on occurrence matrix (unsupervised).

        Args:
            X: (N, vocab_size) occurrence matrix or feature matrix
            y: Ignored (unsupervised learning)
        """
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")
        self.model.fit(X)
        logger.info("Isolation Forest training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.

        Args:
            X: (N, vocab_size) feature matrix

        Returns:
            (N,) array with 1 for anomaly, 0 for normal
        """
        predictions = self.model.predict(X)
        # IsolationForest returns -1 for anomaly, 1 for normal
        # Convert to 0/1 format
        return (predictions == -1).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly score (higher = more anomalous).

        Args:
            X: (N, vocab_size) feature matrix

        Returns:
            (N,) array of anomaly scores in [0, 1] range
        """
        # decision_function returns negative scores for anomalies
        scores = self.model.decision_function(X)

        # Normalize to [0, 1] range
        scores_min = scores.min()
        scores_max = scores.max()

        if scores_max - scores_min > 0:
            normalized_scores = (scores - scores_min) / (scores_max - scores_min)
        else:
            normalized_scores = np.zeros_like(scores)

        # Flip so higher = more anomalous
        return 1 - normalized_scores

    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw anomaly scores from the model.

        Args:
            X: (N, vocab_size) feature matrix

        Returns:
            (N,) array of raw scores (negative = anomaly)
        """
        return self.model.decision_function(X)



def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None
) -> Dict:
    """
    Evaluate model performance.

    Args:
        y_true: True labels (0=normal, 1=anomaly)
        y_pred: Predicted labels (0=normal, 1=anomaly)
        y_proba: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }

    # Add AUC if probabilities provided
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negative'] = int(tn)
        metrics['false_positive'] = int(fp)
        metrics['false_negative'] = int(fn)
        metrics['true_positive'] = int(tp)
    else:
        # Handle edge case where only one class is predicted
        metrics['true_negative'] = 0
        metrics['false_positive'] = 0
        metrics['false_negative'] = 0
        metrics['true_positive'] = 0

    return metrics


def print_metrics(metrics: Dict, model_name: str = "Isolation Forest"):
    """
    Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metrics from evaluate_model()
        model_name: Name of the model (for display)
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name} Performance")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"AUC:       {metrics['auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['true_negative']:6d}  FP: {metrics['false_positive']:6d}")
    print(f"  FN: {metrics['false_negative']:6d}  TP: {metrics['true_positive']:6d}")
    print(f"{'=' * 60}\n")
