"""
Evaluation metrics for log anomaly detection models.

This module provides functions to evaluate model performance
and display results in a human-readable format.
"""

import numpy as np
from typing import Dict, Optional
import json
from pathlib import Path


def evaluate_model(
    predictions: np.ndarray,
    labels: np.ndarray,
    anomaly_scores: Optional[np.ndarray] = None
) -> Dict:
    """
    Evaluate model performance.

    Args:
        predictions (np.ndarray): Binary predictions (0=normal, 1=anomaly)
        labels (np.ndarray): True labels (0=normal, 1=anomaly)
        anomaly_scores (np.ndarray, optional): Continuous anomaly scores

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, roc_auc_score
    )

    results = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0),
        'confusion_matrix': confusion_matrix(labels, predictions).tolist()
    }

    # Add AUC if anomaly scores are provided
    if anomaly_scores is not None:
        try:
            results['auc'] = roc_auc_score(labels, anomaly_scores)
        except ValueError:
            results['auc'] = 0.0

    return results


def print_metrics(metrics: Dict):
    """
    Pretty print evaluation metrics.

    Args:
        metrics (dict): Dictionary of metrics from evaluate_model()
    """
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")

    if 'auc' in metrics:
        print(f"AUC:       {metrics['auc']:.4f}")

    print("\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"              Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:6d}  {cm[0,1]:7d}")
    print(f"       Anomaly  {cm[1,0]:6d}  {cm[1,1]:7d}")
    print("="*60)


def save_experiment_results(
    filepath,
    dataset,
    model_name,
    device,
    model,
    history,
    y_train, y_val, y_test,
    max_len,
    metrics,
    batch_size,
    learning_rate,
    patience,
    top_k=4,
    detection_method="next_event_prediction",
    use_attention=False,
    n_heads=None,
):
    """
    Save experiment metadata, training stats, model config, and metrics to JSON.
    """
    results = {
        "dataset": dataset,
        "model": model_name,
        "device": str(device),
        "architecture": {
            "vocab_size": model.vocab_size,
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim if hasattr(model, 'hidden_dim') else None,
            "num_layers": model.num_layers if hasattr(model, 'num_layers') else None,
            "dropout": model.dropout if hasattr(model, 'dropout') else None ,
        },
        "training": {
            "num_epochs": len(history["train_losses"]),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "early_stopping_patience": patience,
            "best_val_loss": float(history["best_val_loss"]),
            "training_time_seconds": float(history["total_time"]),
        },
        "data": {
            "train_size": len(y_train),
            "val_size": len(y_val),
            "test_size": len(y_test),
            "max_sequence_length": max_len,
            "train_normal_only": True,
        },
        "detection": {
            "top_k": top_k,
            "method": detection_method,
        },
        "metrics": metrics,
    }

    if use_attention:
        results["architecture"]["use_attention"] = True
        results["architecture"]["num_attention_heads"] = n_heads

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {filepath}")
    return results
