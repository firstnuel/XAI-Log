"""
Utilities for evaluating and visualizing log anomaly detection models.

Provides evaluation helpers and plotting utilities (confusion matrix, ROC,
precision/recall, anomaly score distributions) and the UniversalAnomalyVisualizer
class for consistent presentation across different model types.
"""


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

class UniversalAnomalyVisualizer:
    def __init__(self, model_name="Model", save_dir=None):
        """
        Initialize the visualizer.

        Args:
            model_name: Name of the model for plot titles
            save_dir: Optional directory to save plots. If None, plots are only displayed.
        """
        self.model_name = model_name
        self.save_dir = Path(save_dir) if save_dir else None

        # Create save directory if specified
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Style settings
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {'normal': '#1f77b4', 'anomaly': '#d62728'}

    def _save_figure(self, filename):
        """Helper method to save figure if save_dir is set."""
        if self.save_dir:
            filepath = self.save_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {filepath}")

    def plot_training_history(self, history, filename='training_history.png'):
        """
        Generic plotter for Neural Network training curves.
        Compatible with any dict having 'loss', 'val_loss', 'accuracy', etc.

        Args:
            history: Dictionary with training metrics (e.g., 'train_losses', 'val_losses')
            filename: Filename for saving the plot (if save_dir is set)
        """
        if not history:
            print(f"No training history found for {self.model_name} (Expected for non-DL models like IF).")
            return

        keys = [k for k in history.keys() if 'loss' in k]
        if not keys: return

        plt.figure(figsize=(10, 6))
        for key in keys:
            label = key.replace('_', ' ').title()
            linestyle = '--' if 'val' in key else '-'
            plt.plot(history[key], label=label, linestyle=linestyle, linewidth=2)

        plt.title(f'{self.model_name}: Training Dynamics', fontsize=14)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        self._save_figure(filename)
        plt.show()

    def plot_confusion_matrix(self, y_true, y_pred, title_suffix="", filename='confusion_matrix.png'):
        """
        Universal Binary Confusion Matrix.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            title_suffix: Additional text for plot title
            filename: Filename for saving the plot (if save_dir is set)
        """
        cm = confusion_matrix(y_true, y_pred)
        labels = ['Normal', 'Anomaly']

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'})

        plt.title(f'{self.model_name} Confusion Matrix {title_suffix}', fontsize=14)
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        self._save_figure(filename)
        plt.show()

    def plot_anomaly_score_distribution(self, y_true, y_scores, bins=50, filename='anomaly_score_distribution.png'):
        """
        Crucial for Baseline Models (IF, Autoencoders).
        Shows the separation between Normal and Anomalous scores.

        Args:
            y_true: True labels
            y_scores: Anomaly scores (continuous values)
            bins: Number of histogram bins
            filename: Filename for saving the plot (if save_dir is set)
        """
        if y_scores is None: return

        plt.figure(figsize=(10, 6))

        # Plot Normal
        sns.histplot(y_scores[y_true == 0], color=self.colors['normal'],
                     label='Normal', bins=bins, alpha=0.5, kde=True)

        # Plot Anomaly
        sns.histplot(y_scores[y_true == 1], color=self.colors['anomaly'],
                     label='Anomaly', bins=bins, alpha=0.5, kde=True)

        plt.title(f'{self.model_name}: Anomaly Score Distribution', fontsize=14)
        plt.xlabel('Anomaly Score (Higher = More Anomalous)')
        plt.ylabel('Density')
        plt.legend()
        self._save_figure(filename)
        plt.show()

    def plot_roc_curve(self, y_true, y_scores, filename='roc_curve.png'):
        """
        Universal ROC Curve.
        Requires continuous scores (probability or distance), not just 0/1.

        Args:
            y_true: True labels
            y_scores: Anomaly scores (continuous values)
            filename: Filename for saving the plot (if save_dir is set)
        """
        if y_scores is None: return

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{self.model_name}: ROC Curve')
        plt.legend(loc="lower right")
        self._save_figure(filename)
        plt.show()

    def plot_precision_recall_curve(self, y_true, y_scores, filename='precision_recall_curve.png'):
        """
        Universal Precision-Recall Curve.
        Requires continuous scores (probability or distance), not just 0/1.

        Args:
            y_true: True labels
            y_scores: Anomaly scores (continuous values)
            filename: Filename for saving the plot (if save_dir is set)
        """
        if y_scores is None: return

        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{self.model_name}: Precision-Recall Curve')
        plt.legend(loc="lower left")
        self._save_figure(filename)
        plt.show()
