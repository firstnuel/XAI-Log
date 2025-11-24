"""
Explainability Module for Log Anomaly Detection

This module provides interpretable explanations for anomaly predictions
through a Hybrid Intrinsic Explainability Framework.

Components:
- BaselineStatistics: Learn normal patterns from training data
- AnomalyExplainer: Generate human-readable explanations for anomalies
- format_anomaly_report: Format explanations into readable reports

Tiers:
- Tier 0: Critical keyword detection (Exception, Error, Failed)
- Tier 1: Statistical analysis (Frequency, Pattern, Position, Structure)
- Tier 2: Model surprise (deep learning next-token probability)
"""

from .baseline_stats import BaselineStatistics
from .explainer import AnomalyExplainer, format_anomaly_report

__all__ = ['BaselineStatistics', 'AnomalyExplainer', 'format_anomaly_report']
