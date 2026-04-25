"""
Custom Metrics untuk SkillAlign AI.

Menyediakan fungsi-fungsi metrik evaluasi dan konfigurasi monitoring.
"""

import numpy as np
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, mean_absolute_error,
    confusion_matrix, classification_report
)


# Monitoring metrics configuration
MONITORING_METRICS = {
    'training': [
        'loss', 'accuracy', 'precision', 'recall', 'f1_score'
    ],
    'validation': [
        'val_loss', 'val_accuracy', 'val_precision',
        'val_recall', 'val_f1_score'
    ],
    'inference': [
        'latency_ms', 'throughput_qps', 'error_rate'
    ]
}

# Performance targets (dari requirement)
PERFORMANCE_TARGETS = {
    'accuracy': 0.85,       # Minimum 85%
    'mae': 0.02,            # Maksimum 0.02
    'inference_time_ms': 500  # Maksimum 500ms
}


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Hitung semua metrics evaluasi.

    Args:
        y_true: Ground truth labels (0 atau 1).
        y_pred_proba: Predicted probabilities (0-1).
        threshold: Threshold untuk binary classification.
            Default 0.5.

    Returns:
        dict: Dictionary berisi semua metrics.
    """
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    y_true = np.array(y_true).flatten()

    metrics = {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        'recall': round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        'f1_score': round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
        'mae': round(
            mean_absolute_error(y_true, y_pred_proba.flatten()), 4
        ),
    }

    # AUC — hanya jika ada kedua class
    try:
        metrics['auc'] = round(
            roc_auc_score(y_true, y_pred_proba.flatten()), 4
        )
    except ValueError:
        metrics['auc'] = 0.0

    return metrics


def compute_classification_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> str:
    """
    Generate classification report lengkap.

    Args:
        y_true: Ground truth labels.
        y_pred_proba: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        str: Formatted classification report.
    """
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    y_true = np.array(y_true).flatten()

    report = classification_report(
        y_true, y_pred,
        target_names=['Not Match', 'Match'],
        zero_division=0
    )

    return report


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Hitung confusion matrix.

    Args:
        y_true: Ground truth labels.
        y_pred_proba: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        np.ndarray: Confusion matrix (2x2).
    """
    y_pred = (y_pred_proba > threshold).astype(int).flatten()
    y_true = np.array(y_true).flatten()
    return confusion_matrix(y_true, y_pred)


def check_performance_targets(
    metrics: Dict[str, float]
) -> Dict[str, dict]:
    """
    Cek apakah metrics memenuhi performance targets.

    Args:
        metrics: Dictionary berisi evaluation metrics.

    Returns:
        dict: Status per target {target: {value, target, passed}}.
    """
    results = {}

    if 'accuracy' in metrics:
        acc = metrics['accuracy']
        results['accuracy'] = {
            'value': acc,
            'target': f">= {PERFORMANCE_TARGETS['accuracy']}",
            'passed': acc >= PERFORMANCE_TARGETS['accuracy']
        }

    if 'mae' in metrics:
        mae = metrics['mae']
        results['mae'] = {
            'value': mae,
            'target': f"<= {PERFORMANCE_TARGETS['mae']}",
            'passed': mae <= PERFORMANCE_TARGETS['mae']
        }

    return results