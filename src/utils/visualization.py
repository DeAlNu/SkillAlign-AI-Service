"""
Visualization & TensorBoard untuk SkillAlign AI.

Modul untuk visualisasi training metrics, model performance,
dan TensorBoard integration.
"""

import os
import logging
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

logger = logging.getLogger(__name__)


def get_tensorboard_callback(
    log_dir: str = 'logs/training',
    histogram_freq: int = 1,
    profile_batch: int = 0
) -> tf.keras.callbacks.TensorBoard:
    """
    Buat TensorBoard callback yang sudah dikonfigurasi.

    Args:
        log_dir: Directory untuk logs. Default 'logs/training'.
        histogram_freq: Frekuensi histogram logging. Default 1.
        profile_batch: Batch untuk profiling. Default 0 (disabled).

    Returns:
        TensorBoard callback instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=histogram_freq,
        write_graph=True,
        write_images=False,
        update_freq='epoch',
        profile_batch=profile_batch
    )


class MetricsLogger(Callback):
    """
    Custom Callback untuk logging metrics setiap epoch.

    Menyediakan output terformat di console dan
    opsional menyimpan ke file.

    Args:
        log_file: Path ke file log (opsional).
        verbose: Level logging. Default 1.

    Example:
        >>> logger_cb = MetricsLogger(
        ...     log_file='logs/training_metrics.log'
        ... )
        >>> model.fit(..., callbacks=[logger_cb])
    """

    def __init__(
        self,
        log_file: Optional[str] = None,
        verbose: int = 1
    ):
        super(MetricsLogger, self).__init__()
        self.log_file = log_file
        self.verbose = verbose
        self.epoch_logs: List[Dict] = []

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """Log metrics di akhir setiap epoch."""
        logs = logs or {}
        self.epoch_logs.append(logs.copy())

        if self.verbose:
            parts = [f"Epoch {epoch + 1}:"]

            for key in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                if key in logs:
                    parts.append(f"  {key}: {logs[key]:.4f}")

            # Custom metrics
            for key in ['f1_score', 'val_f1_score', 'precision', 'recall']:
                if key in logs:
                    parts.append(f"  {key}: {logs[key]:.4f}")

            print('\n'.join(parts))

        # Write to file
        if self.log_file:
            os.makedirs(
                os.path.dirname(self.log_file), exist_ok=True
            )
            with open(self.log_file, 'a', encoding='utf-8') as f:
                log_line = (
                    f"epoch={epoch + 1}"
                    + ''.join(
                        f",{k}={v:.4f}" if isinstance(v, float)
                        else f",{k}={v}"
                        for k, v in logs.items()
                    )
                    + '\n'
                )
                f.write(log_line)


class TrainingVisualizer:
    """
    Visualisasi untuk training history dan model performance.

    Menyediakan plot untuk:
    - Training/validation loss & accuracy curves
    - Confusion matrix heatmap
    - ROC curve
    - Metrics comparison bar chart

    Args:
        save_dir: Directory untuk menyimpan plot.
            Default 'logs/plots'.

    Example:
        >>> viz = TrainingVisualizer(save_dir='logs/plots')
        >>> viz.plot_training_history(history)
        >>> viz.plot_confusion_matrix(y_true, y_pred)
    """

    def __init__(self, save_dir: str = 'logs/plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette('deep')

    def plot_training_history(
        self,
        history,
        filename: str = 'training_history.png'
    ) -> str:
        """
        Plot training & validation loss/accuracy curves.

        Args:
            history: Keras History object atau dict.
            filename: Nama file output. Default 'training_history.png'.

        Returns:
            str: Path ke saved plot.
        """
        if hasattr(history, 'history'):
            h = history.history
        else:
            h = history

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        axes[0].plot(h.get('loss', []), label='Train Loss', linewidth=2)
        if 'val_loss' in h:
            axes[0].plot(
                h['val_loss'], label='Val Loss',
                linewidth=2, linestyle='--'
            )
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy plot
        axes[1].plot(
            h.get('accuracy', []),
            label='Train Accuracy', linewidth=2
        )
        if 'val_accuracy' in h:
            axes[1].plot(
                h['val_accuracy'],
                label='Val Accuracy',
                linewidth=2, linestyle='--'
            )
        axes[1].set_title(
            'Model Accuracy', fontsize=14, fontweight='bold'
        )
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Training history plot saved: {path}")
        return path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        filename: str = 'confusion_matrix.png'
    ) -> str:
        """
        Plot confusion matrix sebagai heatmap.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels (sudah di-threshold).
            labels: Class label names.
                Default ['Not Match', 'Match'].
            filename: Nama file output.

        Returns:
            str: Path ke saved plot.
        """
        from sklearn.metrics import confusion_matrix

        if labels is None:
            labels = ['Not Match', 'Match']

        cm = confusion_matrix(
            np.array(y_true).flatten(),
            np.array(y_pred).flatten()
        )

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=0.5
        )
        ax.set_title(
            'Confusion Matrix', fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        plt.tight_layout()

        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Confusion matrix plot saved: {path}")
        return path

    def plot_metrics_comparison(
        self,
        metrics: Dict[str, float],
        targets: Optional[Dict[str, float]] = None,
        filename: str = 'metrics_comparison.png'
    ) -> str:
        """
        Plot bar chart perbandingan metrics vs target.

        Args:
            metrics: Dictionary {metric_name: value}.
            targets: Dictionary {metric_name: target_value} (opsional).
            filename: Nama file output.

        Returns:
            str: Path ke saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        metric_names = list(metrics.keys())
        values = list(metrics.values())

        x = np.arange(len(metric_names))
        width = 0.35

        bars = ax.bar(
            x, values, width, label='Achieved',
            color='steelblue', alpha=0.8
        )

        if targets:
            target_values = [
                targets.get(m, 0) for m in metric_names
            ]
            ax.bar(
                x + width, target_values, width,
                label='Target', color='coral', alpha=0.6
            )

        ax.set_title(
            'Model Performance Metrics',
            fontsize=14, fontweight='bold'
        )
        ax.set_xticks(x + width / 2 if targets else x)
        ax.set_xticklabels(metric_names, rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f'{height:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9
            )

        plt.tight_layout()

        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Metrics comparison plot saved: {path}")
        return path

    def plot_score_distribution(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        filename: str = 'score_distribution.png'
    ) -> str:
        """
        Plot distribusi matching scores.

        Args:
            scores: Array of predicted scores.
            labels: Ground truth labels (opsional, untuk coloring).
            filename: Nama file output.

        Returns:
            str: Path ke saved plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if labels is not None:
            for label_val, label_name in [(0, 'Not Match'), (1, 'Match')]:
                mask = np.array(labels).flatten() == label_val
                ax.hist(
                    scores[mask], bins=50, alpha=0.6,
                    label=label_name, density=True
                )
        else:
            ax.hist(
                scores, bins=50, alpha=0.7,
                color='steelblue', density=True
            )

        ax.axvline(
            x=0.5, color='red', linestyle='--',
            alpha=0.7, label='Threshold (0.5)'
        )

        ax.set_title(
            'Matching Score Distribution',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Matching Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        path = os.path.join(self.save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"Score distribution plot saved: {path}")
        return path