"""
Training Pipeline untuk SkillAlign AI.

Modul ini menyediakan konfigurasi training dan class ModelTrainer
untuk menjalankan training pipeline lengkap.
"""

import os
import logging
from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    TensorBoard, ModelCheckpoint
)

from src.models.custom_loss import focal_loss
from src.models.custom_callbacks import F1ScoreCallback
from src.models.model_architecture import SkillAlignMatcher

logger = logging.getLogger(__name__)


# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'validation_split': 0.2,
    'learning_rate': 0.001,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'f1_patience': 8,
    'f1_threshold': 0.80,
    'focal_loss_gamma': 2.0,
    'focal_loss_alpha': 0.25
}


class ModelTrainer:
    """
    Training pipeline lengkap untuk SkillAlign model.

    Mengatur seluruh alur training termasuk:
    - Model compilation
    - Callback configuration
    - Training execution
    - Model saving/export

    Args:
        model: Compiled Keras model.
        config: Training configuration dictionary.
            Jika None, menggunakan TRAINING_CONFIG default.
        log_dir: Directory untuk TensorBoard logs.
            Default 'logs/training'.
        model_dir: Directory untuk menyimpan model.
            Default 'models'.

    Example:
        >>> trainer = ModelTrainer(model)
        >>> history = trainer.train(
        ...     x_train=[cv_train, job_train],
        ...     y_train=y_train,
        ...     x_val=[cv_val, job_val],
        ...     y_val=y_val
        ... )
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: Optional[dict] = None,
        log_dir: str = 'logs/training',
        model_dir: str = 'models'
    ):
        self.model = model
        self.config = config or TRAINING_CONFIG.copy()
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.history = None

        # Buat directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def compile_model(
        self,
        loss_fn=None,
        optimizer=None,
        metrics: Optional[list] = None
    ) -> None:
        """
        Compile model dengan optimizer, loss, dan metrics.

        Args:
            loss_fn: Loss function. Default focal_loss.
            optimizer: Optimizer. Default Adam.
            metrics: List of metrics. Default [accuracy, precision, recall, auc].
        """
        if optimizer is None:
            optimizer = Adam(
                learning_rate=self.config['learning_rate']
            )

        if loss_fn is None:
            loss_fn = focal_loss(
                gamma=self.config.get('focal_loss_gamma', 2.0),
                alpha=self.config.get('focal_loss_alpha', 0.25)
            )

        if metrics is None:
            metrics = [
                'accuracy',
                Precision(name='precision'),
                Recall(name='recall'),
                AUC(name='auc')
            ]

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )

        logger.info("Model compiled successfully.")

    def _build_callbacks(
        self,
        validation_data: Optional[tuple] = None
    ) -> List[tf.keras.callbacks.Callback]:
        """
        Build list of training callbacks.

        Args:
            validation_data: Tuple (x_val, y_val) untuk F1ScoreCallback.

        Returns:
            List of Keras Callback instances.
        """
        callbacks = [
            # Early Stopping berdasarkan val_loss
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),

            # Reduce learning rate saat plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),

            # TensorBoard logging
            TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=False,
                update_freq='epoch',
                profile_batch=0
            ),

            # Model Checkpoint (best val_accuracy)
            ModelCheckpoint(
                filepath=os.path.join(
                    self.model_dir, 'best_model.keras'
                ),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]

        # Custom F1-Score Callback
        if validation_data is not None:
            f1_callback = F1ScoreCallback(
                validation_data=validation_data,
                threshold=self.config.get('f1_threshold', 0.80),
                patience=self.config.get('f1_patience', 8),
                save_best_model=True,
                model_save_path=os.path.join(
                    self.model_dir, 'best_f1_model.keras'
                ),
                verbose=1
            )
            callbacks.append(f1_callback)

        return callbacks

    def train(
        self,
        x_train,
        y_train: np.ndarray,
        x_val=None,
        y_val: Optional[np.ndarray] = None,
        extra_callbacks: Optional[list] = None
    ):
        """
        Jalankan training.

        Args:
            x_train: Training input. Bisa list [cv_seq, job_seq]
                atau single array.
            y_train: Training labels.
            x_val: Validation input (opsional).
            y_val: Validation labels (opsional).
            extra_callbacks: Callbacks tambahan (opsional).

        Returns:
            history: Keras History object.
        """
        # Build callbacks
        val_data_for_callback = None
        if x_val is not None and y_val is not None:
            val_data_for_callback = (x_val, y_val)

        callbacks = self._build_callbacks(val_data_for_callback)
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        # Tentukan validation data
        validation_data = None
        validation_split = None

        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        else:
            validation_split = self.config.get('validation_split', 0.2)

        logger.info(
            f"Starting training: "
            f"epochs={self.config['epochs']}, "
            f"batch_size={self.config['batch_size']}"
        )

        # Train
        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        logger.info("Training completed.")
        return self.history

    def save_model(
        self,
        filename: str = 'skillalign_matcher.keras',
        save_format: str = 'keras'
    ) -> str:
        """
        Simpan trained model.

        Args:
            filename: Nama file model. Default 'skillalign_matcher.keras'.
            save_format: Format penyimpanan ('keras' atau 'tf').

        Returns:
            path: Full path ke saved model.
        """
        path = os.path.join(self.model_dir, filename)

        if save_format == 'tf':
            # SavedModel format
            tf.saved_model.save(self.model, path)
        else:
            # Keras format
            self.model.save(path)

        logger.info(f"Model saved ke: {path}")
        return path

    def evaluate(
        self,
        x_test,
        y_test: np.ndarray
    ) -> dict:
        """
        Evaluasi model pada test data.

        Args:
            x_test: Test input.
            y_test: Test labels.

        Returns:
            dict: Evaluation metrics.
        """
        results = self.model.evaluate(x_test, y_test, verbose=1)
        metric_names = self.model.metrics_names
        evaluation = dict(zip(metric_names, results))

        logger.info(f"Evaluation results: {evaluation}")
        return evaluation

    def get_training_summary(self) -> dict:
        """
        Return summary dari training yang sudah selesai.

        Returns:
            dict: Training summary termasuk best metrics.
        """
        if self.history is None:
            return {'status': 'Not trained yet'}

        h = self.history.history
        summary = {
            'total_epochs': len(h.get('loss', [])),
            'final_loss': h['loss'][-1] if 'loss' in h else None,
            'final_accuracy': h['accuracy'][-1] if 'accuracy' in h else None,
            'best_val_loss': min(h['val_loss']) if 'val_loss' in h else None,
            'best_val_accuracy': max(h['val_accuracy']) if 'val_accuracy' in h else None,
        }

        return summary