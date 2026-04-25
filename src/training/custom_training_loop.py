"""
Custom Training Loop untuk SkillAlign AI.

Implementasi training loop custom menggunakan tf.GradientTape
untuk fine-grained control atas training process (Side Quest).
"""

import time
import logging
from typing import Optional

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


class CustomTrainingLoop:
    """
    Custom training loop menggunakan tf.GradientTape
    untuk fine-grained control atas training process.

    Implementasi ini memberikan kontrol penuh terhadap:
    - Gradient computation dan application
    - Per-step metric tracking
    - Custom logging setiap batch/epoch
    - TensorBoard integration

    Args:
        model: TensorFlow/Keras model.
        optimizer: TF optimizer instance.
        loss_fn: Loss function.
        log_dir: Directory untuk TensorBoard logs.
            Default 'logs/custom_training'.

    Example:
        >>> loop = CustomTrainingLoop(
        ...     model=model,
        ...     optimizer=tf.keras.optimizers.Adam(1e-3),
        ...     loss_fn=focal_loss()
        ... )
        >>> loop.fit(
        ...     train_dataset=train_ds,
        ...     val_dataset=val_ds,
        ...     epochs=50
        ... )
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn,
        log_dir: str = 'logs/custom_training'
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.log_dir = log_dir

        # Metrics trackers
        self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy_metric = tf.keras.metrics.BinaryAccuracy(
            name='train_accuracy'
        )
        self.val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy_metric = tf.keras.metrics.BinaryAccuracy(
            name='val_accuracy'
        )

        # TensorBoard writer
        self.train_summary_writer = tf.summary.create_file_writer(
            f'{log_dir}/train'
        )
        self.val_summary_writer = tf.summary.create_file_writer(
            f'{log_dir}/validation'
        )

        # Training history
        self.history: dict = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1_score': [],
            'learning_rate': [],
            'epoch_time': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.patience_counter = 0

    @tf.function
    def train_step(
        self,
        x_cv: tf.Tensor,
        x_job: tf.Tensor,
        y_true: tf.Tensor
    ) -> tf.Tensor:
        """
        Single training step dengan GradientTape.

        Args:
            x_cv: CV input sequences.
            x_job: Job description input sequences.
            y_true: Ground truth labels.

        Returns:
            loss: Scalar loss value.
        """
        with tf.GradientTape() as tape:
            predictions = self.model(
                [x_cv, x_job], training=True
            )
            loss = self.loss_fn(y_true, predictions)

        # Compute dan apply gradients
        gradients = tape.gradient(
            loss, self.model.trainable_variables
        )

        # Gradient clipping untuk stability
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

        self.optimizer.apply_gradients(
            zip(gradients, self.model.trainable_variables)
        )

        # Update metrics
        self.train_loss_metric.update_state(loss)
        self.train_accuracy_metric.update_state(y_true, predictions)

        return loss

    @tf.function
    def val_step(
        self,
        x_cv: tf.Tensor,
        x_job: tf.Tensor,
        y_true: tf.Tensor
    ) -> tf.Tensor:
        """
        Single validation step (tanpa gradient).

        Args:
            x_cv: CV input sequences.
            x_job: Job description input sequences.
            y_true: Ground truth labels.

        Returns:
            loss: Scalar loss value.
        """
        predictions = self.model(
            [x_cv, x_job], training=False
        )
        loss = self.loss_fn(y_true, predictions)

        self.val_loss_metric.update_state(loss)
        self.val_accuracy_metric.update_state(y_true, predictions)

        return loss

    def _train_epoch(
        self,
        train_dataset: tf.data.Dataset,
        epoch: int
    ) -> None:
        """
        Train satu epoch penuh.

        Args:
            train_dataset: tf.data.Dataset untuk training.
            epoch: Nomor epoch saat ini.
        """
        for batch_idx, (x_cv, x_job, y_true) in enumerate(
            train_dataset
        ):
            loss = self.train_step(x_cv, x_job, y_true)

            if batch_idx % 50 == 0:
                logger.info(
                    f"  Epoch {epoch + 1}, Batch {batch_idx}: "
                    f"Loss={loss.numpy():.4f}"
                )

    def _validate_epoch(
        self,
        val_dataset: tf.data.Dataset
    ) -> float:
        """
        Validasi satu epoch penuh.

        Args:
            val_dataset: tf.data.Dataset untuk validasi.

        Returns:
            val_f1: F1-score pada validation data.
        """
        all_y_true = []
        all_y_pred = []

        for x_cv, x_job, y_true in val_dataset:
            self.val_step(x_cv, x_job, y_true)

            predictions = self.model(
                [x_cv, x_job], training=False
            )
            all_y_true.extend(y_true.numpy().flatten())
            all_y_pred.extend(
                (predictions.numpy() > 0.5).astype(int).flatten()
            )

        val_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)
        return val_f1

    def fit(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 50,
        patience: int = 10
    ) -> dict:
        """
        Training loop utama.

        Args:
            train_dataset: tf.data.Dataset berisi (x_cv, x_job, y).
                Contoh: tf.data.Dataset.from_tensor_slices(
                    (cv_seq, job_seq, labels)
                ).batch(32)
            val_dataset: Validation dataset (opsional).
            epochs: Jumlah epoch. Default 50.
            patience: Early stopping patience. Default 10.

        Returns:
            history: Dictionary berisi metrics per epoch.
        """
        logger.info(
            f"Starting custom training loop: "
            f"{epochs} epochs, patience={patience}"
        )

        for epoch in range(epochs):
            epoch_start = time.time()

            # Reset metrics di awal epoch
            self.train_loss_metric.reset_state()
            self.train_accuracy_metric.reset_state()
            self.val_loss_metric.reset_state()
            self.val_accuracy_metric.reset_state()

            # ===== Training =====
            self._train_epoch(train_dataset, epoch)

            # Log training metrics
            train_loss = self.train_loss_metric.result().numpy()
            train_acc = self.train_accuracy_metric.result().numpy()

            # Write to TensorBoard
            with self.train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss, step=epoch)
                tf.summary.scalar('accuracy', train_acc, step=epoch)
                tf.summary.scalar(
                    'learning_rate',
                    self.optimizer.learning_rate.numpy()
                    if hasattr(self.optimizer.learning_rate, 'numpy')
                    else self.optimizer.learning_rate,
                    step=epoch
                )

            # ===== Validation =====
            val_loss = 0.0
            val_acc = 0.0
            val_f1 = 0.0

            if val_dataset is not None:
                val_f1 = self._validate_epoch(val_dataset)
                val_loss = self.val_loss_metric.result().numpy()
                val_acc = self.val_accuracy_metric.result().numpy()

                with self.val_summary_writer.as_default():
                    tf.summary.scalar('loss', val_loss, step=epoch)
                    tf.summary.scalar('accuracy', val_acc, step=epoch)
                    tf.summary.scalar('f1_score', val_f1, step=epoch)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_weights = self.model.get_weights()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if self.patience_counter >= patience:
                    logger.info(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best val_loss: {self.best_val_loss:.4f}"
                    )
                    if self.best_weights is not None:
                        self.model.set_weights(self.best_weights)
                    break

            epoch_time = time.time() - epoch_start

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_f1_score'].append(val_f1)
            self.history['epoch_time'].append(epoch_time)

            # Print epoch summary
            print(
                f"\nEpoch {epoch + 1}/{epochs} "
                f"({epoch_time:.1f}s)\n"
                f"  Train: loss={train_loss:.4f}, "
                f"acc={train_acc:.4f}\n"
                f"  Val:   loss={val_loss:.4f}, "
                f"acc={val_acc:.4f}, "
                f"f1={val_f1:.4f}"
            )

        logger.info("Custom training loop completed.")
        return self.history

    @staticmethod
    def create_dataset(
        cv_sequences: np.ndarray,
        job_sequences: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        buffer_size: int = 10000
    ) -> tf.data.Dataset:
        """
        Helper untuk membuat tf.data.Dataset dari arrays.

        Args:
            cv_sequences: CV padded sequences.
            job_sequences: Job padded sequences.
            labels: Target labels.
            batch_size: Ukuran batch. Default 32.
            shuffle: Apakah shuffle data. Default True.
            buffer_size: Buffer size untuk shuffle. Default 10000.

        Returns:
            tf.data.Dataset: Dataset yang siap untuk training.
        """
        dataset = tf.data.Dataset.from_tensor_slices(
            (cv_sequences, job_sequences, labels)
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)

        dataset = dataset.batch(batch_size).prefetch(
            tf.data.AUTOTUNE
        )

        return dataset