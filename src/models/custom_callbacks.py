"""
Custom Callbacks untuk SkillAlign AI.

Implementasi custom callback untuk monitoring F1-Score
dan early stopping berbasis F1 selama training.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score


class F1ScoreCallback(Callback):
    """
    Custom Callback untuk monitoring F1-Score
    dan early stopping berbasis F1.

    Callback ini menghitung F1-score pada akhir setiap epoch
    menggunakan validation data, dan menghentikan training
    jika F1-score tidak meningkat selama `patience` epoch.

    Args:
        validation_data: Tuple (x_val, y_val) atau
            Tuple ([x_cv_val, x_job_val], y_val)
        threshold: F1-score threshold target. Default 0.80.
        patience: Jumlah epoch tanpa peningkatan sebelum stop. Default 5.
        save_best_model: Apakah menyimpan model terbaik. Default True.
        model_save_path: Path untuk menyimpan model terbaik.
            Default 'models/best_f1_model.keras'.
        verbose: Level logging (0=silent, 1=progress). Default 1.

    Attributes:
        best_f1: F1-score terbaik yang pernah dicapai.
        wait: Counter epoch tanpa improvement.
        best_weights: Weights model saat F1-score terbaik.

    Example:
        >>> f1_callback = F1ScoreCallback(
        ...     validation_data=([x_cv_val, x_job_val], y_val),
        ...     patience=5,
        ...     threshold=0.80
        ... )
        >>> model.fit(..., callbacks=[f1_callback])
    """

    def __init__(
        self,
        validation_data: tuple,
        threshold: float = 0.80,
        patience: int = 5,
        save_best_model: bool = True,
        model_save_path: str = 'models/best_f1_model.keras',
        verbose: int = 1
    ):
        super(F1ScoreCallback, self).__init__()
        self.x_val, self.y_val = validation_data
        self.threshold = threshold
        self.patience = patience
        self.save_best_model = save_best_model
        self.model_save_path = model_save_path
        self.verbose = verbose

        # State tracking
        self.best_f1: float = 0.0
        self.wait: int = 0
        self.best_weights = None
        self.history: dict = {
            'f1_score': [],
            'precision': [],
            'recall': []
        }

    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        """
        Dipanggil di akhir setiap epoch.

        Menghitung F1-score, precision, recall pada validation data
        dan melakukan early stopping jika diperlukan.

        Args:
            epoch: Nomor epoch saat ini.
            logs: Dictionary berisi metrics training.
        """
        logs = logs or {}

        # Predict pada validation data
        y_pred_proba = self.model.predict(self.x_val, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        y_true = np.array(self.y_val).flatten()

        # Hitung metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # Log ke history
        self.history['f1_score'].append(f1)
        self.history['precision'].append(precision)
        self.history['recall'].append(recall)

        # Tambahkan ke Keras logs agar bisa diakses callback lain
        logs['val_f1_score'] = f1
        logs['val_precision_custom'] = precision
        logs['val_recall_custom'] = recall

        if self.verbose:
            print(
                f"\n  [F1Callback] Epoch {epoch + 1}: "
                f"F1={f1:.4f} | Precision={precision:.4f} | "
                f"Recall={recall:.4f} | Best F1={self.best_f1:.4f}"
            )

        # Check improvement
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.wait = 0
            self.best_weights = self.model.get_weights()

            if self.save_best_model:
                # Pastikan directory ada
                os.makedirs(
                    os.path.dirname(self.model_save_path),
                    exist_ok=True
                )
                self.model.save(self.model_save_path)
                if self.verbose:
                    print(
                        f"  [F1Callback] Model saved to "
                        f"{self.model_save_path} (F1={f1:.4f})"
                    )

            # Check apakah sudah mencapai threshold
            if f1 >= self.threshold:
                if self.verbose:
                    print(
                        f"\n  [F1Callback] F1-score {f1:.4f} "
                        f">= threshold {self.threshold:.4f}. "
                        f"Target tercapai!"
                    )
        else:
            self.wait += 1
            if self.verbose:
                print(
                    f"  [F1Callback] No improvement. "
                    f"Patience: {self.wait}/{self.patience}"
                )

            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.verbose:
                    print(
                        f"\n  [F1Callback] Early stopping at epoch "
                        f"{epoch + 1}. Best F1={self.best_f1:.4f}"
                    )
                # Restore best weights
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                    if self.verbose:
                        print("  [F1Callback] Best weights restored.")

    def on_train_end(self, logs: dict = None) -> None:
        """
        Dipanggil di akhir training. Print summary.
        """
        if self.verbose and self.history['f1_score']:
            print("\n" + "=" * 50)
            print("F1 Score Callback Summary")
            print("=" * 50)
            print(f"  Best F1-Score: {self.best_f1:.4f}")
            print(
                f"  Best Epoch: "
                f"{np.argmax(self.history['f1_score']) + 1}"
            )
            print(f"  Total Epochs: {len(self.history['f1_score'])}")
            print("=" * 50)