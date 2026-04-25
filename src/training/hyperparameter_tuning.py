"""
Hyperparameter Tuning untuk SkillAlign AI.

Modul untuk pencarian hyperparameter optimal menggunakan
Keras Tuner.
"""

import os
import logging
from typing import Optional

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.models.custom_layers import CustomAttentionLayer
from src.models.custom_loss import focal_loss

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuning untuk SkillAlign model
    menggunakan Keras Tuner.

    Args:
        vocab_size: Ukuran vocabulary.
        max_seq_len: Panjang maksimum sequence.
        embedding_dim: Dimensi embedding. Default 128.
        project_name: Nama project untuk Keras Tuner.
            Default 'skillalign_tuning'.
        directory: Directory untuk menyimpan hasil tuning.
            Default 'logs/tuning'.

    Example:
        >>> tuner = HyperparameterTuner(
        ...     vocab_size=10000,
        ...     max_seq_len=500
        ... )
        >>> best_model = tuner.search(
        ...     x_train=[cv_train, job_train],
        ...     y_train=y_train,
        ...     x_val=[cv_val, job_val],
        ...     y_val=y_val
        ... )
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int = 128,
        project_name: str = 'skillalign_tuning',
        directory: str = 'logs/tuning'
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.project_name = project_name
        self.directory = directory
        self.tuner: Optional[kt.Tuner] = None

        os.makedirs(directory, exist_ok=True)

    def _build_model(self, hp: kt.HyperParameters) -> Model:
        """
        Build model dengan hyperparameters dari Keras Tuner.

        Args:
            hp: HyperParameters instance dari Keras Tuner.

        Returns:
            model: Compiled Keras Model.
        """
        # Hyperparameters to tune
        attention_units = hp.Int(
            'attention_units', min_value=64, max_value=256, step=64
        )
        conv_filters_1 = hp.Int(
            'conv_filters_1', min_value=64, max_value=256, step=64
        )
        conv_filters_2 = hp.Int(
            'conv_filters_2', min_value=32, max_value=128, step=32
        )
        dense_units_1 = hp.Int(
            'dense_units_1', min_value=128, max_value=512, step=128
        )
        dense_units_2 = hp.Int(
            'dense_units_2', min_value=64, max_value=256, step=64
        )
        dropout_rate = hp.Float(
            'dropout_rate', min_value=0.2, max_value=0.5, step=0.1
        )
        learning_rate = hp.Float(
            'learning_rate', min_value=1e-4, max_value=1e-2, sampling='log'
        )
        kernel_size = hp.Choice(
            'kernel_size', values=[3, 5, 7]
        )

        # ===== Build Model =====
        cv_input = Input(shape=(self.max_seq_len,), name='cv_input')
        job_input = Input(shape=(self.max_seq_len,), name='job_input')

        # Shared Embedding
        embedding_layer = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            mask_zero=False,
            name='shared_embedding'
        )

        cv_embedded = embedding_layer(cv_input)
        job_embedded = embedding_layer(job_input)

        # Conv1D — CV Branch
        cv_conv = Conv1D(
            conv_filters_1, kernel_size,
            activation='relu', padding='same'
        )(cv_embedded)
        cv_conv = BatchNormalization()(cv_conv)
        cv_conv = Conv1D(
            conv_filters_2, kernel_size,
            activation='relu', padding='same'
        )(cv_conv)
        cv_conv = BatchNormalization()(cv_conv)

        # Conv1D — Job Branch
        job_conv = Conv1D(
            conv_filters_1, kernel_size,
            activation='relu', padding='same'
        )(job_embedded)
        job_conv = BatchNormalization()(job_conv)
        job_conv = Conv1D(
            conv_filters_2, kernel_size,
            activation='relu', padding='same'
        )(job_conv)
        job_conv = BatchNormalization()(job_conv)

        # Custom Attention
        attention = CustomAttentionLayer(
            attention_units=attention_units
        )([cv_conv, job_conv])

        # Pooling
        cv_pooled = GlobalMaxPooling1D()(cv_conv)
        job_pooled = GlobalMaxPooling1D()(job_conv)

        # Merge
        merged = Concatenate()([attention, cv_pooled, job_pooled])

        # Dense Head
        x = Dense(dense_units_1, activation='relu')(merged)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(dense_units_2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

        output = Dense(1, activation='sigmoid')(x)

        model = Model(
            inputs=[cv_input, job_input],
            outputs=output
        )

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=focal_loss(),
            metrics=['accuracy']
        )

        return model

    def search(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        max_trials: int = 20,
        epochs_per_trial: int = 15,
        batch_size: int = 32
    ) -> tf.keras.Model:
        """
        Jalankan hyperparameter search.

        Args:
            x_train: Training input [cv_seq, job_seq].
            y_train: Training labels.
            x_val: Validation input [cv_seq, job_seq].
            y_val: Validation labels.
            max_trials: Maksimum trials. Default 20.
            epochs_per_trial: Epochs per trial. Default 15.
            batch_size: Batch size. Default 32.

        Returns:
            best_model: Model terbaik dari tuning.
        """
        self.tuner = kt.BayesianOptimization(
            self._build_model,
            objective='val_accuracy',
            max_trials=max_trials,
            directory=self.directory,
            project_name=self.project_name,
            overwrite=True
        )

        logger.info(
            f"Starting hyperparameter search: "
            f"{max_trials} trials, {epochs_per_trial} epochs/trial"
        )

        # Early stopping per trial
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        self.tuner.search(
            x_train, y_train,
            epochs=epochs_per_trial,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[early_stop],
            verbose=1
        )

        # Get best model
        best_model = self.tuner.get_best_models(num_models=1)[0]

        logger.info("Hyperparameter search completed.")
        return best_model

    def get_best_hyperparameters(self) -> dict:
        """
        Return best hyperparameters yang ditemukan.

        Returns:
            dict: Best hyperparameters.
        """
        if self.tuner is None:
            return {}

        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        return best_hp.values

    def print_search_summary(self) -> None:
        """Print summary dari search results."""
        if self.tuner is not None:
            self.tuner.results_summary()
