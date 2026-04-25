"""
Model Architecture untuk SkillAlign AI.

Dual-Input Neural Network dengan TensorFlow Functional API
untuk CV-Job Matching menggunakan Custom Attention Layer.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization, Concatenate
)
from tensorflow.keras.models import Model

from .custom_layers import CustomAttentionLayer


class SkillAlignMatcher:
    """
    Dual-Input Neural Network untuk CV-Job Matching.

    Arsitektur ini menggunakan TensorFlow Functional API dengan:
    - Shared embedding layer untuk CV dan Job Description
    - Conv1D layers untuk feature extraction
    - Custom Attention Layer untuk cross-attention
    - Dense layers untuk final matching score

    Args:
        vocab_size: Ukuran vocabulary.
        max_seq_len: Panjang maksimum sequence setelah padding.
        embedding_dim: Dimensi embedding vector. Default 128.
        attention_units: Dimensi attention projection. Default 128.
        embedding_matrix: Pre-trained embedding matrix (opsional).
            Jika None, embedding akan di-train dari awal.
        trainable_embedding: Apakah embedding layer trainable.
            Default True.

    Example:
        >>> matcher = SkillAlignMatcher(
        ...     vocab_size=10000,
        ...     max_seq_len=500,
        ...     embedding_dim=128
        ... )
        >>> model = matcher.build_model()
        >>> model.summary()
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        embedding_dim: int = 128,
        attention_units: int = 128,
        embedding_matrix: np.ndarray = None,
        trainable_embedding: bool = True
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.attention_units = attention_units
        self.embedding_matrix = embedding_matrix
        self.trainable_embedding = trainable_embedding

    def build_model(self) -> Model:
        """
        Build dan return Keras Model.

        Returns:
            model: tf.keras.Model instance dengan dual-input architecture.
        """
        # ===== Input Layers =====
        cv_input = Input(
            shape=(self.max_seq_len,),
            dtype='int32',
            name='cv_input'
        )
        job_input = Input(
            shape=(self.max_seq_len,),
            dtype='int32',
            name='job_input'
        )

        # ===== Shared Embedding Layer =====
        if self.embedding_matrix is not None:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                trainable=self.trainable_embedding,
                mask_zero=False,
                name='shared_embedding'
            )
        else:
            embedding_layer = Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                trainable=True,
                mask_zero=False,
                name='shared_embedding'
            )

        cv_embedded = embedding_layer(cv_input)     # (batch, seq, emb_dim)
        job_embedded = embedding_layer(job_input)    # (batch, seq, emb_dim)

        # ===== Conv1D Feature Extraction =====
        # Conv1D Branch untuk CV
        cv_conv1 = Conv1D(
            filters=128, kernel_size=3,
            activation='relu', padding='same',
            name='cv_conv1d_1'
        )(cv_embedded)
        cv_conv1 = BatchNormalization(name='cv_bn_1')(cv_conv1)

        cv_conv2 = Conv1D(
            filters=64, kernel_size=3,
            activation='relu', padding='same',
            name='cv_conv1d_2'
        )(cv_conv1)
        cv_conv2 = BatchNormalization(name='cv_bn_2')(cv_conv2)

        # Conv1D Branch untuk Job Description
        job_conv1 = Conv1D(
            filters=128, kernel_size=3,
            activation='relu', padding='same',
            name='job_conv1d_1'
        )(job_embedded)
        job_conv1 = BatchNormalization(name='job_bn_1')(job_conv1)

        job_conv2 = Conv1D(
            filters=64, kernel_size=3,
            activation='relu', padding='same',
            name='job_conv1d_2'
        )(job_conv1)
        job_conv2 = BatchNormalization(name='job_bn_2')(job_conv2)

        # ===== Custom Attention Layer =====
        attention_output = CustomAttentionLayer(
            attention_units=self.attention_units,
            name='custom_attention'
        )([cv_conv2, job_conv2])

        # ===== Global Max Pooling Branch =====
        cv_pooled = GlobalMaxPooling1D(name='cv_global_pool')(cv_conv2)
        job_pooled = GlobalMaxPooling1D(name='job_global_pool')(job_conv2)

        # ===== Concatenate semua features =====
        merged = Concatenate(name='feature_merge')(
            [attention_output, cv_pooled, job_pooled]
        )

        # ===== Dense Classification Head =====
        x = Dense(256, activation='relu', name='dense_1')(merged)
        x = BatchNormalization(name='dense_bn_1')(x)
        x = Dropout(0.4, name='dropout_1')(x)

        x = Dense(128, activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='dense_bn_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)

        x = Dense(64, activation='relu', name='dense_3')(x)
        x = Dropout(0.2, name='dropout_3')(x)

        # ===== Output Layer =====
        output = Dense(
            1, activation='sigmoid', name='matching_score'
        )(x)

        # ===== Build Model =====
        model = Model(
            inputs=[cv_input, job_input],
            outputs=output,
            name='SkillAlign_Matcher'
        )

        return model

    def get_model_config(self) -> dict:
        """
        Return model configuration sebagai dictionary.

        Returns:
            config: Dictionary berisi konfigurasi model.
        """
        return {
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'embedding_dim': self.embedding_dim,
            'attention_units': self.attention_units,
            'has_pretrained_embedding': self.embedding_matrix is not None,
            'trainable_embedding': self.trainable_embedding,
            'architecture': 'Dual-Input CNN with Custom Attention'
        }