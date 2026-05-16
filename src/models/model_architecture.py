"""
Model Architecture untuk SkillAlign AI.

Dual-Input Neural Network dengan TensorFlow Functional API
untuk CV-Job Matching menggunakan Custom Attention Layer.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Dropout, BatchNormalization, Concatenate,
    SpatialDropout1D
)
from tensorflow.keras import regularizers
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


class SkillAlignMatcherV4(SkillAlignMatcher):
    """
    Fase 2 upgrade: Multi-Scale CNN + Spatial Dropout + L2 Regularization.

    Improvements over v3 (SkillAlignMatcher):
    - Multi-scale CNN: 3 paralel branch per encoder (kernel_size ∈ {2, 3, 5})
      → Model bisa tangkap pola bigram, trigram, dan 5-gram sekaligus
    - SpatialDropout1D setelah embedding (lebih efektif dari Dropout untuk sequence)
    - L2 regularization pada Conv1D & Dense layers → kurangi overfitting
    - Wider merged representation: 128 (attention) + 192 (cv) + 192 (job) = 512
      vs sebelumnya 256 (128 attention + 64 cv + 64 job)
    - Cross-attention tetap menggunakan k=3 sequential branch (representasi terbaik)

    Args:
        l2_reg: L2 regularization factor. Default 1e-4.
        spatial_dropout: SpatialDropout rate setelah embedding. Default 0.2.
        kernel_sizes: Tuple of kernel sizes untuk multi-scale CNN. Default (2, 3, 5).
        conv_filters: Tuple (filters_layer1, filters_layer2). Default (128, 64).

    Note:
        kernel_sizes HARUS mengandung nilai 3, karena branch k=3 dipakai
        untuk CustomAttentionLayer (sequential input). Jika tidak ada k=3,
        branch terakhir yang dipakai untuk attention.

    Example:
        >>> matcher = SkillAlignMatcherV4(
        ...     vocab_size=15000,
        ...     max_seq_len=300,
        ...     embedding_dim=128,
        ...     embedding_matrix=embedding_matrix,
        ...     l2_reg=1e-4,
        ...     spatial_dropout=0.2,
        ...     kernel_sizes=(2, 3, 5),
        ...     conv_filters=(128, 64),
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
        trainable_embedding: bool = True,
        l2_reg: float = 1e-4,
        spatial_dropout: float = 0.2,
        kernel_sizes: tuple = (2, 3, 5),
        conv_filters: tuple = (128, 64),
    ):
        super().__init__(
            vocab_size, max_seq_len, embedding_dim,
            attention_units, embedding_matrix, trainable_embedding
        )
        self.l2_reg = l2_reg
        self.spatial_dropout = spatial_dropout
        self.kernel_sizes = kernel_sizes
        self.conv_filters = conv_filters

    def _build_multiscale_branch(self, x, name_prefix: str):
        """
        Build multi-scale CNN encoder untuk satu input (CV atau Job).

        Setiap kernel size memiliki 2 Conv1D layers. Output dari setiap branch
        di-pool (GlobalMaxPooling1D), lalu di-concat untuk menghasilkan
        representasi multi-scale yang kaya.

        Args:
            x: Input tensor (batch, seq_len, embedding_dim) setelah SpatialDropout
            name_prefix: 'cv' atau 'job'

        Returns:
            pooled_concat: Tensor (batch, conv_filters[1] * len(kernel_sizes))
                           — dimasukkan ke Dense head
            seq_for_attn:  Tensor (batch, seq_len, conv_filters[1])
                           — branch k=3 sequential, untuk CustomAttentionLayer
        """
        reg = regularizers.l2(self.l2_reg)
        f1, f2 = self.conv_filters
        pooled_branches = []
        seq_for_attn = None
        attn_kernel = 3 if 3 in self.kernel_sizes else self.kernel_sizes[-1]

        for k in self.kernel_sizes:
            # ── Conv Layer 1 ──
            b = Conv1D(
                f1, k, activation='relu', padding='same',
                kernel_regularizer=reg,
                name=f'{name_prefix}_conv_k{k}_1'
            )(x)
            b = BatchNormalization(name=f'{name_prefix}_bn_k{k}_1')(b)

            # ── Conv Layer 2 ──
            b = Conv1D(
                f2, k, activation='relu', padding='same',
                kernel_regularizer=reg,
                name=f'{name_prefix}_conv_k{k}_2'
            )(b)
            b = BatchNormalization(name=f'{name_prefix}_bn_k{k}_2')(b)

            # Branch k=attn_kernel disimpan sequential untuk attention
            if k == attn_kernel:
                seq_for_attn = b   # (batch, seq_len, f2)

            # Global Max Pool → fixed-size per branch
            pooled = GlobalMaxPooling1D(name=f'{name_prefix}_pool_k{k}')(b)
            pooled_branches.append(pooled)

        # Concat semua branch: (batch, f2 * n_kernels)
        if len(pooled_branches) > 1:
            pooled_concat = Concatenate(
                name=f'{name_prefix}_ms_merge'
            )(pooled_branches)
        else:
            pooled_concat = pooled_branches[0]

        return pooled_concat, seq_for_attn

    def build_model(self) -> Model:
        """
        Build Multi-Scale Dual-Input CNN model (v4).

        Architecture summary:
            Input → Embedding(128) → SpatialDropout(0.2) →
            Multi-Scale CNN [k=2,3,5] (2 Conv layers each) →
            Cross-Attention on k=3 branch (128) +
            GlobalMaxPool dari semua branches (192 per encoder) →
            Merged(512) →
            Dense(256, L2) + BN + Drop(0.4) →
            Dense(128, L2) + BN + Drop(0.3) →
            Dense(64) + Drop(0.2) →
            Dense(1, sigmoid)

        Returns:
            model: tf.keras.Model dengan nama 'SkillAlign_Matcher_V4'
        """
        # ── Input Layers ──
        cv_input  = Input(shape=(self.max_seq_len,), dtype='int32', name='cv_input')
        job_input = Input(shape=(self.max_seq_len,), dtype='int32', name='job_input')

        # ── Shared Embedding ──
        if self.embedding_matrix is not None:
            embedding_layer = Embedding(
                self.vocab_size, self.embedding_dim,
                weights=[self.embedding_matrix],
                trainable=self.trainable_embedding,
                mask_zero=False, name='shared_embedding'
            )
        else:
            embedding_layer = Embedding(
                self.vocab_size, self.embedding_dim,
                trainable=True, mask_zero=False, name='shared_embedding'
            )

        cv_emb  = embedding_layer(cv_input)
        job_emb = embedding_layer(job_input)

        # ── Spatial Dropout (lebih efektif untuk sequences) ──
        cv_emb  = SpatialDropout1D(self.spatial_dropout, name='cv_spatial_drop')(cv_emb)
        job_emb = SpatialDropout1D(self.spatial_dropout, name='job_spatial_drop')(job_emb)

        # ── Multi-Scale CNN Encoder ──
        cv_ms,  cv_seq_attn  = self._build_multiscale_branch(cv_emb,  'cv')
        job_ms, job_seq_attn = self._build_multiscale_branch(job_emb, 'job')

        # ── Cross-Attention pada k=3 sequential branch ──
        attention_out = CustomAttentionLayer(
            attention_units=self.attention_units, name='custom_attention'
        )([cv_seq_attn, job_seq_attn])

        # ── Merge: attention(128) + cv_ms(192) + job_ms(192) = 512 ──
        merged = Concatenate(name='feature_merge')(
            [attention_out, cv_ms, job_ms]
        )

        # ── Dense Head dengan L2 regularization ──
        reg = regularizers.l2(self.l2_reg)

        x = Dense(256, activation='relu', kernel_regularizer=reg, name='dense_1')(merged)
        x = BatchNormalization(name='dense_bn_1')(x)
        x = Dropout(0.4, name='dropout_1')(x)

        x = Dense(128, activation='relu', kernel_regularizer=reg, name='dense_2')(x)
        x = BatchNormalization(name='dense_bn_2')(x)
        x = Dropout(0.3, name='dropout_2')(x)

        x = Dense(64, activation='relu', name='dense_3')(x)
        x = Dropout(0.2, name='dropout_3')(x)

        output = Dense(1, activation='sigmoid', name='matching_score')(x)

        return Model(
            inputs=[cv_input, job_input],
            outputs=output,
            name='SkillAlign_Matcher_V4'
        )

    def get_model_config(self) -> dict:
        """Return v4 model configuration."""
        config = super().get_model_config()
        config.update({
            'architecture': 'Multi-Scale Dual-Input CNN + Cross-Attention (v4)',
            'version': 'v4',
            'l2_reg': self.l2_reg,
            'spatial_dropout': self.spatial_dropout,
            'kernel_sizes': list(self.kernel_sizes),
            'conv_filters': list(self.conv_filters),
        })
        return config