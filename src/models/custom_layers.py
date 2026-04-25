"""
Custom Attention Layer untuk SkillAlign AI.

Implementasi mekanisme attention untuk menghitung similarity
antara representasi CV dan Job Description.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense


class CustomAttentionLayer(Layer):
    """
    Custom Attention Mechanism untuk menghitung similarity
    antara CV dan Job Description.

    Layer ini mengimplementasikan scaled dot-product attention
    yang menghitung cross-attention antara dua input sequences
    (CV embedding dan Job Description embedding).

    Args:
        attention_units: Dimensi internal untuk attention projection.
            Default 128.

    Input:
        List dari 2 tensors:
        - cv_embedded: (batch_size, seq_len, embedding_dim)
        - job_embedded: (batch_size, seq_len, embedding_dim)

    Output:
        Tensor: (batch_size, attention_units) — context vector
        yang merepresentasikan cross-attention antara CV dan Job.
    """

    def __init__(self, attention_units: int = 128, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)
        self.attention_units = attention_units

    def build(self, input_shape: list) -> None:
        """Build layer weights berdasarkan input shape."""
        self.W_query = Dense(
            self.attention_units,
            activation='tanh',
            name='attention_query'
        )
        self.W_key = Dense(
            self.attention_units,
            activation='tanh',
            name='attention_key'
        )
        self.W_value = Dense(
            self.attention_units,
            activation=None,
            name='attention_value'
        )
        super(CustomAttentionLayer, self).build(input_shape)

    def call(self, inputs: list, training: bool = None) -> tf.Tensor:
        """
        Forward pass dari attention mechanism.

        Args:
            inputs: List berisi [cv_embedded, job_embedded]
            training: Boolean flag untuk training mode

        Returns:
            context_vector: Tensor (batch_size, attention_units)
        """
        cv_embedded, job_embedded = inputs

        # Project ke query dan key space
        query = self.W_query(cv_embedded)    # (batch, seq_cv, attn_units)
        key = self.W_key(job_embedded)       # (batch, seq_job, attn_units)
        value = self.W_value(job_embedded)   # (batch, seq_job, attn_units)

        # Scaled dot-product attention scores
        scale = tf.math.sqrt(tf.cast(self.attention_units, tf.float32))
        attention_scores = tf.matmul(
            query, key, transpose_b=True
        ) / scale  # (batch, seq_cv, seq_job)

        # Softmax normalization
        attention_weights = tf.nn.softmax(
            attention_scores, axis=-1
        )  # (batch, seq_cv, seq_job)

        # Terapkan attention weights ke value
        context_vector = tf.matmul(
            attention_weights, value
        )  # (batch, seq_cv, attn_units)

        # Global average pooling untuk mendapatkan fixed-size output
        output = tf.reduce_mean(context_vector, axis=1)  # (batch, attn_units)

        return output

    def get_config(self) -> dict:
        """Serialization config untuk model saving/loading."""
        config = super(CustomAttentionLayer, self).get_config()
        config.update({
            'attention_units': self.attention_units
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'CustomAttentionLayer':
        """Deserialization dari config."""
        return cls(**config)