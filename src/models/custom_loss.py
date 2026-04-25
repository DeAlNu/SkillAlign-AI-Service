"""
Custom Loss Functions untuk SkillAlign AI.

Implementasi Focal Loss untuk menangani class imbalance
pada dataset CV-Job matching.
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    Custom Focal Loss untuk menangani class imbalance
    pada dataset CV-Job matching.

    Focal Loss mengurangi kontribusi easy examples terhadap total loss,
    sehingga model lebih fokus pada hard examples yang sulit diklasifikasi.

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: Focusing parameter. Semakin besar gamma, semakin besar
            pengurangan loss untuk easy examples. Default 2.0.
        alpha: Balancing factor untuk class weights. Default 0.25.

    Returns:
        loss: Callable loss function yang compatible dengan TensorFlow/Keras.

    Example:
        >>> model.compile(
        ...     optimizer='adam',
        ...     loss=focal_loss(gamma=2.0, alpha=0.25)
        ... )
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute focal loss.

        Args:
            y_true: Ground truth labels (0 atau 1)
            y_pred: Predicted probabilities (0 sampai 1)

        Returns:
            Scalar tensor: mean focal loss
        """
        # Cast y_true ke float32 untuk kompatibilitas
        y_true = tf.cast(y_true, tf.float32)

        # Clip predictions untuk numerical stability
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

        # Compute p_t (probability of correct class)
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # Compute alpha factor
        alpha_factor = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = alpha_factor * K.pow(1.0 - p_t, gamma)

        # Binary cross-entropy (manual calculation)
        bce = -(
            y_true * K.log(y_pred)
            + (1.0 - y_true) * K.log(1.0 - y_pred)
        )

        # Focal loss = focal_weight * BCE
        focal_loss_val = focal_weight * bce

        return K.mean(focal_loss_val)

    # Set nama fungsi untuk serialization
    loss.__name__ = 'focal_loss'
    return loss


def cosine_similarity_loss():
    """
    Cosine Similarity Loss sebagai alternatif loss function
    untuk matching task.

    Mengukur sejauh mana dua vektor memiliki arah yang sama.

    Returns:
        loss: Callable loss function.
    """
    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute cosine similarity loss."""
        y_true = tf.cast(y_true, tf.float32)
        return tf.reduce_mean(
            tf.keras.losses.cosine_similarity(y_true, y_pred)
        )

    loss.__name__ = 'cosine_similarity_loss'
    return loss