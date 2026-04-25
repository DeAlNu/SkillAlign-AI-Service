"""
Unit Tests untuk SkillAlign Model Components.

Test coverage:
- Custom Attention Layer
- Custom Focal Loss
- Custom Callbacks
- Model Architecture (SkillAlignMatcher)
"""

import pytest
import numpy as np
import tensorflow as tf

from src.models.custom_layers import CustomAttentionLayer
from src.models.custom_loss import focal_loss
from src.models.model_architecture import SkillAlignMatcher


# ==========================================
# Tests: Custom Attention Layer
# ==========================================

class TestCustomAttentionLayer:
    """Tests untuk CustomAttentionLayer."""

    def test_layer_creation(self):
        """Test bahwa layer bisa dibuat tanpa error."""
        layer = CustomAttentionLayer(attention_units=128)
        assert layer is not None
        assert layer.attention_units == 128

    def test_output_shape(self):
        """Test bahwa output shape sesuai ekspektasi."""
        layer = CustomAttentionLayer(attention_units=64)
        batch_size = 4
        seq_len = 10
        emb_dim = 32

        cv_input = tf.random.normal((batch_size, seq_len, emb_dim))
        job_input = tf.random.normal((batch_size, seq_len, emb_dim))

        output = layer([cv_input, job_input])

        # Output shape: (batch_size, attention_units)
        assert output.shape == (batch_size, 64)

    def test_different_attention_units(self):
        """Test dengan berbagai attention_units."""
        for units in [32, 64, 128, 256]:
            layer = CustomAttentionLayer(attention_units=units)
            cv = tf.random.normal((2, 5, 16))
            job = tf.random.normal((2, 5, 16))
            output = layer([cv, job])
            assert output.shape[-1] == units

    def test_serialization(self):
        """Test get_config dan from_config."""
        layer = CustomAttentionLayer(attention_units=128)
        config = layer.get_config()

        assert 'attention_units' in config
        assert config['attention_units'] == 128

        # Recreate dari config
        new_layer = CustomAttentionLayer.from_config(config)
        assert new_layer.attention_units == 128

    def test_output_not_nan(self):
        """Test bahwa output tidak mengandung NaN."""
        layer = CustomAttentionLayer(attention_units=64)
        cv = tf.random.normal((2, 10, 32))
        job = tf.random.normal((2, 10, 32))
        output = layer([cv, job])

        assert not tf.reduce_any(tf.math.is_nan(output)).numpy()


# ==========================================
# Tests: Custom Focal Loss
# ==========================================

class TestFocalLoss:
    """Tests untuk focal_loss function."""

    def test_loss_creation(self):
        """Test bahwa loss function bisa dibuat."""
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        assert callable(loss_fn)

    def test_loss_value_range(self):
        """Test bahwa loss value non-negative."""
        loss_fn = focal_loss()
        y_true = tf.constant([[1.0], [0.0], [1.0], [0.0]])
        y_pred = tf.constant([[0.9], [0.1], [0.8], [0.2]])

        loss_val = loss_fn(y_true, y_pred)
        assert loss_val.numpy() >= 0

    def test_perfect_prediction(self):
        """Test bahwa perfect prediction menghasilkan loss kecil."""
        loss_fn = focal_loss()
        y_true = tf.constant([[1.0], [0.0]])
        y_pred = tf.constant([[0.99], [0.01]])

        loss_val = loss_fn(y_true, y_pred)
        assert loss_val.numpy() < 0.1

    def test_wrong_prediction(self):
        """Test bahwa wrong prediction menghasilkan loss besar."""
        loss_fn = focal_loss()
        y_true = tf.constant([[1.0], [0.0]])
        y_pred_good = tf.constant([[0.9], [0.1]])
        y_pred_bad = tf.constant([[0.1], [0.9]])

        loss_good = loss_fn(y_true, y_pred_good)
        loss_bad = loss_fn(y_true, y_pred_bad)

        assert loss_bad.numpy() > loss_good.numpy()

    def test_different_gamma(self):
        """Test bahwa gamma mempengaruhi loss value."""
        y_true = tf.constant([[1.0], [0.0]])
        y_pred = tf.constant([[0.5], [0.5]])

        loss_g1 = focal_loss(gamma=1.0)(y_true, y_pred)
        loss_g3 = focal_loss(gamma=3.0)(y_true, y_pred)

        # Gamma lebih tinggi -> loss lebih kecil untuk ambiguous pred
        assert loss_g3.numpy() < loss_g1.numpy()

    def test_loss_name(self):
        """Test bahwa loss function memiliki nama."""
        loss_fn = focal_loss()
        assert hasattr(loss_fn, '__name__')
        assert loss_fn.__name__ == 'focal_loss'


# ==========================================
# Tests: Model Architecture
# ==========================================

class TestSkillAlignMatcher:
    """Tests untuk SkillAlignMatcher."""

    def test_model_creation(self):
        """Test bahwa model bisa dibuat tanpa error."""
        matcher = SkillAlignMatcher(
            vocab_size=1000,
            max_seq_len=50,
            embedding_dim=32
        )
        model = matcher.build_model()
        assert model is not None
        assert model.name == 'SkillAlign_Matcher'

    def test_model_inputs(self):
        """Test bahwa model memiliki 2 inputs."""
        matcher = SkillAlignMatcher(
            vocab_size=1000,
            max_seq_len=50,
            embedding_dim=32
        )
        model = matcher.build_model()
        assert len(model.inputs) == 2

    def test_model_output_shape(self):
        """Test bahwa output shape adalah (batch, 1)."""
        matcher = SkillAlignMatcher(
            vocab_size=1000,
            max_seq_len=50,
            embedding_dim=32
        )
        model = matcher.build_model()

        cv_input = np.random.randint(0, 1000, size=(4, 50))
        job_input = np.random.randint(0, 1000, size=(4, 50))

        output = model.predict([cv_input, job_input], verbose=0)
        assert output.shape == (4, 1)

    def test_output_range(self):
        """Test bahwa output antara 0 dan 1 (sigmoid)."""
        matcher = SkillAlignMatcher(
            vocab_size=500,
            max_seq_len=30,
            embedding_dim=16
        )
        model = matcher.build_model()

        cv_input = np.random.randint(0, 500, size=(8, 30))
        job_input = np.random.randint(0, 500, size=(8, 30))

        output = model.predict([cv_input, job_input], verbose=0)

        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)

    def test_model_with_pretrained_embedding(self):
        """Test model dengan pre-trained embedding matrix."""
        vocab_size = 500
        emb_dim = 32
        embedding_matrix = np.random.randn(
            vocab_size, emb_dim
        ).astype(np.float32)

        matcher = SkillAlignMatcher(
            vocab_size=vocab_size,
            max_seq_len=30,
            embedding_dim=emb_dim,
            embedding_matrix=embedding_matrix
        )
        model = matcher.build_model()
        assert model is not None

    def test_model_compilable(self):
        """Test bahwa model bisa di-compile."""
        matcher = SkillAlignMatcher(
            vocab_size=500,
            max_seq_len=30,
            embedding_dim=16
        )
        model = matcher.build_model()
        model.compile(
            optimizer='adam',
            loss=focal_loss(),
            metrics=['accuracy']
        )

    def test_model_config(self):
        """Test get_model_config."""
        matcher = SkillAlignMatcher(
            vocab_size=1000,
            max_seq_len=50,
            embedding_dim=64
        )
        config = matcher.get_model_config()

        assert config['vocab_size'] == 1000
        assert config['max_seq_len'] == 50
        assert config['embedding_dim'] == 64
        assert 'architecture' in config

    def test_model_trainable(self):
        """Test bahwa model bisa di-train (1 step)."""
        matcher = SkillAlignMatcher(
            vocab_size=200,
            max_seq_len=20,
            embedding_dim=16,
            attention_units=16
        )
        model = matcher.build_model()
        model.compile(
            optimizer='adam',
            loss=focal_loss(),
            metrics=['accuracy']
        )

        cv = np.random.randint(0, 200, size=(4, 20))
        job = np.random.randint(0, 200, size=(4, 20))
        labels = np.random.randint(0, 2, size=(4, 1)).astype(np.float32)

        history = model.fit(
            [cv, job], labels,
            epochs=1, batch_size=2, verbose=0
        )
        assert 'loss' in history.history


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
