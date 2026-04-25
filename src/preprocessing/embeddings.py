"""
Embedding Manager untuk SkillAlign AI.

Modul untuk mengelola word embeddings (Word2Vec / FastText)
termasuk training custom embeddings dan loading pre-trained vectors.
"""

import os
import logging
from typing import List, Optional, Tuple

import numpy as np
from gensim.models import Word2Vec, KeyedVectors

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manager untuk Word Embeddings.

    Mendukung:
    - Training Word2Vec dari corpus sendiri
    - Loading pre-trained embeddings (Word2Vec/FastText format)
    - Membuat embedding matrix untuk Keras Embedding layer

    Args:
        embedding_dim: Dimensi embedding vector. Default 128.
        min_count: Minimum frequency untuk word (Word2Vec). Default 2.
        window: Context window size (Word2Vec). Default 5.

    Example:
        >>> emb = EmbeddingManager(embedding_dim=128)
        >>> emb.train_word2vec(tokenized_texts)
        >>> matrix = emb.create_embedding_matrix(word_index, vocab_size)
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        min_count: int = 2,
        window: int = 5
    ):
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.window = window
        self.model: Optional[Word2Vec] = None
        self.keyed_vectors: Optional[KeyedVectors] = None

    def train_word2vec(
        self,
        tokenized_texts: List[List[str]],
        epochs: int = 20,
        workers: int = 4,
        sg: int = 1
    ) -> 'EmbeddingManager':
        """
        Train Word2Vec model pada corpus.

        Args:
            tokenized_texts: List of tokenized sentences.
                Contoh: [['python', 'machine', 'learning'], ...]
            epochs: Jumlah training epochs. Default 20.
            workers: Jumlah parallel workers. Default 4.
            sg: 1 untuk Skip-gram, 0 untuk CBOW. Default 1.

        Returns:
            self: Untuk method chaining.
        """
        logger.info(
            f"Training Word2Vec: dim={self.embedding_dim}, "
            f"epochs={epochs}, sg={'skip-gram' if sg else 'cbow'}"
        )

        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=workers,
            sg=sg,
            epochs=epochs
        )

        self.keyed_vectors = self.model.wv

        logger.info(
            f"Word2Vec trained. Vocabulary: {len(self.keyed_vectors)} words"
        )

        return self

    def load_pretrained(self, path: str, binary: bool = False) -> 'EmbeddingManager':
        """
        Load pre-trained word vectors.

        Mendukung format:
        - Word2Vec text format (.vec, .txt)
        - Word2Vec binary format (.bin)
        - Gensim KeyedVectors format

        Args:
            path: Path ke file pre-trained embeddings.
            binary: True jika format binary. Default False.

        Returns:
            self: Untuk method chaining.

        Raises:
            FileNotFoundError: Jika file tidak ditemukan.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Embedding file tidak ditemukan: {path}"
            )

        logger.info(f"Loading pre-trained embeddings dari: {path}")

        if path.endswith('.bin') or binary:
            self.keyed_vectors = KeyedVectors.load_word2vec_format(
                path, binary=True
            )
        elif path.endswith('.vec') or path.endswith('.txt'):
            self.keyed_vectors = KeyedVectors.load_word2vec_format(
                path, binary=False
            )
        else:
            # Coba load sebagai Gensim model
            self.keyed_vectors = KeyedVectors.load(path)

        # Update embedding_dim berdasarkan loaded vectors
        self.embedding_dim = self.keyed_vectors.vector_size

        logger.info(
            f"Loaded {len(self.keyed_vectors)} vectors "
            f"dengan dim={self.embedding_dim}"
        )

        return self

    def create_embedding_matrix(
        self,
        word_index: dict,
        vocab_size: int
    ) -> np.ndarray:
        """
        Buat embedding matrix untuk Keras Embedding layer.

        Untuk kata yang tidak ditemukan di pre-trained vectors,
        menggunakan random initialization (Xavier/Glorot uniform).

        Args:
            word_index: Dictionary word->index dari Tokenizer.
            vocab_size: Ukuran vocabulary (num_words dari Tokenizer).

        Returns:
            np.ndarray: Embedding matrix (vocab_size, embedding_dim).
        """
        if self.keyed_vectors is None:
            raise RuntimeError(
                "Belum ada embedding vectors. "
                "Panggil train_word2vec() atau load_pretrained() dahulu."
            )

        # Initialize dengan Xavier/Glorot uniform
        limit = np.sqrt(6.0 / (vocab_size + self.embedding_dim))
        embedding_matrix = np.random.uniform(
            -limit, limit,
            size=(vocab_size, self.embedding_dim)
        ).astype(np.float32)

        # Index 0 biasanya padding, set ke zero vector
        embedding_matrix[0] = np.zeros(self.embedding_dim)

        # Fill dengan pre-trained vectors
        found_count = 0
        for word, idx in word_index.items():
            if idx >= vocab_size:
                continue
            if word in self.keyed_vectors:
                embedding_matrix[idx] = self.keyed_vectors[word]
                found_count += 1

        coverage = found_count / min(len(word_index), vocab_size - 1)
        logger.info(
            f"Embedding matrix created: {vocab_size} x {self.embedding_dim}. "
            f"Coverage: {found_count}/{min(len(word_index), vocab_size - 1)} "
            f"({coverage:.1%})"
        )

        return embedding_matrix

    def save_model(self, path: str) -> None:
        """
        Simpan trained Word2Vec model.

        Args:
            path: Path untuk menyimpan model.
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Word2Vec model saved ke: {path}")
        elif self.keyed_vectors is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.keyed_vectors.save(path)
            logger.info(f"KeyedVectors saved ke: {path}")
        else:
            logger.warning("Tidak ada model/vectors untuk disimpan.")

    def get_similar_words(
        self, word: str, topn: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Dapatkan kata-kata paling mirip.

        Args:
            word: Kata target.
            topn: Jumlah kata mirip yang diambil. Default 10.

        Returns:
            List of (word, similarity_score) tuples.
        """
        if self.keyed_vectors is None:
            return []
        try:
            return self.keyed_vectors.most_similar(word, topn=topn)
        except KeyError:
            logger.warning(f"Kata '{word}' tidak ada di vocabulary.")
            return []
