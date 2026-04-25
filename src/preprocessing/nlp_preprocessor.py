"""
NLP Preprocessor untuk SkillAlign AI.

Pipeline preprocessing teks untuk CV dan Job Description
termasuk cleaning, tokenization, stemming, dan padding.
"""

import re
import logging
from typing import List, Optional

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

logger = logging.getLogger(__name__)


class NLPPreprocessor:
    """
    NLP Preprocessing Pipeline untuk CV dan Job Description.

    Pipeline steps:
    1. Lowercase
    2. Remove special characters & digits
    3. Tokenization
    4. Stopword removal
    5. Stemming/Lemmatization
    6. Sequence encoding & padding

    Args:
        max_vocab_size: Ukuran maksimum vocabulary. Default 10000.
        max_seq_len: Panjang maksimum sequence setelah padding. Default 500.
        use_lemmatizer: Gunakan lemmatizer daripada stemmer. Default True.
        language: Bahasa untuk stopwords. Default 'english'.

    Example:
        >>> preprocessor = NLPPreprocessor(
        ...     max_vocab_size=10000,
        ...     max_seq_len=500
        ... )
        >>> preprocessor.fit(train_texts)
        >>> sequences = preprocessor.transform(new_texts)
    """

    def __init__(
        self,
        max_vocab_size: int = 10000,
        max_seq_len: int = 500,
        use_lemmatizer: bool = True,
        language: str = 'english'
    ):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.use_lemmatizer = use_lemmatizer
        self.language = language

        # Tokenizer Keras
        self.tokenizer = Tokenizer(
            num_words=max_vocab_size,
            oov_token='<OOV>',
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        )

        # NLP tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # Download NLTK data jika belum ada
        self._download_nltk_data()

        # Stopwords
        try:
            self.stop_words = set(stopwords.words(self.language))
        except OSError:
            logger.warning(
                f"Stopwords untuk '{self.language}' tidak tersedia. "
                f"Menggunakan set kosong."
            )
            self.stop_words = set()

        # State
        self.is_fitted = False
        self.embedding_matrix: Optional[np.ndarray] = None

    def _download_nltk_data(self) -> None:
        """Download NLTK data yang diperlukan (silent mode)."""
        required_packages = [
            'punkt', 'punkt_tab', 'stopwords', 'wordnet',
            'omw-1.4'
        ]
        for package in required_packages:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                logger.warning(f"Gagal download NLTK package '{package}': {e}")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocessing pipeline untuk satu dokumen teks.

        Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove email addresses
        4. Remove special characters & excessive whitespace
        5. Tokenization
        6. Stopword removal
        7. Stemming / Lemmatization
        8. Reconstruct sebagai string

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        if not text or not isinstance(text, str):
            return ""

        # 1. Lowercase
        text = text.lower().strip()

        # 2. Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)

        # 3. Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # 4. Remove special characters, keep letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)

        # 5. Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # 6. Tokenization
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()

        # 7. Stopword removal
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]

        # 8. Stemming / Lemmatization
        if self.use_lemmatizer:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        else:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return ' '.join(tokens)

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess batch teks.

        Args:
            texts: List of raw text strings.

        Returns:
            List of cleaned text strings.
        """
        return [self.preprocess_text(t) for t in texts]

    def fit(self, texts: List[str]) -> 'NLPPreprocessor':
        """
        Fit tokenizer pada training data.

        Args:
            texts: List of training texts (sudah di-preprocess).

        Returns:
            self: Untuk method chaining.
        """
        # Preprocess dulu
        cleaned_texts = self.preprocess_batch(texts)

        # Fit tokenizer
        self.tokenizer.fit_on_texts(cleaned_texts)
        self.is_fitted = True

        vocab_size = min(
            len(self.tokenizer.word_index) + 1,
            self.max_vocab_size
        )
        logger.info(
            f"Tokenizer fitted. Vocabulary size: {vocab_size}"
        )

        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform teks menjadi padded sequences.

        Args:
            texts: List of texts to transform.

        Returns:
            np.ndarray: Padded sequences (len(texts), max_seq_len).

        Raises:
            RuntimeError: Jika tokenizer belum di-fit.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Tokenizer belum di-fit. "
                "Panggil fit() terlebih dahulu."
            )

        # Preprocess
        cleaned_texts = self.preprocess_batch(texts)

        # Convert ke sequences
        sequences = self.tokenizer.texts_to_sequences(cleaned_texts)

        # Padding
        padded = pad_sequences(
            sequences,
            maxlen=self.max_seq_len,
            padding='post',
            truncating='post'
        )

        return padded

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit tokenizer dan transform dalam satu langkah.

        Args:
            texts: List of training texts.

        Returns:
            np.ndarray: Padded sequences.
        """
        self.fit(texts)
        return self.transform(texts)

    def process(self, text: str) -> np.ndarray:
        """
        Process single text untuk inference.
        Cocok untuk dipanggil dari API endpoint.

        Args:
            text: Single input text.

        Returns:
            np.ndarray: Padded sequence (1, max_seq_len).
        """
        return self.transform([text])

    @property
    def vocab_size(self) -> int:
        """Return actual vocabulary size (termasuk OOV token)."""
        if not self.is_fitted:
            return 0
        return min(
            len(self.tokenizer.word_index) + 1,
            self.max_vocab_size
        )

    @property
    def word_index(self) -> dict:
        """Return word-to-index mapping."""
        if not self.is_fitted:
            return {}
        return self.tokenizer.word_index