"""
Unit Tests untuk SkillAlign Preprocessing Pipeline.

Test coverage:
- NLPPreprocessor
- FeatureEngineer
- EmbeddingManager
"""

import pytest
import numpy as np

from src.preprocessing.nlp_preprocessor import NLPPreprocessor
from src.preprocessing.feature_engineering import FeatureEngineer


# ==========================================
# Tests: NLPPreprocessor
# ==========================================

class TestNLPPreprocessor:
    """Tests untuk NLPPreprocessor."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance untuk testing."""
        return NLPPreprocessor(
            max_vocab_size=500,
            max_seq_len=50
        )

    @pytest.fixture
    def sample_texts(self):
        """Sample texts untuk testing."""
        return [
            "Experienced Python developer with machine learning skills",
            "Data Scientist looking for new opportunities in AI",
            "Full stack developer proficient in React and Node.js",
            "Project manager with 5 years of experience in tech industry",
            "Machine learning engineer specializing in NLP and deep learning"
        ]

    def test_preprocessor_creation(self, preprocessor):
        """Test bahwa preprocessor bisa dibuat."""
        assert preprocessor is not None
        assert preprocessor.max_vocab_size == 500
        assert preprocessor.max_seq_len == 50
        assert preprocessor.is_fitted is False

    def test_preprocess_text_basic(self, preprocessor):
        """Test basic text preprocessing."""
        text = "Hello World! This is a TEST 123."
        cleaned = preprocessor.preprocess_text(text)

        assert cleaned is not None
        assert isinstance(cleaned, str)
        # Should be lowercase
        assert cleaned == cleaned.lower()
        # Digits should be removed
        assert '123' not in cleaned

    def test_preprocess_text_url_removal(self, preprocessor):
        """Test URL removal."""
        text = "Visit https://example.com for more info about Python."
        cleaned = preprocessor.preprocess_text(text)
        assert 'http' not in cleaned
        assert 'example.com' not in cleaned

    def test_preprocess_text_email_removal(self, preprocessor):
        """Test email removal."""
        text = "Contact me at user@example.com for the job."
        cleaned = preprocessor.preprocess_text(text)
        assert '@' not in cleaned

    def test_preprocess_text_empty(self, preprocessor):
        """Test dengan empty/None input."""
        assert preprocessor.preprocess_text("") == ""
        assert preprocessor.preprocess_text(None) == ""

    def test_preprocess_batch(self, preprocessor, sample_texts):
        """Test batch preprocessing."""
        cleaned = preprocessor.preprocess_batch(sample_texts)
        assert len(cleaned) == len(sample_texts)
        assert all(isinstance(t, str) for t in cleaned)

    def test_fit(self, preprocessor, sample_texts):
        """Test tokenizer fitting."""
        preprocessor.fit(sample_texts)
        assert preprocessor.is_fitted is True
        assert preprocessor.vocab_size > 0

    def test_transform(self, preprocessor, sample_texts):
        """Test transform ke sequences."""
        preprocessor.fit(sample_texts)
        sequences = preprocessor.transform(sample_texts)

        assert isinstance(sequences, np.ndarray)
        assert sequences.shape == (
            len(sample_texts), preprocessor.max_seq_len
        )

    def test_transform_without_fit_raises(self, preprocessor):
        """Test bahwa transform tanpa fit raises error."""
        with pytest.raises(RuntimeError):
            preprocessor.transform(["Some text here"])

    def test_fit_transform(self, preprocessor, sample_texts):
        """Test fit_transform."""
        sequences = preprocessor.fit_transform(sample_texts)
        assert sequences.shape[0] == len(sample_texts)
        assert sequences.shape[1] == preprocessor.max_seq_len

    def test_process_single(self, preprocessor, sample_texts):
        """Test process single text."""
        preprocessor.fit(sample_texts)
        result = preprocessor.process("Python developer with ML skills")
        assert result.shape == (1, preprocessor.max_seq_len)

    def test_word_index_property(self, preprocessor, sample_texts):
        """Test word_index property."""
        assert preprocessor.word_index == {}
        preprocessor.fit(sample_texts)
        assert len(preprocessor.word_index) > 0


# ==========================================
# Tests: FeatureEngineer
# ==========================================

class TestFeatureEngineer:
    """Tests untuk FeatureEngineer."""

    @pytest.fixture
    def feature_eng(self):
        """Create FeatureEngineer instance."""
        return FeatureEngineer(max_tfidf_features=100)

    def test_creation(self, feature_eng):
        """Test creation."""
        assert feature_eng is not None
        assert feature_eng.is_fitted is False

    def test_extract_tfidf(self, feature_eng):
        """Test TF-IDF extraction."""
        cv_texts = ["python machine learning", "java web development"]
        job_texts = ["looking for python developer", "need java expert"]

        cv_tfidf, job_tfidf = feature_eng.extract_tfidf_features(
            cv_texts, job_texts
        )
        assert cv_tfidf.shape[0] == len(cv_texts)
        assert job_tfidf.shape[0] == len(job_texts)
        assert feature_eng.is_fitted is True

    def test_compute_skill_overlap_basic(self):
        """Test skill overlap computation."""
        cv_skills = {'Python', 'TensorFlow', 'SQL'}
        job_skills = {'Python', 'TensorFlow', 'Docker', 'AWS'}

        result = FeatureEngineer.compute_skill_overlap(
            cv_skills, job_skills
        )

        assert result['overlap_count'] == 2
        assert result['overlap_ratio'] == 0.5  # 2/4
        assert 'Docker' in result['missing_skills']
        assert 'AWS' in result['missing_skills']

    def test_compute_skill_overlap_empty(self):
        """Test skill overlap dengan set kosong."""
        result = FeatureEngineer.compute_skill_overlap(set(), set())
        assert result['overlap_count'] == 0
        assert result['jaccard_similarity'] == 0.0

    def test_compute_skill_overlap_perfect(self):
        """Test skill overlap sempurna."""
        skills = {'Python', 'SQL'}
        result = FeatureEngineer.compute_skill_overlap(skills, skills)
        assert result['overlap_ratio'] == 1.0
        assert result['jaccard_similarity'] == 1.0

    def test_tfidf_similarity(self):
        """Test TF-IDF cosine similarity."""
        sim = FeatureEngineer.compute_tfidf_similarity(
            "python machine learning developer",
            "python data science machine learning"
        )
        assert 0.0 <= sim <= 1.0
        assert sim > 0.0  # Should have some similarity

    def test_extract_skills(self):
        """Test skill extraction dari text."""
        text = "Experienced in Python, TensorFlow, and data analysis."
        dictionary = ['Python', 'TensorFlow', 'Java', 'Docker']

        found = FeatureEngineer.extract_skills_from_text(
            text, dictionary
        )
        assert 'Python' in found
        assert 'TensorFlow' in found
        assert 'Java' not in found


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
