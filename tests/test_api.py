"""
Unit Tests untuk SkillAlign API Service.

Test coverage:
- FastAPI endpoints (/predict, /health, /)
- Request validation
- Error handling
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from main import app


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def valid_request():
    """Valid prediction request body."""
    return {
        "cv_text": (
            "Experienced Data Scientist with 3 years of experience "
            "in Python, Machine Learning, TensorFlow, scikit-learn, "
            "and data analysis. Strong background in NLP and computer vision."
        ),
        "job_description": (
            "Looking for a Data Scientist with strong Python skills, "
            "experience in ML frameworks, data visualization, and "
            "statistical analysis."
        ),
        "user_id": "test_user_001"
    }


# ==========================================
# Tests: Root & Health Endpoints
# ==========================================

class TestRootEndpoints:
    """Tests untuk root dan health endpoints."""

    def test_root_endpoint(self, client):
        """Test GET / returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "SkillAlign AI Service"
        assert data["version"] == "1.0.0"
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test GET /health returns status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_docs_endpoint(self, client):
        """Test bahwa docs endpoint accessible."""
        response = client.get("/docs")
        assert response.status_code == 200


# ==========================================
# Tests: Predict Endpoint
# ==========================================

class TestPredictEndpoint:
    """Tests untuk POST /predict endpoint."""

    def test_predict_without_model(self, client, valid_request):
        """Test predict ketika model belum loaded."""
        response = client.post("/predict", json=valid_request)
        # Should return 503 (model not loaded)
        assert response.status_code in [503, 500]

    def test_predict_missing_cv_text(self, client):
        """Test predict tanpa cv_text."""
        response = client.post("/predict", json={
            "job_description": (
                "Looking for a developer with strong "
                "programming skills and experience."
            )
        })
        assert response.status_code == 422  # Validation error

    def test_predict_missing_job_description(self, client):
        """Test predict tanpa job_description."""
        response = client.post("/predict", json={
            "cv_text": (
                "Experienced developer with 5 years of experience "
                "in Python and machine learning."
            )
        })
        assert response.status_code == 422

    def test_predict_short_cv_text(self, client):
        """Test predict dengan cv_text terlalu pendek."""
        response = client.post("/predict", json={
            "cv_text": "Too short",
            "job_description": (
                "Looking for a developer with "
                "strong programming skills."
            )
        })
        assert response.status_code == 422

    def test_predict_empty_body(self, client):
        """Test predict dengan body kosong."""
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_predict_invalid_json(self, client):
        """Test predict dengan invalid JSON."""
        response = client.post(
            "/predict",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


# ==========================================
# Tests: API v1 Endpoints
# ==========================================

class TestAPIv1Endpoints:
    """Tests untuk /api/v1 endpoints."""

    def test_api_v1_predict_without_model(self, client, valid_request):
        """Test /api/v1/predict ketika model belum loaded."""
        response = client.post(
            "/api/v1/predict", json=valid_request
        )
        assert response.status_code in [503, 500]

    def test_api_v1_batch_without_model(self, client):
        """Test /api/v1/predict/batch ketika model belum loaded."""
        response = client.post("/api/v1/predict/batch", json={
            "cv_text": (
                "Experienced developer with 5 years of experience "
                "in Python and machine learning engineering."
            ),
            "job_descriptions": [
                "Looking for Python developer with ML experience.",
                "Need Java developer for backend services."
            ]
        })
        assert response.status_code in [503, 500]


# ==========================================
# Tests: Input Validation (Standalone)
# ==========================================

class TestInputValidation:
    """Tests untuk InputValidator (standalone, tanpa API)."""

    def test_validate_cv_text_valid(self):
        """Test valid CV text."""
        from src.utils.validation import InputValidator
        assert InputValidator.validate_cv_text(
            "A" * 50 + " experienced developer with skills"
        )

    def test_validate_cv_text_too_short(self):
        """Test CV text terlalu pendek."""
        from src.utils.validation import InputValidator
        from src.utils.error_handling import DataValidationError
        with pytest.raises(DataValidationError):
            InputValidator.validate_cv_text("Short")

    def test_validate_cv_text_none(self):
        """Test CV text None."""
        from src.utils.validation import InputValidator
        from src.utils.error_handling import DataValidationError
        with pytest.raises(DataValidationError):
            InputValidator.validate_cv_text(None)

    def test_validate_job_description_valid(self):
        """Test valid job description."""
        from src.utils.validation import InputValidator
        assert InputValidator.validate_job_description(
            "Looking for a developer with strong skills"
        )

    def test_validate_prediction_request(self):
        """Test validasi full prediction request."""
        from src.utils.validation import InputValidator
        assert InputValidator.validate_prediction_request(
            cv_text="A" * 60 + " experienced developer",
            job_description="Looking for a developer with strong skills"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
