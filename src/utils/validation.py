"""
Input Validation untuk SkillAlign AI.

Modul untuk memvalidasi input sebelum preprocessing dan inference.
"""

import logging
from typing import List, Optional

from .error_handling import DataValidationError

logger = logging.getLogger(__name__)

# Validation constants
MIN_CV_LENGTH = 50
MAX_CV_LENGTH = 10000
MIN_JOB_LENGTH = 30
MAX_JOB_LENGTH = 10000
MAX_BATCH_SIZE = 50


class InputValidator:
    """
    Validate input sebelum inference.

    Menyediakan static methods untuk validasi
    berbagai jenis input di pipeline SkillAlign.

    Example:
        >>> InputValidator.validate_cv_text("My CV text here...")
        True
        >>> InputValidator.validate_prediction_request(
        ...     cv_text="...",
        ...     job_description="..."
        ... )
        True
    """

    @staticmethod
    def validate_cv_text(text: str) -> bool:
        """
        Validasi teks CV.

        Args:
            text: CV text string.

        Returns:
            True jika valid.

        Raises:
            DataValidationError: Jika tidak valid.
        """
        if text is None:
            raise DataValidationError(
                message="CV text tidak boleh None.",
                field="cv_text"
            )

        if not isinstance(text, str):
            raise DataValidationError(
                message="CV text harus berupa string.",
                field="cv_text",
                value=type(text).__name__
            )

        stripped = text.strip()
        if len(stripped) < MIN_CV_LENGTH:
            raise DataValidationError(
                message=(
                    f"CV text terlalu pendek "
                    f"(minimum {MIN_CV_LENGTH} karakter, "
                    f"diberikan {len(stripped)})."
                ),
                field="cv_text",
                value=len(stripped)
            )

        if len(text) > MAX_CV_LENGTH:
            raise DataValidationError(
                message=(
                    f"CV text terlalu panjang "
                    f"(maximum {MAX_CV_LENGTH} karakter, "
                    f"diberikan {len(text)})."
                ),
                field="cv_text",
                value=len(text)
            )

        return True

    @staticmethod
    def validate_job_description(text: str) -> bool:
        """
        Validasi teks job description.

        Args:
            text: Job description string.

        Returns:
            True jika valid.

        Raises:
            DataValidationError: Jika tidak valid.
        """
        if text is None:
            raise DataValidationError(
                message="Job description tidak boleh None.",
                field="job_description"
            )

        if not isinstance(text, str):
            raise DataValidationError(
                message="Job description harus berupa string.",
                field="job_description",
                value=type(text).__name__
            )

        stripped = text.strip()
        if len(stripped) < MIN_JOB_LENGTH:
            raise DataValidationError(
                message=(
                    f"Job description terlalu pendek "
                    f"(minimum {MIN_JOB_LENGTH} karakter, "
                    f"diberikan {len(stripped)})."
                ),
                field="job_description",
                value=len(stripped)
            )

        if len(text) > MAX_JOB_LENGTH:
            raise DataValidationError(
                message=(
                    f"Job description terlalu panjang "
                    f"(maximum {MAX_JOB_LENGTH} karakter, "
                    f"diberikan {len(text)})."
                ),
                field="job_description",
                value=len(text)
            )

        return True

    @staticmethod
    def validate_prediction_request(
        cv_text: str,
        job_description: str
    ) -> bool:
        """
        Validasi lengkap untuk prediction request.

        Args:
            cv_text: CV text.
            job_description: Job description text.

        Returns:
            True jika semua valid.

        Raises:
            DataValidationError: Jika ada input yang tidak valid.
        """
        InputValidator.validate_cv_text(cv_text)
        InputValidator.validate_job_description(job_description)
        return True

    @staticmethod
    def validate_batch_request(
        cv_texts: List[str],
        job_descriptions: List[str]
    ) -> bool:
        """
        Validasi batch prediction request.

        Args:
            cv_texts: List of CV texts.
            job_descriptions: List of job descriptions.

        Returns:
            True jika valid.

        Raises:
            DataValidationError: Jika tidak valid.
        """
        if len(cv_texts) != len(job_descriptions):
            raise DataValidationError(
                message=(
                    f"Jumlah CV ({len(cv_texts)}) dan "
                    f"job descriptions ({len(job_descriptions)}) "
                    f"harus sama."
                )
            )

        if len(cv_texts) > MAX_BATCH_SIZE:
            raise DataValidationError(
                message=(
                    f"Batch size terlalu besar "
                    f"(maximum {MAX_BATCH_SIZE}, "
                    f"diberikan {len(cv_texts)})."
                )
            )

        if len(cv_texts) == 0:
            raise DataValidationError(
                message="Batch tidak boleh kosong."
            )

        # Validate setiap item
        for i, (cv, job) in enumerate(zip(cv_texts, job_descriptions)):
            try:
                InputValidator.validate_prediction_request(cv, job)
            except DataValidationError as e:
                raise DataValidationError(
                    message=f"Item #{i + 1}: {e.message}",
                    field=e.field,
                    value=e.value
                )

        return True

    @staticmethod
    def validate_score(score: float) -> bool:
        """
        Validasi matching score output.

        Args:
            score: Predicted matching score.

        Returns:
            True jika valid.

        Raises:
            DataValidationError: Jika score di luar range.
        """
        if not isinstance(score, (int, float)):
            raise DataValidationError(
                message="Score harus berupa numeric value.",
                field="matching_score",
                value=type(score).__name__
            )

        if score < 0.0 or score > 1.0:
            raise DataValidationError(
                message=(
                    f"Score harus antara 0 dan 1 "
                    f"(diberikan {score})."
                ),
                field="matching_score",
                value=score
            )

        return True