"""
Error Handling untuk SkillAlign AI.

Custom exception classes untuk penanganan error
di seluruh pipeline AI service.
"""


class AIServiceError(Exception):
    """
    Base exception untuk AI Service.

    Semua custom exception di SkillAlign mewarisi class ini
    untuk memudahkan error handling di API layer.
    """

    def __init__(self, message: str = "AI Service error occurred"):
        self.message = message
        super().__init__(self.message)


class ModelNotLoadedError(AIServiceError):
    """Raised ketika model belum di-load tapi inference dipanggil."""

    def __init__(
        self,
        message: str = "Model belum di-load. Panggil load() dahulu."
    ):
        super().__init__(message)


class PreprocessingError(AIServiceError):
    """Raised ketika terjadi error saat text preprocessing."""

    def __init__(
        self,
        message: str = "Error saat preprocessing teks.",
        original_error: Exception = None
    ):
        self.original_error = original_error
        if original_error:
            message = f"{message} Detail: {str(original_error)}"
        super().__init__(message)


class InferenceError(AIServiceError):
    """Raised ketika terjadi error saat model inference."""

    def __init__(
        self,
        message: str = "Error saat model inference.",
        original_error: Exception = None
    ):
        self.original_error = original_error
        if original_error:
            message = f"{message} Detail: {str(original_error)}"
        super().__init__(message)


class DataValidationError(AIServiceError):
    """Raised ketika input data tidak valid."""

    def __init__(
        self,
        message: str = "Input data tidak valid.",
        field: str = None,
        value=None
    ):
        self.field = field
        self.value = value
        if field:
            message = f"{message} Field: {field}"
        super().__init__(message)


class ModelExportError(AIServiceError):
    """Raised ketika terjadi error saat export model."""

    def __init__(
        self,
        message: str = "Error saat export model.",
        original_error: Exception = None
    ):
        self.original_error = original_error
        if original_error:
            message = f"{message} Detail: {str(original_error)}"
        super().__init__(message)


class EmbeddingError(AIServiceError):
    """Raised ketika terjadi error terkait embedding."""

    def __init__(
        self,
        message: str = "Error terkait embedding.",
        original_error: Exception = None
    ):
        self.original_error = original_error
        if original_error:
            message = f"{message} Detail: {str(original_error)}"
        super().__init__(message)