class CarbynException(Exception):
    """Base exception for Carbyn MVP."""
    pass

class ModelRoutingError(CarbynException):
    """Raised when there is an error routing to a specific model."""
    pass

class ImageProcessingError(CarbynException):
    """Raised when there is an error processing an image."""
    pass

class VectorDBConnectionError(CarbynException):
    """Raised when there is an error connecting to or querying the vector database."""
    pass