class CustomError(Exception):
    """Base class for all custom exceptions."""

    pass


class InvalidPathError(CustomError):
    """Raised when the provided path is neither a str nor a pathlib.Path."""

    def __init__(self, path_name: str, actual_type: type):
        message = (
            f"Invalid path type for '{path_name}': "
            f"Expected str or pathlib.Path, but got {actual_type.__name__}."
        )
        super().__init__(message)
