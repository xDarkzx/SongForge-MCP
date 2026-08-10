from enum import IntEnum


class ErrorCode(IntEnum):
    # Process errors (1000s)
    SUBPROCESS_FAILED = 1000
    SUBPROCESS_TIMEOUT = 1001
    ACESTEP_NOT_CONFIGURED = 1002
    SEPARATOR_NOT_CONFIGURED = 1003
    OMNIZART_NOT_CONFIGURED = 1004

    # Synthesis errors (2000s)
    SYNTHESIS_FAILED = 2000
    SEPARATION_FAILED = 2003
    OMNIZART_TRANSCRIPTION_FAILED = 2004

    # Validation errors (3000s)
    VALIDATION_FAILED = 3000
    INVALID_PARAMETER = 3001
    MISSING_PARAMETER = 3002
    VALUE_OUT_OF_RANGE = 3007
    FILE_NOT_FOUND = 3008


class SongForgeMCPError(Exception):
    def __init__(self, code: ErrorCode, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code.name} ({code.value})] {message}")
