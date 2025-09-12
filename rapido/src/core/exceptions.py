"""
Custom exceptions for Rapido system
"""


class RapidoException(Exception):
    """Base exception for all Rapido errors"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}


class AudioProcessingError(RapidoException):
    """Raised when audio processing fails"""
    pass


class VideoProcessingError(RapidoException):
    """Raised when video processing fails"""
    pass


class SyncTalkConnectionError(RapidoException):
    """Raised when SyncTalk connection fails"""
    pass


class LiveKitConnectionError(RapidoException):
    """Raised when LiveKit connection fails"""
    pass


class DataParsingError(RapidoException):
    """Raised when data parsing fails"""
    pass


class TTSError(RapidoException):
    """Raised when TTS synthesis fails"""
    pass


class ConfigurationError(RapidoException):
    """Raised when system configuration is invalid"""
    pass


class SecurityError(RapidoException):
    """Raised when security validation fails"""
    pass
