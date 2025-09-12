"""
Security utilities and validation for Rapido system
"""

import hashlib
import hmac
import jwt
import os
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta

from .exceptions import SecurityError


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_file_extensions: List[str] = None
    rate_limit_requests_per_minute: int = 60
    
    def __post_init__(self):
        if self.allowed_file_extensions is None:
            self.allowed_file_extensions = ['.json', '.mp3', '.wav', '.png', '.jpg', '.jpeg']


class InputValidator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_lesson_id(lesson_id: str) -> bool:
        """Validate lesson ID format"""
        if not lesson_id or not isinstance(lesson_id, str):
            return False
        
        # Allow alphanumeric, hyphens, underscores, max 64 chars
        pattern = r'^[a-zA-Z0-9_-]{1,64}$'
        return bool(re.match(pattern, lesson_id))
    
    @staticmethod
    def validate_file_path(file_path: str, allowed_directories: List[str] = None) -> bool:
        """Validate file path to prevent directory traversal"""
        if not file_path or not isinstance(file_path, str):
            return False
        
        # Check for directory traversal attempts
        if '..' in file_path or file_path.startswith('/'):
            return False
        
        # Normalize path
        normalized_path = os.path.normpath(file_path)
        if normalized_path != file_path:
            return False
        
        # Check allowed directories if specified
        if allowed_directories:
            return any(normalized_path.startswith(allowed_dir) for allowed_dir in allowed_directories)
        
        return True
    
    @staticmethod
    def validate_audio_data(data: bytes) -> bool:
        """Basic validation of audio data"""
        if not data or len(data) < 44:  # Minimum WAV header size
            return False
        
        # Check for common audio file headers
        audio_headers = [
            b'RIFF',  # WAV
            b'ID3',   # MP3
            b'fLaC',  # FLAC
            b'OggS'   # OGG
        ]
        
        return any(data.startswith(header) for header in audio_headers)
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage"""
        if not filename:
            return "untitled"
        
        # Remove/replace dangerous characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext
        
        return filename
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate JSON data structure"""
        if not isinstance(data, dict):
            return False
        
        return all(field in data for field in required_fields)


class AuthenticationManager:
    """JWT-based authentication manager"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def generate_token(self, lesson_id: str, user_id: str = None, **claims) -> str:
        """Generate a JWT token"""
        now = datetime.utcnow()
        payload = {
            'lesson_id': lesson_id,
            'user_id': user_id,
            'iat': now,
            'exp': now + timedelta(hours=self.config.jwt_expiry_hours),
            'iss': 'rapido-system',
            **claims
        }
        
        return jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.config.jwt_secret, 
                algorithms=[self.config.jwt_algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise SecurityError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid token: {str(e)}")
    
    def extract_lesson_id(self, token: str) -> str:
        """Extract lesson ID from token"""
        payload = self.verify_token(token)
        lesson_id = payload.get('lesson_id')
        
        if not lesson_id or not InputValidator.validate_lesson_id(lesson_id):
            raise SecurityError("Invalid lesson ID in token")
        
        return lesson_id


class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client"""
        now = datetime.utcnow().timestamp()
        minute_ago = now - 60
        
        # Clean old requests
        if client_id in self.requests:
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] 
                if req_time > minute_ago
            ]
        else:
            self.requests[client_id] = []
        
        # Check rate limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Record this request
        self.requests[client_id].append(now)
        return True


class DataEncryption:
    """Data encryption utilities"""
    
    @staticmethod
    def hash_data(data: str, salt: str = None) -> str:
        """Hash data with optional salt"""
        if salt is None:
            salt = os.urandom(32).hex()
        
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000)
        return salt + hash_obj.hex()
    
    @staticmethod
    def verify_hash(data: str, hashed_data: str) -> bool:
        """Verify hashed data"""
        if len(hashed_data) < 64:  # Salt length
            return False
        
        salt = hashed_data[:64]
        expected_hash = hashed_data[64:]
        
        actual_hash = hashlib.pbkdf2_hmac('sha256', data.encode(), salt.encode(), 100000).hex()
        return hmac.compare_digest(expected_hash, actual_hash)
    
    @staticmethod
    def generate_secure_token() -> str:
        """Generate a secure random token"""
        return os.urandom(32).hex()


class SecurityManager:
    """Centralized security manager"""
    
    def __init__(self, config: SecurityConfig = None):
        if config is None:
            jwt_secret = os.getenv('JWT_SECRET')
            if not jwt_secret:
                raise SecurityError("JWT_SECRET environment variable is required")
            config = SecurityConfig(jwt_secret=jwt_secret)
        
        self.config = config
        self.auth_manager = AuthenticationManager(config)
        self.rate_limiter = RateLimiter(config.rate_limit_requests_per_minute)
        self.validator = InputValidator()
    
    def validate_request(self, lesson_id: str, file_path: str = None, 
                        audio_data: bytes = None, client_ip: str = None) -> bool:
        """Comprehensive request validation"""
        # Rate limiting
        if client_ip and not self.rate_limiter.is_allowed(client_ip):
            raise SecurityError("Rate limit exceeded")
        
        # Input validation
        if not self.validator.validate_lesson_id(lesson_id):
            raise SecurityError("Invalid lesson ID format")
        
        if file_path and not self.validator.validate_file_path(file_path):
            raise SecurityError("Invalid file path")
        
        if audio_data and not self.validator.validate_audio_data(audio_data):
            raise SecurityError("Invalid audio data")
        
        return True
    
    def secure_filename(self, filename: str) -> str:
        """Generate secure filename"""
        return self.validator.sanitize_filename(filename)
    
    def generate_session_token(self, lesson_id: str, user_id: str = None) -> str:
        """Generate session token"""
        return self.auth_manager.generate_token(lesson_id, user_id)
    
    def validate_session_token(self, token: str) -> Dict[str, Any]:
        """Validate session token"""
        return self.auth_manager.verify_token(token)
