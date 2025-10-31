"""Enhanced environment configuration manager for Spotify Recommendation System."""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import logging

class Config:
    """Configuration manager that loads environment variables and provides defaults."""
    
    def __init__(self, env_file: str = ".env"):
        """Initialize configuration by loading environment variables."""
        self.env_file = env_file
        self.load_env()
        self._validate_required_keys()
    
    def load_env(self):
        """Load environment variables from .env file."""
        env_path = Path(self.env_file)
        if env_path.exists():
            print(f"📁 Loading environment from {env_path}")
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                # Only set if not already in environment (allows override)
                if key not in os.environ:
                    os.environ[key] = value
        else:
            print(f"⚠️  Environment file {env_path} not found. Using system environment variables only.")
            print(f"💡 Copy .env.example to .env and fill in your API keys!")
    
    def _validate_required_keys(self):
        """Validate that required API keys are present."""
        required_keys = {
            'SPOTIFY_CLIENT_ID': 'Spotify API Client ID',
            'SPOTIFY_CLIENT_SECRET': 'Spotify API Client Secret',
            'GROQ_API_KEY': 'Groq API Key'
        }
        
        missing_keys = []
        for key, description in required_keys.items():
            if not self.get(key):
                missing_keys.append(f"  - {key}: {description}")
        
        if missing_keys:
            print("🚨 Missing required environment variables:")
            print("\n".join(missing_keys))
            print("\n💡 Please set these in your .env file or environment variables.")
            print("📖 See .env.example for reference.")
    
    def get(self, key: str, default: Any = None) -> Optional[str]:
        """Get environment variable with optional default."""
        return os.getenv(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable."""
        value = self.get(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer environment variable."""
        try:
            return int(self.get(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float environment variable."""
        try:
            return float(self.get(key, str(default)))
        except (ValueError, TypeError):
            return default
    
    # Spotify Configuration
    @property
    def spotify_client_id(self) -> str:
        return self.get('SPOTIFY_CLIENT_ID', '')
    
    @property
    def spotify_client_secret(self) -> str:
        return self.get('SPOTIFY_CLIENT_SECRET', '')
    
    # Groq Configuration
    @property
    def groq_api_key(self) -> str:
        return self.get('GROQ_API_KEY', '')
    
    @property
    def groq_api_url(self) -> str:
        return self.get('GROQ_API_URL', 'https://api.groq.com/openai/v1/chat/completions')
    
    # Mistral Configuration (optional)
    @property
    def mistral_api_key(self) -> str:
        return self.get('MISTRAL_API_KEY', '')
    
    # Application Settings
    @property
    def app_name(self) -> str:
        return self.get('APP_NAME', 'Spotify_Hybrid_Recommender')
    
    @property
    def debug(self) -> bool:
        return self.get_bool('DEBUG', False)
    
    @property
    def log_level(self) -> str:
        return self.get('LOG_LEVEL', 'INFO')
    
    # Cache Settings
    @property
    def cache_enabled(self) -> bool:
        return self.get_bool('CACHE_ENABLED', True)
    
    @property
    def cache_ttl(self) -> int:
        return self.get_int('CACHE_TTL', 3600)
    
    # Recommendation Settings
    @property
    def default_recommendation_count(self) -> int:
        return self.get_int('DEFAULT_RECOMMENDATION_COUNT', 20)
    
    @property
    def max_recommendation_count(self) -> int:
        return self.get_int('MAX_RECOMMENDATION_COUNT', 50)
    
    @property
    def similarity_threshold(self) -> float:
        return self.get_float('SIMILARITY_THRESHOLD', 0.7)
    
    # API Rate Limiting
    @property
    def spotify_api_rate_limit(self) -> int:
        return self.get_int('SPOTIFY_API_RATE_LIMIT', 100)
    
    @property
    def groq_api_rate_limit(self) -> int:
        return self.get_int('GROQ_API_RATE_LIMIT', 50)
    
    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate API key formats and return status."""
        validation = {}
        
        # Spotify validation
        spotify_id = self.spotify_client_id
        spotify_secret = self.spotify_client_secret
        validation['spotify'] = bool(spotify_id and spotify_secret and len(spotify_id) > 10)
        
        # Groq validation
        groq_key = self.groq_api_key
        validation['groq'] = bool(groq_key and groq_key.startswith('gsk_'))
        
        # Mistral validation (optional)
        mistral_key = self.mistral_api_key
        validation['mistral'] = bool(mistral_key and len(mistral_key) > 10)
        
        return validation
    
    def print_status(self):
        """Print configuration status."""
        print("🔧 Configuration Status:")
        print(f"  App Name: {self.app_name}")
        print(f"  Debug Mode: {self.debug}")
        print(f"  Cache Enabled: {self.cache_enabled}")
        
        validation = self.validate_api_keys()
        print("\n🔑 API Keys:")
        for service, valid in validation.items():
            status = "✅ Valid" if valid else "❌ Invalid/Missing"
            print(f"  {service.title()}: {status}")


# Global configuration instance
config = Config()


def setup_logging():
    """Setup logging configuration."""
    level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# Auto-setup logging when module is imported
setup_logging()