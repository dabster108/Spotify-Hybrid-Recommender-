"""Utility helpers for the Hybrid Music Recommendation System."""
import os
from pathlib import Path

def load_env(dotenv_path: str = ".env"):
    """Lightweight .env loader (avoids external dependency)."""
    path = Path(dotenv_path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if not line or line.strip().startswith('#'):
            continue
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)
