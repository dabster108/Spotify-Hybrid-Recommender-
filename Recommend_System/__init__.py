"""Recommend_System package initializer.

This module exposes common submodules so scripts can import using
`from Recommend_System import query_analyzer`.
"""
# Import configuration first to ensure environment is loaded
from . import config
from . import query_analyzer
from . import recommend
from . import cache
from . import performance
from . import progress
from . import model
from . import similarity_matching
from . import song_similarity
from . import utils

# Make config easily accessible
from .config import config as settings

__all__ = [
    'config', 'settings', 'query_analyzer', 'recommend', 'cache', 'performance', 'progress',
    'model', 'similarity_matching', 'song_similarity', 'utils'
]
