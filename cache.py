import os
import json
import time
from typing import Dict, Any, List, Optional
import hashlib

class RecommendationCache:
    """
    Cache system for recommendation API calls to reduce latency and API usage
    """
    def __init__(self, cache_dir="./cache", ttl=86400):  # Default TTL: 1 day
        self.cache_dir = cache_dir
        self.ttl = ttl
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize different cache stores
        self.track_cache_file = os.path.join(cache_dir, "track_cache.json")
        self.audio_features_cache_file = os.path.join(cache_dir, "audio_features_cache.json")
        self.query_cache_file = os.path.join(cache_dir, "query_cache.json")
        self.artist_cache_file = os.path.join(cache_dir, "artist_cache.json")
        
        # Load existing cache data
        self.track_cache = self._load_cache(self.track_cache_file)
        self.audio_features_cache = self._load_cache(self.audio_features_cache_file)
        self.query_cache = self._load_cache(self.query_cache_file)
        self.artist_cache = self._load_cache(self.artist_cache_file)
    
    def _load_cache(self, file_path: str) -> Dict[str, Any]:
        """Load cache from file if it exists"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cache {file_path}: {e}")
        return {}
    
    def _save_cache(self, data: Dict[str, Any], file_path: str) -> None:
        """Save cache to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving cache {file_path}: {e}")
    
    def _generate_key(self, data: Any) -> str:
        """Generate a hash key from the data"""
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.md5(serialized.encode('utf-8')).hexdigest()
    
    def get_track(self, track_id: str) -> Optional[Dict]:
        """Get track from cache"""
        entry = self.track_cache.get(track_id)
        if entry and time.time() - entry['timestamp'] < self.ttl:
            return entry['data']
        return None
    
    def save_track(self, track_id: str, track_data: Dict) -> None:
        """Save track to cache"""
        self.track_cache[track_id] = {
            'data': track_data,
            'timestamp': time.time()
        }
        self._save_cache(self.track_cache, self.track_cache_file)
    
    def save_tracks(self, tracks: List[Dict]) -> None:
        """Save multiple tracks to cache"""
        for track in tracks:
            if 'id' in track:
                self.save_track(track['id'], track)
    
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """Get audio features from cache"""
        entry = self.audio_features_cache.get(track_id)
        if entry and time.time() - entry['timestamp'] < self.ttl:
            return entry['data']
        return None
    
    def save_audio_features(self, track_id: str, features: Dict) -> None:
        """Save audio features to cache"""
        self.audio_features_cache[track_id] = {
            'data': features,
            'timestamp': time.time()
        }
        self._save_cache(self.audio_features_cache, self.audio_features_cache_file)
    
    def save_multiple_audio_features(self, features_dict: Dict[str, Dict]) -> None:
        """Save multiple audio features to cache"""
        for track_id, features in features_dict.items():
            self.save_audio_features(track_id, features)
    
    def get_query_results(self, query: str, params: Dict) -> Optional[List]:
        """Get search results for a query from cache"""
        key = self._generate_key({'query': query, 'params': params})
        entry = self.query_cache.get(key)
        if entry and time.time() - entry['timestamp'] < self.ttl:
            return entry['data']
        return None
    
    def save_query_results(self, query: str, params: Dict, results: List) -> None:
        """Save search results for a query to cache"""
        key = self._generate_key({'query': query, 'params': params})
        self.query_cache[key] = {
            'data': results,
            'timestamp': time.time()
        }
        self._save_cache(self.query_cache, self.query_cache_file)
    
    def get_artist_tracks(self, artist_id: str) -> Optional[List]:
        """Get artist tracks from cache"""
        entry = self.artist_cache.get(artist_id)
        if entry and time.time() - entry['timestamp'] < self.ttl:
            return entry['data']
        return None
    
    def save_artist_tracks(self, artist_id: str, tracks: List) -> None:
        """Save artist tracks to cache"""
        self.artist_cache[artist_id] = {
            'data': tracks,
            'timestamp': time.time()
        }
        self._save_cache(self.artist_cache, self.artist_cache_file)
    
    def clear_expired_entries(self) -> None:
        """Clear expired cache entries"""
        current_time = time.time()
        
        # Clear expired track entries
        for key in list(self.track_cache.keys()):
            if current_time - self.track_cache[key]['timestamp'] > self.ttl:
                del self.track_cache[key]
        
        # Clear expired audio features entries
        for key in list(self.audio_features_cache.keys()):
            if current_time - self.audio_features_cache[key]['timestamp'] > self.ttl:
                del self.audio_features_cache[key]
        
        # Clear expired query entries
        for key in list(self.query_cache.keys()):
            if current_time - self.query_cache[key]['timestamp'] > self.ttl:
                del self.query_cache[key]
        
        # Clear expired artist entries
        for key in list(self.artist_cache.keys()):
            if current_time - self.artist_cache[key]['timestamp'] > self.ttl:
                del self.artist_cache[key]
        
        # Save updated caches
        self._save_cache(self.track_cache, self.track_cache_file)
        self._save_cache(self.audio_features_cache, self.audio_features_cache_file)
        self._save_cache(self.query_cache, self.query_cache_file)
        self._save_cache(self.artist_cache, self.artist_cache_file)
