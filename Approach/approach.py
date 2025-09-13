import requests
import base64
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

class SpotifyRecommendationSystem:
    def __init__(self):
        """Initialize the Spotify Recommendation System"""
        self.client_id = "34a3e2ba93db4de89851dfacda6bad8e"
        self.client_secret = "8d0ca8f1faec4437a831fed80ad2dcae"
        self.access_token = None
        self.token_expires = 0
        self.base_url = "https://api.spotify.com/v1"
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Spotify API using Client Credentials Flow"""
        token_url = "https://accounts.spotify.com/api/token"
        headers = {
            "Authorization": f"Basic {base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode()).decode()}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}

        try:
            response = requests.post(token_url, headers=headers, data=data, timeout=10)
            response.raise_for_status()
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.token_expires = time.time() + token_data["expires_in"]
            print("✓ Successfully authenticated with Spotify API")
        except requests.exceptions.RequestException as e:
            print(f"Authentication failed: {e}")
            raise

    def _refresh_token(self) -> None:
        """Refresh the access token if expired"""
        if time.time() < self.token_expires - 60:
            return
        self._authenticate()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to Spotify API"""
        self._refresh_token()
        headers = {"Authorization": f"Bearer {self.access_token}"}
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=headers, params=params or {}, timeout=10)
            response.raise_for_status()
            return response.json() if response.content else {}
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            raise

    def get_artist_id(self, artist_name: str) -> Optional[str]:
        """Search for an artist by name and return their ID"""
        params = {"q": f"artist:{artist_name}", "type": "artist", "limit": 1}
        try:
            results = self._make_request("search", params)
            artists = results.get("artists", {}).get("items", [])
            if artists:
                print(f"✓ Found artist: {artists[0]['name']}")
                return artists[0]["id"]
            else:
                print(f"✗ Artist not found: {artist_name}")
                return None
        except Exception as e:
            print(f"Artist search failed: {e}")
            return None

    def get_artist_genres(self, artist_id: str) -> List[str]:
        """Fetch available genres for an artist"""
        try:
            artist_data = self._make_request(f"artists/{artist_id}")
            genres = artist_data.get("genres", [])
            return genres
        except Exception as e:
            print(f"Failed to fetch artist genres: {e}")
            return []

    def get_artist_top_tracks(self, artist_id: str, limit: int = 20) -> List[Dict]:
        """Fetch top tracks for an artist"""
        params = {"market": "US"}
        try:
            response = self._make_request(f"artists/{artist_id}/top-tracks", params)
            tracks = response.get("tracks", [])
            print(f"✓ Found {len(tracks)} top tracks")
            return tracks[:limit]
        except Exception as e:
            print(f"Failed to get top tracks: {e}")
            return []

    def assign_synthetic_features(self, tracks: List[Dict], genre: str) -> List[Tuple[Dict, List[float]]]:
        """Assign synthetic audio features based on genre"""
        # Predefined feature vectors for known genres
        genre_vectors = {
            "sad": [0.3, 0.3, 0.2, 90.0, 0.7],
            "pop": [0.7, 0.8, 0.6, 120.0, 0.2],
            "romantic": [0.5, 0.4, 0.5, 100.0, 0.6]
        }

        # If genre is known, use its vector; otherwise generate a generic one
        if genre.lower() in genre_vectors:
            base_vector = genre_vectors[genre.lower()]
        else:
            # Generic vector for other Spotify genres
            base_vector = [0.6, 0.7, 0.5, 110.0, 0.3]

        feature_keys = ["danceability", "energy", "valence", "tempo", "acousticness"]
        track_features = []
        for track in tracks:
            if not track or not track.get("id"):
                continue
            variation = np.random.uniform(-0.1, 0.1, len(feature_keys))
            features = [
                min(max(base_vector[i] + variation[i], 0.0), 1.0) if key != "tempo" else
                min(max(base_vector[i] + variation[i]*10, 60.0), 200.0)
                for i, key in enumerate(feature_keys)
            ]
            track_features.append((track, features))
        return track_features

    def recommend_by_genre(self, artist_name: str, genre: str, num_recommendations: int = 8) -> List[Tuple[Dict, float]]:
        """Recommend tracks by artist and genre using cosine similarity"""
        artist_id = self.get_artist_id(artist_name)
        if not artist_id:
            return []

        top_tracks = self.get_artist_top_tracks(artist_id, 20)
        if not top_tracks:
            return []

        # Assign features
        track_features = self.assign_synthetic_features(top_tracks, genre)
        if not track_features:
            return []

        # Reference vector
        predefined_vectors = {
            "sad": [0.3, 0.3, 0.2, 90.0, 0.7],
            "pop": [0.7, 0.8, 0.6, 120.0, 0.2],
            "romantic": [0.5, 0.4, 0.5, 100.0, 0.6]
        }
        reference_vector = np.array([predefined_vectors.get(genre.lower(), [0.6, 0.7, 0.5, 110.0, 0.3])])

        # Normalize tempo for cosine similarity
        recommendations = []
        for track, features in track_features:
            features_norm = features.copy()
            features_norm[3] = (features_norm[3] - 60.0) / (200.0 - 60.0)
            track_vector = np.array([features_norm])
            similarity = cosine_similarity(reference_vector, track_vector)[0][0]
            recommendations.append((track, similarity))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        print(f"✓ Generated {len(recommendations[:num_recommendations])} {genre} recommendations")
        return recommendations[:num_recommendations]

    def print_recommendations(self, recommendations: List[Tuple[Dict, float]], genre: str, artist_name: str):
        """Print the recommendations nicely"""
        print(f"\n🎵 {genre.capitalize()} Songs by {artist_name}")
        print("=" * (len(genre) + len(artist_name) + 12))
        if not recommendations:
            print("No recommendations found.")
            return
        for i, (track, score) in enumerate(recommendations, 1):
            artist_names = ", ".join([artist["name"] for artist in track.get("artists", [])])
            print(f"{i:2d}. {track.get('name', 'Unknown')} - {artist_names} (Similarity: {score:.3f})")


def main():
    recommender = SpotifyRecommendationSystem()

    artist_name = input("Enter artist name: ").strip()
    artist_id = recommender.get_artist_id(artist_name)
    if not artist_id:
        return

    genres = recommender.get_artist_genres(artist_id)
    if genres:
        print(f"🎵 Available Spotify genres: {genres}")
        print("\nPick a genre for recommendations (or type one of 'sad', 'pop', 'romantic'):")
        for idx, g in enumerate(genres, 1):
            print(f"{idx}. {g}")
        genre_input = input("Enter genre: ").strip()
        # If input is number, map to Spotify genre
        if genre_input.isdigit():
            idx = int(genre_input) - 1
            if 0 <= idx < len(genres):
                genre_input = genres[idx]
    else:
        print("No genres available for this artist on Spotify.")
        genre_input = input("Pick a genre (or 'sad', 'pop', 'romantic'): ").strip()

    recommendations = recommender.recommend_by_genre(artist_name, genre_input, 8)
    recommender.print_recommendations(recommendations, genre_input, artist_name)


if __name__ == "__main__":
    main()

##main branch
