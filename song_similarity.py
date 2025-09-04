"""Song similarity module for Spotify recommendation system."""

import requests
from typing import Dict, List, Optional, Tuple, Any

def find_specific_song(spotify_token: str, song_name: str, artist_name: str = None) -> Optional[Dict]:
    """Find a specific song in Spotify's database."""
    if not spotify_token:
        return None
    
    try:
        headers = {'Authorization': f'Bearer {spotify_token}'}
        
        # Try different query strategies to improve matching
        queries = []
        
        # First, try exact song name with artist if provided
        if artist_name:
            queries.append(f'track:"{song_name}" artist:"{artist_name}"')
            queries.append(f'{song_name} artist:{artist_name}')
        
        # Then try just the song name with quotes for exact matching
        queries.append(f'track:"{song_name}"')
        
        # Finally, try a general search with the song name
        queries.append(song_name)
        
        print(f"🔍 Searching for song: {song_name}{' by ' + artist_name if artist_name else ''}")
        
        for query in queries:
            search_params = {
                'q': query,
                'type': 'track',
                'limit': 10
            }
            
            try:
                response = requests.get('https://api.spotify.com/v1/search', 
                                    headers=headers, params=search_params, timeout=10)
                
                if response.status_code != 200:
                    print(f"Song search failed: {response.status_code}")
                    continue
                    
                data = response.json()
                tracks = data.get('tracks', {}).get('items', [])
                
                if not tracks:
                    print(f"No tracks found for query: '{query}'")
                    continue
                    
                # If artist is specified, try to find an exact match first
                if artist_name:
                    for track in tracks:
                        track_artists = [a.get('name', '').lower() for a in track.get('artists', [])]
                        if any(artist_name.lower() in artist.lower() for artist in track_artists):
                            print(f"Found exact match: {track['name']} by {track['artists'][0]['name']}")
                            return track
                
                # If no exact match or no artist specified, return first result
                print(f"Found song: {tracks[0]['name']} by {tracks[0]['artists'][0]['name']}")
                return tracks[0]
                
            except Exception as e:
                print(f"Error during search with query '{query}': {e}")
                continue
        
        # If all search strategies failed
        print(f"Could not find song: {song_name}")
        return None
            
    except Exception as e:
        print(f"Error searching for song: {e}")
        return None
        
def get_recommendations_by_song(spotify_token: str, seed_track_id: str, seed_features: Dict, 
                               exclude_artists: List[str] = None, limit: int = 50) -> List[Dict]:
    """Get recommendations based on a seed song's audio features."""
    if not spotify_token or not seed_track_id:
        return []
        
    # Now search for tracks with similar audio features
    headers = {'Authorization': f'Bearer {spotify_token}'}
    
    # Use Spotify's recommendations API
    recommendations_params = {
        'seed_tracks': seed_track_id,
        'target_danceability': seed_features.get('danceability', 0.5),
        'target_energy': seed_features.get('energy', 0.5),
        'target_valence': seed_features.get('valence', 0.5),
        'target_acousticness': seed_features.get('acousticness', 0.5),
        'limit': limit
    }
    
    try:
        response = requests.get('https://api.spotify.com/v1/recommendations', 
                               headers=headers, params=recommendations_params, timeout=10)
        
        if response.status_code != 200:
            print(f"Recommendations request failed: {response.status_code}")
            return []
            
        data = response.json()
        recommended_tracks = []
        exclude_artists = [artist.lower() for artist in (exclude_artists or [])]
        
        for track_data in data.get('tracks', []):
            track_artists = [artist.get('name', '').lower() for artist in track_data.get('artists', [])]
            
            # Skip songs by excluded artists
            if exclude_artists and any(artist in exclude_artists for artist in track_artists):
                continue
                
            recommended_tracks.append(track_data)
        
        print(f"Found {len(recommended_tracks)} similar recommendations")
        return recommended_tracks
        
    except Exception as e:
        print(f"Error getting song recommendations: {e}")
        return []
