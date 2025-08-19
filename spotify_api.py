import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
import random
import re


def get_spotify_client():
    """Initialize Spotify client with authentication and retry logic"""
    try:
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            raise ValueError("Spotify credentials are not properly configured")
            
        # Initialize with required scopes and retry strategy
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
        )
        
        sp = spotipy.Spotify(
            auth_manager=auth_manager,
            requests_timeout=15,  # Increased timeout
            retries=5,           # Increased retries
            backoff_factor=0.5   # Add exponential backoff
        )
        
        # Test the connection with error handling
        try:
            test_response = sp.search(q="test", limit=1)
            if not test_response or 'tracks' not in test_response:
                raise ConnectionError("Invalid API response format")
            return sp
        except spotipy.exceptions.SpotifyException as se:
            if se.http_status == 429:  # Rate limit error
                print("Rate limit reached. Retrying with backoff...")
                import time
                time.sleep(int(se.headers.get('Retry-After', 3)))
                # Retry the test query
                sp.search(q="test", limit=1)
                return sp
            raise
            
    except ValueError as ve:
        print(f"Configuration error: {ve}")
    except spotipy.exceptions.SpotifyException as se:
        print(f"Spotify API error (Status {se.http_status}): {se.msg}")
    except ConnectionError as ce:
        print(f"Connection error: {ce}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")
    
    print("Please check your Spotify API credentials and network connection.")
    return None


def get_valid_genres():
    """Get list of valid genres from Spotify"""
    sp = get_spotify_client()
    if sp:
        try:
            return sp.recommendation_genre_seeds()['genres']
        except Exception as e:
            print(f"Error fetching genres: {e}")
            print("Spotify API failed, falling back to default genres")
    else:
        print("Spotify API failed, falling back to default genres")
    
    # Fallback to common genres if API fails
    return [
        'acoustic', 'afrobeat', 'alt-rock', 'alternative', 'ambient', 'blues', 
        'classical', 'country', 'dance', 'electronic', 'folk', 'hip-hop', 
        'indie', 'jazz', 'latin', 'metal', 'pop', 'punk', 'r-n-b', 'rock'
    ]


def search_track(query, limit=5):
    """Search for tracks based on user input - returns single track object or None"""
    sp = get_spotify_client()
    if not sp:
        print("❌ Failed to initialize Spotify client")
        return None
    
    print(f"🔍 Searching Spotify for: {query}")
    
    try:
        # Enhanced patterns for better detection including cultural/language requests
        cultural_patterns = [
            r'(?:hindi|bollywood|indian)\s+(?:classical|songs?|music)',
            r'(?:classical|traditional)\s+(?:hindi|bollywood|indian)',
            r'(?:punjabi|tamil|telugu|bengali|marathi)\s+(?:songs?|music)',
            r'(?:arabic|persian|urdu|sanskrit)\s+(?:songs?|music)',
            r'(?:chinese|japanese|korean|thai)\s+(?:songs?|music)',
        ]
        
        # Check for cultural/language-specific requests
        cultural_match = None
        for pattern in cultural_patterns:
            if re.search(pattern, query.lower()):
                cultural_match = pattern
                break
        
        # If cultural request, modify search strategy
        if cultural_match:
            # Extract key terms for better search
            terms = []
            if 'hindi' in query.lower() or 'bollywood' in query.lower():
                terms = ['bollywood', 'hindi', 'indian classical']
            elif 'punjabi' in query.lower():
                terms = ['punjabi']
            elif 'tamil' in query.lower():
                terms = ['tamil']
            elif 'classical' in query.lower() and any(x in query.lower() for x in ['hindi', 'indian']):
                terms = ['hindustani', 'raga', 'classical indian']
            
            if terms:
                for term in terms:
                    print(f"🎵 Searching for {term} music")
                    results = sp.search(q=term, type='track', limit=10, market='IN')  # Use Indian market
                    if results and results['tracks']['items']:
                        # Filter for tracks that seem relevant
                        for track in results['tracks']['items']:
                            # Check if any artist or track name contains relevant terms
                            track_text = f"{track['name']} {' '.join([artist['name'] for artist in track['artists']])}"
                            if any(keyword in track_text.lower() for keyword in terms):
                                return track
                        # If no perfect match, return first result
                        return results['tracks']['items'][0]
        
        # Original search patterns for artist/song detection
        specific_patterns = [
            r'(.+?)\s+by\s+(.+?)(?:\s|$)',         # "[song] by [artist]"  
            r'songs?\s+by\s+(.+?)(?:\s|$)',        # "songs by [artist]"
            r'something\s+by\s+(.+?)(?:\s|$)',     # "something by [artist]"
            r'music\s+by\s+(.+?)(?:\s|$)',         # "music by [artist]"
            r'(?:play|want|give me|i like)\s+(.+?)\s+by\s+(.+?)(?:\s|$)', # "play [song] by [artist]"
        ]
        
        # Initialize variables
        results = None
        track = None
        
        # Try specific patterns first
        for pattern in specific_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:  # song and artist
                    song_name = groups[0].strip()
                    artist_name = groups[1].strip()
                    # Skip generic words
                    if song_name.lower() not in ['something', 'songs', 'song', 'music']:
                        print(f"🎵 Looking for '{song_name}' by {artist_name}")
                        results = sp.search(q=f"track:{song_name} artist:{artist_name}", type='track', limit=1)
                    else:  # Just search by artist
                        print(f"🎵 Looking for tracks by {artist_name}")
                        results = sp.search(q=f"artist:{artist_name}", type='track', limit=5)
                elif len(groups) == 1:  # just artist
                    artist_name = groups[0].strip()
                    print(f"🎵 Looking for tracks by {artist_name}")
                    results = sp.search(q=f"artist:{artist_name}", type='track', limit=5)
                
                if results and results['tracks']['items']:
                    # Filter for tracks by the exact artist if possible
                    for candidate_track in results['tracks']['items']:
                        if any(artist['name'].lower() == artist_name.lower() 
                               for artist in candidate_track['artists']):
                            track = candidate_track
                            break
                    # If no exact match, take the first result
                    if not track:
                        track = results['tracks']['items'][0]
                    break
        
        # If no specific matches, do a general search with better query processing
        if not track:
            print("🔍 Performing general search")
            
            # Improve query for better results by extracting key music terms
            processed_query = query.lower()
            
            # Extract genre and mood keywords
            genre_keywords = []
            mood_keywords = []
            
            # Genre detection
            if any(word in processed_query for word in ['pop', 'popular']):
                genre_keywords.append('pop')
            if any(word in processed_query for word in ['rock', 'alternative']):
                genre_keywords.append('rock')
            if any(word in processed_query for word in ['electronic', 'edm', 'dance']):
                genre_keywords.append('electronic')
            if any(word in processed_query for word in ['hip-hop', 'hiphop', 'rap']):
                genre_keywords.append('hip-hop')
            if any(word in processed_query for word in ['jazz', 'blues']):
                genre_keywords.append('jazz')
            if any(word in processed_query for word in ['classical', 'orchestra']):
                genre_keywords.append('classical')
            
            # Mood detection
            if any(word in processed_query for word in ['chill', 'relaxing', 'calm', 'peaceful']):
                mood_keywords.extend(['chill', 'ambient', 'relaxing'])
            if any(word in processed_query for word in ['upbeat', 'energetic', 'party', 'dance']):
                mood_keywords.extend(['upbeat', 'dance', 'energetic'])
            if any(word in processed_query for word in ['sad', 'melancholy', 'emotional']):
                mood_keywords.extend(['sad', 'emotional'])
            
            # Build better search query
            search_terms = []
            if genre_keywords:
                search_terms.extend(genre_keywords[:2])  # Max 2 genres
            if mood_keywords:
                search_terms.extend(mood_keywords[:2])  # Max 2 moods
            
            # Try the improved query first
            if search_terms:
                improved_query = ' '.join(search_terms)
                print(f"🎵 Trying improved search: {improved_query}")
                results = sp.search(q=improved_query, type='track', limit=limit*2)
                if results and results['tracks']['items']:
                    tracks = results['tracks']['items']
                    # Filter for popular tracks
                    popular_tracks = [t for t in tracks if t.get('popularity', 0) > 50]
                    if popular_tracks:
                        tracks = popular_tracks
                    tracks.sort(key=lambda x: x.get('popularity', 0), reverse=True)
                    track = tracks[0]
                    print(f"✅ Found {len(tracks)} tracks with improved search")
            
            # Fallback to original query if improved search didn't work
            if not track:
                results = sp.search(q=query, type='track', limit=limit)
                if results and results['tracks']['items']:
                    tracks = results['tracks']['items']
                    print(f"✅ Found {len(tracks)} tracks with original search")
                    # Return the most popular track
                    if tracks:
                        tracks.sort(key=lambda x: x.get('popularity', 0), reverse=True)
                        track = tracks[0]
        
        # Verify the track has required fields and return Spotify track object
        if track and 'name' in track and 'artists' in track and track['artists']:
            return track
        
        print("❌ No matching tracks found")
        return None
            
    except Exception as e:
        print(f"Error searching tracks: {e}")
        return None

def get_recommendations(seed_track=None, seed_tracks=None, seed_genre=None, limit=5):
    """Get track recommendations - uses search-based approach due to API limitations"""
    sp = get_spotify_client()
    if not sp:
        return []
    
    print("ℹ️ Note: Using search-based recommendations (Spotify Recommendations API not available)")
    
    # Always use our enhanced search-based recommendation system
    if seed_track:
        return get_recommendations_by_seed(seed_track=seed_track, limit=limit)
    elif seed_tracks:
        # Use the first track as the main seed
        if seed_tracks:
            main_track = seed_tracks[0] if isinstance(seed_tracks[0], dict) else {"id": seed_tracks[0]}
            return get_recommendations_by_seed(seed_track=main_track, limit=limit)
    elif seed_genre:
        return get_recommendations_by_seed(seed_genre=seed_genre, limit=limit)
    
    # Fallback to popular tracks
    return get_top_tracks(limit)

def get_recommendations_by_seed(seed_track=None, seed_genre=None, limit=5):
    """Get song recommendations from Spotify API using advanced search"""
    sp = get_spotify_client()
    if not sp:
        return []
    
    try:
        if seed_track:
            print(f"🔄 Using search-based recommendations for: {seed_track['name']} by {seed_track['artist']}")
            artist_name = seed_track['artist']
            track_name = seed_track['name']
            
            songs = []
            seen_urls = {seed_track['spotify_url']}  # Skip the seed track
            
            # Strategy 1: Get more songs by the same artist
            try:
                artist_results = sp.search(q=f"artist:{artist_name}", type='track', limit=15)
                for track in artist_results['tracks']['items']:
                    if track['external_urls']['spotify'] not in seen_urls:
                        seen_urls.add(track['external_urls']['spotify'])
                        songs.append({
                            "name": track['name'],
                            "artist": track['artists'][0]['name'],
                            "spotify_url": track['external_urls']['spotify'],
                            "preview_url": track.get('preview_url'),
                            "popularity": track.get('popularity', 0),
                            "source": "same_artist"
                        })
            except Exception as e:
                print(f"Error in artist search: {e}")
            
            # Strategy 2: Search for similar genre/style if we have enough info
            try:
                # Extract potential genre keywords from track name
                genre_keywords = []
                track_lower = track_name.lower()
                if any(word in track_lower for word in ['chill', 'relax', 'calm']):
                    genre_keywords.extend(['chill', 'ambient', 'downtempo'])
                elif any(word in track_lower for word in ['dance', 'party', 'beat']):
                    genre_keywords.extend(['dance', 'electronic', 'pop'])
                elif any(word in track_lower for word in ['rock', 'metal']):
                    genre_keywords.extend(['rock', 'alternative'])
                elif any(word in track_lower for word in ['hip', 'rap']):
                    genre_keywords.extend(['hip-hop', 'rap'])
                
                for keyword in genre_keywords[:2]:  # Limit to avoid too many requests
                    genre_results = sp.search(q=keyword, type='track', limit=10)
                    for track in genre_results['tracks']['items']:
                        if track['external_urls']['spotify'] not in seen_urls and len(songs) < limit * 2:
                            seen_urls.add(track['external_urls']['spotify'])
                            songs.append({
                                "name": track['name'],
                                "artist": track['artists'][0]['name'],
                                "spotify_url": track['external_urls']['spotify'],
                                "preview_url": track.get('preview_url'),
                                "popularity": track.get('popularity', 0),
                                "source": f"genre_{keyword}"
                            })
            except Exception as e:
                print(f"Error in genre search: {e}")
            
            # Sort by popularity and source preference
            def sort_key(song):
                popularity_score = song['popularity']
                source_bonus = 20 if song['source'] == 'same_artist' else 0
                return popularity_score + source_bonus
            
            songs.sort(key=sort_key, reverse=True)
            return songs[:limit]
            
        elif seed_genre:
            # Genre-based search with popularity filter
            search_query = f"genre:{seed_genre}"
            results = sp.search(q=search_query, type='track', limit=limit*2)
            
            songs = []
            for track in results['tracks']['items']:
                songs.append({
                    "name": track['name'],
                    "artist": track['artists'][0]['name'],
                    "spotify_url": track['external_urls']['spotify'],
                    "preview_url": track.get('preview_url'),
                    "popularity": track.get('popularity', 0)
                })
            
            # Sort by popularity
            songs.sort(key=lambda x: x['popularity'], reverse=True)
            return songs[:limit]
        
        else:
            # Default to popular tracks
            return get_top_tracks(limit)
            
    except Exception as e:
        print(f"Error getting recommendations: {e}")
        return get_top_tracks(limit)


def search_by_genre(genre, limit=5):
    """Search for tracks by genre using Spotify search"""
    sp = get_spotify_client()
    if not sp:
        return []
    
    try:
        # Search for tracks with the genre as a query
        query = f'genre:{genre}' if genre in get_valid_genres() else genre
        results = sp.search(q=query, type='track', limit=limit)
        
        songs = []
        for track in results['tracks']['items']:
            songs.append({
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "spotify_url": track['external_urls']['spotify'],
                "preview_url": track.get('preview_url'),
                "popularity": track.get('popularity', 0)
            })
        return songs
    except Exception as e:
        print(f"Error in genre search: {e}")
        return []


def test_recommendations_api():
    """Test the Spotify Recommendations API with known good track IDs"""
    sp = get_spotify_client()
    if not sp:
        return False
    
    # Test with some popular, well-known tracks
    test_tracks = [
        "4iV5W9uYEdYUVa79Axb7Rh",  # Never Gonna Give You Up - Rick Astley
        "7GhIk7Il098yCjg4BQjzvb",  # Bohemian Rhapsody - Queen
        "4uLU6hMCjMI75M1A2tKUQC"   # Never Gonna Give You Up (another version)
    ]
    
    print("\n🧪 Testing Spotify Recommendations API...")
    
    for i, track_id in enumerate(test_tracks, 1):
        print(f"\nTest {i}: Track ID {track_id}")
        try:
            # First verify the track exists
            track_info = sp.track(track_id)
            print(f"✅ Track exists: '{track_info['name']}' by {track_info['artists'][0]['name']}")
            
            # Test recommendations
            results = sp.recommendations(seed_tracks=[track_id], limit=3, market='US')
            if results and results.get('tracks'):
                print(f"✅ Got {len(results['tracks'])} recommendations")
                return True
            else:
                print("❌ No recommendations returned")
                
        except spotipy.exceptions.SpotifyException as se:
            print(f"❌ API Error {se.http_status}: {se.msg}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n❌ All recommendation tests failed")
    return False


    """Find the closest matching genre from valid Spotify genres"""
    input_genre = input_genre.lower()
    
    # Direct mapping for common variations
    genre_mappings = {
        'chill': 'chill',
        'relax': 'ambient',
        'relaxing': 'ambient',
        'dance': 'dance',
        'edm': 'electronic',
        'rap': 'hip-hop',
        'hiphop': 'hip-hop',
        'rnb': 'r-n-b',
        'indie': 'indie',
        'metal': 'metal',
        'punk': 'punk',
        'folk': 'folk',
        'blues': 'blues'
    }
    
    if input_genre in genre_mappings and genre_mappings[input_genre] in valid_genres:
        return genre_mappings[input_genre]
    
    # Find partial matches
    for genre in valid_genres:
        if input_genre in genre or genre in input_genre:
            return genre
    
    # Default fallback
    return 'pop'


def get_top_tracks(limit=5):
    """Get current top tracks as a fallback for collaborative filtering"""
    sp = get_spotify_client()
    if not sp:
        return []
    
    try:
        # Try multiple search strategies for popular tracks
        search_queries = [
            'year:2023-2024',
            'genre:pop',
            'genre:rock',
            'playlist:today\'s top hits'
        ]
        
        all_songs = []
        seen_urls = set()
        
        for query in search_queries:
            try:
                results = sp.search(q=query, type='track', limit=15)
                for track in results['tracks']['items']:
                    if (track['external_urls']['spotify'] not in seen_urls and 
                        track.get('popularity', 0) > 60):  # Only popular tracks
                        seen_urls.add(track['external_urls']['spotify'])
                        all_songs.append({
                            "name": track['name'],
                            "artist": track['artists'][0]['name'],
                            "spotify_url": track['external_urls']['spotify'],
                            "preview_url": track.get('preview_url'),
                            "popularity": track.get('popularity', 0)
                        })
                        if len(all_songs) >= limit * 2:  # Get more than needed to sort
                            break
                if len(all_songs) >= limit * 2:
                    break
            except Exception as e:
                print(f"Error with query '{query}': {e}")
                continue
        
        # Sort by popularity and return top results
        all_songs.sort(key=lambda x: x['popularity'], reverse=True)
        return all_songs[:limit]
        
    except Exception as e:
        print(f"Error getting top tracks: {e}")
    
    # Final fallback to basic search
    return search_by_genre('pop', limit)
