import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from config import SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
import random
import re

    
def enforce_artist_diversity(tracks, max_per_artist=1, artist_specific_limit=None):
    """
    Enforce artist diversity in recommendations by limiting tracks per artist.
    """
    if not tracks:
        return tracks

    if artist_specific_limit is None:
        artist_specific_limit = {}

    artist_counts = {}
    diverse_tracks = []

    # Sort tracks by hybrid_score (descending) to prioritize best tracks
    sorted_tracks = sorted(tracks, key=lambda x: x.get('hybrid_score', 0), reverse=True)

    for track in sorted_tracks:
        artist_name = track['artists'][0]['name'] if track.get('artists') else 'Unknown Artist'

        # Determine the limit for the current artist
        limit = artist_specific_limit.get(artist_name, max_per_artist)

        # Count tracks per artist
        if artist_name not in artist_counts:
            artist_counts[artist_name] = 0

        # Add track if we haven't exceeded the limit for this artist
        if artist_counts[artist_name] < limit:
            diverse_tracks.append(track)
            artist_counts[artist_name] += 1

    print(f"🎭 Artist diversity enforced: {len(diverse_tracks)} tracks from {len(artist_counts)} artists")
    return diverse_tracks


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


def search_track_by_cultural_context(query, limit=5):
    """Enhanced search for cultural/regional music requests"""
    sp = get_spotify_client()
    if not sp:
        return []
    
    # Cultural/regional mapping with specific search terms
    cultural_mappings = {
        'nepali': ['nepali', 'nepal', 'devanagari'],
        'hindi': ['hindi', 'bollywood', 'filmi'],  
        'bollywood': ['bollywood', 'hindi', 'filmi'],
        'punjabi': ['punjabi', 'bhangra'],
        'tamil': ['tamil', 'kollywood'],
        'telugu': ['telugu', 'tollywood'],
        'bengali': ['bengali', 'rabindra sangeet'],
        'classical indian': ['hindustani', 'carnatic', 'raga', 'indian classical'],
        'arabic': ['arabic', 'middle east'],
        'korean': ['korean', 'k-pop', 'kpop'],
        'chinese': ['chinese', 'mandarin', 'c-pop'],
        'japanese': ['japanese', 'j-pop', 'jpop']
    }
    
    # Extract cultural context from query
    query_lower = query.lower()
    matched_culture = None
    search_terms = []
    
    for culture, terms in cultural_mappings.items():
        if any(term in query_lower for term in [culture] + terms):
            matched_culture = culture
            search_terms = terms
            break
    
    tracks = []
    
    if matched_culture:
        print(f"🌍 Detected {matched_culture} music request")
        
        # Set appropriate market for better results
        markets = {
            'nepali': 'NP',
            'hindi': 'IN', 
            'bollywood': 'IN',
            'punjabi': 'IN',
            'tamil': 'IN',
            'telugu': 'IN',
            'bengali': 'IN',
            'classical indian': 'IN',
            'korean': 'KR',
            'japanese': 'JP'
        }
        
        market = markets.get(matched_culture, None)
        
        # Try different search strategies
        for term in search_terms[:3]:  # Try top 3 terms
            try:
                print(f"🔍 Searching for: {term}")
                search_params = {
                    'q': f"{term} {query}",
                    'type': 'track',
                    'limit': limit * 2
                }
                if market:
                    search_params['market'] = market
                    
                results = sp.search(**search_params)
                
                if results and results['tracks']['items']:
                    for track in results['tracks']['items']:
                        # Validate cultural relevance
                        track_text = f"{track['name']} {' '.join([artist['name'] for artist in track['artists']])}"
                        
                        # Check if track contains cultural indicators
                        if any(indicator in track_text.lower() for indicator in search_terms):
                            track_info = {
                                "name": track['name'],
                                "artist": track['artists'][0]['name'],
                                "spotify_url": track['external_urls']['spotify'],
                                "preview_url": track.get('preview_url'),
                                "popularity": track.get('popularity', 0),
                                "is_regional": True,
                                "culture": matched_culture
                            }
                            tracks.append(track_info)
                            
                            if len(tracks) >= limit:
                                break
                
                if len(tracks) >= limit:
                    break
                    
            except Exception as e:
                print(f"Error searching for {term}: {e}")
                continue
    
    return tracks[:limit]


def search_track(query, limit=5):
    """Enhanced search function with better fallback strategies and duplicate handling"""
    sp = get_spotify_client()
    if not sp:
        print("❌ Failed to initialize Spotify client")
        return []
    
    print(f"🔍 Searching Spotify for: {query}")
    
    try:
        all_tracks = []
        
        # First, try cultural/regional search
        cultural_tracks = search_track_by_cultural_context(query, limit)
        if cultural_tracks:
            all_tracks.extend(cultural_tracks)
            print(f"✅ Found {len(cultural_tracks)} cultural/regional tracks")
        
        # If not enough, continue with other search strategies
        if len(all_tracks) < limit:
            # Artist-song patterns
            specific_patterns = [
                r'(.+?)\s+by\s+(.+?)(?:\s+|$)',         # "[song] by [artist]"  
                r'songs?\s+by\s+(.+?)(?:\s+|$)',        # "songs by [artist]"
                r'something\s+by\s+(.+?)(?:\s+|$)',     # "something by [artist]"
                r'music\s+by\s+(.+?)(?:\s+|$)',         # "music by [artist]"
                r'(?:play|want|give me|i like)\s+(.+?)\s+by\s+(.+?)(?:\s+|$)', # "play [song] by [artist]"
            ]
            
            # Check for specific song/artist patterns
            for pattern in specific_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:  # Song by artist
                        song, artist = match.groups()
                        print(f"🎵 Searching for specific song: '{song}' by '{artist}'")
                        results = sp.search(q=f"track:{song} artist:{artist}", type='track', limit=min(limit*2, 50))
                        tracks = results['tracks']['items']
                        
                        # Add non-duplicate tracks
                        existing_ids = {track['id'] for track in all_tracks}
                        new_tracks = [track for track in tracks if track['id'] not in existing_ids]
                        all_tracks.extend(new_tracks)
                        
                        # Also try broader search for the artist
                        if len(all_tracks) < limit:
                            artist_results = sp.search(q=f"artist:{artist}", type='track', limit=min(limit*2, 50))
                            artist_tracks = artist_results['tracks']['items']
                            existing_ids = {track['id'] for track in all_tracks}
                            new_artist_tracks = [track for track in artist_tracks if track['id'] not in existing_ids]
                            all_tracks.extend(new_artist_tracks)
                            
                    elif len(match.groups()) == 1:  # Just artist
                        artist = match.groups()[0]
                        print(f"🎤 Searching for artist: '{artist}'")
                        results = sp.search(q=f"artist:{artist}", type='track', limit=min(limit*2, 50))
                        tracks = results['tracks']['items']
                        
                        # Add non-duplicate tracks
                        existing_ids = {track['id'] for track in all_tracks}
                        new_tracks = [track for track in tracks if track['id'] not in existing_ids]
                        all_tracks.extend(new_tracks)
                    
                    break  # Found a pattern, don't try others
            
            # If still not enough results, try general search with enhancements
            if len(all_tracks) < limit:
                print("🔍 Performing enhanced general search")
                search_variations = [
                    query,
                    f"{query} pop",
                    f"{query} rock", 
                    f"{query} electronic",
                    f"{query} music",
                    f"{query} upbeat"
                ]
                
                for search_query in search_variations:
                    if len(all_tracks) >= limit * 2:  # Get extra for better selection
                        break
                        
                    print(f"🔍 Enhanced search: {search_query}")
                    results = sp.search(q=search_query, type='track', limit=20)
                    tracks = results['tracks']['items']
                    
                    if tracks:
                        # Add non-duplicate tracks
                        existing_ids = {track['id'] for track in all_tracks}
                        new_tracks = [track for track in tracks if track['id'] not in existing_ids]
                        all_tracks.extend(new_tracks)
                        print(f"✅ Added {len(new_tracks)} new tracks")
        
        # Apply artist diversity to get better variety
        diverse_tracks = enforce_artist_diversity(all_tracks, max_per_artist=max(1, limit // max(len(all_tracks)//limit, 1)))
        
        # If we still don't have enough, relax the diversity requirement
        if len(diverse_tracks) < limit and len(all_tracks) > len(diverse_tracks):
            print(f"⚠️ Only {len(diverse_tracks)} diverse tracks found, relaxing artist constraint...")
            diverse_tracks = enforce_artist_diversity(all_tracks, max_per_artist=limit)
        
        print(f"✅ Final search results: {len(diverse_tracks)} tracks")
        return diverse_tracks[:limit]
        
    except Exception as e:
        print(f"❌ Search error: {str(e)}")
        return []
        
        # Try specific song-artist patterns
        for pattern in specific_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) == 2:  # song and artist
                    song_name = groups[0].strip()
                    artist_name = groups[1].strip()
                    
                    # Skip generic words
                    if song_name.lower() not in ['something', 'songs', 'music']:
                        print(f"🎵 Searching for specific song: '{song_name}' by '{artist_name}'")
                        search_query = f"track:{song_name} artist:{artist_name}"
                    else:
                        print(f"🎤 Searching for artist: '{artist_name}'")
                        search_query = f"artist:{artist_name}"
                        
                elif len(groups) == 1:  # just artist
                    artist_name = groups[0].strip()
                    print(f"🎤 Searching for artist: '{artist_name}'")
                    search_query = f"artist:{artist_name}"
                else:
                    continue
                
                try:
                    results = sp.search(q=search_query, type='track', limit=limit*2)
                    if results and results['tracks']['items']:
                        for track in results['tracks']['items']:
                            track_info = {
                                "name": track['name'],
                                "artist": track['artists'][0]['name'],
                                "spotify_url": track['external_urls']['spotify'],
                                "preview_url": track.get('preview_url'),
                                "popularity": track.get('popularity', 0),
                                "is_regional": False
                            }
                            tracks.append(track_info)
                            
                        return tracks[:limit]
                except Exception as e:
                    print(f"Error in specific search: {e}")
                    continue
        
        # If no specific patterns found, do general search with genre/mood enhancement
        print("🔍 Performing enhanced general search")
        
        # Extract genre and mood keywords for better search
        genre_keywords = []
        mood_keywords = []
        
        query_lower = query.lower()
        
        # Genre detection
        if any(word in query_lower for word in ['pop', 'popular']):
            genre_keywords.append('pop')
        if any(word in query_lower for word in ['rock', 'alternative']):
            genre_keywords.append('rock')
        if any(word in query_lower for word in ['electronic', 'edm', 'dance']):
            genre_keywords.append('electronic')
        if any(word in query_lower for word in ['hip-hop', 'hiphop', 'rap']):
            genre_keywords.append('hip-hop')
        if any(word in query_lower for word in ['jazz', 'blues']):
            genre_keywords.append('jazz')
        if any(word in query_lower for word in ['classical', 'orchestra']):
            genre_keywords.append('classical')
        if any(word in query_lower for word in ['old', 'vintage', 'retro']):
            genre_keywords.append('oldies')
            
        # Mood detection
        if any(word in query_lower for word in ['chill', 'relaxing', 'calm', 'peaceful']):
            mood_keywords.append('chill')
        if any(word in query_lower for word in ['upbeat', 'energetic', 'party', 'dance']):
            mood_keywords.append('upbeat')
        if any(word in query_lower for word in ['sad', 'melancholy', 'emotional']):
            mood_keywords.append('sad')
        if any(word in query_lower for word in ['romantic', 'love']):
            mood_keywords.append('romantic')
        
        # Build enhanced search query
        search_terms = []
        if genre_keywords:
            search_terms.extend(genre_keywords)
        if mood_keywords:
            search_terms.extend(mood_keywords)
        
        # Try enhanced search first
        if search_terms:
            enhanced_query = f"{query} {' '.join(search_terms)}"
            print(f"🔍 Enhanced search: {enhanced_query}")
            
            try:
                results = sp.search(q=enhanced_query, type='track', limit=limit*2)
                if results and results['tracks']['items']:
                    for track in results['tracks']['items']:
                        track_info = {
                            "name": track['name'],
                            "artist": track['artists'][0]['name'],
                            "spotify_url": track['external_urls']['spotify'],
                            "preview_url": track.get('preview_url'),
                            "popularity": track.get('popularity', 0),
                            "is_regional": False
                        }
                        tracks.append(track_info)
                    
                    if tracks:
                        return tracks[:limit]
            except Exception as e:
                print(f"Error in enhanced search: {e}")
        
        # Fallback to original query
        try:
            print(f"🔍 Fallback search: {query}")
            results = sp.search(q=query, type='track', limit=limit*2)
            if results and results['tracks']['items']:
                for track in results['tracks']['items']:
                    track_info = {
                        "name": track['name'],
                        "artist": track['artists'][0]['name'],
                        "spotify_url": track['external_urls']['spotify'],
                        "preview_url": track.get('preview_url'),
                        "popularity": track.get('popularity', 0),
                        "is_regional": False
                    }
                    tracks.append(track_info)
                
                return tracks[:limit]
        except Exception as e:
            print(f"Error in fallback search: {e}")
        
        print("❌ No matching tracks found")
        return []
            
    except Exception as e:
        print(f"Error searching tracks: {e}")
        return []


def get_recommendations(seed_track=None, seed_tracks=None, seed_genre=None, limit=5):
    """Get track recommendations using search-based approach"""
    sp = get_spotify_client()
    if not sp:
        return []
    
    print("ℹ️ Using search-based recommendations")
    
    if seed_track:
        return get_recommendations_by_seed(seed_track=seed_track, limit=limit)
    elif seed_tracks:
        if seed_tracks:
            main_track = seed_tracks[0] if isinstance(seed_tracks[0], dict) else {"id": seed_tracks[0]}
            return get_recommendations_by_seed(seed_track=main_track, limit=limit)
    elif seed_genre:
        return get_recommendations_by_seed(seed_genre=seed_genre, limit=limit)
    
    # Fallback to popular tracks
    return get_top_tracks(limit)


def get_recommendations_by_seed(seed_track=None, seed_genre=None, limit=5):
    """Get song recommendations from Spotify using search-based approach"""
    sp = get_spotify_client()
    if not sp:
        return []
    
    try:
        if seed_track:
            print(f"🔄 Getting recommendations for: {seed_track['name']} by {seed_track['artist']}")
            artist_name = seed_track['artist']
            
            songs = []
            seen_urls = set()
            if 'spotify_url' in seed_track:
                seen_urls.add(seed_track['spotify_url'])
            
            # Strategy 1: More songs by the same artist
            try:
                artist_results = sp.search(q=f"artist:{artist_name}", type='track', limit=15)
                if artist_results and artist_results['tracks']['items']:
                    for track in artist_results['tracks']['items']:
                        url = track['external_urls']['spotify']
                        if url not in seen_urls:
                            track_info = {
                                "name": track['name'],
                                "artist": track['artists'][0]['name'],
                                "spotify_url": url,
                                "preview_url": track.get('preview_url'),
                                "popularity": track.get('popularity', 0),
                                "source": "same_artist"
                            }
                            songs.append(track_info)
                            seen_urls.add(url)
            except Exception as e:
                print(f"Error getting artist tracks: {e}")
            
            # Strategy 2: Similar artists/genres
            try:
                # Get artist genres
                artist_search = sp.search(q=artist_name, type='artist', limit=1)
                if artist_search and artist_search['artists']['items']:
                    artist_info = artist_search['artists']['items'][0]
                    genres = artist_info.get('genres', [])
                    
                    if genres:
                        # Search by first genre
                        genre_results = sp.search(q=f"genre:{genres[0]}", type='track', limit=10)
                        if genre_results and genre_results['tracks']['items']:
                            for track in genre_results['tracks']['items']:
                                url = track['external_urls']['spotify']
                                if url not in seen_urls:
                                    track_info = {
                                        "name": track['name'],
                                        "artist": track['artists'][0]['name'],
                                        "spotify_url": url,
                                        "preview_url": track.get('preview_url'),
                                        "popularity": track.get('popularity', 0),
                                        "source": "genre_match"
                                    }
                                    songs.append(track_info)
                                    seen_urls.add(url)
            except Exception as e:
                print(f"Error getting genre matches: {e}")
            
            # Sort by popularity and source preference
            def sort_key(song):
                popularity_score = song.get('popularity', 0)
                source_bonus = 10 if song.get('source') == 'same_artist' else 0
                return popularity_score + source_bonus
            
            songs.sort(key=sort_key, reverse=True)
            return songs[:limit]
            
        elif seed_genre:
            return search_by_genre(seed_genre, limit)
        
        else:
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
        # Search for tracks with the genre
        query = f'genre:{genre}' if genre in get_valid_genres() else genre
        results = sp.search(q=query, type='track', limit=limit*2)
        
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
        songs.sort(key=lambda x: x.get('popularity', 0), reverse=True)
        return songs[:limit]
        
    except Exception as e:
        print(f"Error in genre search: {e}")
        return []


def get_top_tracks(limit=10):
    """Get popular tracks from various sources"""
    sp = get_spotify_client()
    if not sp:
        return []
    
    try:
        print("🔥 Getting top tracks...")
        
        from datetime import datetime
        current_year = datetime.now().year
        
        # Search queries for popular music
        search_queries = [
            f'year:{current_year}',
            f'year:{current_year-1}', 
            'genre:pop',
            'genre:hip-hop',
            'genre:rock',
            'genre:dance'
        ]
        
        all_songs = []
        seen_urls = set()
        
        for query in search_queries:
            if len(all_songs) >= limit * 3:
                break
                
            try:
                results = sp.search(q=query, type='track', limit=20)
                if results and results['tracks']['items']:
                    for track in results['tracks']['items']:
                        url = track['external_urls']['spotify']
                        if url not in seen_urls and track.get('popularity', 0) > 40:
                            track_info = {
                                "name": track['name'],
                                "artist": track['artists'][0]['name'],
                                "spotify_url": url,
                                "preview_url": track.get('preview_url'),
                                "popularity": track.get('popularity', 0)
                            }
                            all_songs.append(track_info)
                            seen_urls.add(url)
            except Exception as e:
                print(f"Error in search query '{query}': {e}")
                continue
        
        if not all_songs:
            # Emergency fallback
            results = sp.search(q="popular music", type='track', limit=limit)
            if results and results['tracks']['items']:
                for track in results['tracks']['items']:
                    all_songs.append({
                        "name": track['name'],
                        "artist": track['artists'][0]['name'],
                        "spotify_url": track['external_urls']['spotify'],
                        "preview_url": track.get('preview_url'),
                        "popularity": track.get('popularity', 0)
                    })
        
        # Sort by popularity
        all_songs.sort(key=lambda x: x.get('popularity', 0), reverse=True)
        
        print(f"✅ Found {len(all_songs)} popular tracks, returning top {limit}")
        return all_songs[:limit]
        
    except Exception as e:
        print(f"❌ Error in get_top_tracks: {e}")
        return []
