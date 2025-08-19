from utils import clean_text, extract_keywords, calculate_hybrid_score
from spotify_api import (
    get_recommendations, 
    get_recommendations_by_seed,
    get_top_tracks, 
    search_track
)
from groq import Groq
import os
import re

def enforce_artist_diversity(tracks, max_per_artist=1):
    """
    Enforce artist diversity in recommendations by limiting tracks per artist
    
    Args:
        tracks: List of track dictionaries with 'artist' field
        max_per_artist: Maximum number of tracks allowed per artist
    
    Returns:
        List of tracks with enforced diversity
    """
    if not tracks:
        return tracks
    
    artist_counts = {}
    diverse_tracks = []
    
    # Sort tracks by hybrid_score (descending) to prioritize best tracks
    sorted_tracks = sorted(tracks, key=lambda x: x.get('hybrid_score', 0), reverse=True)
    
    for track in sorted_tracks:
        artist_name = track.get('artist', 'Unknown Artist')
        
        # Count tracks per artist
        if artist_name not in artist_counts:
            artist_counts[artist_name] = 0
        
        # Add track if we haven't exceeded the limit for this artist
        if artist_counts[artist_name] < max_per_artist:
            diverse_tracks.append(track)
            artist_counts[artist_name] += 1
    
    print(f"🎭 Artist diversity enforced: {len(diverse_tracks)} tracks from {len(artist_counts)} artists")
    
    return diverse_tracks

def get_diverse_tracks_by_search(query, limit=20):
    """
    Get diverse tracks using multiple search strategies to ensure variety
    
    Args:
        query: Search query string
        limit: Number of tracks to retrieve (should be higher than final limit)
    
    Returns:
        List of track dictionaries with diversity
    """
    from spotify_api import get_spotify_client
    import spotipy
    
    sp = get_spotify_client()
    if not sp:
        return []
    
    all_tracks = []
    seen_urls = set()
    
    print(f"🔍 Searching for diverse tracks: {query}")
    
    try:
        # Strategy 1: Direct search with original query
        try:
            results = sp.search(q=query, type='track', limit=min(50, limit*2), market='US')
            if results and results['tracks']['items']:
                for track in results['tracks']['items']:
                    if track['external_urls']['spotify'] not in seen_urls:
                        seen_urls.add(track['external_urls']['spotify'])
                        track_info = {
                            "name": track['name'],
                            "artist": track['artists'][0]['name'],
                            "spotify_url": track['external_urls']['spotify'],
                            "preview_url": track.get('preview_url'),
                            "popularity": track.get('popularity', 0),
                            "source": "direct_search"
                        }
                        all_tracks.append(track_info)
                        
                print(f"✅ Direct search found {len([t for t in all_tracks if t['source'] == 'direct_search'])} tracks")
        except Exception as e:
            print(f"❌ Direct search error: {e}")
        
        # Strategy 2: Search with modified queries for more diversity
        query_variations = []
        original_query = query.lower()
        
        # Add genre-based variations
        if any(word in original_query for word in ['sad', 'emotional', 'melancholy']):
            query_variations.extend(['sad songs', 'emotional music', 'melancholic'])
        elif any(word in original_query for word in ['happy', 'upbeat', 'energetic']):
            query_variations.extend(['upbeat songs', 'happy music', 'energetic'])
        elif any(word in original_query for word in ['chill', 'relax', 'calm']):
            query_variations.extend(['chill music', 'relaxing songs', 'ambient'])
        
        # Add cultural/language variations
        if any(word in original_query for word in ['nepali', 'nepal']):
            query_variations.extend(['nepali music', 'nepal songs', 'himalayan music'])
        elif any(word in original_query for word in ['hindi', 'bollywood', 'indian']):
            query_variations.extend(['bollywood', 'hindi songs', 'indian music'])
        elif any(word in original_query for word in ['korean', 'k-pop']):
            query_variations.extend(['k-pop', 'korean music', 'kpop'])
        
        # Execute variation searches
        for variation in query_variations[:3]:  # Limit to 3 variations to avoid too many API calls
            if len(all_tracks) >= limit:
                break
                
            try:
                results = sp.search(q=variation, type='track', limit=20, market='US')
                if results and results['tracks']['items']:
                    for track in results['tracks']['items']:
                        if (track['external_urls']['spotify'] not in seen_urls and 
                            len(all_tracks) < limit*3):  # Get more tracks for diversity
                            seen_urls.add(track['external_urls']['spotify'])
                            track_info = {
                                "name": track['name'],
                                "artist": track['artists'][0]['name'],
                                "spotify_url": track['external_urls']['spotify'],
                                "preview_url": track.get('preview_url'),
                                "popularity": track.get('popularity', 0),
                                "source": f"variation_{variation}"
                            }
                            all_tracks.append(track_info)
                            
                print(f"✅ Variation '{variation}' found {len([t for t in all_tracks if t['source'].startswith('variation')])} additional tracks")
            except Exception as e:
                print(f"❌ Variation search error for '{variation}': {e}")
        
        # Strategy 3: Genre-based search if we still need more diversity
        if len(all_tracks) < limit:
            genre_searches = ['pop', 'rock', 'alternative', 'indie']
            if any(word in original_query for word in ['electronic', 'edm']):
                genre_searches = ['electronic', 'dance', 'edm', 'ambient']
            elif any(word in original_query for word in ['rock', 'metal']):
                genre_searches = ['rock', 'alternative', 'metal', 'indie']
            elif any(word in original_query for word in ['rap', 'hip-hop']):
                genre_searches = ['hip-hop', 'rap', 'r-n-b']
            
            for genre in genre_searches[:2]:  # Limit to 2 genres
                if len(all_tracks) >= limit*2:
                    break
                    
                try:
                    results = sp.search(q=f'genre:{genre}', type='track', limit=15, market='US')
                    if results and results['tracks']['items']:
                        for track in results['tracks']['items']:
                            if (track['external_urls']['spotify'] not in seen_urls and 
                                len(all_tracks) < limit*3):
                                seen_urls.add(track['external_urls']['spotify'])
                                track_info = {
                                    "name": track['name'],
                                    "artist": track['artists'][0]['name'],
                                    "spotify_url": track['external_urls']['spotify'],
                                    "preview_url": track.get('preview_url'),
                                    "popularity": track.get('popularity', 0),
                                    "source": f"genre_{genre}"
                                }
                                all_tracks.append(track_info)
                except Exception as e:
                    print(f"❌ Genre search error for '{genre}': {e}")
        
        # Sort by popularity to get the best tracks first
        all_tracks.sort(key=lambda x: x.get('popularity', 0), reverse=True)
        
        print(f"📊 Total diverse tracks collected: {len(all_tracks)} from various sources")
        return all_tracks[:limit]
        
    except Exception as e:
        print(f"❌ Error in diverse track search: {e}")
        return []

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_song_info(user_text):
    """Extract potential song and artist information from user input"""
    # Common patterns in user requests
    patterns = [
        r'play\s+(.*?)(?:\s+by\s+(.*?))?(?:\s+|$)',
        r'like\s+(.*?)(?:\s+by\s+(.*?))?(?:\s+|$)',
        r'similar\s+to\s+(.*?)(?:\s+by\s+(.*?))?(?:\s+|$)',
        r'recommend.*like\s+(.*?)(?:\s+by\s+(.*?))?(?:\s+|$)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, user_text.lower())
        if match:
            song = match.group(1)
            artist = match.group(2) if match.group(2) else None
            if song:
                return song, artist
    
    return None, None

def preprocess_input(user_text):
    """Clean and extract keywords from user input"""
    cleaned = clean_text(user_text)
    keywords = extract_keywords(cleaned)
    return cleaned, keywords

def llm_semantic_analysis(user_text):
    """
    Use Groq API to extract semantic info or preferences and potential song references.
    Returns a dictionary with extracted music preferences and keywords.
    """
    try:
        # Enhanced prompt for better music preference extraction including cultural context
        prompt = f"""Analyze this music request: "{user_text}"
        
1. If there's a specific song or artist mentioned, extract them in the format:
   SONG: <song name>
   ARTIST: <artist name>

2. Extract musical preferences:
   GENRES: <list of relevant genres including cultural/regional genres>
   MOOD: <mood/emotion of requested music>
   TEMPO: <slow/medium/fast if mentioned>
   OCCASION: <any mentioned occasion/activity>
   LANGUAGE: <if specific language/culture mentioned like hindi, bollywood, punjabi, etc>
   REGION: <if specific region mentioned like indian, arabic, korean, etc>

3. Additional keywords that could help find similar music:
   KEYWORDS: <relevant descriptive words including cultural terms>

Format the response in a structured way using the exact labels above.
If any field is not applicable, use 'N/A'.

Special attention to cultural/regional music requests:
- Hindi/Bollywood = Indian pop/film music
- Classical Hindi = Hindustani classical music
- Punjabi = Punjabi folk/pop music
- etc."""

        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3  # Lower temperature for more focused responses
        )
        
        # Extract response
        response = completion.choices[0].message.content
        
        # Parse the structured response
        result = {
            'song': None,
            'artist': None,
            'genres': [],
            'mood': None,
            'tempo': None,
            'occasion': None,
            'language': None,
            'region': None,
            'keywords': []
        }
        
        # Extract information using regex with better error handling
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('SONG:'):
                song = line.replace('SONG:', '').strip()
                result['song'] = song if song != 'N/A' else None
            elif line.startswith('ARTIST:'):
                artist = line.replace('ARTIST:', '').strip()
                result['artist'] = artist if artist != 'N/A' else None
            elif line.startswith('GENRES:'):
                genres = line.replace('GENRES:', '').strip()
                if genres != 'N/A':
                    result['genres'] = [g.strip() for g in genres.split(',') if g.strip()]
            elif line.startswith('MOOD:'):
                mood = line.replace('MOOD:', '').strip()
                result['mood'] = mood if mood != 'N/A' else None
            elif line.startswith('TEMPO:'):
                tempo = line.replace('TEMPO:', '').strip()
                result['tempo'] = tempo if tempo != 'N/A' else None
            elif line.startswith('OCCASION:'):
                occasion = line.replace('OCCASION:', '').strip()
                result['occasion'] = occasion if occasion != 'N/A' else None
            elif line.startswith('LANGUAGE:'):
                language = line.replace('LANGUAGE:', '').strip()
                result['language'] = language if language != 'N/A' else None
            elif line.startswith('REGION:'):
                region = line.replace('REGION:', '').strip()
                result['region'] = region if region != 'N/A' else None
            elif line.startswith('KEYWORDS:'):
                keywords = line.replace('KEYWORDS:', '').strip()
                if keywords != 'N/A':
                    result['keywords'] = [k.strip() for k in keywords.split(',') if k.strip()]
        
        return result
        
    except Exception as e:
        print(f"Error in LLM analysis: {type(e).__name__}: {e}")
        # Return basic structure with extracted song/artist if available
        song, artist = extract_song_info(user_text)
        return {
            'song': song,
            'artist': artist,
            'genres': [],
            'mood': None,
            'tempo': None,
            'occasion': None,
            'language': None,
            'region': None,
            'keywords': []
        }

def content_based_filtering(user_text, llm_analysis, limit=5):
    """
    Enhanced Content-Based Filtering using Spotify search with LLM insights
    Returns tracks matching user preferences with popularity scores
    """
    search_results = []
    
    # Strategy 1: Direct song/artist search if detected by LLM
    if llm_analysis.get('song') and llm_analysis.get('artist'):
        song_name = llm_analysis['song']
        artist_name = llm_analysis['artist']
        print(f"🎵 LLM detected: '{song_name}' by {artist_name}")
        
        track = search_track(f"{song_name} {artist_name}", limit=3)
        if track:
            # Get recommendations based on this specific track
            seed_track = {
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "spotify_url": track['external_urls']['spotify'],
                "preview_url": track.get('preview_url'),
                "popularity": track.get('popularity', 0)
            }
            print(f"✅ Found seed track: {seed_track['name']} by {seed_track['artist']}")
            recommendations = get_recommendations(seed_track=seed_track, limit=limit)
            if recommendations:
                return recommendations
    
    # Strategy 2: Artist-only search
    elif llm_analysis.get('artist'):
        artist_name = llm_analysis['artist']
        print(f"🎵 LLM detected artist: {artist_name}")
        
        track = search_track(f"artist:{artist_name}", limit=3)
        if track:
            seed_track = {
                "name": track['name'],
                "artist": track['artists'][0]['name'],
                "spotify_url": track['external_urls']['spotify'],
                "preview_url": track.get('preview_url'),
                "popularity": track.get('popularity', 0)
            }
            recommendations = get_recommendations(seed_track=seed_track, limit=limit)
            if recommendations:
                return recommendations
    
    # Strategy 3: Genre-based search with cultural context
    elif llm_analysis.get('genres') or llm_analysis.get('language') or llm_analysis.get('region'):
        search_terms = []
        
        # Add cultural/regional context
        if llm_analysis.get('language'):
            search_terms.append(llm_analysis['language'])
        if llm_analysis.get('region'):
            search_terms.append(llm_analysis['region'])
        
        # Add genres
        if llm_analysis.get('genres'):
            search_terms.extend(llm_analysis['genres'][:2])  # Max 2 genres
        
        # Add mood if available
        if llm_analysis.get('mood'):
            search_terms.append(llm_analysis['mood'])
        
        # Create search query with cultural context
        if search_terms:
            search_query = ' '.join(search_terms)
            print(f"🎵 Genre/Cultural search: {search_query}")
            
            track = search_track(search_query, limit=3)
            if track:
                seed_track = {
                    "name": track['name'],
                    "artist": track['artists'][0]['name'],
                    "spotify_url": track['external_urls']['spotify'],
                    "preview_url": track.get('preview_url'),
                    "popularity": track.get('popularity', 0)
                }
                recommendations = get_recommendations(seed_track=seed_track, limit=limit)
                if recommendations:
                    return recommendations
    
    # Strategy 4: Keyword-based search
    if llm_analysis.get('keywords'):
        keywords = llm_analysis['keywords'][:3]  # Use top 3 keywords
        if keywords:
            keyword_query = ' '.join(keywords)
            print(f"🔍 Keyword search: {keyword_query}")
            
            track = search_track(keyword_query, limit=3)
            if track:
                seed_track = {
                    "name": track['name'],
                    "artist": track['artists'][0]['name'],
                    "spotify_url": track['external_urls']['spotify'],
                    "preview_url": track.get('preview_url'),
                    "popularity": track.get('popularity', 0)
                }
                recommendations = get_recommendations(seed_track=seed_track, limit=limit)
                if recommendations:
                    return recommendations
    
    # Strategy 5: Fallback to original text search
    print(f"� Fallback search with original text: {user_text}")
    track = search_track(user_text, limit=3)
    if track:
        seed_track = {
            "name": track['name'],
            "artist": track['artists'][0]['name'],
            "spotify_url": track['external_urls']['spotify'],
            "preview_url": track.get('preview_url'),
            "popularity": track.get('popularity', 0)
        }
        recommendations = get_recommendations(seed_track=seed_track, limit=limit)
        if recommendations:
            return recommendations
    
    # Final fallback to genre-based recommendations
    print("⚠️ No specific tracks found, using genre-based recommendations")
    if llm_analysis.get('genres'):
        primary_genre = llm_analysis['genres'][0]
        return get_recommendations(seed_genre=primary_genre, limit=limit)
    else:
        return get_recommendations(seed_genre="pop", limit=limit)

def collaborative_filtering(user_id=None, limit=5):
    """
    Popularity-Based Collaborative Filtering
    Returns popular tracks across all Spotify users as lightweight CF
    """
    try:
        print("📊 Getting popular tracks for collaborative filtering...")
        popular_tracks = get_top_tracks(limit=limit*2)  # Get more to have variety
        
        if popular_tracks:
            # Filter to only highly popular tracks (popularity > 70)
            highly_popular = [track for track in popular_tracks 
                            if track.get('popularity', 0) > 70]
            
            # If we don't have enough highly popular tracks, use regular popular tracks
            if len(highly_popular) < limit:
                highly_popular = popular_tracks
            
            # Sort by popularity descending
            highly_popular.sort(key=lambda x: x.get('popularity', 0), reverse=True)
            
            print(f"✅ Found {len(highly_popular)} popular tracks for CF")
            return highly_popular[:limit]
        
    except Exception as e:
        print(f"❌ Error in popularity-based CF: {e}")
    
    # Fallback to genre-based popular tracks
    print("🔄 Fallback: Using genre-based popular recommendations")
    return get_recommendations(seed_genre="pop", limit=limit)

def hybrid_recommend(user_text, user_id=None, limit=5):
    """
    Enhanced Hybrid Recommendation System
    
    Flow:
    1. Content-Based Filtering: LLM analysis + Spotify search
    2. Popularity-Based Ranking: Use Spotify popularity scores
    3. Combine results with relevance and popularity weighting
    """
    if not user_text.strip():
        return get_recommendations(seed_genre="pop", limit=limit)
    
    try:
        print("🤖 Step 1: Analyzing your request with LLM...")
        
        # Step 1: LLM Semantic Analysis
        cleaned, basic_keywords = preprocess_input(user_text)
        llm_analysis = llm_semantic_analysis(cleaned)
        
        print("🔍 Step 2: Content-based filtering...")
        
        # Step 2: Content-Based Filtering
        content_candidates = content_based_filtering(cleaned, llm_analysis, limit=limit*2)
        
        print("📈 Step 3: Applying popularity-based ranking...")
        
        # Step 3: Popularity-Based Ranking
        if content_candidates:
            # Enhance each candidate with popularity-based scoring
            for song in content_candidates:
                # Use the utility function for consistent scoring
                song['hybrid_score'] = calculate_hybrid_score(song, llm_analysis)
                song['relevance_score'] = song['hybrid_score'] * 0.6  # Content portion
                song['popularity_score'] = song.get('popularity', 0) / 100.0
        
        print("🎯 Step 4: Lightweight collaborative filtering...")
        
        # Step 4: Lightweight Collaborative Filtering (Popular tracks as backup)
        popular_tracks = collaborative_filtering(user_id, limit=max(3, limit//2))
        
        # Add popularity boost to popular tracks
        for track in popular_tracks:
            track['hybrid_score'] = track.get('popularity', 0) / 100.0 * 0.8  # Lower weight for CF
            track['relevance_score'] = 0.3  # Lower relevance since not query-specific
            track['popularity_score'] = track.get('popularity', 0) / 100.0
        
        print("🔀 Step 5: Combining and ranking results...")
        
        # Step 5: Combine and rank all results
        all_candidates = []
        seen_urls = set()
        
        # Prioritize content-based results
        if content_candidates:
            for song in content_candidates:
                if song['spotify_url'] not in seen_urls:
                    seen_urls.add(song['spotify_url'])
                    all_candidates.append(song)
        
        # Add popular tracks as variety (avoid duplicates)
        if popular_tracks:
            for track in popular_tracks:
                if track['spotify_url'] not in seen_urls and len(all_candidates) < limit * 2:
                    seen_urls.add(track['spotify_url'])
                    all_candidates.append(track)
        
        # Sort by hybrid score (relevance + popularity)
        all_candidates.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        # Return top results
        final_recommendations = all_candidates[:limit]
        
        # Add some debug info
        print(f"✅ Returning {len(final_recommendations)} recommendations")
        for i, rec in enumerate(final_recommendations[:3], 1):
            print(f"   {i}. {rec['name']} by {rec['artist']} "
                  f"(Hybrid: {rec.get('hybrid_score', 0):.2f}, "
                  f"Pop: {rec.get('popularity', 0)})")
        
        return final_recommendations
        
    except Exception as e:
        print(f"❌ Error in hybrid recommendation: {e}")
        # Fallback to basic recommendations
        return get_recommendations(seed_genre="pop", limit=limit)
