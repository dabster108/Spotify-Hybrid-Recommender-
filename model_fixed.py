from utils import clean_text, extract_keywords, calculate_hybrid_score
from spotify_api_fixed import search_track, get_top_tracks, get_spotify_client
from groq import Groq
import os
import re

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def enforce_artist_diversity(tracks, max_per_artist=1, artist_specific_limit=None):
    if not tracks:
        return tracks
    
    if artist_specific_limit is None:
        artist_specific_limit = {}
    
    artist_counts = {}
    diverse_tracks = []
    sorted_tracks = sorted(tracks, key=lambda x: x.get('hybrid_score', 0), reverse=True)
    
    for track in sorted_tracks:
        artist_name = track['artists'][0]['name'] if track.get('artists') else 'Unknown Artist'
        limit_for_artist = artist_specific_limit.get(artist_name, max_per_artist)
        
        if artist_name not in artist_counts:
            artist_counts[artist_name] = 0
        
        if artist_counts[artist_name] < limit_for_artist:
            diverse_tracks.append(track)
            artist_counts[artist_name] += 1
    
    print(f"🎭 Artist diversity enforced: {len(diverse_tracks)} tracks from {len(artist_counts)} artists")
    return diverse_tracks

def llm_semantic_analysis(user_text):
    try:
        prompt = f"""Analyze this music request: "{user_text}"

SONG: [specific song name or N/A]
ARTIST: [specific artist name or N/A]
GENRES: [genres, comma-separated]
MOOD: [mood or N/A]
LANGUAGE: [language or N/A]
REGION: [region or N/A]
KEYWORDS: [keywords, comma-separated]
LIMIT: [number or N/A]

For regional music, extract language/region precisely:
- "nepali songs" → LANGUAGE: nepali, REGION: nepali
- "hindi songs" → LANGUAGE: hindi, REGION: indian  
- "korean music" → LANGUAGE: korean, REGION: korean

Analyze: "{user_text}"
"""
        
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1
        )
        
        response = completion.choices[0].message.content
        print(f"🤖 LLM Response: {response[:200]}...")
        
        result = {
            'song': None, 'artist': None, 'genres': [], 'mood': None,
            'language': None, 'region': None, 'keywords': [], 'limit': None
        }
        
        # First, check for common patterns manually
        user_lower = user_text.lower()
        if 'nepali' in user_lower or 'nepal' in user_lower:
            result['language'] = 'nepali'
            result['region'] = 'nepali'
        elif 'hindi' in user_lower:
            result['language'] = 'hindi'
            result['region'] = 'indian'
        elif 'punjabi' in user_lower:
            result['language'] = 'punjabi'
            result['region'] = 'indian'
        elif 'korean' in user_lower:
            result['language'] = 'korean'
            result['region'] = 'korean'
        
        # Extract mood
        if 'sad' in user_lower:
            result['mood'] = 'sad'
        elif 'happy' in user_lower:
            result['mood'] = 'happy'
        elif 'upbeat' in user_lower:
            result['mood'] = 'upbeat'
        elif 'chill' in user_lower or 'relax' in user_lower:
            result['mood'] = 'relaxing'
        
        # Enhanced artist pattern detection before parsing LLM response
        artist_patterns = [
            r'song\s+(?:by|from)\s+([A-Za-z\s\'-]+?)(?:\s*$|\s+(?:songs?|music|track))',  # "song by [Artist]"
            r'songs?\s+(?:by|from)\s+([A-Za-z\s\'-]+?)(?:\s*$|\s+(?:please|now))',  # "songs by/from [Artist]"
            r'(?:play|want|give me|i like)\s+([A-Za-z\s\'-]+?)(?:\s+songs?|\s+music|\s*$)',  # "play [Artist] songs"
            r'music\s+(?:by|from)\s+([A-Za-z\s\'-]+?)(?:\s*$|\s+(?:please|now))',   # "music by/from [Artist]"
            r'tracks?\s+(?:by|from)\s+([A-Za-z\s\'-]+?)(?:\s*$|\s+(?:please|now))', # "tracks by/from [Artist]"
            r'listen\s+to\s+([A-Za-z\s\'-]+?)(?:\s+songs?|\s+music|\s*$)',  # "listen to [Artist]"
            r'by\s+([A-Za-z\s\'-]+?)(?:\s*$)',  # simple "by [Artist]" at end
        ]
        
        for pattern in artist_patterns:
            match = re.search(pattern, user_text, re.IGNORECASE)
            if match:
                artist_name = match.group(1).strip()
                # Clean up common false positives and improve filtering
                excluded_words = ['music', 'songs', 'tracks', 'some', 'good', 'nice', 'best', 'the', 'a', 'an']
                # Check if artist name is not just excluded words
                artist_words = artist_name.lower().split()
                if (len(artist_words) > 0 and 
                    not all(word in excluded_words for word in artist_words) and 
                    len(artist_name) > 2):
                    result['artist'] = artist_name
                    print(f"🎤 Artist detected: {artist_name}")
                    break
        
        # Parse LLM response
        for line in response.splitlines():
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().upper()
                value = value.strip()
                
                if value and value.upper() != 'N/A':
                    if key == 'SONG':
                        result['song'] = value
                    elif key == 'ARTIST' and not result['artist']:  # Don't override pattern detection
                        result['artist'] = value
                    elif key == 'GENRES':
                        result['genres'] = [g.strip() for g in value.split(',') if g.strip()][:3]
                    elif key == 'MOOD' and not result['mood']:  # Don't override manual detection
                        result['mood'] = value
                    elif key == 'LANGUAGE' and not result['language']:  # Don't override manual detection
                        result['language'] = value
                    elif key == 'REGION' and not result['region']:  # Don't override manual detection
                        result['region'] = value
                    elif key == 'KEYWORDS':
                        result['keywords'] = [k.strip() for k in value.split(',') if k.strip()][:5]
                    elif key == 'LIMIT':
                        try:
                            numbers = re.findall(r'\d+', value)
                            if numbers:
                                result['limit'] = int(numbers[0])
                        except:
                            pass
        
        # Extract keywords from original text
        if not result['keywords']:
            result['keywords'] = [word for word in user_text.lower().split() if len(word) > 2][:5]
        
        if not result['limit']:
            patterns = [r'(\d+)\s+(?:songs?|tracks?)', r'give\s+me\s+(\d+)', r'want\s+(\d+)']
            for pattern in patterns:
                match = re.search(pattern, user_text.lower())
                if match:
                    result['limit'] = int(match.group(1))
                    break
        
        return result
        
    except Exception as e:
        print(f"❌ Error in LLM analysis: {e}")
        return {'song': None, 'artist': None, 'genres': [], 'mood': None,
                'language': None, 'region': None, 'keywords': [], 'limit': None}

def is_featured_artist(track_name, track_artists, target_artist):
    """
    Detect if an artist is featured vs being the main artist
    Returns: ('main', 'featured', 'not_found')
    """
    target_lower = target_artist.lower()
    track_name_lower = track_name.lower()
    
    # Check if target artist is the main artist (first in list)
    if track_artists and track_artists[0]['name'].lower() == target_lower:
        return 'main'
    
    # Check for featuring patterns in track name
    featuring_patterns = [
        r'\(feat\.?\s+([^)]*' + re.escape(target_lower) + r'[^)]*)\)',
        r'\(ft\.?\s+([^)]*' + re.escape(target_lower) + r'[^)]*)\)',
        r'\(featuring\s+([^)]*' + re.escape(target_lower) + r'[^)]*)\)',
        r'\(with\s+([^)]*' + re.escape(target_lower) + r'[^)]*)\)',
        r'feat\.?\s+([^-]*' + re.escape(target_lower) + r'[^-]*)',
        r'ft\.?\s+([^-]*' + re.escape(target_lower) + r'[^-]*)',
        r'featuring\s+([^-]*' + re.escape(target_lower) + r'[^-]*)',
        r'with\s+([^-]*' + re.escape(target_lower) + r'[^-]*)'
    ]
    
    for pattern in featuring_patterns:
        if re.search(pattern, track_name_lower, re.IGNORECASE):
            return 'featured'
    
    # Check if target artist appears in the artists list (but not as main)
    for artist in track_artists[1:]:  # Skip first artist (main)
        if target_lower in artist['name'].lower():
            return 'featured'
    
    return 'not_found'

def filter_artist_tracks(tracks, target_artist, prioritize_main=True):
    """
    Filter and sort tracks based on whether the target artist is main or featured
    """
    main_artist_tracks = []
    featured_tracks = []
    
    for track in tracks:
        track_artists = track.get('artists', [])
        track_name = track.get('name', '')
        
        artist_status = is_featured_artist(track_name, track_artists, target_artist)
        
        if artist_status == 'main':
            main_artist_tracks.append(track)
        elif artist_status == 'featured':
            # Add a marker to indicate this is a featured track
            track['is_featured'] = True
            featured_tracks.append(track)
    
    if prioritize_main:
        # Return main artist tracks first, then featured tracks
        return main_artist_tracks + featured_tracks
    else:
        # Return all tracks mixed
        return main_artist_tracks + featured_tracks

def content_based_filtering(user_text, llm_analysis, limit=5):
    print(f"🔍 Content-based filtering (target: {limit} tracks)")
    
    search_query = user_text
    tracks = []
    
    # If we have specific song/artist, prioritize that
    if llm_analysis.get('song') and llm_analysis.get('artist'):
        search_query = f"{llm_analysis['song']} by {llm_analysis['artist']}"
        print(f"🎵 Searching for specific song: {search_query}")
        tracks = search_track(search_query, limit=limit * 2)
    elif llm_analysis.get('artist'):
        artist_name = llm_analysis['artist']
        print(f"🎤 Searching for artist: {artist_name}")
        
        # Try multiple search strategies for artist
        search_strategies = [
            f"artist:{artist_name}",
            f"artist:\"{artist_name}\"",
            f"{artist_name} songs"
        ]
        
        for search_query in search_strategies:
            print(f"🔍 Trying artist search: {search_query}")
            try:
                tracks = search_track(search_query, limit=limit * 2)
                if tracks:
                    # Use new filtering logic to prioritize main artist tracks
                    filtered_tracks = filter_artist_tracks(tracks, artist_name, prioritize_main=True)
                    
                    # Separate main and featured tracks for better reporting
                    main_tracks = [t for t in filtered_tracks if not t.get('is_featured', False)]
                    featured_tracks = [t for t in filtered_tracks if t.get('is_featured', False)]
                    
                    if main_tracks:
                        # Apply very strong boost to main tracks to ensure they always rank higher
                        for track in main_tracks:
                            # Add a massive boost of +1.0 to ensure main tracks always come first
                            track['hybrid_score'] = track.get('hybrid_score', 0.5) + 1.0
                        # Apply penalty to featured tracks
                        for track in featured_tracks:
                            # Subtract 0.3 from featured tracks to ensure they rank lower
                            track['hybrid_score'] = track.get('hybrid_score', 0.5) - 0.3
                        
                        # Prioritize main tracks first, then add featured tracks to fill up to limit
                        tracks = main_tracks[:limit] + featured_tracks[:max(0, limit - len(main_tracks))]
                        print(f"✅ Found {len(main_tracks)} main tracks and {len(featured_tracks)} featured tracks by {artist_name}")
                        break
                    elif featured_tracks:
                        tracks = featured_tracks
                        print(f"✅ Found {len(featured_tracks)} featured tracks by {artist_name}")
                        break
                    else:
                        print(f"⚠️ No matching tracks by {artist_name} found")
                        tracks = []
                        
            except Exception as e:
                print(f"⚠️ Artist search failed for '{search_query}': {e}")
                continue
        
        # If still no tracks, try fallback with the new filtering logic
        if not tracks:
            try:
                print(f"🔄 Fallback search for: {artist_name}")
                tracks = search_track(artist_name, limit=limit * 2)
                if tracks:
                    # Apply the new filtering logic for fallback as well
                    filtered_tracks = filter_artist_tracks(tracks, artist_name, prioritize_main=True)
                    
                    main_tracks = [t for t in filtered_tracks if not t.get('is_featured', False)]
                    featured_tracks = [t for t in filtered_tracks if t.get('is_featured', False)]
                    
                    if main_tracks or featured_tracks:
                        # Boost main tracks significantly to ensure they rank higher
                        for track in main_tracks:
                            track['hybrid_score'] = track.get('hybrid_score', 0.5) + 0.5  # Strong boost for main tracks
                        # Lower featured tracks slightly  
                        for track in featured_tracks:
                            track['hybrid_score'] = track.get('hybrid_score', 0.5) - 0.2  # Lower featured tracks
                            
                        tracks = main_tracks[:limit] + featured_tracks[:max(0, limit - len(main_tracks))]
                        print(f"🔄 Fallback found {len(main_tracks)} main + {len(featured_tracks)} featured tracks")
                    else:
                        tracks = []
            except Exception as e:
                print(f"⚠️ Fallback artist search failed: {e}")
                tracks = []
    elif llm_analysis.get('language') or llm_analysis.get('region'):
        # For regional/language requests, build better search query
        regional_terms = []
        if llm_analysis.get('language'):
            regional_terms.append(llm_analysis['language'])
        if llm_analysis.get('region') and llm_analysis['region'] != llm_analysis.get('language'):
            regional_terms.append(llm_analysis['region'])
        if llm_analysis.get('mood'):
            regional_terms.append(llm_analysis['mood'])
        if llm_analysis.get('genres'):
            regional_terms.extend(llm_analysis['genres'])
            
        search_query = ' '.join(regional_terms)
        print(f"🌍 Regional/Cultural search: {search_query}")
        
        # Try multiple search strategies for regional music
        regional_searches = [
            search_query,
            llm_analysis.get('language', ''),
            f"{llm_analysis.get('language', '')} music",
            f"{llm_analysis.get('region', '')} songs"
        ]
        
        for query in regional_searches:
            if query.strip():
                print(f"🔍 Trying regional search: {query}")
                try:
                    tracks = search_track(query.strip(), limit=limit * 2)
                    if tracks and len(tracks) > 0:
                        print(f"✅ Found {len(tracks)} regional tracks with query: {query}")
                        break
                except Exception as e:
                    print(f"⚠️ Regional search failed for '{query}': {e}")
                    continue
                    
    elif llm_analysis.get('genres'):
        # Genre-based search
        genre_terms = llm_analysis['genres'][:]
        if llm_analysis.get('mood'):
            genre_terms.append(llm_analysis['mood'])
        search_query = ' '.join(genre_terms)
        print(f"🎼 Genre-based search: {search_query}")
        tracks = search_track(search_query, limit=limit * 2)
    
    # If no tracks found with primary search, try fallbacks
    if not tracks:
        print("⚠️ No tracks found, trying fallback approaches...")
        
        # Fallback 1: Try with mood + keywords
        if llm_analysis.get('mood') and llm_analysis.get('keywords'):
            fallback_query = f"{llm_analysis['mood']} {' '.join(llm_analysis['keywords'][:3])}"
            print(f"🔍 Fallback search: {fallback_query}")
            try:
                tracks = search_track(fallback_query, limit=limit)
            except Exception as e:
                print(f"⚠️ Fallback search failed: {e}")
        
        # Fallback 2: Try with just keywords
        if not tracks and llm_analysis.get('keywords'):
            keyword_query = ' '.join(llm_analysis['keywords'][:3])
            print(f"🔍 Keyword search: {keyword_query}")
            try:
                tracks = search_track(keyword_query, limit=limit)
            except Exception as e:
                print(f"⚠️ Keyword search failed: {e}")
        
        # Fallback 3: Genre-based search
        if not tracks and llm_analysis.get('genres'):
            for genre in llm_analysis['genres']:
                print(f"🔍 Individual genre search: {genre}")
                try:
                    tracks = search_track(genre, limit=limit)
                    if tracks:
                        break
                except Exception as e:
                    print(f"⚠️ Genre search failed for '{genre}': {e}")
                    continue
    
    # Final fallback to popular tracks
    if not tracks:
        print("❌ All search strategies failed, using popular tracks as fallback")
        try:
            tracks = get_top_tracks(limit)
        except Exception as e:
            print(f"❌ Even popular tracks failed: {e}")
            return []
    
    # Calculate hybrid scores and add compatibility fields
    for track in tracks:
        try:
            if track.get('artists') and not track.get('artist'):
                track['artist'] = track['artists'][0]['name']
            
            track['hybrid_score'] = calculate_hybrid_score(track, llm_analysis)
            track['relevance_score'] = track['hybrid_score'] * 0.6
            track['popularity_score'] = track.get('popularity', 0) / 100.0
        except Exception as e:
            print(f"⚠️ Error processing track: {e}")
            # Set default values
            track['hybrid_score'] = 0.5
            track['relevance_score'] = 0.3
            track['popularity_score'] = 0.5
    
    # Sort by hybrid score
    tracks.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
    
    # Determine artist diversity based on request type
    # If specific artist requested, allow more songs from that artist
    if llm_analysis.get('artist'):
        max_per_artist = min(limit, 5)  # Allow up to 5 songs from requested artist
        print(f"🎤 Artist request detected: allowing up to {max_per_artist} songs from {llm_analysis['artist']}")
    else:
        max_per_artist = 1 if limit <= 3 else 2  # Standard diversity for other requests
    
    try:
        final_tracks = enforce_artist_diversity(tracks, max_per_artist=max_per_artist)
        
        # If we don't have enough tracks and it's an artist request, be more lenient
        if len(final_tracks) < limit and len(tracks) >= limit:
            print(f"⚠️ Only {len(final_tracks)} tracks found, relaxing constraints...")
            if llm_analysis.get('artist'):
                # For artist requests, prioritize that artist's songs
                final_tracks = tracks[:limit]
            else:
                final_tracks = enforce_artist_diversity(tracks, max_per_artist=max_per_artist + 1)
    except Exception as e:
        print(f"⚠️ Artist diversity enforcement failed: {e}")
        final_tracks = tracks[:limit]  # Just take the first N tracks
    
    print(f"✅ Content-based filtering complete: {len(final_tracks)} tracks")
    
    # Log results for transparency
    try:
        artists = list(set(track.get('artist', 'Unknown') for track in final_tracks[:limit]))
        print(f"🎭 Artists in results: {', '.join(artists[:3])}{'...' if len(artists) > 3 else ''}")
    except Exception as e:
        print(f"⚠️ Error logging artists: {e}")
    
    return final_tracks[:limit]

def hybrid_recommend(user_text, user_id=None, limit=3):
    if not user_text.strip():
        return get_top_tracks(limit)
    
    try:
        print("🔍 Step 1: Analyzing your request with LLM...")
        
        cleaned_text = clean_text(user_text)
        llm_analysis = llm_semantic_analysis(cleaned_text)
        
        if llm_analysis.get('limit') and 1 <= llm_analysis['limit'] <= 20:
            limit = llm_analysis['limit']
            print(f"🎯 User requested {limit} songs")
        else:
            print(f"🎯 Using default limit: {limit} songs")
            print(f"🎯 User requested {limit} songs")
        
        print(f"🔍 LLM Analysis:")
        if llm_analysis.get('song'):
            print(f"   Song: {llm_analysis['song']}")
        if llm_analysis.get('artist'):
            print(f"   Artist: {llm_analysis['artist']}")
        if llm_analysis.get('language'):
            print(f"   Language: {llm_analysis['language']}")
        if llm_analysis.get('region'):
            print(f"   Region: {llm_analysis['region']}")
        if llm_analysis.get('genres'):
            print(f"   Genres: {', '.join(llm_analysis['genres'])}")
        if llm_analysis.get('mood'):
            print(f"   Mood: {llm_analysis['mood']}")
        if llm_analysis.get('limit'):
            print(f"   Requested Limit: {llm_analysis['limit']}")
        
        print("�� Step 2: Content-based filtering...")
        
        content_candidates = content_based_filtering(cleaned_text, llm_analysis, limit=limit)
        
        print("🎯 Step 3: Final ranking and results...")
        
        if content_candidates:
            content_candidates.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
            
            print(f"✅ Returning {len(content_candidates)} recommendations")
            
            for i, rec in enumerate(content_candidates[:3], 1):
                artist_name = rec['artists'][0]['name'] if rec.get('artists') else 'Unknown'
                print(f"   {i}. {rec.get('name', 'Unknown')} by {artist_name} (Score: {rec.get('hybrid_score', 0):.2f})")
            
            return content_candidates
        else:
            print("❌ No content-based results, using popular tracks")
            return get_top_tracks(limit)
        
    except Exception as e:
        print(f"❌ Error in hybrid recommendation: {e}")
        return get_top_tracks(limit)
