
# """Test/Alternative entrypoint for the Hybrid Recommendation System.

# This provides an enhanced CLI interface with additional features for testing
# and demonstration purposes.
# """
# import os
# from utils import load_env

# # Load environment variables
# load_env()

# from model import HybridRecommendationSystem


# def main():
#     """Enhanced test interface for the Hybrid Recommendation System."""
#     system = HybridRecommendationSystem()
    
#     print("🎵 Hybrid Music Recommendation System (Enhanced Test Mode)")
#     print("=" * 60)
#     print("Features:")
#     print("• Natural language music queries")
#     print("• Artist-specific recommendations")
#     print("• Multi-language support")
#     print("• Advanced audio feature analysis")
#     print("=" * 60)
#     print("Type a query (or 'quit'/'exit' to stop)\n")
    
#     while True:
#         try:
#             user_input = input("🎤 You: ").strip()
            
#             if not user_input:
#                 continue
                
#             # Check for exit commands
#             if user_input.lower() in {"quit", "exit", "bye", "goodbye", "stop"}:
#                 print("\n🎵 System: Goodbye! Thanks for using the hybrid system!")
#                 break
            
#             print()  # Add spacing
#             response = system.chat(user_input)
#             print(f"🎯 System: {response}\n")
#             print("─" * 80)
            
#         except KeyboardInterrupt:
#             print("\n🎵 System: Goodbye! Thanks for using the hybrid system!")
#             break
#         except Exception as e:
#             print(f"❌ System error: {e}")
#             print("Please try again.\n")
#                 token_info = response.json()
#                 self.spotify_token = token_info['access_token']
#                 self.spotify_token_expires = current_time + token_info['expires_in'] - 60
#                 return True
#             else:
#                 print(f"Spotify auth failed: {response.status_code}")
#                 return False
                
#         except Exception as e:
#             print(f"Spotify auth error: {e}")
#             return False

#     def interpret_query_with_llm(self, query: str) -> UserPreferences:
#         """Use Groq LLM to interpret natural language query into structured preferences with artist detection."""
#         try:
#             headers = {
#                 'Authorization': f'Bearer {self.groq_api_key}',
#                 'Content-Type': 'application/json'
#             }
            
#             prompt = f"""Analyze this music request and extract structured preferences: "{query}"

# CRITICAL: First determine if the user explicitly mentions a specific artist name.

# Return a JSON object with these fields:
# - is_artist_specified: Boolean (true if user mentions specific artist, false for general requests)
# - artist_name: String (exact artist name if specified, null if not)
# - genres: List of music genres - BE SMART about cultural/language contexts
# - moods: List of moods (e.g., ["happy", "sad", "energetic", "relaxing", "romantic"])
# - energy_level: Float 0-1 (0=very calm, 1=very energetic)
# - valence_level: Float 0-1 (0=very sad, 1=very happy)
# - tempo_preference: "slow", "medium", or "fast"
# - language_preference: Language if specified (e.g., "nepali", "hindi", "korean", "spanish", "japanese")
# - activity_context: Context like "workout", "study", "party", "sleep" if mentioned
# - requested_count: Number if user specifies how many songs (e.g., "3 songs", "5 tracks")

# MOOD DETECTION RULES - BE VERY CAREFUL:
# - "sad" → moods: ["sad"], valence_level: 0.2, energy_level: 0.3
# - "happy" → moods: ["happy"], valence_level: 0.8, energy_level: 0.7
# - "energetic" → moods: ["energetic"], valence_level: 0.7, energy_level: 0.9
# - "relaxing/chill" → moods: ["chill"], valence_level: 0.6, energy_level: 0.3
# - "romantic" → moods: ["romantic"], valence_level: 0.7, energy_level: 0.4

# GENRE DETECTION RULES - BE CONTEXT AWARE:
# For Nepali music: Use ["folk", "world-music", "traditional"] NOT "pop"
# For Hindi/Bollywood: Use ["bollywood", "filmi", "indian-classical"] NOT "pop"  
# For Korean: Use ["k-pop", "korean-folk"] NOT just "pop"
# For Japanese: Use ["j-pop", "japanese-folk"] NOT just "pop"
# For Arabic: Use ["arabic", "world-music", "traditional"] NOT "pop"
# For Spanish: Use ["latin", "spanish-folk", "flamenco"] NOT "pop"
# For devotional/religious: Use ["devotional", "spiritual", "folk"] NOT "pop"
# For traditional/folk requests: Use ["folk", "traditional", "world-music"] NOT "pop"

# EXAMPLES:
# - "sad songs by Taylor Swift" → is_artist_specified: true, artist_name: "Taylor Swift", moods: ["sad"], valence_level: 0.2
# - "happy pop songs" → moods: ["happy"], genres: ["pop"], valence_level: 0.8
# - "some sad nepali songs" → is_artist_specified: false, artist_name: null, moods: ["sad"], valence_level: 0.2

# ARTIST DETECTION EXAMPLES:
# - "songs by Taylor Swift" → is_artist_specified: true, artist_name: "Taylor Swift"
# - "Taylor Swift songs" → is_artist_specified: true, artist_name: "Taylor Swift"
# - "sad songs by Taylor Swift" → is_artist_specified: true, artist_name: "Taylor Swift", moods: ["sad"]

# Only return valid JSON, no explanations."""

#             payload = {
#                 "model": "llama-3.3-70b-versatile",
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.2,
#                 "max_tokens": 400
#             }
            
#             response = requests.post(self.groq_api_url, headers=headers, json=payload, timeout=15)
            
#             if response.status_code == 200:
#                 result = response.json()
#                 llm_output = result['choices'][0]['message']['content'].strip()
                
#                 # Extract JSON from response
#                 json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
#                 if json_match:
#                     prefs_dict = json.loads(json_match.group())
                    
#                     # Smart genre selection with cultural context validation
#                     genres = prefs_dict.get('genres') or ['pop']
#                     language_pref = prefs_dict.get('language_preference')
                    
#                     # Override inappropriate genre selection for cultural requests
#                     if language_pref and 'pop' in genres and len(genres) == 1:
#                         if language_pref.lower() == 'nepali':
#                             genres = ['folk', 'world-music', 'traditional']
#                         elif language_pref.lower() == 'hindi':
#                             genres = ['bollywood', 'filmi', 'indian-classical']
#                         elif language_pref.lower() == 'korean':
#                             genres = ['k-pop', 'korean-folk']
#                         elif language_pref.lower() == 'japanese':
#                             genres = ['j-pop', 'japanese-folk']
#                         elif language_pref.lower() == 'arabic':
#                             genres = ['arabic', 'world-music', 'traditional']
#                         elif language_pref.lower() == 'spanish':
#                             genres = ['latin', 'spanish-folk']
#                         elif language_pref.lower() in ['devotional', 'spiritual']:
#                             genres = ['devotional', 'spiritual', 'folk']
                    
#                     # Convert to UserPreferences format
#                     preferences = UserPreferences(
#                         genres=genres,
#                         moods=prefs_dict.get('moods') or ['neutral'],
#                         energy_level=prefs_dict.get('energy_level', 0.5),
#                         valence_level=prefs_dict.get('valence_level', 0.5),
#                         tempo_preference=prefs_dict.get('tempo_preference', 'medium'),
#                         artists_similar_to=[prefs_dict.get('artist_name')] if prefs_dict.get('artist_name') else [],
#                         decades=[],
#                         language_preference=language_pref,
#                         activity_context=prefs_dict.get('activity_context'),
#                         is_artist_specified=prefs_dict.get('is_artist_specified', False),
#                         requested_count=prefs_dict.get('requested_count')
#                     )
                    
#                     print(f"🎯 LLM extracted preferences:")
#                     print(f"   - Genres: {preferences.genres}")
#                     print(f"   - Moods: {preferences.moods}")
#                     print(f"   - Valence (sad=0.2, happy=0.8): {preferences.valence_level}")
#                     print(f"   - Energy: {preferences.energy_level}")
#                     print(f"   - Artist specified: {preferences.is_artist_specified}")
                    
#                     return preferences
                    
#         except Exception as e:
#             print(f"LLM interpretation failed: {e}")
        
#         # Fallback: Enhanced rule-based interpretation with advanced analyzer
#         return self._fallback_query_interpretation(query)

#     def _fallback_query_interpretation(self, query: str) -> UserPreferences:
#         """Enhanced fallback using Advanced Query Analyzer when available."""
        
#         # Use Advanced Query Analyzer if available
#         if self.query_analyzer:
#             try:
#                 analysis = self.query_analyzer.analyze_to_dict(query)
                
#                 # Convert analysis to UserPreferences format
#                 genres = analysis.get('genres', [])
#                 moods = analysis.get('moods', [])
#                 languages = analysis.get('languages', [])
#                 artists = analysis.get('artists', [])
#                 situations = analysis.get('situations', [])
#                 religious_cultural = analysis.get('religious_cultural', [])
                
#                 # Determine energy and valence from moods
#                 energy, valence = self._map_moods_to_audio_features(moods)
                
#                 # Determine activity context
#                 activity_context = None
#                 if situations:
#                     activity_context = situations[0]  # Take first situation
#                 elif religious_cultural:
#                     activity_context = religious_cultural[0]  # Take first religious/cultural context
                
#                 # Determine language preference
#                 language_preference = languages[0] if languages else None
                
#                 # Combine genres with cultural contexts
#                 final_genres = genres.copy()
#                 if religious_cultural:
#                     final_genres.extend(religious_cultural)
                
#                 # Fallback to pop if no genres detected
#                 if not final_genres:
#                     final_genres = ['pop']
                
#                 # Use fallback if query was too vague
#                 if analysis.get('fallback') == 'general music':
#                     moods = ['neutral']
#                     energy, valence = 0.5, 0.5
                
#                 return UserPreferences(
#                     genres=final_genres,
#                     moods=moods or ['neutral'],
#                     energy_level=energy,
#                     valence_level=valence,
#                     tempo_preference='medium',
#                     artists_similar_to=artists,
#                     decades=[],
#                     language_preference=language_preference,
#                     activity_context=activity_context
#                 )
                
#             except Exception as e:
#                 print(f"Advanced analyzer failed: {e}, using basic fallback")
        
#         # Basic fallback interpretation (original code)
#         return self._basic_fallback_interpretation(query)
    
#     def _map_moods_to_audio_features(self, moods: List[str]) -> Tuple[float, float]:
#         """Map mood keywords to energy and valence values."""
#         energy = 0.5
#         valence = 0.5
        
#         mood_mappings = {
#             'happy': (0.7, 0.8),
#             'sad': (0.3, 0.2),
#             'energetic': (0.9, 0.7),
#             'chill': (0.3, 0.6),
#             'romantic': (0.4, 0.7),
#             'focus': (0.4, 0.5),
#             'spiritual': (0.4, 0.7),
#             'nostalgic': (0.4, 0.5),
#             'aggressive': (0.9, 0.3),
#             'party': (0.8, 0.8),
#             'motivational': (0.8, 0.7)
#         }
        
#         if moods:
#             # Take the first mood's mapping
#             primary_mood = moods[0]
#             if primary_mood in mood_mappings:
#                 energy, valence = mood_mappings[primary_mood]
        
#         return energy, valence
    
#     def _basic_fallback_interpretation(self, query: str) -> UserPreferences:
#         """Enhanced fallback rule-based query interpretation with cultural and contextual awareness."""
#         query_lower = query.lower()
        
#         # Basic mood mapping with expanded detection
#         moods = []
#         energy = 0.5
#         valence = 0.5
#         activity_context = None
        
#         print(f"🔍 Fallback analysis for: '{query_lower}'")
        
#         # Emotional/Mood Detection - IMPROVED SAD DETECTION
#         if any(word in query_lower for word in ['sad', 'melancholy', 'emotional', 'heartbreak', 'crying', 'depression', 'lonely', 'sorrow', 'grief', 'blue', 'down']):
#             moods.extend(['sad', 'emotional', 'melancholy'])
#             energy, valence = 0.3, 0.2
#             print(f"🎭 Detected SAD mood from keywords in '{query_lower}'")
#         elif any(word in query_lower for word in ['happy', 'upbeat', 'energetic', 'party', 'celebration', 'festive', 'joyful', 'cheerful']):
#             moods.extend(['happy', 'upbeat', 'energetic'])
#             energy, valence = 0.8, 0.8
#         elif any(word in query_lower for word in ['sad', 'melancholy', 'emotional', 'heartbreak', 'crying', 'depression', 'lonely']):
#             moods.extend(['sad', 'emotional', 'melancholy'])
#             energy, valence = 0.3, 0.2
#         elif any(word in query_lower for word in ['relaxing', 'chill', 'calm', 'peaceful', 'soothing', 'meditation', 'zen']):
#             moods.extend(['relaxing', 'chill', 'calm'])
#             energy, valence = 0.3, 0.6
#         elif any(word in query_lower for word in ['romantic', 'love', 'valentine', 'intimate', 'sensual']):
#             moods.extend(['romantic', 'love'])
#             energy, valence = 0.4, 0.7
#         elif any(word in query_lower for word in ['nostalgic', 'memories', 'throwback', 'old times', 'vintage']):
#             moods.extend(['nostalgic', 'sentimental'])
#             energy, valence = 0.4, 0.5
#         elif any(word in query_lower for word in ['aggressive', 'angry', 'intense', 'hardcore', 'metal']):
#             moods.extend(['aggressive', 'intense'])
#             energy, valence = 0.9, 0.3
            
#         # Activity Context Detection
#         if any(word in query_lower for word in ['workout', 'gym', 'exercise', 'running', 'fitness', 'training']):
#             activity_context = 'workout'
#             if not moods: moods.extend(['energetic', 'motivational'])
#             energy = max(energy, 0.8)
#         elif any(word in query_lower for word in ['study', 'studying', 'focus', 'concentration', 'work', 'productivity']):
#             activity_context = 'study'
#             if not moods: moods.extend(['focus', 'instrumental'])
#             energy = min(energy, 0.5)
#         elif any(word in query_lower for word in ['sleep', 'sleeping', 'bedtime', 'lullaby', 'night']):
#             activity_context = 'sleep'
#             if not moods: moods.extend(['calm', 'peaceful'])
#             energy, valence = 0.2, 0.6
#         elif any(word in query_lower for word in ['driving', 'road trip', 'travel', 'journey']):
#             activity_context = 'driving'
#             if not moods: moods.extend(['upbeat', 'adventure'])
#             energy = max(energy, 0.6)
#         elif any(word in query_lower for word in ['party', 'dancing', 'club', 'nightlife']):
#             activity_context = 'party'
#             if not moods: moods.extend(['dance', 'party'])
#             energy = max(energy, 0.8)
#         elif any(word in query_lower for word in ['morning', 'wake up', 'breakfast', 'start day']):
#             activity_context = 'morning'
#             if not moods: moods.extend(['fresh', 'uplifting'])
#             energy, valence = 0.6, 0.7
            
#         # Basic genre detection (expanded)
#         genres = []
#         if 'pop' in query_lower: genres.append('pop')
#         if 'rock' in query_lower: genres.append('rock')
#         if any(term in query_lower for term in ['hip hop', 'rap', 'hiphop']): genres.append('hip-hop')
#         if 'electronic' in query_lower or 'edm' in query_lower: genres.append('electronic')
#         if 'classical' in query_lower or 'orchestra' in query_lower: genres.append('classical')
#         if 'jazz' in query_lower: genres.append('jazz')
#         if 'folk' in query_lower: genres.append('folk')
#         if 'country' in query_lower: genres.append('country')
#         if 'reggae' in query_lower: genres.append('reggae')
#         if 'blues' in query_lower: genres.append('blues')
#         if any(term in query_lower for term in ['r&b', 'rnb', 'soul']): genres.append('r&b')
#         if 'funk' in query_lower: genres.append('funk')
#         if any(term in query_lower for term in ['metal', 'heavy metal']): genres.append('metal')
#         if 'indie' in query_lower: genres.append('indie')
#         if 'alternative' in query_lower: genres.append('alternative')
        
#         # Language/Cultural detection (greatly expanded)
#         language_preference = None
#         cultural_genres = []
#         religious_context = None
        
#         # South Asian
#         if any(term in query_lower for term in ['nepali', 'nepal', 'nepalese']):
#             language_preference = 'nepali'
#             cultural_genres.extend(['folk', 'pop', 'traditional'])
#         elif any(term in query_lower for term in ['hindi', 'bollywood', 'indian', 'bharat']):
#             language_preference = 'hindi'
#             cultural_genres.extend(['bollywood', 'pop', 'classical'])
#         elif any(term in query_lower for term in ['punjabi', 'bhangra', 'sikh']):
#             language_preference = 'punjabi'
#             cultural_genres.extend(['bhangra', 'pop', 'folk'])
#         elif any(term in query_lower for term in ['tamil', 'kollywood']):
#             language_preference = 'tamil'
#             cultural_genres.extend(['tamil pop', 'folk'])
#         elif any(term in query_lower for term in ['bengali', 'bangla']):
#             language_preference = 'bengali'
#             cultural_genres.extend(['bengali pop', 'folk'])
#         elif any(term in query_lower for term in ['urdu', 'pakistan', 'qawwali']):
#             language_preference = 'urdu'
#             cultural_genres.extend(['qawwali', 'pop', 'folk'])
            
#         # East Asian
#         elif any(term in query_lower for term in ['korean', 'kpop', 'k-pop', 'korea']):
#             language_preference = 'korean'
#             cultural_genres.extend(['k-pop', 'pop', 'indie'])
#         elif any(term in query_lower for term in ['japanese', 'jpop', 'j-pop', 'japan', 'anime']):
#             language_preference = 'japanese'
#             cultural_genres.extend(['j-pop', 'pop', 'anime'])
#         elif any(term in query_lower for term in ['chinese', 'mandarin', 'cantopop', 'cpop']):
#             language_preference = 'chinese'
#             cultural_genres.extend(['c-pop', 'pop', 'traditional'])
#         elif any(term in query_lower for term in ['thai', 'thailand']):
#             language_preference = 'thai'
#             cultural_genres.extend(['thai pop', 'folk'])
            
#         # Latin/Hispanic
#         elif any(term in query_lower for term in ['spanish', 'latino', 'reggaeton', 'latin']):
#             language_preference = 'spanish'
#             cultural_genres.extend(['latin', 'reggaeton', 'pop'])
#         elif any(term in query_lower for term in ['mexican', 'mariachi', 'ranchera']):
#             language_preference = 'spanish'
#             cultural_genres.extend(['mariachi', 'ranchera', 'latin'])
#         elif any(term in query_lower for term in ['brazilian', 'portuguese', 'bossa nova', 'samba']):
#             language_preference = 'portuguese'
#             cultural_genres.extend(['bossa nova', 'samba', 'brazilian'])
            
#         # European
#         elif any(term in query_lower for term in ['french', 'chanson', 'france']):
#             language_preference = 'french'
#             cultural_genres.extend(['chanson', 'pop', 'french'])
#         elif any(term in query_lower for term in ['german', 'deutsch', 'germany']):
#             language_preference = 'german'
#             cultural_genres.extend(['german pop', 'pop'])
#         elif any(term in query_lower for term in ['italian', 'italy']):
#             language_preference = 'italian'
#             cultural_genres.extend(['italian pop', 'opera'])
#         elif any(term in query_lower for term in ['russian', 'russia']):
#             language_preference = 'russian'
#             cultural_genres.extend(['russian pop', 'folk'])
            
#         # Middle Eastern/Arabic
#         elif any(term in query_lower for term in ['arabic', 'arab', 'middle eastern']):
#             language_preference = 'arabic'
#             cultural_genres.extend(['arabic pop', 'folk', 'traditional'])
#         elif any(term in query_lower for term in ['persian', 'farsi', 'iran']):
#             language_preference = 'persian'
#             cultural_genres.extend(['persian pop', 'folk'])
#         elif any(term in query_lower for term in ['turkish', 'turkey']):
#             language_preference = 'turkish'
#             cultural_genres.extend(['turkish pop', 'folk'])
            
#         # African
#         elif any(term in query_lower for term in ['african', 'afrobeat', 'nigeria', 'ghana']):
#             language_preference = 'african'
#             cultural_genres.extend(['afrobeat', 'african', 'world'])
#         elif any(term in query_lower for term in ['swahili', 'kenya', 'tanzania']):
#             language_preference = 'swahili'
#             cultural_genres.extend(['african pop', 'folk'])
            
#         # Religious/Spiritual Context Detection
#         if any(term in query_lower for term in ['bhajan', 'kirtan', 'devotional', 'spiritual', 'mantra']):
#             religious_context = 'hindu_spiritual'
#             cultural_genres.extend(['devotional', 'spiritual', 'traditional'])
#             energy, valence = 0.4, 0.7
#         elif any(term in query_lower for term in ['gospel', 'christian', 'hymn', 'praise', 'worship']):
#             religious_context = 'christian'
#             cultural_genres.extend(['gospel', 'christian', 'contemporary christian'])
#             energy, valence = 0.6, 0.8
#         elif any(term in query_lower for term in ['christmas', 'xmas', 'holiday']):
#             religious_context = 'christmas'
#             cultural_genres.extend(['christmas', 'holiday', 'seasonal'])
#             energy, valence = 0.5, 0.8
#         elif any(term in query_lower for term in ['ramadan', 'eid', 'islamic', 'quran']):
#             religious_context = 'islamic'
#             cultural_genres.extend(['islamic', 'spiritual', 'traditional'])
#             energy, valence = 0.4, 0.7
#         elif any(term in query_lower for term in ['meditation', 'zen', 'buddhist', 'mindfulness']):
#             religious_context = 'meditation'
#             cultural_genres.extend(['meditation', 'ambient', 'new age'])
#             energy, valence = 0.2, 0.6
            
#         # Festival/Celebration Detection
#         if any(term in query_lower for term in ['diwali', 'festival of lights']):
#             religious_context = 'diwali'
#             cultural_genres.extend(['festive', 'celebration', 'traditional'])
#             energy, valence = 0.7, 0.9
#         elif any(term in query_lower for term in ['holi', 'color festival']):
#             religious_context = 'holi'
#             cultural_genres.extend(['festive', 'celebration', 'folk'])
#             energy, valence = 0.8, 0.9
#         elif any(term in query_lower for term in ['wedding', 'marriage', 'shaadi']):
#             activity_context = 'wedding'
#             cultural_genres.extend(['wedding', 'celebration', 'traditional'])
#             energy, valence = 0.7, 0.8
            
#         # Use cultural genres if detected, otherwise fallback to regular genres
#         final_genres = cultural_genres + genres if cultural_genres else (genres or ['pop'])
        
#         return UserPreferences(
#             genres=final_genres,
#             moods=moods or ['neutral'],
#             energy_level=energy,
#             valence_level=valence,
#             tempo_preference='medium',
#             artists_similar_to=[],
#             decades=[],
#             language_preference=language_preference,
#             activity_context=activity_context or religious_context,
#             is_artist_specified=False,  # Fallback doesn't detect artists
#             requested_count=None
#         )

#     def search_spotify_tracks(self, preferences: UserPreferences, limit: int = 100, specific_artist: str = None) -> List[Track]:
#         """Enhanced search with cultural, religious, and contextual awareness, plus specific artist support."""
#         if not self.get_spotify_token():
#             return []
        
#         headers = {'Authorization': f'Bearer {self.spotify_token}'}
#         all_tracks = []
#         seen_ids = set()
        
#         # Build search queries from preferences with cultural precision
#         search_queries = []
        
#         # SPECIFIC ARTIST REQUEST - Override other searches
#         if specific_artist:
#             print(f"Focusing search on specific artist: {specific_artist}")
#             # Primary artist searches - more precise queries
#             search_queries.extend([
#                 f'artist:"{specific_artist}"',  # Exact artist match
#                 f'artist:{specific_artist}',    # Artist field search
#                 f'"{specific_artist}"',         # Exact name in quotes
#             ])
            
#             # Add collaboration searches (feat., with, collab) - more specific
#             collaboration_terms = ['feat', 'featuring', 'with', 'ft']
#             for term in collaboration_terms:
#                 search_queries.extend([
#                     f'artist:"{specific_artist}" {term}',
#                     f'{term} "{specific_artist}"',
#                     f'artist:{specific_artist} {term}',
#                     f'{term} {specific_artist}',
#                 ])
            
#             # Add genre if specified
#             if preferences.genres:
#                 for genre in preferences.genres[:2]:
#                     search_queries.extend([
#                         f'artist:"{specific_artist}" genre:{genre}',
#                         f'{specific_artist} {genre}'
#                     ])
            
#             print(f"Using {len(search_queries)} artist-specific search queries")
            
#         else:
#             # Language/Cultural searches (highest priority) - More specific approach
#             if preferences.language_preference:
#                 lang = preferences.language_preference.lower()
                
#                 # Use specific cultural/regional terms first, then broader searches
#                 # Cultural/Regional variations with known Nepali artists and terms
#                 if lang == 'nepali':
#                     # Known Nepali artists and specific terms
#                     search_queries.extend([
#                         'Narayan Gopal', 'Aruna Lama', 'Ani Choying Drolma', 'Phatteman',
#                         'Bipul Chettri', 'Sugam Pokhrel', 'Pramod Kharel', 'Raju Lama',
#                         'Nepal Idol', 'Deusi Bhailo', 'Tihar songs', 'Dashain songs',
#                         'Lok Dohori', 'Adhunik Geet', 'Modern Song', 'Nepali Lok Geet',
#                         'artist:"Narayan Gopal"', 'artist:"Aruna Lama"', 'artist:"Bipul Chettri"',
#                         'genre:"world-music" nepal', 'genre:"folk" himalaya', 'nepal traditional folk'
#                     ])
#                     # Add mood + Nepali combinations
#                     for mood in preferences.moods[:2]:
#                         search_queries.extend([
#                             f'nepali {mood}', f'nepal {mood} song', f'himalayan {mood}'
#                         ])
#                 elif lang == 'hindi':
#                     search_queries.extend([
#                         'artist:"Lata Mangeshkar"', 'artist:"Kishore Kumar"', 'artist:"Arijit Singh"',
#                         'artist:"Shreya Ghoshal"', 'artist:"Rahat Fateh Ali Khan"',
#                         'bollywood', 'hindi film music', 'playback singer', 'desi music',
#                         'genre:"bollywood" hindi', 'mumbai film music'
#                     ])
#                 elif lang == 'korean':
#                     search_queries.extend([
#                         'artist:"BTS"', 'artist:"BLACKPINK"', 'artist:"IU"', 'artist:"TWICE"',
#                         'artist:"Red Velvet"', 'artist:"EXO"', 'artist:"Girls Generation"',
#                         'genre:"k-pop"', 'korean pop', 'hallyu wave', 'seoul music'
#                     ])
#                 elif lang == 'spanish':
#                     search_queries.extend([
#                         'artist:"Jesse & Joy"', 'artist:"Manu Chao"', 'artist:"Gipsy Kings"',
#                         'genre:"latin" spanish', 'hispanic music', 'latin america'
#                     ])
#                 elif lang == 'japanese':
#                     search_queries.extend([
#                         'artist:"Utada Hikaru"', 'artist:"Mr.Children"', 'artist:"ONE OK ROCK"',
#                         'genre:"j-pop"', 'japanese pop', 'anime soundtrack'
#                     ])
#                 elif lang == 'arabic':
#                     search_queries.extend([
#                         'artist:"Fairuz"', 'artist:"Um Kulthum"', 'artist:"Marcel Khalife"',
#                         'genre:"world-music" arabic', 'middle eastern music', 'oud music'
#                     ])
#                 elif lang == 'punjabi':
#                     search_queries.extend([
#                         'artist:"Gurdas Maan"', 'artist:"Diljit Dosanjh"', 'artist:"Sidhu Moose Wala"',
#                         'genre:"bhangra"', 'punjabi folk', 'dhol music'
#                     ])
#                 elif lang == 'chinese':
#                     search_queries.extend([
#                         'artist:"Jay Chou"', 'artist:"Faye Wong"', 'artist:"Teresa Teng"',
#                         'genre:"c-pop"', 'mandarin music', 'chinese ballad'
#                     ])
#                 elif lang == 'french':
#                     search_queries.extend([
#                         'artist:"Édith Piaf"', 'artist:"Charles Aznavour"', 'artist:"Stromae"',
#                         'genre:"chanson"', 'french pop', 'francophone music'
#                     ])
#                 elif lang == 'portuguese':
#                     search_queries.extend([
#                         'artist:"Caetano Veloso"', 'artist:"Gilberto Gil"', 'artist:"Marisa Monte"',
#                         'genre:"bossa-nova"', 'brazilian music', 'samba'
#                     ])
#                 elif lang == 'african':
#                     search_queries.extend([
#                         'artist:"Fela Kuti"', 'artist:"Miriam Makeba"', 'artist:"Youssou N\'Dour"',
#                         'genre:"afrobeat"', 'african traditional', 'world music africa'
#                     ])
                
#                 # Add broader cultural searches after specific ones
#                 search_queries.extend([
#                     f'genre:"world-music" {preferences.language_preference}',
#                     f'{preferences.language_preference} traditional',
#                     f'{preferences.language_preference} folk music'
#                 ])
            
#             # Activity/Context-based searches
#             if preferences.activity_context:
#                 context = preferences.activity_context.lower()
                
#                 if context == 'workout':
#                     search_queries.extend(['workout music', 'gym playlist', 'fitness', 'training music'])
#                 elif context == 'study':
#                     search_queries.extend(['study music', 'focus playlist', 'concentration', 'lo-fi'])
#                 elif context == 'sleep':
#                     search_queries.extend(['sleep music', 'lullaby', 'ambient', 'calm music'])
#                 elif context == 'driving':
#                     search_queries.extend(['road trip', 'driving music', 'highway songs'])
#                 elif context == 'party':
#                     search_queries.extend(['party music', 'dance hits', 'club music'])
#                 elif context == 'wedding':
#                     search_queries.extend(['wedding music', 'marriage songs', 'celebration'])
#                 elif context == 'morning':
#                     search_queries.extend(['morning music', 'wake up songs', 'breakfast music'])
#                 elif context == 'meditation':
#                     search_queries.extend(['meditation music', 'zen', 'spiritual', 'mindfulness'])
#                 elif context == 'hindu_spiritual':
#                     search_queries.extend(['bhajan', 'kirtan', 'devotional', 'mantra', 'spiritual'])
#                 elif context == 'christian':
#                     search_queries.extend(['gospel', 'christian music', 'worship', 'hymn'])
#                 elif context == 'christmas':
#                     search_queries.extend(['christmas music', 'holiday songs', 'xmas carols'])
#                 elif context == 'islamic':
#                     search_queries.extend(['islamic music', 'nasheed', 'spiritual'])
#                 elif context == 'diwali':
#                     search_queries.extend(['diwali songs', 'festival music', 'celebration'])
#                 elif context == 'holi':
#                     search_queries.extend(['holi songs', 'color festival', 'celebration'])
            
#             # Genre + mood combinations
#             genres = preferences.genres or []  # Handle None case
#             moods = preferences.moods or []    # Handle None case
#             for genre in genres[:3]:
#                 for mood in moods[:2]:
#                     search_queries.append(f'genre:{genre} {mood}')
            
#             # Pure genre searches
#             for genre in genres[:3]:
#                 search_queries.append(f'genre:{genre}')
            
#             # Artist-based searches
#             artists = preferences.artists_similar_to or []  # Handle None case
#             for artist in artists[:2]:
#                 search_queries.append(f'artist:{artist}')
            
#             # General mood searches
#             search_queries.extend(moods[:3])  # Use the safe moods variable
        
#         print(f"Searching with {len(search_queries)} {'artist-specific' if specific_artist else 'culturally-aware'} queries...")
        
#         # Search with cultural priority: prioritize specific cultural searches first
#         cultural_tracks = []
#         general_tracks = []
        
#         search_limit = 25 if specific_artist else 20  # More results for specific artist requests
        
#         for i, query in enumerate(search_queries[:search_limit]):
#             try:
#                 params = {
#                     'q': query,
#                     'type': 'track',
#                     'limit': min(30 if specific_artist else 25, max(10, limit // len(search_queries[:search_limit]))),
#                     'market': 'US'
#                 }
                
#                 response = requests.get('https://api.spotify.com/v1/search', 
#                                       headers=headers, params=params, timeout=10)
                
#                 if response.status_code == 200:
#                     data = response.json()
#                     tracks = data.get('tracks', {}).get('items', [])
                    
#                     for track_data in tracks:
#                         if track_data['id'] not in seen_ids:
#                             seen_ids.add(track_data['id'])
#                             track = self._convert_spotify_track(track_data)
                            
#                             # For specific artist requests, validate artist match more strictly
#                             if specific_artist:
#                                 track_artists = [artist.lower().strip() for artist in track.artists]
#                                 specific_artist_lower = specific_artist.lower().strip()
                                
#                                 # Check if the specific artist is actually in the track's artists
#                                 artist_match = False
#                                 for artist in track_artists:
#                                     if (artist == specific_artist_lower or 
#                                         specific_artist_lower in artist.split() or
#                                         any(specific_artist_lower in part.strip() for part in artist.split(' feat')) or
#                                         any(specific_artist_lower in part.strip() for part in artist.split(' featuring')) or
#                                         any(specific_artist_lower in part.strip() for part in artist.split(' with')) or
#                                         any(specific_artist_lower in part.strip() for part in artist.split(' & '))):
#                                         artist_match = True
#                                         break
                                
#                                 if artist_match:
#                                     cultural_tracks.append(track)
#                                 # Skip tracks where the artist doesn't match
#                             else:
#                                 # Prioritize cultural authenticity for non-artist requests
#                                 if preferences.language_preference and i < 10:  # First 10 queries are most culturally specific
#                                     cultural_tracks.append(track)
#                                 else:
#                                     general_tracks.append(track)
                            
#             except Exception as e:
#                 print(f"Search error for '{query}': {e}")
#                 continue
        
#         # Combine results with cultural/artist tracks first
#         all_tracks = cultural_tracks + general_tracks[:max(20, limit - len(cultural_tracks))]
        
#         if specific_artist:
#             print(f"Found {len(cultural_tracks)} tracks by/featuring {specific_artist} + {len(general_tracks)} related tracks")
#         else:
#             print(f"Found {len(cultural_tracks)} culturally-specific + {len(general_tracks)} general tracks")
        
#         print(f"Total: {len(all_tracks)} {'artist-focused' if specific_artist else 'culturally-relevant'} tracks from Spotify")
#         return all_tracks

#     def _convert_spotify_track(self, track_data: dict) -> Track:
#         """Convert Spotify API response to Track object."""
#         return Track(
#             id=track_data.get('id', ''),
#             name=track_data.get('name', 'Unknown'),
#             artists=[artist['name'] for artist in track_data.get('artists', [])],
#             album=track_data.get('album', {}).get('name', 'Unknown'),
#             popularity=track_data.get('popularity', 0),
#             duration_ms=track_data.get('duration_ms', 0),
#             explicit=track_data.get('explicit', False),
#             external_url=track_data.get('external_urls', {}).get('spotify', ''),
#             preview_url=track_data.get('preview_url'),
#             release_date=track_data.get('album', {}).get('release_date', '')
#         )

#     def get_audio_features(self, track_ids: List[str]) -> Dict[str, dict]:
#         """Fetch audio features for multiple tracks from Spotify."""
#         if not self.get_spotify_token():
#             return {}
        
#         headers = {'Authorization': f'Bearer {self.spotify_token}'}
#         features = {}
        
#         # Process in batches of 100 (Spotify limit)
#         for i in range(0, len(track_ids), 100):
#             batch = track_ids[i:i+100]
            
#             try:
#                 params = {'ids': ','.join(batch)}
#                 response = requests.get('https://api.spotify.com/v1/audio-features',
#                                       headers=headers, params=params, timeout=15)
                
#                 if response.status_code == 200:
#                     data = response.json()
#                     for feature in data.get('audio_features', []):
#                         if feature:  # Some tracks might not have features
#                             features[feature['id']] = feature
                            
#             except Exception as e:
#                 print(f"Audio features error: {e}")
#                 continue
        
#         return features

#     def enhance_tracks_with_features(self, tracks: List[Track]) -> List[Track]:
#         """Add audio features to track objects."""
#         track_ids = [track.id for track in tracks]
#         audio_features = self.get_audio_features(track_ids)
        
#         enhanced_tracks = []
#         for track in tracks:
#             if track.id in audio_features:
#                 features = audio_features[track.id]
#                 track.danceability = features.get('danceability', 0.5)
#                 track.energy = features.get('energy', 0.5)
#                 track.valence = features.get('valence', 0.5)
#                 track.acousticness = features.get('acousticness', 0.5)
#                 track.instrumentalness = features.get('instrumentalness', 0.5)
#                 track.speechiness = features.get('speechiness', 0.5)
#                 track.tempo = features.get('tempo', 120.0)
            
#             enhanced_tracks.append(track)
        
#         return enhanced_tracks

#     def fetch_sad_recommendations(self, seed_artist_name: Optional[str] = None, limit: int = 30) -> List[Track]:
#         """Fetch additional 'sad' recommendations directly from Spotify recommendations endpoint.

#         Uses low valence / moderate-low energy / moderate-slower tempo constraints.
#         Optionally seeds by an artist name for artist-focused sad tracks.
#         """
#         if not self.get_spotify_token():
#             return []
#         headers = {'Authorization': f'Bearer {self.spotify_token}'}
#         seed_artist_id = None
#         if seed_artist_name:
#             try:
#                 resp = requests.get('https://api.spotify.com/v1/search', headers=headers,
#                                     params={'q': seed_artist_name, 'type': 'artist', 'limit': 1}, timeout=10)
#                 if resp.status_code == 200:
#                     items = resp.json().get('artists', {}).get('items', [])
#                     if items:
#                         seed_artist_id = items[0]['id']
#             except Exception as e:
#                 print(f"⚠️ Artist seed lookup failed: {e}")
#         params = {
#             'limit': min(limit, 100),
#             # Conservative sad profile
#             'max_valence': 0.35,
#             'target_valence': 0.22,
#             'min_valence': 0.0,
#             'max_energy': 0.55,
#             'target_energy': 0.35,
#             'min_energy': 0.1,
#             'max_tempo': 105,
#             'target_tempo': 82,
#             'min_tempo': 40,
#         }
#         if seed_artist_id:
#             params['seed_artists'] = seed_artist_id
#         else:
#             # Provide generic seed genres if no artist seed; Spotify needs at least one seed
#             params['seed_genres'] = 'acoustic,folk,indie'
#         try:
#             r = requests.get('https://api.spotify.com/v1/recommendations', headers=headers, params=params, timeout=10)
#             if r.status_code != 200:
#                 print(f"⚠️ Sad recommendations request failed: {r.status_code}")
#                 return []
#             data = r.json()
#             rec_tracks = []
#             for item in data.get('tracks', []):
#                 try:
#                     rec_tracks.append(self._convert_spotify_track(item))
#                 except Exception:
#                     continue
#             if rec_tracks:
#                 print(f"🎯 Added {len(rec_tracks)} direct low-valence recommendations")
#             return rec_tracks
#         except Exception as e:
#             print(f"⚠️ Sad recommendation fetch error: {e}")
#             return []

#     def sequential_recommendations(self, candidate_tracks: List[Track], top_k: int = 10) -> List[Tuple[Track, float]]:
#         """Sequential modeling based on listening history patterns."""
#         if not self.user_history:
#             # Return random selection if no history
#             selected = random.sample(candidate_tracks, min(top_k, len(candidate_tracks)))
#             return [(track, 0.5) for track in selected]
        
#         print("🔄 Applying sequential modeling...")
        
#         # Analyze recent listening patterns
#         recent_history = list(self.user_history)[-50:]  # Last 50 plays
        
#         # Extract patterns from history
#         genre_counts = defaultdict(int)
#         mood_scores = {'energy': 0, 'valence': 0, 'tempo': 0}
#         artist_preferences = defaultdict(int)
        
#         # Mock analysis (in real system, you'd have actual history)
#         # For demo, we'll simulate preferences
#         for track in candidate_tracks:
#             # Calculate sequential score based on simulated patterns
#             seq_score = 0.0
            
#             # Artist diversity bonus
#             if len(set(track.artists)) > 1:
#                 seq_score += 0.1
            
#             # Popularity momentum (trending tracks)
#             if track.popularity > 70:
#                 seq_score += 0.2
            
#             # Duration preference (avoid very long/short tracks)
#             duration_min = track.duration_ms / 60000
#             if 2.5 <= duration_min <= 5.0:
#                 seq_score += 0.1
            
#             # Recent release bonus
#             if track.release_date and track.release_date.startswith('202'):
#                 seq_score += 0.15
            
#             candidate_tracks_with_scores = [(track, min(seq_score + 0.3, 1.0)) for track in candidate_tracks]
        
#         # Sort by sequential score
#         scored_tracks = sorted([(track, min(seq_score + 0.3, 1.0)) for track in candidate_tracks],
#                               key=lambda x: x[1], reverse=True)
        
#         return scored_tracks[:top_k]

#     def ranking_recommendations(self, tracks: List[Track], preferences: UserPreferences, top_k: int = 10) -> List[Tuple[Track, float]]:
#         """Ranking-based recommendations using audio features and preferences with cultural authenticity."""
#         print("Applying ranking-based scoring...")
#         print(f"🎯 Debug: Processing {len(tracks)} tracks with moods: {preferences.moods}")
#         print(f"🎯 Debug: Target valence for sad songs: {preferences.valence_level}")
        
#         scored_tracks = []
#         tracks_with_features = 0
#         tracks_without_features = 0
        
#         for track in tracks:
#             score = 0.0
            
#             # Debug: Check if track has audio features
#             has_features = hasattr(track, 'valence') and hasattr(track, 'energy')
#             if has_features and track.valence > 0:
#                 tracks_with_features += 1
#             else:
#                 tracks_without_features += 1
#                 print(f"⚠️ Track '{track.name}' missing audio features (valence: {getattr(track, 'valence', 'N/A')}, energy: {getattr(track, 'energy', 'N/A')})")
            
#             # Cultural authenticity boost (NEW)
#             if preferences.language_preference:
#                 lang = preferences.language_preference.lower()
#                 track_text = f"{track.name} {' '.join(track.artists)} {track.album}".lower()
                
#                 # Known cultural indicators for authenticity
#                 cultural_indicators = {
#                     'nepali': ['narayan gopal', 'aruna lama', 'bipul chettri', 'sugam pokhrel', 'pramod kharel', 
#                               'raju lama', 'ani choying', 'phatteman', 'kathmandu', 'nepal', 'himalayan'],
#                     'hindi': ['lata mangeshkar', 'kishore kumar', 'arijit singh', 'shreya ghoshal', 'bollywood', 
#                              'mumbai', 'delhi', 'hindi', 'bollywood'],
#                     'korean': ['bts', 'blackpink', 'twice', 'red velvet', 'exo', 'iu', 'seoul', 'korean', 'kpop'],
#                     'spanish': ['manu chao', 'gipsy kings', 'latino', 'spanish', 'hispanic', 'latin'],
#                     'japanese': ['utada hikaru', 'mr.children', 'one ok rock', 'japanese', 'jpop', 'tokyo'],
#                     'arabic': ['fairuz', 'um kulthum', 'marcel khalife', 'arabic', 'middle eastern'],
#                     'punjabi': ['gurdas maan', 'diljit dosanjh', 'sidhu moose wala', 'punjabi', 'bhangra'],
#                     'chinese': ['jay chou', 'faye wong', 'teresa teng', 'chinese', 'mandarin'],
#                     'french': ['édith piaf', 'charles aznavour', 'stromae', 'french', 'chanson'],
#                     'portuguese': ['caetano veloso', 'gilberto gil', 'marisa monte', 'brazilian', 'bossa'],
#                     'african': ['fela kuti', 'miriam makeba', 'youssou n\'dour', 'african', 'afrobeat']
#                 }
                
#                 if lang in cultural_indicators:
#                     for indicator in cultural_indicators[lang]:
#                         if indicator in track_text:
#                             score += 0.4  # Strong cultural authenticity boost
#                             break
            
#             # Popularity score (normalized, but lower weight for cultural queries)
#             weight = 0.1 if preferences.language_preference else 0.2
#             score += (track.popularity / 100) * weight
            
#             # Audio feature matching with strong mood emphasis
#             if hasattr(track, 'energy') and hasattr(track, 'valence') and track.energy >= 0 and track.valence >= 0:
#                 # Energy level matching
#                 energy_diff = abs(track.energy - preferences.energy_level)
#                 score += (1 - energy_diff) * 0.25
                
#                 # Valence (happiness) matching - CRITICAL for mood matching
#                 valence_diff = abs(track.valence - preferences.valence_level)
#                 valence_score = (1 - valence_diff) * 0.3  # Increased weight for mood matching
#                 score += valence_score
                
#                 # Strong penalty for mood mismatches (especially for sad songs)
#                 if preferences.moods and 'sad' in preferences.moods:
#                     if track.valence > 0.6:  # Too happy for sad request
#                         score *= 0.3  # Heavy penalty
#                         print(f"🎭 Applied sad mood penalty to '{track.name}' (valence: {track.valence:.2f})")
#                     elif track.valence < 0.4:  # Good match for sad
#                         score += 0.3  # Bonus for good sad match
#                         print(f"🎭 Applied sad mood bonus to '{track.name}' (valence: {track.valence:.2f})")
                
#                 # Similar penalties/bonuses for other moods
#                 elif preferences.moods and 'happy' in preferences.moods:
#                     if track.valence < 0.4:  # Too sad for happy request
#                         score *= 0.3  # Heavy penalty
#                     elif track.valence > 0.6:  # Good match for happy
#                         score += 0.3  # Bonus
                
#                 # Tempo preference matching
#                 tempo_score = 0.0
#                 if hasattr(track, 'tempo') and track.tempo > 0:
#                     if preferences.tempo_preference == 'slow' and track.tempo < 90:
#                         tempo_score = 0.8
#                     elif preferences.tempo_preference == 'medium' and 90 <= track.tempo <= 140:
#                         tempo_score = 0.8
#                     elif preferences.tempo_preference == 'fast' and track.tempo > 140:
#                         tempo_score = 0.8
#                     else:
#                         tempo_score = 0.4
                
#                 score += tempo_score * 0.15
#             else:
#                 # Fallback scoring for tracks without audio features
#                 print(f"⚠️ Using fallback scoring for '{track.name}' (no audio features)")
                
#                 # Use track name and artist analysis for mood matching
#                 track_text = f"{track.name} {' '.join(track.artists)}".lower()
                
#                 # Look for sad keywords in track names for sad requests
#                 if preferences.moods and 'sad' in preferences.moods:
#                     sad_keywords = ['sad', 'cry', 'tears', 'broken', 'heart', 'hurt', 'pain', 'sorry', 'miss', 'alone', 'lonely', 'goodbye', 'lost', 'empty', 'blue', 'down']
#                     if any(keyword in track_text for keyword in sad_keywords):
#                         score += 0.4  # Bonus for sad-sounding track names
#                         print(f"🎭 Text-based sad bonus for '{track.name}'")
            
#             # Artist preference bonus
#             for artist in track.artists:
#                 if artist.lower() in [a.lower() for a in preferences.artists_similar_to]:
#                     score += 0.3
#                     break
            
#             # Recency bonus (smaller for cultural queries)
#             recency_weight = 0.05 if preferences.language_preference else 0.1
#             if track.release_date and ('2023' in track.release_date or '2024' in track.release_date):
#                 score += recency_weight
            
#             scored_tracks.append((track, min(score, 1.0)))
        
#         print(f"🎯 Debug: {tracks_with_features} tracks with audio features, {tracks_without_features} without")
#         # Fallback: if we need sad songs but have zero featureful tracks, try recommendation endpoint
#         if tracks_with_features == 0 and preferences.moods and 'sad' in preferences.moods:
#             print("⚠️ No audio-feature data; attempting direct low-valence fetch from Spotify recommendations API...")
#             seed_artist = None
#             if preferences.artists_similar_to:
#                 seed_artist = preferences.artists_similar_to[0]
#             sad_extra = self.fetch_sad_recommendations(seed_artist_name=seed_artist, limit=25)
#             if sad_extra:
#                 # Enrich sad_extra with features for better scoring
#                 sad_extra_enhanced = self.enhance_tracks_with_features(sad_extra)
#                 # Recurse once with enriched list (avoid infinite loop by checking features)
#                 featureful = any(getattr(t, 'valence', 0) > 0 for t in sad_extra_enhanced)
#                 if featureful:
#                     print("🔄 Re-scoring with newly fetched low-valence tracks")
#                     return self.ranking_recommendations(sad_extra_enhanced, preferences, top_k)
#                 else:
#                     print("⚠️ Still missing audio features after recommendations fetch")
        
#         # Sort by ranking score
#         scored_tracks.sort(key=lambda x: x[1], reverse=True)
#         return scored_tracks[:top_k]

#     def embedding_recommendations(self, tracks: List[Track], query: str, top_k: int = 10) -> List[Tuple[Track, float]]:
#         """Text embedding-based recommendations using semantic similarity."""
#         print("Applying embedding-based matching...")
        
#         # Generate query embedding using LLM
#         query_embedding = self._generate_text_embedding(query)
#         if not query_embedding:
#             # Fallback to text similarity
#             return self._text_similarity_fallback(tracks, query, top_k)
        
#         scored_tracks = []
        
#         for track in tracks:
#             # Generate track text representation
#             track_text = f"{track.name} {' '.join(track.artists)} {track.album}"
#             track_embedding = self._generate_text_embedding(track_text)
            
#             if track_embedding:
#                 # Calculate cosine similarity
#                 similarity = self._cosine_similarity(query_embedding, track_embedding)
#                 scored_tracks.append((track, similarity))
#             else:
#                 # Fallback to basic text matching
#                 text_score = self._basic_text_similarity(query, track_text)
#                 scored_tracks.append((track, text_score))
        
#         # Sort by embedding similarity
#         scored_tracks.sort(key=lambda x: x[1], reverse=True)
#         return scored_tracks[:top_k]

#     def _generate_text_embedding(self, text: str) -> Optional[List[float]]:
#         """Generate text embedding using Groq LLM (simulated)."""
#         try:
#             # In a real implementation, you'd use a proper embedding model
#             # For now, we'll simulate embeddings based on text features
#             words = text.lower().split()
            
#             # Create a simple feature vector based on text characteristics
#             embedding = []
            
#             # Mood features
#             mood_words = {
#                 'happy': [1, 0, 0], 'sad': [0, 1, 0], 'energetic': [0, 0, 1],
#                 'chill': [0.5, 0, 0.5], 'relaxing': [0.8, 0, 0.2]
#             }
            
#             mood_vector = [0, 0, 0]
#             for word in words:
#                 if word in mood_words:
#                     for i, val in enumerate(mood_words[word]):
#                         mood_vector[i] += val
            
#             # Normalize mood vector
#             mood_sum = sum(mood_vector) or 1
#             embedding.extend([v / mood_sum for v in mood_vector])
            
#             # Genre features (simplified)
#             genre_words = ['pop', 'rock', 'hip-hop', 'electronic', 'jazz', 'classical']
#             genre_vector = [1 if genre in text.lower() else 0 for genre in genre_words]
#             embedding.extend(genre_vector)
            
#             # Text length and complexity features
#             embedding.extend([
#                 len(words) / 10,  # Length feature
#                 len(set(words)) / len(words) if words else 0,  # Diversity feature
#                 sum(1 for w in words if len(w) > 5) / len(words) if words else 0  # Complexity
#             ])
            
#             return embedding if len(embedding) == 12 else None
            
#         except Exception as e:
#             print(f"Embedding generation failed: {e}")
#             return None

#     def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
#         """Calculate cosine similarity between two vectors."""
#         if len(vec1) != len(vec2):
#             return 0.0
        
#         dot_product = sum(a * b for a, b in zip(vec1, vec2))
#         norm1 = math.sqrt(sum(a * a for a in vec1))
#         norm2 = math.sqrt(sum(b * b for b in vec2))
        
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
        
#         return dot_product / (norm1 * norm2)

#     def _text_similarity_fallback(self, tracks: List[Track], query: str, top_k: int) -> List[Tuple[Track, float]]:
#         """Fallback text similarity when embeddings fail."""
#         query_words = set(query.lower().split())
#         scored_tracks = []
        
#         for track in tracks:
#             track_text = f"{track.name} {' '.join(track.artists)} {track.album}".lower()
#             track_words = set(track_text.split())
            
#             # Jaccard similarity
#             intersection = len(query_words & track_words)
#             union = len(query_words | track_words)
#             similarity = intersection / union if union > 0 else 0
            
#             scored_tracks.append((track, similarity))
        
#         scored_tracks.sort(key=lambda x: x[1], reverse=True)
#         return scored_tracks[:top_k]

#     def _basic_text_similarity(self, query: str, track_text: str) -> float:
#         """Basic text similarity scoring."""
#         query_words = set(query.lower().split())
#         track_words = set(track_text.lower().split())
        
#         intersection = len(query_words & track_words)
#         union = len(query_words | track_words)
        
#         return intersection / union if union > 0 else 0

#     def hybrid_merge(self, sequential_results: List[Tuple[Track, float]], 
#                     ranking_results: List[Tuple[Track, float]], 
#                     embedding_results: List[Tuple[Track, float]], 
#                     preferences: UserPreferences,
#                     existing_songs: List[str] = None,
#                     specific_artist: str = None,
#                     requested_count: int = None) -> List[Track]:
#         """Enhanced merge with strict artist diversity and language filtering, but bypass diversity for specific artist requests."""
#         if specific_artist:
#             if requested_count:
#                 print(f"🔄 Merging recommendations for {requested_count} songs by {specific_artist} (diversity bypassed)...")
#             else:
#                 print(f"🔄 Merging recommendations for specific artist: {specific_artist} (diversity bypassed)...")
#         else:
#             print("🔄 Merging recommendations with enhanced rules...")
        
#         # Create a unified scoring system
#         track_scores = defaultdict(float)
#         track_objects = {}
        
#         # Add sequential scores
#         for track, score in sequential_results:
#             track_scores[track.id] += score * self.sequential_weight
#             track_objects[track.id] = track
        
#         # Add ranking scores
#         for track, score in ranking_results:
#             track_scores[track.id] += score * self.ranking_weight
#             track_objects[track.id] = track
        
#         # Add embedding scores
#         for track, score in embedding_results:
#             track_scores[track.id] += score * self.embedding_weight
#             track_objects[track.id] = track
        
#         # Apply strict language filtering (but not for specific artist requests)
#         filtered_tracks = []
#         for track_id, score in track_scores.items():
#             track = track_objects[track_id]
            
#             # STRICT ENGLISH LANGUAGE FILTERING (skip if specific artist requested)
#             if not specific_artist and preferences.language_preference and preferences.language_preference.lower() == 'english':
#                 track_text = f"{track.name} {' '.join(track.artists)} {track.album}".lower()
                
#                 # Exclude non-English content indicators
#                 non_english_indicators = [
#                     'bollywood', 'hindi', 'korean', 'kpop', 'k-pop', 'japanese', 'jpop', 'j-pop',
#                     'chinese', 'mandarin', 'cantopop', 'cpop', 'c-pop', 'arabic', 'spanish', 
#                     'french', 'german', 'nepali', 'punjabi', 'tamil', 'bengali', 'urdu', 
#                     'persian', 'turkish', 'russian', 'portuguese', 'italian', 'korean',
#                     # Artist name patterns that indicate non-English
#                     'narayan gopal', 'aruna lama', 'bipul chettri', 'arijit singh', 'shreya ghoshal',
#                     'lata mangeshkar', 'kishore kumar', 'rahat fateh', 'bts', 'blackpink', 'twice'
#                 ]
                
#                 # Skip if any non-English indicator found
#                 if any(indicator in track_text for indicator in non_english_indicators):
#                     continue
            
#             filtered_tracks.append((track_id, score))
        
#         # Sort by combined score
#         sorted_tracks = sorted(filtered_tracks, key=lambda x: x[1], reverse=True)
        
#         # ARTIST DIVERSITY HANDLING
#         final_recommendations = []
#         seen_artists = set()
#         seen_artist_variants = set()  # Track artist name variations
        
#         # Determine recommendation count based on request
#         if requested_count:
#             max_recommendations = requested_count
#         elif existing_songs:
#             max_recommendations = 3
#         else:
#             max_recommendations = 3  # Default to 3 songs when no specific count requested
        
#         # SPECIFIC ARTIST REQUEST - PURE ARTIST-FOCUSED RESULTS (NO COLLABORATIVE FILTERING)
#         if specific_artist:
#             print(f"PURE ARTIST FOCUS: Including only tracks by/featuring {specific_artist}")
            
#             # Filter tracks to only include the requested artist (primary or collaborations)
#             artist_tracks = []
#             seen_track_names = set()  # Track duplicate song titles
            
#             for track_id, score in sorted_tracks:
#                 track = track_objects[track_id]
#                 track_artists = [artist.lower().strip() for artist in track.artists]
#                 specific_artist_lower = specific_artist.lower().strip()
                
#                 # Check if requested artist is exactly one of the track's artists
#                 artist_match = False
#                 for artist in track_artists:
#                     # More strict matching for exact artist names
#                     if (artist == specific_artist_lower or 
#                         # Handle common variations like "The Artist" vs "Artist"
#                         artist.replace("the ", "") == specific_artist_lower.replace("the ", "") or
#                         # Handle featuring/collaboration patterns
#                         specific_artist_lower in artist and (
#                             "feat" in artist or "featuring" in artist or 
#                             "with" in artist or "&" in artist or "x" in artist
#                         )):
#                         artist_match = True
#                         break
                
#                 # Also check if any of the track artists contains the specific artist as a significant part
#                 if not artist_match:
#                     # Check if the artist name is a significant part of any track artist
#                     for artist in track_artists:
#                         artist_words = set(artist.split())
#                         specific_words = set(specific_artist_lower.split())
#                         # If most words of the specific artist appear in the track artist
#                         if len(specific_words.intersection(artist_words)) >= max(1, len(specific_words) * 0.7):
#                             artist_match = True
#                             break
                
#                 # Skip duplicates (same track name) - keep only the first occurrence
#                 track_name_clean = track.name.lower().strip()
#                 # Remove feat/featuring parts for comparison
#                 track_name_clean = re.sub(r'\s*\(.*?(feat|featuring).*?\)', '', track_name_clean)
#                 track_name_clean = re.sub(r'\s*feat\.?.*$', '', track_name_clean)
                
#                 if artist_match and track_name_clean not in seen_track_names:
#                     seen_track_names.add(track_name_clean)
#                     # Apply mood filtering for artist requests too
#                     mood_match = True
#                     if preferences.moods and 'sad' in preferences.moods:
#                         # Check if track has sad characteristics (low valence)
#                         if hasattr(track, 'valence') and track.valence > 0:
#                             if track.valence > 0.5:  # Not sad enough
#                                 mood_match = False
#                                 print(f"Filtered out '{track.name}' - not sad (valence: {track.valence:.2f})")
                    
#                     if mood_match:
#                         artist_tracks.append((track_id, score))
            
#             # Prioritize solo tracks over collaborations if user asked for sad songs
#             if preferences.moods and 'sad' in preferences.moods:
#                 solo_tracks = []
#                 collab_tracks = []
                
#                 for track_id, score in artist_tracks:
#                     track = track_objects[track_id]
#                     is_collab = (len(track.artists) > 1 or 
#                                'feat' in track.name.lower() or 
#                                'featuring' in track.name.lower())
                    
#                     if is_collab:
#                         collab_tracks.append((track_id, score))
#                     else:
#                         solo_tracks.append((track_id, score))
                
#                 # Prefer solo tracks first, then collaborations
#                 artist_tracks = solo_tracks + collab_tracks
#                 print(f"Prioritized {len(solo_tracks)} solo tracks over {len(collab_tracks)} collaborations for sad mood")
            
#             # Return up to max_recommendations tracks from this specific artist
#             for track_id, score in artist_tracks[:max_recommendations]:
#                 track = track_objects[track_id]
#                 final_recommendations.append(track)
                
#                 # Track the artists we've included (for logging)
#                 if track.artists:
#                     seen_artists.add(track.artists[0])
            
#             if requested_count:
#                 print(f"Found {len(final_recommendations)}/{requested_count} requested tracks by/featuring {specific_artist}")
#             else:
#                 print(f"Found {len(final_recommendations)} tracks by/featuring {specific_artist}")
            
#             # If we didn't find enough tracks and the user requested a specific count, let them know
#             if requested_count and len(final_recommendations) < requested_count:
#                 print(f"⚠️ Only found {len(final_recommendations)} out of {requested_count} requested tracks for {specific_artist}")
            
#         else:
#             # NORMAL DIVERSITY ENFORCEMENT (for non-specific requests)
#             for track_id, score in sorted_tracks:
#                 track = track_objects[track_id]
                
#                 # Get primary artist and normalize name
#                 primary_artist = track.artists[0] if track.artists else 'Unknown'
#                 normalized_artist = primary_artist.lower().strip()
                
#                 # Check for artist diversity (strict enforcement)
#                 artist_already_used = False
#                 for seen_variant in seen_artist_variants:
#                     # Check for similar artist names (handle variations like "Artist" vs "Artist feat. Someone")
#                     if (seen_variant in normalized_artist or normalized_artist in seen_variant or 
#                         any(word in seen_variant.split() for word in normalized_artist.split() if len(word) > 3)):
#                         artist_already_used = True
#                         break
                
#                 # Only allow duplicate artists if we have fewer than minimum recommendations
#                 if not artist_already_used or len(final_recommendations) < 2:
#                     final_recommendations.append(track)
#                     seen_artists.add(primary_artist)
#                     seen_artist_variants.add(normalized_artist)
                    
#                     if len(final_recommendations) >= max_recommendations:
#                         break
            
#             print(f"Final selection: {len(final_recommendations)} tracks with {len(seen_artists)} unique artists")
        
#         return final_recommendations

#     def evaluate_recommendations(self, recommendations: List[Track], ground_truth: List[str] = None) -> Dict[str, float]:
#         """Evaluate recommendation quality using multiple metrics."""
#         print("Evaluating recommendation quality...")
        
#         metrics = {}
        
#         # Diversity metrics - safely handle None values
#         artists = [track.artists[0] for track in recommendations if track.artists and len(track.artists) > 0]
#         unique_artists = len(set(artists))
#         metrics['artist_diversity'] = unique_artists / len(recommendations) if recommendations else 0
        
#         # Popularity distribution - safely handle None values
#         popularities = [track.popularity for track in recommendations if track.popularity is not None]
#         if popularities:
#             metrics['avg_popularity'] = sum(popularities) / len(popularities)
#             metrics['popularity_std'] = np.std(popularities) if len(popularities) > 1 else 0
#         else:
#             metrics['avg_popularity'] = 0
#             metrics['popularity_std'] = 0
        
#         # Audio feature diversity (if available) - safely handle None values
#         if recommendations and hasattr(recommendations[0], 'energy'):
#             energy_values = [track.energy for track in recommendations if hasattr(track, 'energy') and track.energy is not None]
#             valence_values = [track.valence for track in recommendations if hasattr(track, 'valence') and track.valence is not None]
            
#             if energy_values:
#                 metrics['energy_diversity'] = np.std(energy_values)
#                 metrics['avg_energy'] = np.mean(energy_values)
            
#             if valence_values:
#                 metrics['valence_diversity'] = np.std(valence_values)
#                 metrics['avg_valence'] = np.mean(valence_values)
        
#         # Novelty (how recent are the tracks) - safely handle None values
#         recent_tracks = sum(1 for track in recommendations 
#                           if track.release_date and ('2023' in track.release_date or '2024' in track.release_date or '2025' in track.release_date))
#         metrics['novelty'] = recent_tracks / len(recommendations) if recommendations else 0
        
#         return metrics

#     def get_hybrid_recommendations(self, query: str, existing_songs: List[str] = None, specific_artist: str = None, requested_count: int = None) -> str:
#         """Main hybrid recommendation pipeline with LLM-driven artist detection."""
#         print("Starting Enhanced Hybrid Recommendation Pipeline...")
#         print("=" * 60)
        
#         # Parse existing songs from input if provided
#         if existing_songs:
#             print(f"Found {len(existing_songs)} existing songs in input")
        
#         # Step 1: LLM Query Interpretation with Artist Detection
#         print(" Step 1: Interpreting query with LLM for artist detection...")
#         preferences = self.interpret_query_with_llm(query)
        
#         # Use LLM-detected artist information or override
#         if not specific_artist and preferences.is_artist_specified and preferences.artists_similar_to:
#             specific_artist = preferences.artists_similar_to[0]
#             print(f"LLM detected specific artist request: '{specific_artist}'")
        
#         # Use LLM-detected count or override
#         if not requested_count and preferences.requested_count:
#             requested_count = preferences.requested_count
#             print(f"LLM detected requested count: {requested_count}")
        
#         # Determine recommendation strategy
#         if specific_artist:
#             print(f"ARTIST-SPECIFIC STRATEGY: Fetching tracks by/featuring '{specific_artist}'")
#             if requested_count:
#                 print(f" Target count: {requested_count} songs")
#             else:
#                 print("Target count: All available tracks (no limit)")
#         else:
#             print("GENERAL STRATEGY: Hybrid recommendations with diversity")
#             final_count = requested_count or 3
#             print(f"   Target count: {final_count} songs (default)")
        
#         print(f"   Extracted preferences: {preferences.genres}, {preferences.moods}")
#         print(f"   Language preference: {preferences.language_preference}")
        
#         # Step 2: Conditional Spotify Search Strategy
#         if specific_artist:
#             # ARTIST-SPECIFIC SEARCH: Use Spotify artist API for precision
#             print("Step 2: Artist-specific Spotify search...")
#             candidate_tracks = self._search_artist_tracks(specific_artist, preferences, limit=100)
#         else:
#             # GENERAL SEARCH: Use hybrid content-based search
#             print(" Step 2: General hybrid content search...")
#             candidate_tracks = self.search_spotify_tracks(preferences, limit=80)
        
#         if not candidate_tracks:
#             if specific_artist:
#                 return f"❌ No tracks found for artist '{specific_artist}'.\n\n**Possible reasons:**\n• Artist name might be misspelled\n• Artist might not be available on Spotify\n• Artist might use a different name on Spotify\n\n**Try:**\n• Check the spelling of the artist name\n• Use the artist's most common or official name\n• Try a more general request like 'songs similar to [artist]'"
#             return "❌ No tracks found. Please try a different query."
        
#         # Step 3: Enhance with Audio Features
#         print("Step 3: Fetching audio features...")
#         enhanced_tracks = self.enhance_tracks_with_features(candidate_tracks)
        
#         # Step 4: Apply Recommendation Strategies Based on Type
#         print("⚡ Step 4: Applying recommendation strategies...")
        
#         if specific_artist:
#             # For artist-specific: ONLY use ranking algorithm on artist-specific tracks
#             print("   Using PURE artist-focused ranking (no collaborative filtering)")
#             print("   Skipping sequential and embedding algorithms to avoid collaborative filtering")
            
#             # Only apply ranking algorithm to rank tracks by the specific artist
#             ranking_results = self.ranking_recommendations(enhanced_tracks, preferences, top_k=50)
            
#             # Create empty results for other algorithms to maintain interface consistency
#             sequential_results = []
#             embedding_results = []
            
#             print(f"   Artist-specific ranking: {len(ranking_results)} candidates")
#             print(f"   Sequential: 0 candidates (skipped for artist-specific requests)")
#             print(f"   Embedding: 0 candidates (skipped for artist-specific requests)")
#         else:
#             # For general: Use full hybrid approach with diversity
#             print("   Using full hybrid approach with diversity")
#             sequential_results = self.sequential_recommendations(enhanced_tracks, top_k=15)
#             ranking_results = self.ranking_recommendations(enhanced_tracks, preferences, top_k=15)
#             embedding_results = self.embedding_recommendations(enhanced_tracks, query, top_k=15)
            
#             print(f"   Sequential: {len(sequential_results)} candidates")
#             print(f"   Ranking: {len(ranking_results)} candidates")
#             print(f"   Embedding: {len(embedding_results)} candidates")
        
#         # Step 5: Enhanced Hybrid Merge with Strategy-Aware Rules
#         print("🔄 Step 5: Merging with strategy-aware filtering...")
#         final_recommendations = self.hybrid_merge(sequential_results, ranking_results, embedding_results, 
#                                                 preferences, existing_songs, specific_artist, requested_count)
        
#         # Step 6: Evaluation
#         print("Step 6: Evaluating recommendation quality...")
#         metrics = self.evaluate_recommendations(final_recommendations)
        
#         # Step 7: Format Results with Strategy Indication
#         return self.format_enhanced_results(final_recommendations, metrics, preferences, existing_songs, specific_artist, requested_count)

#     def _search_artist_tracks(self, artist_name: str, preferences: UserPreferences, limit: int = 100) -> List[Track]:
#         """Search for tracks by a specific artist using Spotify Artist API for precision."""
#         if not self.get_spotify_token():
#             return []
            
#         # Special handling for "different artists" query
#         if artist_name and any(artist_name.lower() == term for term in ["different artists", "different", "various artists", "various"]):
#             print("'Different artists' is not a specific artist - searching for varied artists instead")
#             # Return empty list to trigger general recommendation logic with explicit unique artist flag
#             preferences.is_artist_specified = False  # Override any previous setting
#             preferences.artists_similar_to = []  # Clear any artist references
            
#             # Add a genre to help with diversity
#             if not preferences.genres:
#                 preferences.genres = ["pop"]
                
#             return []
        
#         headers = {'Authorization': f'Bearer {self.spotify_token}'}
#         all_tracks = []
#         seen_ids = set()
        
#         print(f"🔍 Searching for artist: {artist_name}")
        
#         try:
#             # Step 1: Find the artist using Spotify's artist search
#             artist_search_params = {
#                 'q': artist_name,
#                 'type': 'artist',
#                 'limit': 10
#             }
            
#             artist_response = requests.get('https://api.spotify.com/v1/search', 
#                                          headers=headers, params=artist_search_params, timeout=10)
            
#             if artist_response.status_code != 200:
#                 print(f"Artist search failed: {artist_response.status_code}")
#                 return []
            
#             artist_data = artist_response.json()
#             artists = artist_data.get('artists', {}).get('items', [])
            
#             if not artists:
#                 print(f"No artist found matching '{artist_name}'")
#                 return []
            
#             # Find the best matching artist (exact match or closest)
#             target_artist = None
#             for artist in artists:
#                 if artist['name'].lower() == artist_name.lower():
#                     target_artist = artist
#                     break
            
#             if not target_artist:
#                 target_artist = artists[0]  # Use first result as fallback
            
#             artist_id = target_artist['id']
#             artist_name_exact = target_artist['name']
            
#             print(f"Found artist: {artist_name_exact} (ID: {artist_id})")
            
#             # Step 2: Get top tracks by this artist
#             top_tracks_response = requests.get(f'https://api.spotify.com/v1/artists/{artist_id}/top-tracks',
#                                              headers=headers, params={'market': 'US'}, timeout=10)
            
#             if top_tracks_response.status_code == 200:
#                 top_tracks_data = top_tracks_response.json()
#                 for track_data in top_tracks_data.get('tracks', []):
#                     if track_data['id'] not in seen_ids:
#                         seen_ids.add(track_data['id'])
#                         track = self._convert_spotify_track(track_data)
#                         all_tracks.append(track)
            
#             # Step 3: Get albums by this artist and extract tracks
#             albums_response = requests.get(f'https://api.spotify.com/v1/artists/{artist_id}/albums',
#                                          headers=headers, 
#                                          params={'market': 'US', 'limit': 20, 'include_groups': 'album,single'}, 
#                                          timeout=10)
            
#             if albums_response.status_code == 200:
#                 albums_data = albums_response.json()
#                 for album in albums_data.get('items', []):
#                     album_id = album['id']
                    
#                     # Get tracks from this album
#                     album_tracks_response = requests.get(f'https://api.spotify.com/v1/albums/{album_id}/tracks',
#                                                        headers=headers, params={'limit': 50}, timeout=10)
                    
#                     if album_tracks_response.status_code == 200:
#                         album_tracks_data = album_tracks_response.json()
#                         for track_data in album_tracks_data.get('items', []):
#                             if track_data['id'] not in seen_ids:
#                                 # Add album info to track data for conversion
#                                 track_data['album'] = album
#                                 track_data['external_urls'] = {'spotify': f"https://open.spotify.com/track/{track_data['id']}"}
#                                 track_data['popularity'] = 50  # Default popularity for album tracks
                                
#                                 # Verify this track is actually by the requested artist
#                                 # (since some albums might have guest artists)
#                                 track_artists = [artist['name'].lower() for artist in track_data.get('artists', [])]
#                                 if artist_name_exact.lower() in track_artists:
#                                     seen_ids.add(track_data['id'])
#                                     track = self._convert_spotify_track(track_data)
#                                     all_tracks.append(track)
                                    
#                                     if len(all_tracks) >= limit:
#                                         break
                    
#                     if len(all_tracks) >= limit:
#                         break
            
#             # Step 4: Search for collaborations/features
#             collab_queries = [
#                 f'artist:"{artist_name_exact}" feat',
#                 f'feat "{artist_name_exact}"',
#                 f'artist:"{artist_name_exact}" with',
#                 f'with "{artist_name_exact}"',
#                 f'"{artist_name_exact}" collaboration'
#             ]
            
#             for query in collab_queries[:3]:  # Limit collaboration searches
#                 try:
#                     params = {
#                         'q': query,
#                         'type': 'track',
#                         'limit': 20,
#                         'market': 'US'
#                     }
                    
#                     response = requests.get('https://api.spotify.com/v1/search', 
#                                           headers=headers, params=params, timeout=10)
                    
#                     if response.status_code == 200:
#                         data = response.json()
#                         tracks = data.get('tracks', {}).get('items', [])
                        
#                         for track_data in tracks:
#                             if track_data['id'] not in seen_ids:
#                                 # Verify the artist is actually in this track
#                                 track_artists = [artist['name'].lower() for artist in track_data['artists']]
#                                 if artist_name_exact.lower() in [ta for ta in track_artists]:
#                                     seen_ids.add(track_data['id'])
#                                     track = self._convert_spotify_track(track_data)
#                                     all_tracks.append(track)
                                    
#                                     if len(all_tracks) >= limit:
#                                         break
                    
#                     if len(all_tracks) >= limit:
#                         break
                        
#                 except Exception as e:
#                     print(f"Collaboration search error: {e}")
#                     continue
            
#             print(f"Found {len(all_tracks)} tracks by/featuring {artist_name_exact}")
            
#         except Exception as e:
#             print(f"❌ Artist search failed: {e}")
#             # Fallback to general search
#             return self.search_spotify_tracks(preferences, limit=limit, specific_artist=artist_name)
        
#         return all_tracks

#     def format_enhanced_results(self, tracks: List[Track], metrics: Dict[str, float], 
#                               preferences: UserPreferences, existing_songs: List[str] = None, 
#                               specific_artist: str = None, requested_count: int = None) -> str:
#         """Enhanced formatting with structured output and language information."""
#         if not tracks:
#             if specific_artist:
#                 if requested_count:
#                     return f"I couldn't find {requested_count} tracks by '{specific_artist}'. Please check the artist name and try again!"
#                 return f"I couldn't find tracks by '{specific_artist}'. Please check the artist name and try again!"
#             return "I couldn't find tracks matching your preferences. Try a different query or let me know what specific culture/genre interests you!"
        
#         result = "**Enhanced Music Recommendations**\n"
#         result += "=" * 50 + "\n\n"
        
#         # Decide whether to hide generic genre output (e.g., only 'pop' inferred for artist-specific request)
#         hide_genre = (
#             specific_artist is not None and
#             preferences.genres and len(preferences.genres) == 1 and
#             preferences.genres[0].lower() == 'pop'
#         )
        
#         # Show analysis summary with strategy indication
#         result += "**Analysis Summary:**\n"
#         if specific_artist:
#             if requested_count:
#                 result += f"   **Strategy**: Artist-Specific Search ({requested_count} songs by {specific_artist})\n"
#             else:
#                 result += f"   **Strategy**: Artist-Specific Search (All available tracks by {specific_artist})\n"
#         else:
#             final_count = requested_count or 3
#             result += f"   **Strategy**: General Hybrid Recommendations ({final_count} diverse tracks)\n"
            
#         if preferences.language_preference:
#             result += f"   **Language**: {preferences.language_preference.title()}\n"
#         if preferences.genres and not hide_genre:
#             result += f"   **Genres**: {', '.join(preferences.genres[:3])}\n"
#         if preferences.moods:
#             result += f"   **Mood**: {', '.join(preferences.moods[:2])}\n"
#         if hide_genre:
#             result += "   **Note**: Genre omitted (generic placeholder removed)\n"
#         if preferences.activity_context:
#             result += f"   **Context**: {preferences.activity_context.replace('_', ' ').title()}\n"
        
#         # Input-based recommendation count
#         rec_count = len(tracks)
#         if existing_songs:
#             result += f"   **Input Songs**: {len(existing_songs)} provided → Recommending {rec_count} additional songs\n"
#         else:
#             if specific_artist:
#                 if requested_count:
#                     result += f"   **Specific Request**: Found {rec_count}/{requested_count} songs by/featuring {specific_artist}\n"
#                 else:
#                     result += f"   **Artist Focus**: Recommending {rec_count} songs by/featuring {specific_artist}\n"
#             else:
#                 result += f"   **New Discovery**: Recommending {rec_count} songs\n"
#         result += "\n"
        
#         # Structured recommendations with enhanced metadata
#         if specific_artist:
#             result += f"**Songs by/featuring {specific_artist}:**\n\n"
#         else:
#             result += "**Structured Recommendations:**\n\n"
        
#         for i, track in enumerate(tracks, 1):
#             artists_str = ", ".join(track.artists)
            
#             # Determine language from track characteristics
#             track_language = self._determine_track_language(track, preferences.language_preference)
            
#             # Determine primary genre only if not hidden
#             if not hide_genre:
#                 primary_genre = self._determine_primary_genre(track, preferences.genres)
            
#             # Highlight collaborations for specific artist requests
#             collaboration_indicator = ""
#             if specific_artist:
#                 if len(track.artists) > 1:
#                     collaboration_indicator = " (Collab)"  # Collaboration indicator
            
#             # Structured output format
#             result += f"**{i}. {track.name}{collaboration_indicator}**\n"
#             result += f"   **Artist**: {artists_str}\n"
#             if not hide_genre:
#                 result += f"   **Genre**: {primary_genre}\n"
#             result += f"   **Language**: {track_language}\n"
#             result += f"   **Album**: {track.album}\n"
            
#             # Popularity visualization
#             popularity_bar = "●" * (track.popularity // 20) + "○" * (5 - track.popularity // 20)
#             result += f"   **Popularity**: {popularity_bar} ({track.popularity}/100)\n"
            
#             # Duration
#             duration_min = track.duration_ms // 60000
#             duration_sec = (track.duration_ms // 1000) % 60
#             result += f"   **Duration**: {duration_min}:{duration_sec:02d}\n"
            
#             # Audio features (if available)
#             if hasattr(track, 'energy') and track.energy > 0:
#                 result += f"   **Energy**: {track.energy:.2f} | **Mood**: {track.valence:.2f}\n"
            
#             # Links
#             result += f"   **Spotify**: {track.external_url}\n"
#             if track.preview_url:
#                 result += f"   **Preview**: {track.preview_url}\n"
#             result += "\n"
        
#         # Enhanced Quality Metrics
#         result += "📈 **Quality Metrics:**\n"
#         if specific_artist:
#             result += f"   � **Artist Focus**: 100% tracks by/featuring {specific_artist}\n"
#             collaboration_count = sum(1 for track in tracks if len(track.artists) > 1)
#             if collaboration_count > 0:
#                 result += f"   🤝 **Collaborations**: {collaboration_count}/{len(tracks)} tracks feature other artists\n"
#         else:
#             result += f"   �🎨 **Artist Diversity**: {metrics.get('artist_diversity', 0):.2f} (Perfect: 1.0)\n"
        
#         result += f"   **Avg Popularity**: {metrics.get('avg_popularity', 0):.1f}/100\n"
#         result += f"   **Recent Content**: {metrics.get('novelty', 0):.1%}\n"
        
#         if preferences.language_preference and preferences.language_preference.lower() == 'english':
#             result += f"   **Language Filter**: English-only content verified\n"
#         elif preferences.language_preference:
#             result += f"   **Cultural Focus**: {preferences.language_preference.title()} music prioritized\n"
        
#         # Closing message
#         if specific_artist:
#             if requested_count:
#                 if len(tracks) == requested_count:
#                     result += f"\nPerfect! Found all {requested_count} requested tracks by/featuring {specific_artist}!"
#                 else:
#                     result += f"\nFound {len(tracks)}/{requested_count} requested tracks by/featuring {specific_artist}!"
#                     if len(tracks) < requested_count:
#                         result += f" (Limited by available content)"
#             elif existing_songs:
#                 result += f"\nBased on your {len(existing_songs)} input songs, here are {len(tracks)} tracks by/featuring {specific_artist}!"
#             else:
#                 result += f"\nHere are {len(tracks)} amazing tracks by/featuring {specific_artist}!"
                
#             if any(len(track.artists) > 1 for track in tracks):
#                 result += f" Including collaborations!"
#         else:
#             if existing_songs:
#                 result += f"\nBased on your {len(existing_songs)} input songs, here are {len(tracks)} carefully selected additions!"
#             else:
#                 result += f"\nDiscovered {len(tracks)} amazing tracks just for you!"
            
#         result += "\n\n*Powered by Enhanced Cultural AI with Artist Diversity & Language Filtering*"
        
#         return result

#     def _determine_track_language(self, track: Track, preference_lang: str = None) -> str:
#         """Determine track language based on artist and track information."""
#         if preference_lang:
#             return preference_lang.title()
        
#         track_text = f"{track.name} {' '.join(track.artists)} {track.album}".lower()
        
#         # Language detection patterns
#         language_patterns = {
#             'Korean': ['korean', 'kpop', 'k-pop', 'bts', 'blackpink', 'twice', 'red velvet', 'exo'],
#             'Hindi': ['bollywood', 'hindi', 'arijit singh', 'shreya ghoshal', 'lata mangeshkar'],
#             'Nepali': ['nepali', 'narayan gopal', 'aruna lama', 'bipul chettri', 'himalayan'],
#             'Spanish': ['spanish', 'latino', 'reggaeton', 'latin'],
#             'Japanese': ['japanese', 'jpop', 'j-pop', 'anime', 'utada hikaru'],
#             'Chinese': ['chinese', 'mandarin', 'cpop', 'c-pop', 'jay chou'],
#             'Arabic': ['arabic', 'middle eastern', 'fairuz', 'um kulthum'],
#             'French': ['french', 'chanson', 'stromae', 'édith piaf'],
#             'Portuguese': ['portuguese', 'brazilian', 'bossa nova', 'caetano veloso']
#         }
        
#         for language, patterns in language_patterns.items():
#             if any(pattern in track_text for pattern in patterns):
#                 return language
        
#         return 'English'  # Default assumption
    
#     def _determine_primary_genre(self, track: Track, preference_genres: List[str]) -> str:
#         """Determine primary genre from preferences or track characteristics."""
#         if preference_genres:
#             return preference_genres[0].title()
        
#         track_text = f"{track.name} {' '.join(track.artists)} {track.album}".lower()
        
#         # Genre detection patterns
#         genre_patterns = {
#             'K-Pop': ['kpop', 'k-pop', 'korean pop'],
#             'Bollywood': ['bollywood', 'hindi film'],
#             'Folk': ['folk', 'traditional', 'acoustic'],
#             'Pop': ['pop', 'mainstream'],
#             'Rock': ['rock', 'alternative'],
#             'Hip-Hop': ['hip hop', 'rap', 'hiphop'],
#             'Electronic': ['electronic', 'edm', 'dance'],
#             'Classical': ['classical', 'orchestra'],
#             'Jazz': ['jazz'],
#             'Blues': ['blues'],
#             'Country': ['country'],
#             'Reggae': ['reggae'],
#             'Latin': ['latin', 'reggaeton', 'salsa'],
#             'World': ['world music', 'ethnic', 'cultural']
#         }
        
#         for genre, patterns in genre_patterns.items():
#             if any(pattern in track_text for pattern in patterns):
#                 return genre
        
#         return 'Pop'  # Default genre

#     def parse_input_songs(self, query: str) -> Tuple[str, List[str], str, int]:
#         """Parse input to extract existing songs, clean query, specific artist requests, and requested count."""
#         # Common song indication patterns
#         song_indicators = [
#             r'(?:songs?|tracks?)\s*(?:like|similar to|such as)?\s*["\']([^"\']+)["\']',
#             r'(?:artist|by)\s+([A-Za-z\s&]+)(?:\s*-\s*|\s+)([A-Za-z\s]+)',
#             r'([A-Za-z\s]+)\s*by\s+([A-Za-z\s&]+)',
#             r'["\']([^"\']+)["\'](?:\s*by\s+([A-Za-z\s&]+))?',
#         ]
        
#         # Artist-specific request patterns with count detection
#         artist_request_patterns = [
#             r'(\d+)\s+(?:songs?|tracks?|music)\s+(?:by|from)\s+(.+?)(?:\s*$)',  # "3 songs by Artist"
#             r'(?:songs?|tracks?|music)\s+(?:by|from)\s+(.+?)(?:\s*$)',
#             r'(?:play|recommend|find|get|give)\s+(?:me\s+)?(\d+)?\s*(.+?)(?:\s+(?:songs?|tracks?|music))?(?:\s*$)',  # "play 5 Artist songs"
#             r'(?:play|recommend|find|get|give)\s+(?:me\s+)?(.+?)(?:\s+(?:songs?|tracks?|music))?(?:\s*$)',
#             r'(.+?)(?:\'s)?\s+(?:songs?|tracks?|music)(?:\s*$)',
#             r'(?:artist|singer):\s*(.+?)(?:\s*$)',
#             r'(?:only|just)\s+(\d+)?\s*(.+?)(?:\s+(?:songs?|tracks?))?(?:\s*$)'  # "just 3 Artist songs"
#         ]
        
#         existing_songs = []
#         clean_query = query
#         specific_artist = None
#         requested_count = None
        
#         # Check for specific artist requests first
#         for pattern in artist_request_patterns:
#             match = re.search(pattern, query, re.IGNORECASE)
#             if match:
#                 groups = match.groups()
                
#                 # Handle different pattern structures
#                 if len(groups) == 2 and groups[0] is not None and groups[0].isdigit():
#                     # Pattern: "3 songs by Artist"
#                     requested_count = int(groups[0])
#                     artist_name = groups[1].strip()
#                 elif len(groups) == 2 and groups[0] is not None and not groups[0].isdigit():
#                     # Pattern: "play Artist" or similar
#                     if groups[1] is not None and groups[1].isdigit():
#                         requested_count = int(groups[1])
#                         artist_name = groups[0].strip()
#                     else:
#                         artist_name = groups[0].strip()
#                 elif len(groups) == 1 and groups[0] is not None:
#                     # Single capture group
#                     artist_name = groups[0].strip()
#                 else:
#                     continue
                
#                 # Clean up the artist name
#                 artist_name = re.sub(r'\s+', ' ', artist_name)  # Normalize spaces
                
#                 # Clean up common false positives
#                 if artist_name.lower().startswith('just '):
#                     artist_name = artist_name[5:]
#                 if artist_name.lower().startswith('only '):
#                     artist_name = artist_name[5:]
#                 if artist_name.lower().endswith(' songs'):
#                     artist_name = artist_name[:-6]
#                 if artist_name.lower().endswith(' tracks'):
#                     artist_name = artist_name[:-7]
#                 if artist_name.lower().endswith(' music'):
#                     artist_name = artist_name[:-6]
                
#                 # Remove any remaining numbers from artist name
#                 artist_name = re.sub(r'\b\d+\b', '', artist_name).strip()
                
#                 # Validate it looks like an artist name (not too generic)
#                 if (len(artist_name.split()) <= 5 and len(artist_name) > 1 and 
#                     not any(generic in artist_name.lower() for generic in 
#                            ['music', 'song', 'track', 'playlist', 'album', 'genre', 'some', 'any', 'good', 'best', 'me',
#                             'relaxing', 'evening', 'morning', 'night', 'chill', 'upbeat', 'happy', 'sad', 'energetic',
#                             'romantic', 'dance', 'workout', 'study', 'sleep', 'party', 'driving', 'meditation',
#                             'nepali', 'hindi', 'korean', 'spanish', 'japanese', 'arabic', 'chinese', 'punjabi',
#                             'french', 'english', 'pop', 'rock', 'hip', 'rap', 'jazz', 'classical', 'folk', 'country',
#                             'electronic', 'indie', 'alternative', 'metal', 'blues', 'reggae', 'latin', 'r&b', 'soul'])):
#                     specific_artist = artist_name
                    
#                     if requested_count:
#                         print(f"Detected specific request: {requested_count} songs by '{specific_artist}'")
#                     else:
#                         print(f"Detected specific artist request: '{specific_artist}'")
                    
#                     clean_query = f"music by {specific_artist}"  # Simplify query for processing
#                     break
        
#         # Parse existing songs
#         for pattern in song_indicators:
#             matches = re.findall(pattern, query, re.IGNORECASE)
#             for match in matches:
#                 if isinstance(match, tuple):
#                     if len(match) == 2 and match[0] and match[1]:
#                         song_info = f"{match[0].strip()} by {match[1].strip()}"
#                         existing_songs.append(song_info)
#                 else:
#                     existing_songs.append(match.strip())
        
#         # Clean the query by removing song references for better analysis
#         for indicator in ['songs like', 'similar to', 'such as', 'tracks like']:
#             clean_query = re.sub(rf'{indicator}[^,]*,?', '', clean_query, flags=re.IGNORECASE)
        
#         # Remove quotes and artist references
#         clean_query = re.sub(r'["\']([^"\']+)["\'](?:\s*by\s+[^,]*)?', '', clean_query)
#         clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        
#         return clean_query, list(set(existing_songs)), specific_artist, requested_count

#     def recommend_music(self, query: str) -> str:
#         """Main entry point for music recommendations with enhanced rules."""
#         print("Enhanced Music Recommendation Assistant")
#         print("=" * 50)
        
#         # Process the query using similarity matching module or fallback
#         try:
#             from similarity_matching import process_query
#             clean_query, song_references, specific_artist, requested_count = process_query(query)
#             existing_songs = song_references
#         except ImportError:
#             # Fallback to original parse_input_songs method
#             clean_query, existing_songs, specific_artist, requested_count = self.parse_input_songs(query)
        
#         if specific_artist:
#             if requested_count:
#                 print(f"Detected specific request: {requested_count} songs by '{specific_artist}'")
#             else:
#                 print(f"Detected specific artist request: '{specific_artist}'")
#             print(f"Will search for songs by/featuring this artist")
#         elif existing_songs:
#             print(f"Detected {len(existing_songs)} existing songs in input:")
#             for song in existing_songs:
#                 print(f"   • {song}")
#             print(f"🔍 Processing clean query: '{clean_query}'")
#         else:
#             print(f"🔍 Processing new music discovery query: '{query}'")
        
#         # Get recommendations with enhanced rules
#         return self.get_hybrid_recommendations(clean_query, existing_songs, specific_artist, requested_count)

#     def normal_chat_response(self, query: str) -> str:
#         """Generate intelligent chat responses using GROQ LLM."""
#         try:
#             headers = {
#                 'Authorization': f'Bearer {self.groq_api_key}',
#                 'Content-Type': 'application/json'
#             }
            
#             prompt = f"""You are a friendly AI assistant specializing in music and recommendations. The user said: "{query}"

# Respond naturally and conversationally. Keep it brief (1-2 sentences), friendly, and engaging. 
# If appropriate, mention that you can provide sophisticated music recommendations using hybrid AI algorithms.
# Don't be overly formal - be casual and warm."""

#             payload = {
#                 "model": "llama3-8b-8192",
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.8,
#                 "max_tokens": 150
#             }
            
#             response = requests.post(self.groq_api_url, headers=headers, json=payload, timeout=10)
            
#             if response.status_code == 200:
#                 result = response.json()
#                 if 'choices' in result and result['choices']:
#                     return result['choices'][0]['message']['content'].strip()
                    
#         except Exception as e:
#             print(f"LLM chat failed: {e}")
            
#         # Fallback responses
#         query_lower = query.lower().strip()
        
#         if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'greetings']):
#             responses = [
#                 "Hello! I'm your AI music assistant with hybrid recommendation algorithms. What can I help you discover today?",
#                 "Hi there! Ready to find some amazing music using advanced AI recommendations?",
#                 "Hey! I use sequential, ranking, and embedding models to find perfect music matches. What's your mood?",
#             ]
#             return random.choice(responses)
        
#         if any(phrase in query_lower for phrase in ['how are you', 'whats up']):
#             responses = [
#                 "I'm doing great! Just fine-tuning my hybrid recommendation algorithms. How can I help you find some music?",
#                 "All systems go! My sequential, ranking, and embedding models are ready to find you perfect tracks. What are you in the mood for?",
#                 "Fantastic! I'm excited to use my advanced music AI to recommend something amazing. What genre interests you?"
#             ]
#             return random.choice(responses)
        
#         if any(word in query_lower for word in ['help', 'what can you do', 'capabilities']):
#             return """**I'm an Advanced Hybrid Music Recommendation System!**

# My capabilities include:
# **LLM Query Understanding** - I interpret natural language like "relaxing evening songs"
# **Sequential Modeling** - Learn from listening patterns and history
# **Ranking Algorithms** - Score tracks using audio features and preferences  
# **Embedding Similarity** - Semantic matching using vector representations
# **Spotify Integration** - Real track data, audio features, and metadata
# **Quality Evaluation** - Diversity, novelty, and satisfaction metrics

# Just describe what you want to hear and I'll use all three AI approaches to find perfect matches!"""
        
#         # Default responses
#         responses = [
#             "That's interesting! I'd love to help you discover some music that matches your vibe. What are you in the mood for?",
#             "Cool! While we chat, feel free to ask me for music recommendations - I use advanced hybrid AI algorithms to find perfect matches.",
#             "Thanks for sharing! Want to discover some new music? I can analyze your preferences and find amazing tracks.",
#             f"I appreciate that! Speaking of '{query}' - there might be some great songs related to that theme. Want me to find some?"
#         ]
        
#         return random.choice(responses)

#     def simulate_listening_history(self, tracks: List[Track]):
#         """Simulate adding tracks to listening history (for demo purposes)."""
#         for track in tracks:
#             history_item = ListeningHistory(
#                 track_id=track.id,
#                 timestamp=int(time.time()),
#                 play_duration_ms=track.duration_ms,
#                 skipped=random.choice([True, False]),
#                 liked=random.choice([True, False]),
#                 context='recommendation'
#             )
#             self.user_history.append(history_item)

#     def chat(self, query: str) -> str:
#         """Enhanced main chat interface with ambiguity handling and cultural awareness."""
#         if not query.strip():
#             return "Please ask me something! I can chat or provide advanced music recommendations using hybrid AI"
        
#         qlower = query.lower()
#         if any(phrase in qlower for phrase in ["list genres", "available genres", "show genres", "genre seeds"]):
#             genres = self.get_available_genres()
#             if not genres:
#                 return "Couldn't fetch Spotify genre list right now. Try again later."
#             # Compact formatting in columns
#             cols = 4
#             lines = []
#             width = max(len(g) for g in genres)
#             for i in range(0, len(genres), cols):
#                 chunk = genres[i:i+cols]
#                 lines.append("  " + "  |  ".join(g.ljust(width) for g in chunk))
#             return "**Spotify Available Genre Seeds** (" + str(len(genres)) + ")\n" + "\n".join(lines)
        
#         # Check if it's a music query
#         if self.is_music_query(query):
#             # Check for ambiguous queries that need clarification
#             ambiguous_response = self._check_for_ambiguous_query(query)
#             if ambiguous_response:
#                 return ambiguous_response
            
#             return self.get_hybrid_recommendations(query)
#         else:
#             return self.normal_chat_response(query)

#     def get_available_genres(self) -> List[str]:
#         """Fetch and cache Spotify's available genre seeds list."""
#         if self.available_genres_cache is not None:
#             return self.available_genres_cache
#         if not self.get_spotify_token():
#             return []
#         try:
#             headers = {'Authorization': f'Bearer {self.spotify_token}'}
#             resp = requests.get('https://api.spotify.com/v1/recommendations/available-genre-seeds', headers=headers, timeout=10)
#             if resp.status_code == 200:
#                 data = resp.json()
#                 genres = data.get('genres', [])
#                 genres.sort()
#                 self.available_genres_cache = genres
#                 print(f"Fetched {len(genres)} Spotify genre seeds")
#                 return genres
#             else:
#                 print(f"Genre seed fetch failed: {resp.status_code}")
#         except Exception as e:
#             print(f"Genre seed fetch error: {e}")
#         return []
    
#     def _check_for_ambiguous_query(self, query: str) -> Optional[str]:
#         """Check if query is ambiguous and needs clarification."""
#         query_lower = query.lower().strip()
        
#         # Very vague requests
#         if query_lower in ['music', 'songs', 'song', 'play music', 'recommend', 'suggest']:
#             return """I'd love to help you find perfect music! Could you be a bit more specific?

# **Try asking like:**
# • "Relaxing evening music" or "Energetic workout songs"
# • "Nepali folk songs" or "Korean pop music" 
# • "Romantic songs for dinner" or "Party music for dancing"
# • "Bollywood hits" or "Christmas music"
# • "Something like Taylor Swift" or "90s rock music"

# **What's your mood or preference today?**"""

#         # Cultural but too vague
#         elif any(term in query_lower for term in ['cultural music', 'ethnic music', 'world music', 'traditional music']):
#             return """**Great choice for exploring world music!** Which culture or region interests you?

# **Popular options:**
# • **South Asian**: Nepali, Hindi/Bollywood, Punjabi, Tamil
# • **East Asian**: Korean (K-pop), Japanese (J-pop), Chinese (C-pop)
# • **Latin**: Spanish, Brazilian, Mexican, Reggaeton
# • **Middle Eastern**: Arabic, Persian, Turkish
# • **African**: Afrobeat, South African, West African
# • **European**: French Chanson, German, Italian

# **Or tell me more specifically** what you're in the mood for!"""

#         # Religious but unclear
#         elif any(term in query_lower for term in ['spiritual music', 'religious music', 'devotional']):
#             return """**Spiritual music is so enriching!** Which tradition or style speaks to you?

# **Options include:**
# • **Hindu/Indian**: Bhajans, Kirtans, Mantras
# • **Christian**: Gospel, Contemporary Christian, Hymns
# • **Islamic**: Nasheeds, Spiritual recitations
# • **Buddhist**: Meditation music, Zen sounds
# • **Seasonal**: Christmas carols, Diwali songs, Eid music
# • **General**: Meditation, Mindfulness, Prayer music

# **What type of spiritual experience** are you seeking? ✨"""

#         # Activity context but vague
#         elif query_lower in ['music for activity', 'background music', 'mood music']:
#             return """**Perfect! Let me find music for your specific activity.**

# **What are you doing?**
# • **Workout**: Gym, running, high-energy training
# • **Study/Work**: Focus music, lo-fi, instrumental
# • **Relaxation**: Sleep, meditation, chill evening
# • **Travel**: Road trip, driving, adventure music
# • **Social**: Party, dancing, celebration, wedding
# • **Daily routine**: Morning energy, dinner ambiance

# **Tell me your activity** and I'll curate the perfect soundtrack!"""

#         return None

# def main():
#     """Main function to run the hybrid recommendation system."""
#     print("Advanced Hybrid Music Recommendation System")
#     print("=" * 65)
#     print(" Powered by Sequential + Ranking + Embedding AI Models")
#     print(" Real Spotify data with audio feature analysis")
#     print(" Quality metrics and evaluation built-in")
#     print("\nCommands:")
#     print("  • Normal chat: 'Hello', 'How are you?'")
#     print("  • Music requests: 'Relaxing evening music', 'Energetic workout songs'")
#     print("  • Exit: 'quit'")
#     print("\nType your request...\n")
    
#     # Initialize system
#     system = HybridRecommendationSystem()
    
#     while True:
#         try:
#             user_input = input(" You: ").strip()
            
#             if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
#                 print("System: Thanks for using the Hybrid Music Recommendation System!")
#                 break
                
#             if not user_input:
#                 continue
                
#             print()  # Add spacing
#             response = system.chat(user_input)
#             print(f"System: {response}\n")
#             print("─" * 80)
            
#         except KeyboardInterrupt:
#             print("\nSystem: Goodbye! Thanks for using the hybrid system!")
#             break
#         except Exception as e:
#             print(f"System error: {e}")
#             print("Please try again.\n")

# if __name__ == "__main__":
#     main()