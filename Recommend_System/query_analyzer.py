#!/usr/bin/env python3
"""
Advanced Music Query Analyzer
Interprets natural language user queries into structured JSON for hybrid recommendation systems.
Goes beyond basic fallback systems to extract cultural, religious, and contextual metadata.
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class QueryAnalysis:
    """Structured representation of analyzed music query."""
    intent: str  # "playlist" | "song" | "artist" | "mood_based" | "mixed"
    genres: List[str]
    moods: List[str]
    languages: List[str]
    artists: List[str]
    situations: List[str]
    religious_cultural: List[str]
    fallback: Optional[str]

class AdvancedMusicQueryAnalyzer:
    def __init__(self):
        """Initialize the advanced query analyzer with comprehensive keyword mappings."""
        
        # Intent detection patterns
        self.intent_patterns = {
            'song': ['song', 'track', 'tune', 'piece', 'number'],
            'artist': ['artist', 'singer', 'musician', 'band', 'performer'],
            'playlist': ['playlist', 'songs', 'music', 'tracks', 'collection'],
            'mood_based': ['mood', 'feeling', 'vibe', 'atmosphere', 'energy']
        }
        
        # Comprehensive genre mapping
        self.genre_keywords = {
            # Western genres
            'pop': ['pop', 'popular'],
            'rock': ['rock', 'metal', 'punk', 'grunge', 'alternative'],
            'hip-hop': ['hip hop', 'rap', 'hiphop', 'hip-hop', 'trap'],
            'electronic': ['electronic', 'edm', 'techno', 'house', 'dubstep', 'trance'],
            'jazz': ['jazz', 'blues', 'swing'],
            'classical': ['classical', 'orchestra', 'symphony', 'opera'],
            'country': ['country', 'folk', 'bluegrass'],
            'reggae': ['reggae', 'ska', 'dancehall'],
            'r&b': ['r&b', 'rnb', 'soul', 'funk'],
            'indie': ['indie', 'independent', 'alternative'],
            
            # South Asian genres
            'bollywood': ['bollywood', 'filmi', 'hindi cinema', 'playback'],
            'lok': ['lok', 'folk', 'traditional', 'lok geet'],
            'bhajan': ['bhajan', 'devotional', 'spiritual'],
            'qawwali': ['qawwali', 'sufi'],
            'bhangra': ['bhangra', 'punjabi pop'],
            'classical indian': ['indian classical', 'raga', 'hindustani', 'carnatic'],
            
            # East Asian genres
            'k-pop': ['kpop', 'k-pop', 'korean pop'],
            'j-pop': ['jpop', 'j-pop', 'japanese pop'],
            'c-pop': ['cpop', 'c-pop', 'mandopop', 'cantopop'],
            
            # Latin genres
            'latin': ['latin', 'latino', 'hispanic'],
            'reggaeton': ['reggaeton', 'perreo'],
            'salsa': ['salsa', 'bachata', 'merengue'],
            'bossa nova': ['bossa nova', 'samba'],
            
            # Middle Eastern/Arabic
            'arabic': ['arabic', 'middle eastern', 'oud'],
            'nasheed': ['nasheed', 'islamic music'],
            
            # African
            'afrobeat': ['afrobeat', 'african', 'highlife'],
            
            # Religious/Spiritual
            'gospel': ['gospel', 'christian music', 'hymn'],
            'kirtan': ['kirtan', 'chanting'],
            'meditation': ['meditation', 'ambient', 'new age']
        }
        
        # Comprehensive mood mapping
        self.mood_keywords = {
            'happy': ['happy', 'cheerful', 'joyful', 'upbeat', 'positive', 'bright'],
            'sad': ['sad', 'melancholy', 'sorrowful', 'depressing', 'emotional', 'tearjerker'],
            'energetic': ['energetic', 'high energy', 'pumped', 'intense', 'powerful'],
            'chill': ['chill', 'relaxing', 'calm', 'peaceful', 'laid back', 'mellow'],
            'romantic': ['romantic', 'love', 'intimate', 'sensual', 'passionate'],
            'focus': ['focus', 'concentration', 'study', 'productive', 'instrumental'],
            'spiritual': ['spiritual', 'divine', 'sacred', 'holy', 'blessed'],
            'nostalgic': ['nostalgic', 'memories', 'throwback', 'vintage', 'retro'],
            'aggressive': ['aggressive', 'angry', 'hardcore', 'brutal', 'fierce'],
            'party': ['party', 'dance', 'club', 'festive', 'celebration'],
            'motivational': ['motivational', 'inspiring', 'uplifting', 'empowering']
        }
        
        # Language and culture mapping
        self.language_keywords = {
            # South Asian
            'nepali': ['nepali', 'nepal', 'nepalese', 'himalayan'],
            'hindi': ['hindi', 'indian', 'bharat', 'hindustani'],
            'punjabi': ['punjabi', 'sikh', 'punjab'],
            'tamil': ['tamil', 'kollywood', 'south indian'],
            'bengali': ['bengali', 'bangla', 'calcutta'],
            'urdu': ['urdu', 'pakistan', 'pakistani'],
            'gujarati': ['gujarati', 'gujarat'],
            'marathi': ['marathi', 'maharashtra'],
            
            # East Asian
            'korean': ['korean', 'korea', 'seoul', 'hallyu'],
            'japanese': ['japanese', 'japan', 'tokyo', 'anime'],
            'chinese': ['chinese', 'mandarin', 'cantonese', 'taiwan', 'hong kong'],
            'thai': ['thai', 'thailand', 'bangkok'],
            'vietnamese': ['vietnamese', 'vietnam'],
            
            # European
            'english': ['english', 'british', 'american', 'uk', 'usa'],
            'spanish': ['spanish', 'español', 'spain'],
            'french': ['french', 'français', 'france'],
            'german': ['german', 'deutsch', 'germany'],
            'italian': ['italian', 'italy', 'italiano'],
            'russian': ['russian', 'russia'],
            'portuguese': ['portuguese', 'brazil', 'portugal'],
            
            # Middle Eastern/Arabic
            'arabic': ['arabic', 'arab', 'middle east'],
            'persian': ['persian', 'farsi', 'iran'],
            'turkish': ['turkish', 'turkey'],
            'hebrew': ['hebrew', 'israel'],
            
            # African
            'swahili': ['swahili', 'kenya', 'tanzania'],
            'yoruba': ['yoruba', 'nigeria'],
            'amharic': ['amharic', 'ethiopia'],
            
            # Latin American
            'latin': ['latin american', 'latino', 'hispanic'],
            'mexican': ['mexican', 'mexico', 'mariachi'],
            'brazilian': ['brazilian', 'brazil', 'samba'],
            'argentinian': ['argentinian', 'argentina', 'tango']
        }
        
        # Situational context mapping
        self.situation_keywords = {
            'workout': ['workout', 'gym', 'exercise', 'fitness', 'training', 'running'],
            'study': ['study', 'studying', 'focus', 'concentration', 'work', 'productivity'],
            'sleep': ['sleep', 'sleeping', 'bedtime', 'lullaby', 'night', 'rest'],
            'driving': ['driving', 'road trip', 'travel', 'highway', 'journey'],
            'party': ['party', 'dancing', 'club', 'nightlife', 'celebration'],
            'wedding': ['wedding', 'marriage', 'shaadi', 'ceremony'],
            'morning': ['morning', 'wake up', 'breakfast', 'start day'],
            'evening': ['evening', 'dinner', 'sunset', 'end of day'],
            'meditation': ['meditation', 'yoga', 'mindfulness', 'zen'],
            'heartbreak': ['heartbreak', 'breakup', 'loss', 'grief'],
            'cooking': ['cooking', 'kitchen', 'baking', 'chef'],
            'gaming': ['gaming', 'video games', 'streaming'],
            'reading': ['reading', 'books', 'literature']
        }
        
        # Religious and cultural context mapping
        self.religious_cultural_keywords = {
            # Hindu/Indian
            'hindu_spiritual': ['bhajan', 'kirtan', 'aarti', 'mantra', 'devotional'],
            'indian_classical': ['raga', 'hindustani', 'carnatic', 'tabla', 'sitar'],
            'bollywood': ['bollywood', 'filmi', 'hindi cinema', 'playback singer'],
            'nepali_folk': ['nepali lok', 'lok geet', 'himalayan folk', 'newari'],
            
            # Islamic
            'islamic': ['nasheed', 'islamic music', 'quran', 'ramadan', 'eid'],
            'sufi': ['qawwali', 'sufi', 'dervish', 'mystical'],
            
            # Christian
            'christian': ['gospel', 'hymn', 'praise', 'worship', 'christian music'],
            'christmas': ['christmas', 'xmas', 'carols', 'holiday'],
            
            # Buddhist
            'buddhist': ['buddhist chanting', 'tibetan', 'zen music', 'meditation'],
            
            # Jewish
            'jewish': ['hebrew', 'klezmer', 'jewish music', 'synagogue'],
            
            # Cultural festivals
            'diwali': ['diwali', 'festival of lights', 'deepavali'],
            'holi': ['holi', 'color festival', 'spring festival'],
            'eid': ['eid', 'eid mubarak', 'islamic festival'],
            'chinese_new_year': ['chinese new year', 'lunar new year', 'spring festival'],
            
            # Regional cultural
            'punjabi_culture': ['bhangra', 'punjabi folk', 'gurdwara'],
            'south_indian': ['carnatic', 'tamil', 'malayalam', 'telugu'],
            'bengali_culture': ['rabindra sangeet', 'bengali folk', 'durga puja']
        }
        
        # Artist name patterns (for common artists)
        self.common_artists = [
            # International
            'taylor swift', 'ed sheeran', 'adele', 'bruno mars', 'beyonce',
            'drake', 'eminem', 'kanye west', 'billie eilish', 'ariana grande',
            
            # Bollywood/Hindi
            'arijit singh', 'shreya ghoshal', 'lata mangeshkar', 'kishore kumar',
            'sonu nigam', 'rahat fateh ali khan', 'a r rahman',
            
            # Nepali
            'narayan gopal', 'aruna lama', 'bipul chettri', 'sajjan raj vaidya',
            
            # K-pop
            'bts', 'blackpink', 'twice', 'stray kids', 'red velvet',
            
            # Regional Indian
            'gurdas maan', 'diljit dosanjh', 'sidhu moose wala'
        ]

    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Main method to analyze a music query and return structured data.
        
        Args:
            query: Natural language music query
            
        Returns:
            QueryAnalysis object with extracted metadata
        """
        query_lower = query.lower().strip()
        
        # Initialize result containers
        intent = "playlist"  # Default intent
        genres = []
        moods = []
        languages = []
        artists = []
        situations = []
        religious_cultural = []
        fallback = None
        
        # Detect intent
        intent = self._detect_intent(query_lower)
        
        # Extract genres
        genres = self._extract_genres(query_lower)
        
        # Extract moods
        moods = self._extract_moods(query_lower)
        
        # Extract languages/cultures
        languages = self._extract_languages(query_lower)
        
        # Extract artists
        artists = self._extract_artists(query_lower)
        
        # Extract situations
        situations = self._extract_situations(query_lower)
        
        # Extract religious/cultural contexts
        religious_cultural = self._extract_religious_cultural(query_lower)
        
        # Check if query is too vague
        if not any([genres, moods, languages, artists, situations, religious_cultural]):
            if self._is_vague_query(query_lower):
                fallback = "general music"
        
        return QueryAnalysis(
            intent=intent,
            genres=list(set(genres)),  # Remove duplicates
            moods=list(set(moods)),
            languages=list(set(languages)),
            artists=list(set(artists)),
            situations=list(set(situations)),
            religious_cultural=list(set(religious_cultural)),
            fallback=fallback
        )
    
    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query."""
        for intent, keywords in self.intent_patterns.items():
            if any(keyword in query for keyword in keywords):
                return intent
        
        # Check for specific patterns
        if any(word in query for word in ['recommend', 'suggest', 'find']):
            return "playlist"
        elif any(word in query for word in ['feeling', 'mood']):
            return "mood_based"
        elif 'by' in query or 'from' in query:
            return "artist"
        
        return "playlist"  # Default
    
    def _extract_genres(self, query: str) -> List[str]:
        """Extract genre information from query."""
        found_genres = []
        
        for genre, keywords in self.genre_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_genres.append(genre)
        
        return found_genres
    
    def _extract_moods(self, query: str) -> List[str]:
        """Extract mood information from query."""
        found_moods = []
        
        for mood, keywords in self.mood_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_moods.append(mood)
        
        return found_moods
    
    def _extract_languages(self, query: str) -> List[str]:
        """Extract language/culture information from query."""
        found_languages = []
        
        for language, keywords in self.language_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_languages.append(language)
        
        return found_languages
    
    def _extract_artists(self, query: str) -> List[str]:
        """Extract artist names from query."""
        found_artists = []
        
        # Check for common artists
        for artist in self.common_artists:
            if artist in query:
                found_artists.append(artist.title())
        
        # Look for patterns like "by [artist]", "from [artist]", "like [artist]"
        patterns = [
            r'by ([a-zA-Z\s]+)',
            r'from ([a-zA-Z\s]+)',
            r'like ([a-zA-Z\s]+)',
            r'similar to ([a-zA-Z\s]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                artist_name = match.strip().title()
                if len(artist_name.split()) <= 3:  # Reasonable artist name length
                    found_artists.append(artist_name)
        
        return found_artists
    
    def _extract_situations(self, query: str) -> List[str]:
        """Extract situational context from query."""
        found_situations = []
        
        for situation, keywords in self.situation_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_situations.append(situation)
        
        return found_situations
    
    def _extract_religious_cultural(self, query: str) -> List[str]:
        """Extract religious and cultural context from query."""
        found_contexts = []
        
        for context, keywords in self.religious_cultural_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_contexts.append(context)
        
        return found_contexts
    
    def _is_vague_query(self, query: str) -> bool:
        """Check if query is too vague and needs fallback."""
        vague_patterns = [
            'music', 'songs', 'song', 'something', 'anything',
            'good music', 'nice songs', 'recommend', 'suggest'
        ]
        
        # If query is very short and only contains vague terms
        words = query.split()
        if len(words) <= 3:
            return any(pattern in query for pattern in vague_patterns)
        
        return False
    
    def analyze_to_json(self, query: str) -> str:
        """Analyze query and return JSON string."""
        analysis = self.analyze_query(query)
        return json.dumps(asdict(analysis), indent=2, ensure_ascii=False)
    
    def analyze_to_dict(self, query: str) -> Dict[str, Any]:
        """Analyze query and return dictionary."""
        analysis = self.analyze_query(query)
        return asdict(analysis)

def main():
    """Test the analyzer with example queries."""
    analyzer = AdvancedMusicQueryAnalyzer()
    
    test_queries = [
        "Play some Nepali romantic lok songs for a wedding",
        "I want chill Korean songs for studying",
        "Suggest me something good",
        "Bollywood bhajans for meditation",
        "Energetic hip hop tracks by Eminem for workout",
        "Sad Arabic songs for heartbreak",
        "Traditional Chinese music for tea ceremony",
        "Gospel music for Sunday morning",
        "Punjabi bhangra for Holi celebration",
        "Jazz music for reading",
        "Spanish guitar for romantic dinner",
        "Sufi qawwali by Nusrat Fateh Ali Khan",
        "K-pop dance tracks for party",
        "Nepali lok geet",
        "Islamic nasheed for Ramadan",
        "Classical Indian raga for yoga"
    ]
    
    print("🎵 Advanced Music Query Analyzer Test Results")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: \"{query}\"")
        print("-" * 40)
        
        analysis = analyzer.analyze_to_dict(query)
        
        print(f"Intent: {analysis['intent']}")
        if analysis['genres']:
            print(f"Genres: {', '.join(analysis['genres'])}")
        if analysis['moods']:
            print(f"Moods: {', '.join(analysis['moods'])}")
        if analysis['languages']:
            print(f"Languages: {', '.join(analysis['languages'])}")
        if analysis['artists']:
            print(f"Artists: {', '.join(analysis['artists'])}")
        if analysis['situations']:
            print(f"Situations: {', '.join(analysis['situations'])}")
        if analysis['religious_cultural']:
            print(f"Religious/Cultural: {', '.join(analysis['religious_cultural'])}")
        if analysis['fallback']:
            print(f"Fallback: {analysis['fallback']}")

if __name__ == "__main__":
    main()
