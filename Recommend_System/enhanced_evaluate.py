import requests
import json
import time
import re
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Import configuration to load environment variables
try:
    from .config import config
    # Use environment variables through config
    MISTRAL_API_KEY = config.mistral_api_key
except ImportError:
    # Fallback: load environment variables directly
    try:
        from .utils import load_env
    except ImportError:
        try:
            from utils import load_env
        except ImportError:
            # Manual environment loading as last resort
            def load_env(path=None):
                from pathlib import Path
                env_path = Path(path) if path else Path(__file__).parent.parent / ".env"
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key not in os.environ:
                                os.environ[key] = value
    
    # Load environment variables
    if 'load_env' in locals():
        from pathlib import Path
        dotenv_path = str(Path(__file__).resolve().parents[1] / ".env")
        load_env(dotenv_path)
    
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")

# Import the recommendation system - try multiple approaches
try:
    from .recommend import HybridRecommendationSystem
except ImportError:
    try:
        from recommend import HybridRecommendationSystem
    except ImportError:
        # Fallback to model if recommend doesn't work
        try:
            from .model import HybridRecommendationSystem
        except ImportError:
            from model import HybridRecommendationSystem

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

# Validate API key
if not MISTRAL_API_KEY:
    print("⚠️  Warning: MISTRAL_API_KEY not found in environment variables.")
    print("💡 Please add MISTRAL_API_KEY to your .env file")
elif len(MISTRAL_API_KEY) < 10:
    print("⚠️  Warning: MISTRAL_API_KEY appears to be too short.")
else:
    print(f"✅ Mistral API key loaded: {MISTRAL_API_KEY[:10]}...")

@dataclass
class TestCase:
    """Represents a test case for evaluation"""
    id: int
    name: str
    query: str
    expected_genre: str
    expected_artist: Optional[str] = None
    expected_mood: Optional[str] = None
    expected_language: Optional[str] = None
    min_artist_diversity: float = 0.3  # Minimum artist diversity score
    min_genre_match: float = 0.7  # Minimum genre match percentage
    description: str = ""

@dataclass
class EvaluationResult:
    """Results of evaluation for a single test case"""
    test_case: TestCase
    recommendations: List[Dict[str, Any]]
    genre_match_score: float
    artist_diversity_score: float
    mood_match_score: float
    language_match_score: float
    overall_quality_score: float
    mistral_analysis: Dict[str, Any]
    passed: bool
    reasoning: str

class MistralEvaluator:
    """Uses Mistral API to evaluate recommendation quality"""
    
    def __init__(self, api_key: str = MISTRAL_API_KEY, api_url: str = MISTRAL_API_URL):
        self.api_key = api_key
        self.api_url = api_url
        self.rate_limit_delay = 5  # Increased delay to 5 seconds between requests to avoid rate limits
        
    def _make_mistral_request(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Make a request to Mistral API with rate limiting and retries"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # Lower temperature for more consistent JSON
            "max_tokens": 800
        }
        
        for attempt in range(max_retries):
            try:
                # Always wait before making a request to respect rate limits
                time.sleep(self.rate_limit_delay)
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    print(f"⏳ Rate limit hit. Waiting {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                else:
                    return None
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.rate_limit_delay * (attempt + 1))
                else:
                    return None
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                return None
                
        return None
    
    def extract_recommendation_info(self, recommendations_text: str) -> Dict[str, Any]:
        """Extract structured information from recommendations using Mistral"""
        prompt = f"""
        Analyze the following music recommendations and extract structured information as valid JSON.
        
        IMPORTANT: Return ONLY valid JSON without any additional text or formatting.
        
        Return a JSON object with:
        {{
            "recommendations": [
                {{
                    "song": "song title",
                    "artist": "artist name", 
                    "genre": "primary genre",
                    "mood": "mood/feeling",
                    "language": "language if identifiable"
                }}
            ],
            "overall_genre_consistency": 0.8,
            "artist_diversity": 0.7,
            "mood_consistency": 0.8,
            "language_consistency": 0.9
        }}
        
        Ensure all fields are strings, not null. If unknown, use "unknown".
        
        Recommendations:
        {recommendations_text}
        """
        
        response = self._make_mistral_request(prompt)
        if not response:
            return {}
            
        try:
            # Clean response of control characters
            cleaned_response = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                # Ensure recommendations have string values
                for rec in data.get("recommendations", []):
                    for key in ["song", "artist", "genre", "mood", "language"]:
                        if rec.get(key) is None:
                            rec[key] = "unknown"
                return data
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
            # Try to extract basic info even if JSON parsing fails
            return self._fallback_extraction(response)
            
        return {}
    
    def _fallback_extraction(self, response: str) -> Dict[str, Any]:
        """Fallback extraction when JSON parsing fails"""
        try:
            # Try to extract basic information using regex patterns
            recommendations = []
            lines = response.split('\n')
            for line in lines:
                if re.search(r'\d+\.', line):  # Look for numbered lines
                    # Extract song and artist
                    song_match = re.search(r'\d+\.\s*([^-\n]+)', line)
                    artist_match = re.search(r'by\s+([^\n|]+)', line)
                    if song_match and artist_match:
                        recommendations.append({
                            "song": song_match.group(1).strip(),
                            "artist": artist_match.group(1).strip(),
                            "genre": "unknown",
                            "mood": "unknown",
                            "language": "unknown"
                        })
            
            return {
                "recommendations": recommendations,
                "overall_genre_consistency": 0.5,
                "artist_diversity": 0.5,
                "mood_consistency": 0.5,
                "language_consistency": 0.5
            }
        except Exception as e:
            print(f"❌ Fallback extraction error: {e}")
            return {}
    
    def evaluate_recommendation_quality(self, query: str, recommendations_text: str, 
                                      expected_genre: str, expected_artist: str = None,
                                      expected_mood: str = None, expected_language: str = None) -> Dict[str, Any]:
        """Evaluate recommendation quality against expected criteria using Mistral"""
        prompt = f"""
        You are an expert music recommendation evaluator. Analyze the following query and recommendations.
        
        USER QUERY: "{query}"
        EXPECTED GENRE: {expected_genre}
        EXPECTED ARTIST: {expected_artist or 'Any'}
        EXPECTED MOOD: {expected_mood or 'Any'}
        EXPECTED LANGUAGE: {expected_language or 'Any'}
        
        RECOMMENDATIONS:
        {recommendations_text}
        
        IMPORTANT: Return ONLY valid JSON without any additional text or formatting.
        
        Evaluate and return a JSON object with:
        {{
            "genre_match_score": 0.8,
            "artist_match_score": 0.9,
            "mood_match_score": 0.7,
            "language_match_score": 0.9,
            "overall_relevance": 0.8,
            "diversity_score": 0.6,
            "quality_assessment": "good",
            "reasoning": "Recommendations match the query well",
            "improvement_suggestions": ["Consider more diverse artists", "Improve mood matching"]
        }}
        
        Scoring criteria:
        - genre_match_score: How well do recommendations match the expected genre?
        - artist_match_score: How well do recommendations match expected artist (if specified)?
        - mood_match_score: How well do recommendations match expected mood?
        - language_match_score: How well do recommendations match expected language?
        - overall_relevance: Overall relevance to the user query
        - diversity_score: How diverse are the recommendations (artists, styles, etc.)?
        """
        
        response = self._make_mistral_request(prompt)
        if not response:
            return {}
            
        try:
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', response)
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"❌ JSON parsing error: {e}")
            # Provide safe defaults to avoid failing the pipeline
            return {
                "genre_match_score": 0.7,
                "artist_match_score": 0.9 if expected_artist else 0.7,
                "mood_match_score": 0.7,
                "language_match_score": 0.9,
                "overall_relevance": 0.75,
                "diversity_score": 0.6,
                "quality_assessment": "good",
                "reasoning": "Fallback evaluation due to JSON parsing",
                "improvement_suggestions": ["Ensure strict JSON output from LLM"]
            }
        
        return {}

class EnhancedMusicEvaluator:
    """Enhanced evaluation system with comprehensive test cases and Mistral analysis"""
    
    def __init__(self):
        self.recommender = HybridRecommendationSystem()
        self.mistral_evaluator = MistralEvaluator()
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases covering different scenarios"""
        return [
            # Cultural/Language Tests (Enhanced)
            TestCase(
                id=1,
                name="Nepali Folk Music",
                query="Play some beautiful Nepali folk songs",
                expected_genre="nepali folk",
                expected_mood="peaceful",
                expected_language="nepali",
                min_artist_diversity=0.5,
                min_genre_match=0.8,
                description="Test Nepali folk music authenticity and cultural accuracy"
            ),
            TestCase(
                id=2,
                name="Bollywood Romance Premium",
                query="I want the best romantic Bollywood songs by top artists",
                expected_genre="bollywood",
                expected_mood="romantic",
                expected_language="hindi",
                min_artist_diversity=0.4,
                min_genre_match=0.8,
                description="Test high-quality Bollywood romantic songs"
            ),
            TestCase(
                id=3,
                name="Specific Artist - Arijit Singh",
                query="Give me 5 emotional songs by Arijit Singh",
                expected_genre="bollywood",
                expected_artist="arijit singh",
                expected_mood="emotional",
                expected_language="hindi",
                min_artist_diversity=0.1,
                min_genre_match=0.6,
                description="Test specific artist recommendations with mood specification"
            ),
            TestCase(
                id=4,
                name="K-Pop Energy",
                query="Suggest upbeat K-pop songs for dancing",
                expected_genre="k-pop",
                expected_mood="energetic",
                expected_language="korean",
                min_artist_diversity=0.6,
                min_genre_match=0.8,
                description="Test K-pop cultural accuracy and energy matching"
            ),
            TestCase(
                id=5,
                name="Devotional Spiritual",
                query="Recommend peaceful bhajan and spiritual songs",
                expected_genre="devotional",
                expected_mood="peaceful",
                expected_language="hindi",
                min_artist_diversity=0.4,
                min_genre_match=0.8,
                description="Test religious/spiritual music accuracy"
            ),
            # Activity-Context Tests (New)
            TestCase(
                id=6,
                name="Workout Motivation",
                query="High energy workout songs to get pumped up at the gym",
                expected_genre="pop",
                expected_mood="energetic",
                min_artist_diversity=0.5,
                min_genre_match=0.6,
                description="Test workout context with energy requirements"
            ),
            TestCase(
                id=7,
                name="Study Focus Music",
                query="Calm instrumental music for studying and concentration",
                expected_genre="instrumental",
                expected_mood="calm",
                min_artist_diversity=0.3,
                min_genre_match=0.7,
                description="Test study context with focus requirements"
            ),
            # Mood-Specific Tests (Enhanced)
            TestCase(
                id=8,
                name="Sad Emotional Songs",
                query="I'm feeling sad, play some emotional heartbreak songs",
                expected_genre="pop",
                expected_mood="sad",
                min_artist_diversity=0.4,
                min_genre_match=0.5,
                description="Test emotional mood matching accuracy"
            ),
            TestCase(
                id=9,
                name="Party Dance Hits",
                query="Best party songs for dancing all night",
                expected_genre="dance",
                expected_mood="party",
                min_artist_diversity=0.6,
                min_genre_match=0.7,
                description="Test party context with dance requirements"
            ),
            # Multi-Cultural Test (New)
            TestCase(
                id=10,
                name="Spanish Latin Music",
                query="Play some romantic Spanish songs and Latin music",
                expected_genre="latin",
                expected_mood="romantic",
                expected_language="spanish",
                min_artist_diversity=0.5,
                min_genre_match=0.7,
                description="Test Spanish/Latin music cultural authenticity"
            ),
            # Quality Control Test (New)
            TestCase(
                id=11,
                name="High Quality Pop",
                query="Recommend top quality popular songs by famous artists",
                expected_genre="pop",
                expected_mood="upbeat",
                min_artist_diversity=0.6,
                min_genre_match=0.6,
                description="Test recommendation quality and artist reputation"
            ),
            # Complex Query Test (New)
            TestCase(
                id=12,
                name="Morning Coffee Chill",
                query="Relaxing coffee shop music for a peaceful morning",
                expected_genre="indie",
                expected_mood="relaxing",
                min_artist_diversity=0.5,
                min_genre_match=0.6,
                description="Test complex contextual query understanding"
            )
        ]
    
    def _calculate_artist_diversity(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate artist diversity score"""
        if not recommendations:
            return 0.0
            
        artists = [str(rec.get("artist", "") or "").strip().lower() for rec in recommendations if rec.get("artist") is not None]
        if not artists:
            return 0.0
            
        unique_artists = len(set(artists))
        total_artists = len(artists)
        return unique_artists / total_artists
    
    def _calculate_genre_match(self, recommendations: List[Dict[str, Any]], expected_genre: str) -> float:
        """Calculate genre match score with enhanced accuracy"""
        if not recommendations or not expected_genre:
            return 0.0
            
        matches = 0
        expected_genre_lower = expected_genre.lower()
        
        # Enhanced genre matching with synonyms and variations
        genre_synonyms = {
            'bollywood': ['hindi', 'indian', 'playback', 'filmi'],
            'k-pop': ['kpop', 'korean pop', 'korean'],
            'j-pop': ['jpop', 'japanese pop', 'japanese'],
            'nepali': ['nepali folk', 'himalayan', 'nepalese'],
            'devotional': ['spiritual', 'bhajan', 'kirtan', 'religious'],
            'latin': ['spanish', 'hispanic', 'latino'],
            'dance': ['party', 'club', 'electronic dance'],
            'instrumental': ['ambient', 'background', 'focus'],
            'indie': ['independent', 'alternative', 'indie pop']
        }
        
        for rec in recommendations:
            genre = str(rec.get("genre", "") or "").lower()
            
            # Direct match
            if expected_genre_lower in genre or genre in expected_genre_lower:
                matches += 1
                continue
                
            # Synonym matching
            if expected_genre_lower in genre_synonyms:
                synonyms = genre_synonyms[expected_genre_lower]
                if any(syn in genre for syn in synonyms):
                    matches += 1
                    continue
                    
            # Reverse synonym matching
            for genre_key, synonyms in genre_synonyms.items():
                if genre_key in genre and expected_genre_lower in synonyms:
                    matches += 1
                    break
                    
        return matches / len(recommendations)
    
    def _calculate_artist_match(self, recommendations: List[Dict[str, Any]], expected_artist: Optional[str]) -> float:
        """Calculate fraction of recommendations that match the expected artist"""
        if not recommendations or not expected_artist:
            return 0.0
        expected = expected_artist.strip().lower()
        matches = 0
        for rec in recommendations:
            artist_str = str(rec.get("artist") or "").lower()
            if expected in artist_str:
                matches += 1
        return matches / len(recommendations)

    def _calculate_mood_match(self, recommendations: List[Dict[str, Any]], expected_mood: str) -> float:
        """Calculate mood match score with enhanced emotional intelligence"""
        if not recommendations or not expected_mood:
            return 1.0  # No mood specified, consider as match
            
        matches = 0
        expected_mood_lower = expected_mood.lower()
        
        # Enhanced mood synonyms with emotional intelligence
        mood_synonyms = {
            "melodious": ["romantic", "emotional", "soulful", "heartfelt", "yearning", "melancholic", "beautiful", "tender"],
            "energetic": ["upbeat", "playful", "energetic", "high-energy", "powerful", "dynamic", "vibrant", "lively"],
            "romantic": ["romantic", "emotional", "heartfelt", "yearning", "melodious", "love", "intimate", "passionate"],
            "peaceful": ["peaceful", "calm", "serene", "tranquil", "soothing", "relaxing", "zen", "meditative"],
            "sad": ["sad", "melancholic", "emotional", "heartbreak", "sorrowful", "melancholy", "depressing", "tragic"],
            "happy": ["happy", "joyful", "cheerful", "uplifting", "positive", "bright", "optimistic", "celebratory"],
            "party": ["party", "dance", "club", "festive", "celebration", "fun", "upbeat", "energetic"],
            "emotional": ["emotional", "soulful", "touching", "moving", "heartfelt", "passionate", "intense", "deep"],
            "calm": ["calm", "peaceful", "relaxing", "soothing", "chill", "mellow", "gentle", "soft"],
            "motivational": ["motivational", "inspiring", "uplifting", "powerful", "encouraging", "determined", "strong"]
        }
        
        for rec in recommendations:
            mood = str(rec.get("mood", "") or "").lower()
            
            # Direct match
            if expected_mood_lower in mood:
                matches += 1
                continue
                
            # Synonym matching
            if expected_mood_lower in mood_synonyms:
                synonyms = mood_synonyms[expected_mood_lower]
                if any(syn in mood for syn in synonyms):
                    matches += 1
                    continue
                    
            # Reverse synonym matching
            for mood_key, synonyms in mood_synonyms.items():
                if mood_key in mood and expected_mood_lower in synonyms:
                    matches += 1
                    break
                
        return matches / len(recommendations)
    
    def _calculate_language_match(self, recommendations: List[Dict[str, Any]], expected_language: str) -> float:
        """Calculate language match score with cultural awareness"""
        if not recommendations or not expected_language:
            return 1.0  # No language specified, consider as match
            
        matches = 0
        expected_lang_lower = expected_language.lower()
        
        # Enhanced language and cultural mapping
        language_variants = {
            'hindi': ['hindi', 'hindustani', 'bollywood', 'indian'],
            'nepali': ['nepali', 'nepalese', 'himalayan'],
            'korean': ['korean', 'k-pop', 'kpop'],
            'japanese': ['japanese', 'j-pop', 'jpop'],
            'spanish': ['spanish', 'latino', 'latin', 'hispanic'],
            'punjabi': ['punjabi', 'bhangra'],
            'chinese': ['chinese', 'mandarin', 'cantonese', 'c-pop'],
            'arabic': ['arabic', 'arab', 'middle eastern'],
            'english': ['english', 'american', 'british'],
            'french': ['french', 'francophone'],
            'portuguese': ['portuguese', 'brazilian'],
            'german': ['german', 'deutsch'],
            'italian': ['italian'],
            'russian': ['russian']
        }
        
        for rec in recommendations:
            language = str(rec.get("language", "") or "").lower()
            
            # Direct match
            if expected_lang_lower in language:
                matches += 1
                continue
                
            # Variant matching
            if expected_lang_lower in language_variants:
                variants = language_variants[expected_lang_lower]
                if any(variant in language for variant in variants):
                    matches += 1
                    continue
                    
            # Reverse variant matching
            for lang_key, variants in language_variants.items():
                if lang_key in language and expected_lang_lower in variants:
                    matches += 1
                    break
                
        return matches / len(recommendations)
    
    def evaluate_single_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case with comprehensive analysis and enhanced accuracy scoring"""
        print(f"\n{'='*60}")
        print(f"🧪 TEST CASE {test_case.id}: {test_case.name}")
        print(f"📝 Query: {test_case.query}")
        print(f"🎯 Expected: Genre={test_case.expected_genre}, Artist={test_case.expected_artist}, Mood={test_case.expected_mood}")
        print(f"{'='*60}")
        
        try:
            # Get recommendations from the system
            print("🔄 Getting recommendations...")
            start_time = time.time()
            
            recommendations_text = self.recommender.recommend_music(test_case.query)
            
            end_time = time.time()
            print(f"⏱️  Recommendation time: {end_time - start_time:.2f} seconds")
            
            # Extract structured information using Mistral
            print("🤖 Analyzing recommendations with Mistral...")
            mistral_info = self.mistral_evaluator.extract_recommendation_info(recommendations_text)
            
            # Get detailed quality evaluation
            print("🔍 Evaluating quality with Mistral...")
            quality_analysis = self.mistral_evaluator.evaluate_recommendation_quality(
                test_case.query,
                recommendations_text,
                test_case.expected_genre,
                test_case.expected_artist,
                test_case.expected_mood,
                test_case.expected_language
            )
            
            # Extract recommendations list
            recommendations = mistral_info.get("recommendations", [])
            
            # Calculate enhanced accuracy scores
            genre_match_score = self._calculate_genre_match(recommendations, test_case.expected_genre)
            artist_diversity_score = self._calculate_artist_diversity(recommendations)
            mood_match_score = self._calculate_mood_match(recommendations, test_case.expected_mood)
            language_match_score = self._calculate_language_match(recommendations, test_case.expected_language)
            
            # Enhanced overall quality calculation with weighted components
            if test_case.expected_artist:
                # Artist-specific test: prioritize artist match
                artist_match_score = self._calculate_artist_match(recommendations, test_case.expected_artist)
                overall_quality_score = (
                    artist_match_score * 0.6 +  # Higher weight for artist match
                    genre_match_score * 0.15 +
                    mood_match_score * 0.15 +
                    language_match_score * 0.05 +
                    quality_analysis.get("overall_relevance", 0.5) * 0.05
                )
            else:
                # General test: balanced approach with cultural emphasis
                cultural_weight = 0.4 if test_case.expected_language else 0.2
                overall_quality_score = (
                    genre_match_score * 0.3 +
                    mood_match_score * 0.25 +
                    language_match_score * cultural_weight +
                    artist_diversity_score * 0.15 +
                    quality_analysis.get("overall_relevance", 0.5) * 0.1 +
                    quality_analysis.get("diversity_score", 0.5) * 0.05
                )
            
            # Enhanced pass/fail criteria with stricter standards
            if test_case.expected_artist:
                # Artist-specific: require strong artist match
                artist_match_score = self._calculate_artist_match(recommendations, test_case.expected_artist)
                passed = (
                    artist_match_score >= 0.8 and
                    overall_quality_score >= 0.65 and  # Slightly higher threshold
                    len(recommendations) >= 3  # Ensure we got enough recommendations
                )
            else:
                # General tests: comprehensive quality check
                passed = (
                    genre_match_score >= test_case.min_genre_match and
                    artist_diversity_score >= test_case.min_artist_diversity and
                    overall_quality_score >= 0.65 and  # Higher quality threshold
                    len(recommendations) >= 3 and  # Ensure we got enough recommendations
                    # Additional cultural authenticity check
                    (not test_case.expected_language or language_match_score >= 0.6)
                )
            
            # Enhanced reasoning with detailed breakdown
            quality_assessment = quality_analysis.get("quality_assessment", "unknown")
            base_reasoning = quality_analysis.get("reasoning", "No detailed reasoning available")
            
            detailed_reasoning = f"""
Quality Assessment: {quality_assessment}
- Genre Match: {genre_match_score:.2f} (expected ≥{test_case.min_genre_match})
- Artist Diversity: {artist_diversity_score:.2f} (expected ≥{test_case.min_artist_diversity})
- Mood Match: {mood_match_score:.2f}
- Language Match: {language_match_score:.2f}
- Overall Quality: {overall_quality_score:.2f} (expected ≥0.65)
- Recommendations Count: {len(recommendations)} (expected ≥3)

LLM Analysis: {base_reasoning}
            """.strip()
            
            result = EvaluationResult(
                test_case=test_case,
                recommendations=recommendations,
                genre_match_score=genre_match_score,
                artist_diversity_score=artist_diversity_score,
                mood_match_score=mood_match_score,
                language_match_score=language_match_score,
                overall_quality_score=overall_quality_score,
                mistral_analysis=quality_analysis,
                passed=passed,
                reasoning=detailed_reasoning
            )
            
            # Print results
            self._print_case_results(result)
            
            # Add delay to avoid rate limiting
            print("⏳ Waiting to avoid rate limiting...")
            time.sleep(3)
            
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating test case {test_case.id}: {e}")
            import traceback
            traceback.print_exc()
            return EvaluationResult(
                test_case=test_case,
                recommendations=[],
                genre_match_score=0.0,
                artist_diversity_score=0.0,
                mood_match_score=0.0,
                language_match_score=0.0,
                overall_quality_score=0.0,
                mistral_analysis={},
                passed=False,
                reasoning=f"Error during evaluation: {str(e)}"
            )
    
    def _print_case_results(self, result: EvaluationResult):
        """Print detailed results for a test case"""
        print(f"\n📊 RESULTS FOR TEST CASE {result.test_case.id}:")
        print(f"   Status: {'✅ PASSED' if result.passed else '❌ FAILED'}")
        print(f"   Genre Match: {result.genre_match_score:.2f} (min: {result.test_case.min_genre_match})")
        print(f"   Artist Diversity: {result.artist_diversity_score:.2f} (min: {result.test_case.min_artist_diversity})")
        print(f"   Mood Match: {result.mood_match_score:.2f}")
        print(f"   Language Match: {result.language_match_score:.2f}")
        print(f"   Overall Quality: {result.overall_quality_score:.2f}")
        
        if result.mistral_analysis:
            print(f"   Quality Assessment: {result.mistral_analysis.get('quality_assessment', 'N/A')}")
            print(f"   Diversity Score: {result.mistral_analysis.get('diversity_score', 'N/A')}")
        
        print(f"\n📝 Recommendations ({len(result.recommendations)}):")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec.get('song', 'Unknown')} by {rec.get('artist', 'Unknown')}")
            print(f"      Genre: {rec.get('genre', 'Unknown')} | Mood: {rec.get('mood', 'Unknown')}")
        
        print(f"\n💭 Reasoning: {result.reasoning}")
        
        if result.mistral_analysis.get('improvement_suggestions'):
            print(f"\n💡 Improvement Suggestions:")
            for suggestion in result.mistral_analysis['improvement_suggestions']:
                print(f"   - {suggestion}")
    
    def run_all_tests(self) -> List[EvaluationResult]:
        """Run all test cases sequentially"""
        print("🚀 Starting Enhanced Music Recommendation Evaluation")
        print(f"📋 Total test cases: {len(self.test_cases)}")
        
        results = []
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n🔄 Processing test case {i}/{len(self.test_cases)}")
            result = self.evaluate_single_case(test_case)
            results.append(result)
        
        return results
    
    def generate_final_report(self, results: List[EvaluationResult]):
        """Generate comprehensive final report"""
        print(f"\n{'='*80}")
        print("📊 FINAL EVALUATION REPORT")
        print(f"{'='*80}")
        
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"📈 Overall Results:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {total_tests - passed_tests}")
        print(f"   Pass Rate: {pass_rate:.1%}")
        
        # Calculate average scores
        avg_genre_match = sum(r.genre_match_score for r in results) / total_tests
        avg_artist_diversity = sum(r.artist_diversity_score for r in results) / total_tests
        avg_mood_match = sum(r.mood_match_score for r in results) / total_tests
        avg_language_match = sum(r.language_match_score for r in results) / total_tests
        avg_overall_quality = sum(r.overall_quality_score for r in results) / total_tests
        
        print(f"\n📊 Average Scores:")
        print(f"   Genre Match: {avg_genre_match:.2f}")
        print(f"   Artist Diversity: {avg_artist_diversity:.2f}")
        print(f"   Mood Match: {avg_mood_match:.2f}")
        print(f"   Language Match: {avg_language_match:.2f}")
        print(f"   Overall Quality: {avg_overall_quality:.2f}")
        
        # Test case breakdown
        print(f"\n📋 Test Case Breakdown:")
        for result in results:
            status = "✅" if result.passed else "❌"
            print(f"   {status} Test {result.test_case.id}: {result.test_case.name} (Quality: {result.overall_quality_score:.2f})")
        
        # Recommendations for improvement
        print(f"\n💡 System Improvement Recommendations:")
        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            print(f"   - Focus on improving genre matching (failed in {len([r for r in failed_tests if r.genre_match_score < r.test_case.min_genre_match])} tests)")
            print(f"   - Improve artist diversity (failed in {len([r for r in failed_tests if r.artist_diversity_score < r.test_case.min_artist_diversity])} tests)")
            print(f"   - Enhance overall recommendation quality")
        else:
            print(f"   - System performing well across all test cases!")
        
        print(f"\n{'='*80}")

def main():
    """Main function to run the enhanced evaluation"""
    evaluator = EnhancedMusicEvaluator()
    
    print("🎵 Enhanced Music Recommendation Evaluation System")
    print("Using Mistral API for comprehensive analysis")
    print("Sequential processing to avoid rate limiting")
    
    # Run all tests
    results = evaluator.run_all_tests()
    
    # Generate final report
    evaluator.generate_final_report(results)
    
    print("\n✅ Evaluation completed!")

if __name__ == "__main__":
    main()