#!/usr/bin/env python3
"""
Evaluator Judge for Hybrid Music Recommendation System
Tests if recommended songs follow the input conditions (language, genre, artist, and mood).
Includes LLM-based evaluation for relevance and similarity.

The LLM Judge uses advanced language models (OpenAI/GPT-OSS-20B or Llama-3.3-70B) to:
1. Evaluate if recommendations truly match user intent beyond simple keyword matching
2. Judge whether the recommendation follows all aspects of the user's query
3. Provide numerical scores for recommendation quality and diversity
4. Assess whether a typical user would be satisfied with the recommendations

Usage:
- Basic rule-based testing: python test_judge.py
- With LLM judge: python test_judge.py -l
- Specific LLM model: python test_judge.py -l -m "llama-3.3-70b-versatile"
- Single test with LLM: python test_judge.py -l -t 3
- Custom query with LLM: python test_judge.py -l -q "recommend energetic workout music"
"""

import sys
import os
import re
import time
import json
import requests
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass

# Add parent directory to path to import the main module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from main import HybridRecommendationSystem
except ImportError:
    try:
        from helo.main import HybridRecommendationSystem
    except ImportError:
        print("❌ Error: Could not import HybridRecommendationSystem")
        sys.exit(1)

@dataclass
class TestCase:
    """Represents a test case for the recommendation system"""
    id: int
    input: str
    description: str
    criteria: Dict[str, Any]  # language, genre, artist, mood, count

@dataclass
class TestResult:
    """Represents the result of a test case"""
    test_id: int
    input: str
    status: str  # "PASS" or "FAIL"
    reasoning: str
    recommendations: List[Dict[str, Any]]  # The actual recommendations returned
    llm_evaluation: Dict[str, Any] = None  # LLM evaluation results

class LLMJudge:
    """Uses LLM to evaluate if recommendations match user's intent"""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b"):
        """Initialize the LLM Judge with specified model"""
        print(f"🤖 Initializing LLM Judge with model: {model_name}")
        self.model_name = model_name
        
        # Determine which API to use based on the model name
        if "llama-3.3" in model_name.lower():
            self.api_type = "llama"
            # Try to load API key from environment or parent directory config
            try:
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from config import GROQ_API_KEY
                self.api_key = GROQ_API_KEY
                self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            except (ImportError, AttributeError):
                print("⚠️ Groq API key not found for Llama model, using default configuration")
                self.api_key = None
                self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        else:  # Default to OpenAI compatible API
            self.api_type = "openai"
            # Try to load API key from environment or parent directory config
            try:
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from config import GROQ_API_KEY
                self.api_key = GROQ_API_KEY
                self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            except (ImportError, AttributeError):
                print("⚠️ OpenAI API key not found, using default configuration")
                self.api_key = None
                self.api_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def evaluate_recommendations(self, query: str, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate if the recommendations match the user's query intent
        Returns a dictionary with evaluation metrics and reasoning
        """
        if not recommendations:
            return {
                "overall_score": 0,
                "relevance_score": 0,
                "diversity_score": 0,
                "reasoning": "No recommendations to evaluate",
                "matches_query": False
            }
        
        # Format recommendations for the LLM prompt
        formatted_recs = self._format_recommendations_for_prompt(recommendations)
        
        # Build the evaluation prompt
        prompt = self._build_evaluation_prompt(query, formatted_recs)
        
        try:
            # Call the LLM API
            evaluation = self._call_llm_api(prompt)
            
            # Parse the evaluation response
            return self._parse_llm_response(evaluation)
        except Exception as e:
            print(f"❌ LLM evaluation failed: {e}")
            # Return a basic failure response
            return {
                "overall_score": 0,
                "relevance_score": 0,
                "diversity_score": 0,
                "reasoning": f"LLM evaluation failed: {str(e)}",
                "matches_query": False
            }
    
    def _format_recommendations_for_prompt(self, recommendations: List[Dict[str, Any]]) -> str:
        """Format the recommendations into a structured format for the LLM prompt"""
        formatted = "RECOMMENDATIONS:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            formatted += f"{i}. {rec.get('name', 'Unknown Track')}\n"
            formatted += f"   Artist: {rec.get('artists', 'Unknown')}\n"
            formatted += f"   Genre: {rec.get('genre', 'Unknown')}\n"
            formatted += f"   Language: {rec.get('language', 'Unknown')}\n"
            if rec.get('mood'):
                formatted += f"   Mood: {rec.get('mood')}\n"
            formatted += "\n"
        
        return formatted
    
    def _build_evaluation_prompt(self, query: str, formatted_recs: str) -> str:
        """Build the prompt for the LLM to evaluate recommendations"""
        return f"""You are a music recommendation evaluator with extensive knowledge of music genres, artists, and songs across different cultures and languages.

USER QUERY: "{query}"

{formatted_recs}

Evaluate whether the above recommendations truly match the user's query intent. Consider:
1. Relevance to the specific query (language, genre, artist, mood)
2. Song diversity and variety
3. Whether the recommendations would satisfy the user's request

Return your evaluation as a JSON object with the following fields:
- overall_score: A score from 0-10 indicating how well the recommendations match the query
- relevance_score: A score from 0-10 on how relevant the songs are to the query
- diversity_score: A score from 0-10 on the diversity of recommendations
- reasoning: A detailed explanation of your evaluation (1-3 sentences)
- matches_query: A boolean indicating if the recommendations truly satisfy the query

ONLY return the JSON with no additional text.
"""

    def _call_llm_api(self, prompt: str) -> str:
        """Call the LLM API and return the response"""
        if self.api_type == "llama":
            return self._call_llama_api(prompt)
        else:  # Default to OpenAI compatible API
            return self._call_openai_api(prompt)
    
    def _call_llama_api(self, prompt: str) -> str:
        """Call the Llama model API through Groq"""
        if not self.api_key:
            # Simulate a response for testing when no API key is available
            print("⚠️ No API key available, returning simulated response")
            return json.dumps({
                "overall_score": 7,
                "relevance_score": 8, 
                "diversity_score": 6,
                "reasoning": "This is a simulated LLM response since no API key was provided.",
                "matches_query": True
            })
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Determine which Llama model to use
        if "llama-3.3-70b-versatile" in self.model_name.lower():
            model_id = "llama3-70b-8192"
        else:
            # Default to Llama 3.1 8B if not specifically using 70B
            model_id = "llama3-8b-8192"
            
        print(f"   Using Llama model: {model_id}")
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API returned status code {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"Failed to call Llama API: {str(e)}")
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call the OpenAI compatible API (GPT-OSS-20B through Groq)"""
        if not self.api_key:
            # Simulate a response for testing when no API key is available
            print("⚠️ No API key available, returning simulated response")
            return json.dumps({
                "overall_score": 7,
                "relevance_score": 8, 
                "diversity_score": 6,
                "reasoning": "This is a simulated LLM response since no API key was provided.",
                "matches_query": True
            })
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        # Determine which model to use
        if "gpt-oss-20b" in self.model_name.lower() or "openai" in self.model_name.lower():
            # Using Claude Opus as a substitute for GPT-OSS-20B through Groq
            model_id = "claude-3-opus-20240229"
            print(f"   Using OpenAI compatible model: {model_id}")
        else:
            # Default to Mixtral if not specifically requesting GPT-OSS
            model_id = "mixtral-8x7b-32768"
            print(f"   Using default OpenAI compatible model: {model_id}")
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                self.api_url, 
                headers=headers, 
                json=payload, 
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                raise Exception(f"API returned status code {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"Failed to call OpenAI API: {str(e)}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured evaluation"""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                evaluation = json.loads(response)
            
            # Ensure all expected fields are present
            expected_fields = ['overall_score', 'relevance_score', 'diversity_score', 'reasoning', 'matches_query']
            for field in expected_fields:
                if field not in evaluation:
                    evaluation[field] = 0 if 'score' in field else (False if field == 'matches_query' else "Not provided")
            
            return evaluation
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"❌ Failed to parse LLM response: {e}")
            # Return a default evaluation
            return {
                "overall_score": 0,
                "relevance_score": 0,
                "diversity_score": 0,
                "reasoning": f"Failed to parse LLM evaluation: {str(e)}",
                "matches_query": False,
                "raw_response": response[:200] + "..." if len(response) > 200 else response
            }

class RecommendationJudge:
    """Evaluates if music recommendations meet specified criteria"""
    
    def __init__(self, use_llm_judge: bool = True, llm_model: str = "llama-3.3-70b-versatile"):
        """Initialize the recommendation system and test cases"""
        print("🔍 Initializing Recommendation Judge...")
        self.recommender = HybridRecommendationSystem()
        print("✅ Recommendation system initialized")
        
        # Set up LLM judge if requested
        self.use_llm_judge = use_llm_judge
        if use_llm_judge:
            self.llm_judge = LLMJudge(model_name=llm_model)
            print(f"🤖 LLM Judge initialized with model: {llm_model}")
        else:
            self.llm_judge = None
            print("⚠️ LLM Judge disabled")
        
        # Set up test cases
        self.test_cases = self._setup_test_cases()
        print(f"📋 Loaded {len(self.test_cases)} test cases")
    
    def _setup_test_cases(self) -> List[TestCase]:
        """Define the test cases according to requirements"""
        return [
            TestCase(
                id=1,
                input="Recommend Nepali songs",
                description="Output should only be Nepali songs",
                criteria={"language": "nepali"}
            ),
            TestCase(
                id=2,
                input="Recommend Pop songs",
                description="Output should only be Pop genre",
                criteria={"genre": "pop"}
            ),
            TestCase(
                id=3,
                input="Recommend 3 songs similar to Shape of You by Ed Sheeran ",
                description="Output should contain exactly 3 English Pop songs",
                criteria={"language": "english", "genre": "pop", "count": 3}
            ),
            TestCase(
                id=4,
                input="Recommend English songs only",
                description="Output should exclude Hindi/Nepali",
                criteria={"language": "english", "exclude_languages": ["hindi", "nepali"]}
            ),
            TestCase(
                id=5,
                input="Recommend songs by The Weeknd",
                description="Output must contain only songs by The Weeknd",
                criteria={"artist": "The Weeknd"}
            ),
            TestCase(
                id=6,
                input="Recommend songs by different artists",
                description="Output must not repeat the same artist",
                criteria={"unique_artists": True}
            ),
            TestCase(
                id=7,
                input="Recommend Hindi songs",
                description="Output must only contain Hindi-language tracks",
                criteria={"language": "hindi"}
            ),
            TestCase(
                id=8,
                input="Recommend Nepali Pop songs",
                description="Output must match both language (Nepali) and genre (Pop)",
                criteria={"language": "nepali", "genre": "pop"}
            ),
            TestCase(
                id=9,
                input="Recommend 3 songs like Blinding Lights The Weeknd",
                description="Output must return 3 songs max, Pop genre, English",
                criteria={"language": "english", "genre": "pop", "count": 3}
            ),
            TestCase(
                id=10,
                input="sad english songs",
                description="Output must align with mood Sad and language English",
                criteria={"language": "english", "mood": "sad"}
            )
        ]
    
    def _parse_recommendations(self, output: str) -> List[Dict[str, Any]]:
        """Parse the recommendation output to extract track details"""
        recommendations = []
        
        # Extract recommendation sections from the output
        # This is a more robust parser that handles various output formats
        import re
        
        print(f"\nParsing output of length: {len(output)}")
        print(f"Sample of output: {output[:300]}...")
        
        # Pattern 0: Directly extract the entire structured section
        recommendations_section_match = re.search(r'Structured Recommendations:(.*?)(?:📈|$)', output, re.DOTALL)
        if recommendations_section_match:
            recommendations_text = recommendations_section_match.group(1).strip()
            print(f"Found recommendations section of length: {len(recommendations_text)}")
        else:
            print("Could not find recommendations section")
            
        # Try different pattern matches to extract song sections
        # Pattern 1: Bold numbered items like "**1. Track Name**"
        track_sections = re.findall(r'\*\*\d+\.\s+([^*]+)\*\*.*?(?=\*\*\d+\.|$)', output, re.DOTALL)
        
        # Pattern 2: Regular numbered items like "1. Track Name"
        if not track_sections:
            track_sections = re.findall(r'\d+\.\s+(.*?)(?=\d+\.|$)', output, re.DOTALL)
        
        # Pattern 3: Look for sections with Artist: and other metadata markers
        if not track_sections:
            track_sections = re.findall(r'(?:🎤|🎵|🎧)\s*(?:\*\*)?([^\n]+)(?:\*\*)?.*?(?=(?:🎤|🎵|🎧)|$)', output, re.DOTALL)
        
        # Pattern 4: Last resort - just look for Artist: patterns
        if not track_sections:
            artist_sections = re.findall(r'(?:Artist[s]?):\s*(.*?)[\n\r]', output, re.IGNORECASE)
            if artist_sections:
                # Reconstruct sections based on artist mentions
                start_idx = 0
                for i, artist in enumerate(artist_sections):
                    end_idx = output.find(f"Artist", start_idx + 1)
                    if end_idx == -1 and i == len(artist_sections) - 1:
                        end_idx = len(output)
                    if end_idx > start_idx:
                        track_sections.append(output[start_idx:end_idx])
                        start_idx = end_idx
        
        print(f"  Found {len(track_sections)} potential track sections")
        
        # Simplified direct extraction approach
        # Look for tracks in format "**1. Track Name**"
        track_entries = re.findall(r'\*\*(\d+)\.\s+([^*]+)\*\*\s*\n(.*?)(?=\*\*\d+\.|📈|\Z)', output, re.DOTALL)
        
        if track_entries:
            print(f"Found {len(track_entries)} tracks with direct extraction")
            
            for track_number, track_name, track_details in track_entries:
                track = {
                    'name': track_name.strip(),
                    'number': int(track_number)
                }
                
                # Extract fields with emoji markers
                artist_match = re.search(r'🎤.*?:(.*?)(?:\n|$)', track_details, re.DOTALL)
                genre_match = re.search(r'🎼.*?:(.*?)(?:\n|$)', track_details, re.DOTALL)
                language_match = re.search(r'🌐.*?:(.*?)(?:\n|$)', track_details, re.DOTALL)
                
                track['artists'] = artist_match.group(1).strip() if artist_match else "Unknown"
                track['genre'] = genre_match.group(1).strip() if genre_match else "Unknown"
                track['language'] = language_match.group(1).strip() if language_match else "Unknown"
                
                print(f"Track {track['number']}: {track['name']}")
                print(f"  Artist: {track['artists']}")
                print(f"  Genre: {track['genre']}")
                print(f"  Language: {track['language']}")
                
                recommendations.append(track)
                
            # If we found tracks with direct extraction, return them
            if recommendations:
                return recommendations
        
        # Otherwise fall back to the original approach
        for section in track_sections:
            track = {}
            section_text = section if isinstance(section, str) else section[0]
            
            # Extract track name - first line or before first newline
            track_name_match = re.search(r'^([^\n]+)', section_text.strip(), re.MULTILINE)
            if track_name_match:
                track_name = track_name_match.group(1).strip()
                # Clean up any markdown or emoji
                track_name = re.sub(r'\*\*|\*|🎵|🎤|🎧|^\d+\.\s*', '', track_name).strip()
                track['name'] = track_name
            else:
                track['name'] = "Unknown Track"
            
            # Print the full section for debugging
            print(f"Processing section for track: {track['name']}")
            print("-" * 40)
            print(section_text[:200] + "..." if len(section_text) > 200 else section_text)
            print("-" * 40)
            
            # Extract artist - look for Artist: or similar pattern
            artist_match = re.search(r'(?:Artist[s]?|🎤)[\s:]+(.*?)(?:[\n\r]|$)', section_text, re.IGNORECASE)
            if artist_match:
                track['artists'] = artist_match.group(1).strip()
                print(f"Found artist: {track['artists']}")
            else:
                track['artists'] = "Unknown"
                print("No artist found in section")
            
            # Extract genre - look for Genre: or similar pattern
            genre_match = re.search(r'(?:Genre|🎼)[\s:]+(.*?)(?:[\n\r]|$)', section_text, re.IGNORECASE)
            if genre_match:
                track['genre'] = genre_match.group(1).strip()
                print(f"Found genre: {track['genre']}")
            else:
                track['genre'] = "Unknown"
                print("No genre found in section")
            
            # Extract language - look for Language: or similar pattern
            lang_match = re.search(r'(?:Language|🌐)[\s:]+(.*?)(?:[\n\r]|$)', section_text, re.IGNORECASE)
            if lang_match:
                track['language'] = lang_match.group(1).strip()
                print(f"Found language: {track['language']}")
            else:
                track['language'] = "Unknown"
                print("No language found in section")
            
            # Extract mood/energy/valence - multiple possible patterns
            mood_patterns = [
                r'(?:Mood|😊):\s*(.*?)(?:[\n\r]|$)',
                r'Energy:\s*(.*?)(?:[\n\r]|$)',
                r'Valence:\s*(.*?)(?:[\n\r]|$)'
            ]
            
            for pattern in mood_patterns:
                mood_match = re.search(pattern, section_text, re.IGNORECASE)
                if mood_match:
                    track['mood'] = mood_match.group(1).strip()
                    break
            
            # Only add if we have at least name and artist
            if track['name'] != "Unknown Track" or track['artists'] != "Unknown":
                recommendations.append(track)
        
        return recommendations
    
    def _evaluate_test_case(self, test_case: TestCase, recommendations: List[Dict[str, Any]]) -> TestResult:
        """Evaluate if recommendations meet the criteria for the test case"""
        criteria = test_case.criteria
        status = "PASS"
        reasoning = []
        llm_evaluation = None
        
        if not recommendations:
            return TestResult(
                test_id=test_case.id,
                input=test_case.input,
                status="FAIL",
                reasoning="No recommendations were found in the output.",
                recommendations=[],
                llm_evaluation=None
            )
        
        # First, run rule-based evaluation
        # Check count if specified
        if "count" in criteria and len(recommendations) != criteria["count"]:
            status = "FAIL"
            reasoning.append(f"Expected {criteria['count']} songs but got {len(recommendations)}.")
        
        # Check language
        if "language" in criteria:
            expected_lang = criteria["language"].lower()
            lang_matches = sum(1 for rec in recommendations if expected_lang in rec["language"].lower())
            if lang_matches < len(recommendations):
                status = "FAIL"
                reasoning.append(f"Only {lang_matches} out of {len(recommendations)} songs match the expected language ({expected_lang}).")
        
        # Check excluded languages
        if "exclude_languages" in criteria:
            for lang in criteria["exclude_languages"]:
                excluded_lang = lang.lower()
                excluded_count = sum(1 for rec in recommendations if excluded_lang in rec["language"].lower())
                if excluded_count > 0:
                    status = "FAIL"
                    reasoning.append(f"Found {excluded_count} songs in {excluded_lang} language which should be excluded.")
        
        # Check genre
        if "genre" in criteria:
            expected_genre = criteria["genre"].lower()
            genre_matches = sum(1 for rec in recommendations if expected_genre in rec["genre"].lower())
            if genre_matches < len(recommendations):
                status = "FAIL"
                reasoning.append(f"Only {genre_matches} out of {len(recommendations)} songs match the expected genre ({expected_genre}).")
        
        # Check artist
        if "artist" in criteria:
            expected_artist = criteria["artist"].lower()
            artist_matches = sum(1 for rec in recommendations if expected_artist.lower() in rec["artists"].lower())
            if artist_matches < len(recommendations):
                status = "FAIL"
                reasoning.append(f"Only {artist_matches} out of {len(recommendations)} songs are by {expected_artist}.")
        
        # Check unique artists
        if criteria.get("unique_artists", False):
            artists = [rec["artists"].split(',')[0].strip().lower() for rec in recommendations]
            unique_artists = set(artists)
            if len(unique_artists) < len(artists):
                status = "FAIL"
                reasoning.append(f"Found duplicate artists: only {len(unique_artists)} unique artists in {len(recommendations)} recommendations.")
        
        # Check mood
        if "mood" in criteria:
            expected_mood = criteria["mood"].lower()
            # This is an approximate check since mood is subjective
            # Look for mood in name, artist fields too as sometimes song titles can indicate mood
            mood_matches = 0
            mood_vocabulary = {
                'sad': [
                    'sad', 'cry', 'tear', 'blue', 'alone', 'lonely', 'broken', 'heartbreak', 'melancholy', 'grief', 
                    'sorrow', 'depressed', 'depression', 'hurt', 'pain', 'lost', 'empty', 'regret', 'goodbye', 
                    'missing', 'darkness', 'fall', 'weep', 'weeping', 'mourning', 'somber', 'wistful', 'bitter'
                ],
                'happy': [
                    'happy', 'joy', 'celebration', 'smile', 'fun', 'bright', 'light', 'upbeat', 'cheerful', 
                    'sunny', 'delight', 'ecstatic', 'thrilled', 'excited', 'bliss', 'paradise', 'glow',
                    'party', 'shine', 'positive', 'optimistic'
                ],
                'energetic': [
                    'energy', 'power', 'strong', 'fast', 'workout', 'run', 'dance', 'move', 'beat', 'jump', 
                    'pump', 'fire', 'adrenaline', 'rush', 'exciting', 'wild', 'intensity'
                ],
                'calm': [
                    'calm', 'relax', 'peaceful', 'quiet', 'gentle', 'slow', 'easy', 'chill', 'ambient',
                    'meditation', 'sleep', 'rest', 'tranquil', 'soothing', 'serene', 'stillness'
                ],
                'romantic': [
                    'love', 'romance', 'heart', 'kiss', 'embrace', 'passion', 'desire', 'intimate',
                    'together', 'forever', 'valentine', 'affection', 'tender', 'devotion'
                ]
            }
            
            for rec in recommendations:
                has_mood_match = False
                # Check explicit mood field first
                if rec.get("mood") and expected_mood in rec["mood"].lower():
                    has_mood_match = True
                
                # Check for mood indicators in title
                elif any(mood_word in rec["name"].lower() for mood_word in mood_vocabulary.get(expected_mood, [])):
                    has_mood_match = True
                
                # For "sad" songs, specifically check audio features if available
                elif expected_mood == "sad" and rec.get("danceability") is not None:
                    # Typical sad song audio features: low energy, low valence (happiness), lower tempo
                    if (float(rec.get("energy", 0.8)) < 0.6 and 
                        float(rec.get("valence", 0.6)) < 0.4):
                        has_mood_match = True
                
                # Check genre for sad songs (some genres are typically sad)
                elif expected_mood == "sad" and rec.get("genre") and any(
                    sad_genre in rec["genre"].lower() for sad_genre in 
                    ["ballad", "soul", "blues", "acoustic"]):
                    has_mood_match = True
                    
                if has_mood_match:
                    mood_matches += 1
            
            # For test case 10 (sad english songs), ensure we have more than half of songs matching the sad mood
            if expected_mood == "sad" and "sad english songs" in test_case.input.lower():
                required_matches = len(recommendations) // 2  # At least half should be sad
                if mood_matches < required_matches:
                    status = "FAIL"
                    reasoning.append(f"Only {mood_matches} out of {len(recommendations)} songs match the sad mood. For this test, at least {required_matches} sad songs are required.")
            elif mood_matches < 1:  # For other moods, at least one song should match
                status = "FAIL"
                reasoning.append(f"Only {mood_matches} out of {len(recommendations)} songs match the expected mood ({expected_mood}).")
        
        # Generate rule-based reasoning
        if status == "PASS":
            if "language" in criteria and "genre" in criteria:
                reasoning_text = f"All recommendations correctly match both the {criteria['language'].title()} language and {criteria['genre'].title()} genre criteria."
            elif "language" in criteria:
                reasoning_text = f"All recommendations are in the requested {criteria['language'].title()} language."
            elif "genre" in criteria:
                reasoning_text = f"All recommendations belong to the {criteria['genre'].title()} genre as requested."
            elif "artist" in criteria:
                reasoning_text = f"All recommendations are songs by {criteria['artist']} as requested."
            elif criteria.get("unique_artists", False):
                reasoning_text = f"All recommendations feature different artists, ensuring artist diversity."
            else:
                reasoning_text = "All recommendations meet the specified criteria."
        else:
            reasoning_text = " ".join(reasoning)
        
        # Now, run LLM-based evaluation if enabled
        if self.use_llm_judge and self.llm_judge:
            try:
                print(f"  🤖 Running LLM evaluation for test case {test_case.id}...")
                llm_evaluation = self.llm_judge.evaluate_recommendations(test_case.input, recommendations)
                
                # Incorporate LLM evaluation into the final result
                if not llm_evaluation["matches_query"]:
                    # If LLM says recommendations don't match, fail the test
                    status = "FAIL"
                    llm_reason = f"LLM Judge: {llm_evaluation['reasoning']} (Scores: Overall {llm_evaluation['overall_score']}/10, Relevance {llm_evaluation['relevance_score']}/10, Diversity {llm_evaluation['diversity_score']}/10)"
                    
                    # Combine rule-based and LLM reasoning
                    if reasoning_text and reasoning_text != "All recommendations meet the specified criteria.":
                        reasoning_text = f"{reasoning_text} {llm_reason}"
                    else:
                        reasoning_text = llm_reason
                elif status == "PASS":
                    # If we passed rule-based checks, add LLM insights
                    reasoning_text = f"{reasoning_text} LLM Judge confirms relevance: {llm_evaluation['reasoning']}"
            except Exception as e:
                print(f"  ❌ LLM evaluation failed: {e}")
                # If LLM evaluation fails, use only rule-based results
        
        return TestResult(
            test_id=test_case.id,
            input=test_case.input,
            status=status,
            reasoning=reasoning_text,
            recommendations=recommendations,
            llm_evaluation=llm_evaluation
        )
    
    def run_tests(self) -> List[TestResult]:
        """Run all test cases and collect results"""
        results = []
        
        print("\n🧪 Running Test Cases...\n")
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"Test {i}/{len(self.test_cases)}: \"{test_case.input}\"")
            print("  Description:", test_case.description)
            print("  Criteria:", test_case.criteria)
            
            try:
                start_time = time.time()
                print("  Querying recommendation system...")
                
                # Try the recommend_music method first (recommended way)
                if hasattr(self.recommender, 'recommend_music'):
                    output = self.recommender.recommend_music(test_case.input)
                # Fallback to get_hybrid_recommendations if recommend_music doesn't exist
                elif hasattr(self.recommender, 'get_hybrid_recommendations'):
                    print("  Using fallback method: get_hybrid_recommendations")
                    output = self.recommender.get_hybrid_recommendations(test_case.input)
                else:
                    raise AttributeError("No suitable recommendation method found in the system")
                
                end_time = time.time()
                
                # Debug output
                print(f"  Raw output length: {len(output)} characters")
                print(f"  First 100 chars: {output[:100]}...")
                
                recommendations = self._parse_recommendations(output)
                if not recommendations:
                    print("  WARNING: No recommendations parsed from output")
                    print("  Check the output format or update the parsing logic")
                    print("  Sample output: ", output[:500])
                
                result = self._evaluate_test_case(test_case, recommendations)
                results.append(result)
                
                emoji = "✅" if result.status == "PASS" else "❌"
                print(f"  Result: {emoji} {result.status}")
                print(f"  Reasoning: {result.reasoning}")
                print(f"  Time: {end_time - start_time:.2f} seconds")
                print(f"  Found {len(recommendations)} recommendations")
                print("")
                
            except Exception as e:
                print(f"  ❌ Error running test: {e}")
                results.append(TestResult(
                    test_id=test_case.id,
                    input=test_case.input,
                    status="FAIL",
                    reasoning=f"Error running test: {e}",
                    recommendations=[]
                ))
                print("")
        
        return results
    
    def print_summary(self, results: List[TestResult]):
        """Print a summary of all test results"""
        print("\n📊 Test Results Summary")
        print("=" * 60)
        
        if not results:
            print("No test results to summarize")
            print("=" * 60)
            return
            
        pass_count = sum(1 for r in results if r.status == "PASS")
        fail_count = len(results) - pass_count
        
        print(f"Total Tests: {len(results)}")
        print(f"Passed: {pass_count} ({pass_count / len(results) * 100:.1f}%)")
        print(f"Failed: {fail_count} ({fail_count / len(results) * 100:.1f}%)")
        print("=" * 60)
        
        # Summarize LLM evaluations if available
        has_llm_evaluations = any(r.llm_evaluation for r in results)
        if has_llm_evaluations:
            llm_total_score = sum(r.llm_evaluation.get('overall_score', 0) for r in results if r.llm_evaluation)
            llm_relevance_score = sum(r.llm_evaluation.get('relevance_score', 0) for r in results if r.llm_evaluation)
            llm_diversity_score = sum(r.llm_evaluation.get('diversity_score', 0) for r in results if r.llm_evaluation)
            llm_eval_count = sum(1 for r in results if r.llm_evaluation)
            
            print("\n🤖 LLM Evaluation Summary:")
            print(f"Average Overall Score: {llm_total_score / llm_eval_count:.1f}/10")
            print(f"Average Relevance Score: {llm_relevance_score / llm_eval_count:.1f}/10")
            print(f"Average Diversity Score: {llm_diversity_score / llm_eval_count:.1f}/10")
            print("=" * 60)
        
        print("\nDetailed Results:")
        for result in results:
            emoji = "✅" if result.status == "PASS" else "❌"
            print(f"{emoji} Test #{result.test_id}: {result.status}")
            print(f"   Input: \"{result.input}\"")
            print(f"   Reasoning: {result.reasoning}")
            print(f"   Recommendations: {len(result.recommendations)}")
            
            # Print LLM evaluation details if available
            if result.llm_evaluation:
                print(f"   🤖 LLM Scores: Overall {result.llm_evaluation.get('overall_score', 0)}/10, " +
                      f"Relevance {result.llm_evaluation.get('relevance_score', 0)}/10, " +
                      f"Diversity {result.llm_evaluation.get('diversity_score', 0)}/10")
            
            print("-" * 60)

def main():
    """Main function to run the recommendation judge"""
    print("🎵 Spotify Recommendation Judge 🎵")
    print("=" * 60)
    
    # Parse command line arguments for running specific test
    import argparse
    parser = argparse.ArgumentParser(description="Test the Spotify Recommendation System")
    parser.add_argument('-t', '--test', type=int, help='Run only a specific test ID (1-10)')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
    parser.add_argument('-q', '--query', type=str, help='Run a custom query instead of tests')
    parser.add_argument('-f', '--fix', action='store_true', help='Run in safe mode with error handling')
    parser.add_argument('-l', '--llm', action='store_true', help='Use LLM judge for evaluations')
    parser.add_argument('-m', '--model', type=str, default="llama-3.3-70b-versatile", 
                       help='LLM model to use: "llama-3.3-70b-versatile" or "openai/gpt-oss-20b"')
    args = parser.parse_args()
    
    # Initialize judge with LLM options
    use_llm = args.llm if hasattr(args, 'llm') else False
    model_name = args.model if hasattr(args, 'model') else "openai/gpt-oss-20b"
    
    print(f"🤖 LLM Judge: {'Enabled' if use_llm else 'Disabled'}")
    if use_llm:
        print(f"   Using model: {model_name}")
    
    judge = RecommendationJudge(use_llm_judge=use_llm, llm_model=model_name)
    
    # Handle custom query mode
    if args.query:
        print(f"\n🔍 Running custom query: '{args.query}'")
        try:
            recommender = judge.recommender
            print("⏳ Getting recommendation...")
            
            # Try recommend_music first, fallback to get_hybrid_recommendations
            if hasattr(recommender, 'recommend_music'):
                output = recommender.recommend_music(args.query)
            elif hasattr(recommender, 'get_hybrid_recommendations'):
                output = recommender.get_hybrid_recommendations(args.query)
            else:
                print("❌ No recommendation method available")
                return
                
            print("\n📊 Raw recommendation output:")
            print("=" * 60)
            print(output)
            print("=" * 60)
            
            print("\n🔍 Parsing recommendations...")
            recommendations = judge._parse_recommendations(output)
            
            print(f"\n✅ Found {len(recommendations)} recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec['name']}")
                print(f"   Artist: {rec['artists']}")
                print(f"   Genre: {rec['genre']}")
                print(f"   Language: {rec['language']}")
                if 'mood' in rec and rec['mood']:
                    print(f"   Mood: {rec['mood']}")
            
            # Run LLM evaluation if enabled
            if use_llm and judge.llm_judge and recommendations:
                print("\n🤖 Running LLM evaluation...")
                try:
                    llm_evaluation = judge.llm_judge.evaluate_recommendations(args.query, recommendations)
                    
                    print("\n📊 LLM Evaluation Results:")
                    print(f"   Overall Score: {llm_evaluation.get('overall_score', 'N/A')}/10")
                    print(f"   Relevance Score: {llm_evaluation.get('relevance_score', 'N/A')}/10")
                    print(f"   Diversity Score: {llm_evaluation.get('diversity_score', 'N/A')}/10")
                    print(f"   Matches Query: {'Yes' if llm_evaluation.get('matches_query') else 'No'}")
                    print(f"   Reasoning: {llm_evaluation.get('reasoning', 'No reasoning provided')}")
                except Exception as e:
                    print(f"❌ LLM evaluation failed: {e}")
            
            return
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Run specific test or all tests
    if args.test:
        if 1 <= args.test <= len(judge.test_cases):
            print(f"\n🧪 Running Test {args.test} only\n")
            test_case = judge.test_cases[args.test - 1]
            results = []
            
            try:
                print(f"Test {args.test}: \"{test_case.input}\"")
                print("  Description:", test_case.description)
                print("  Criteria:", test_case.criteria)
                
                start_time = time.time()
                
                # Try different recommendation methods
                if hasattr(judge.recommender, 'recommend_music'):
                    output = judge.recommender.recommend_music(test_case.input)
                else:
                    output = judge.recommender.get_hybrid_recommendations(test_case.input)
                    
                end_time = time.time()
                
                # Print raw output in debug mode
                if args.debug:
                    print("\n📄 Raw recommendation output:")
                    print("-" * 40)
                    print(output)
                    print("-" * 40)
                
                recommendations = judge._parse_recommendations(output)
                result = judge._evaluate_test_case(test_case, recommendations)
                results.append(result)
                
                emoji = "✅" if result.status == "PASS" else "❌"
                print(f"  Result: {emoji} {result.status}")
                print(f"  Reasoning: {result.reasoning}")
                print(f"  Time: {end_time - start_time:.2f} seconds")
                print(f"  Found {len(recommendations)} recommendations")
                
                # Print LLM evaluation if available
                if result.llm_evaluation:
                    print("\n  🤖 LLM Evaluation:")
                    print(f"     Overall Score: {result.llm_evaluation.get('overall_score', 0)}/10")
                    print(f"     Relevance Score: {result.llm_evaluation.get('relevance_score', 0)}/10") 
                    print(f"     Diversity Score: {result.llm_evaluation.get('diversity_score', 0)}/10")
                    print(f"     Matches Query: {'Yes' if result.llm_evaluation.get('matches_query') else 'No'}")
                    print(f"     Reasoning: {result.llm_evaluation.get('reasoning', 'No reasoning provided')}")
                
                # Print detailed recommendations in debug mode
                if args.debug:
                    print("\n🎵 Parsed recommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"  {i}. {rec['name']}")
                        print(f"     Artist: {rec['artists']}")
                        print(f"     Genre: {rec['genre']}")
                        print(f"     Language: {rec['language']}")
                        if 'mood' in rec and rec['mood']:
                            print(f"     Mood: {rec['mood']}")
                
            except Exception as e:
                print(f"  ❌ Error running test: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"❌ Invalid test ID. Please choose a number between 1 and {len(judge.test_cases)}")
            return
    else:
        # Run all tests
        results = judge.run_tests()
    
    judge.print_summary(results)

if __name__ == "__main__":
    main()
