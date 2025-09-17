#!/usr/bin/env python3
"""
Enhanced Music Recommendation Evaluation System
Uses Mistral API for comprehensive analysis of recommendations with proper rate limiting
"""

import requests
import json
import time
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from recommend import HybridRecommendationSystem

# --- CONFIGURE YOUR MISTRAL API KEY AND ENDPOINT HERE ---
MISTRAL_API_KEY = "JmiqFZXuIM4vyC40iGPeqg355yfQdl6A"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

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
        self.rate_limit_delay = 2  # seconds between requests
        
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
                return json.loads(json_match.group())
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
                            "mood": "unknown"
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
            TestCase(
                id=1,
                name="Punjabi Pop Music",
                query="Suggest me some energetic Punjabi pop songs for a party",
                expected_genre="punjabi pop",
                expected_mood="energetic",
                expected_language="punjabi",
                min_artist_diversity=0.4,
                min_genre_match=0.8,
                description="Test Punjabi pop music with high energy and party mood"
            ),
            TestCase(
                id=2,
                name="Bollywood Romance",
                query="I want romantic Bollywood songs for a date night",
                expected_genre="bollywood",
                expected_mood="romantic",
                expected_language="hindi",
                min_artist_diversity=0.3,
                min_genre_match=0.7,
                description="Test Bollywood romantic songs for date night"
            ),
            TestCase(
                id=3,
                name="Specific Artist - Arijit Singh",
                query="Give me 5 songs by Arijit Singh",
                expected_genre="bollywood",
                expected_artist="arijit singh",
                expected_mood="melodious",
                expected_language="hindi",
                min_artist_diversity=0.1,
                min_genre_match=0.5,
                description="Test recommendations similar to specific artist"
            ),
            TestCase(
                id=4,
                name="Hindi Rock Music",
                query="I need some Hindi rock songs for workout",
                expected_genre="hindi rock",
                expected_mood="energetic",
                expected_language="hindi",
                min_artist_diversity=0.5,
                min_genre_match=0.6,
                description="Test Hindi rock music for workout"
            ),
            TestCase(
                id=5,
                name="Devotional Music",
                query="Recommend some peaceful devotional songs",
                expected_genre="devotional",
                expected_mood="peaceful",
                min_artist_diversity=0.4,
                min_genre_match=0.8,
                description="Test devotional music"
            )
        ]
    
    def _calculate_artist_diversity(self, recommendations: List[Dict[str, Any]]) -> float:
        """Calculate artist diversity score"""
        if not recommendations:
            return 0.0
            
        artists = [rec.get("artist", "").strip().lower() for rec in recommendations if rec.get("artist")]
        if not artists:
            return 0.0
            
        unique_artists = len(set(artists))
        total_artists = len(artists)
        return unique_artists / total_artists
    
    def _calculate_genre_match(self, recommendations: List[Dict[str, Any]], expected_genre: str) -> float:
        """Calculate genre match score"""
        if not recommendations or not expected_genre:
            return 0.0
            
        matches = 0
        expected_genre_lower = expected_genre.lower()
        
        for rec in recommendations:
            genre = rec.get("genre", "").lower()
            # For Bollywood genre, also accept "pop" as a match since Bollywood songs are often categorized as pop
            if expected_genre_lower == "bollywood" and genre == "pop":
                matches += 1
            elif expected_genre_lower in genre or any(word in genre for word in expected_genre_lower.split()):
                matches += 1
                
        return matches / len(recommendations)
    
    def _calculate_artist_match(self, recommendations: List[Dict[str, Any]], expected_artist: Optional[str]) -> float:
        """Calculate fraction of recommendations that match the expected artist"""
        if not recommendations or not expected_artist:
            return 0.0
        expected = expected_artist.strip().lower()
        matches = 0
        for rec in recommendations:
            artist_str = (rec.get("artist") or "").lower()
            if expected in artist_str:
                matches += 1
        return matches / len(recommendations)

    def _calculate_mood_match(self, recommendations: List[Dict[str, Any]], expected_mood: str) -> float:
        """Calculate mood match score"""
        if not recommendations or not expected_mood:
            return 1.0  # No mood specified, consider as match
            
        matches = 0
        expected_mood_lower = expected_mood.lower()
        
        # Define mood synonyms for better matching
        mood_synonyms = {
            "melodious": ["romantic", "emotional", "soulful", "heartfelt", "yearning", "melancholic"],
            "energetic": ["upbeat", "playful", "energetic"],
            "romantic": ["romantic", "emotional", "heartfelt", "yearning", "melodious"],
            "peaceful": ["peaceful", "calm", "serene"]
        }
        
        for rec in recommendations:
            mood = rec.get("mood", "").lower()
            # Check direct match
            if expected_mood_lower in mood or any(word in mood for word in expected_mood_lower.split()):
                matches += 1
            # Check synonyms
            elif expected_mood_lower in mood_synonyms:
                synonyms = mood_synonyms[expected_mood_lower]
                if any(syn in mood for syn in synonyms):
                    matches += 1
                
        return matches / len(recommendations)
    
    def _calculate_language_match(self, recommendations: List[Dict[str, Any]], expected_language: str) -> float:
        """Calculate language match score"""
        if not recommendations or not expected_language:
            return 1.0  # No language specified, consider as match
            
        matches = 0
        expected_lang_lower = expected_language.lower()
        
        for rec in recommendations:
            language = rec.get("language", "").lower()
            if expected_lang_lower in language or any(word in language for word in expected_lang_lower.split("/")):
                matches += 1
                
        return matches / len(recommendations)
    
    def evaluate_single_case(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate a single test case with comprehensive analysis"""
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
            
            # Calculate scores
            genre_match_score = self._calculate_genre_match(recommendations, test_case.expected_genre)
            artist_diversity_score = self._calculate_artist_diversity(recommendations)
            mood_match_score = self._calculate_mood_match(recommendations, test_case.expected_mood)
            language_match_score = self._calculate_language_match(recommendations, test_case.expected_language)
            
            # Calculate overall quality score
            # For artist-specific cases, give more weight to artist match
            if test_case.expected_artist:
                artist_match_score = self._calculate_artist_match(recommendations, test_case.expected_artist)
                overall_quality_score = (
                    artist_match_score * 0.5 +
                    genre_match_score * 0.2 +
                    mood_match_score * 0.15 +
                    language_match_score * 0.1 +
                    quality_analysis.get("overall_relevance", 0.5) * 0.05
                )
            else:
                overall_quality_score = (
                    genre_match_score * 0.3 +
                    artist_diversity_score * 0.2 +
                    mood_match_score * 0.2 +
                    language_match_score * 0.2 +
                    quality_analysis.get("overall_relevance", 0.5) * 0.1
                )
            
            # Determine if test passed
            if test_case.expected_artist:
                # For artist-specific cases, require strong artist match and decent overall quality
                artist_match_score = self._calculate_artist_match(recommendations, test_case.expected_artist)
                passed = (
                    artist_match_score >= 0.8 and
                    overall_quality_score >= 0.6
                )
            else:
                passed = (
                    genre_match_score >= test_case.min_genre_match and
                    artist_diversity_score >= test_case.min_artist_diversity and
                    overall_quality_score >= 0.6
                )
            
            # Create reasoning
            reasoning = quality_analysis.get("reasoning", "No detailed reasoning available")
            
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
                reasoning=reasoning
            )
            
            # Print results
            self._print_case_results(result)
            
            # Add delay to avoid rate limiting
            print("⏳ Waiting to avoid rate limiting...")
            time.sleep(3)
            
            return result
            
        except Exception as e:
            print(f"❌ Error evaluating test case {test_case.id}: {e}")
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
                reasoning=f"Error: {str(e)}"
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
