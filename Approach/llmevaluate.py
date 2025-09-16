import requests
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import precision_score, recall_score
import time
from approach import SpotifyRecommendationSystem

class MistralLLMJudge:
    def __init__(self, api_key: str):
        """Initialize Mistral LLM Judge"""
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def evaluate_song(self, song_name: str, artist_name: str, target_genre: str, 
                     target_artist: str, case_type: str) -> Dict:
        """
        Evaluate a song using Mistral LLM as judge
        Returns evaluation scores for different criteria
        """
        
        # Define evaluation prompts based on case type
        prompts = {
            "genre_match": f"""
Rate if "{song_name}" by {artist_name} matches the "{target_genre}" genre (0.0-1.0).
Consider if the song's style, mood, and characteristics fit this genre.

Return only this JSON:
{{"genre_match_score": 0.8, "reasoning": "Brief explanation"}}
            """,
            
            "artist_consistency": f"""
Rate if "{song_name}" by {artist_name} represents the style of {target_artist} (0.0-1.0).
Is this song typical of what {target_artist} would create or sing?

Return only this JSON:
{{"artist_consistency_score": 0.7, "reasoning": "Brief explanation"}}
            """,
            
            "overall_quality": f"""
Rate the overall recommendation quality (0.0-1.0): "{song_name}" by {artist_name} 
for someone wanting {target_genre} music by {target_artist}.
Consider both genre fit AND artist match.

Return only this JSON:
{{"overall_quality_score": 0.9, "reasoning": "Brief explanation"}}
            """
        }
        
        # Select prompt based on case type
        if case_type == "case1":
            prompt = prompts["genre_match"]
        elif case_type == "case2": 
            prompt = prompts["artist_consistency"]
        else:  # case3
            prompt = prompts["overall_quality"]
        
        try:
            payload = {
                "model": "mistral-large-latest",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a music evaluation expert. Always respond with valid JSON only, no additional text or formatting."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 150
            }
            
            response = requests.post(self.base_url, headers=self.headers, 
                                   json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Parse JSON response with multiple fallback methods
            try:
                # First try direct JSON parsing
                evaluation = json.loads(content)
                return evaluation
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code blocks
                try:
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                        evaluation = json.loads(json_content)
                        return evaluation
                    elif "```" in content:
                        json_start = content.find("```") + 3
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                        evaluation = json.loads(json_content)
                        return evaluation
                except:
                    pass
                
                # Try to extract score from text using regex
                try:
                    import re
                    # Look for score patterns like "score": 0.8 or "0.8"
                    score_patterns = [
                        r'"(?:genre_match_score|artist_consistency_score|overall_quality_score)":\s*([0-9]*\.?[0-9]+)',
                        r'score["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)',
                        r'([0-9]*\.?[0-9]+)'
                    ]
                    
                    for pattern in score_patterns:
                        match = re.search(pattern, content)
                        if match:
                            score = float(match.group(1))
                            if 0.0 <= score <= 1.0:
                                score_key = "genre_match_score" if case_type == "case1" else \
                                          "artist_consistency_score" if case_type == "case2" else \
                                          "overall_quality_score"
                                return {score_key: score, "reasoning": "Extracted from text"}
                    
                    # If no valid score found, use a default
                    print(f"Warning: Could not parse response for {song_name}. Content: {content[:100]}...")
                    score_key = "genre_match_score" if case_type == "case1" else \
                              "artist_consistency_score" if case_type == "case2" else \
                              "overall_quality_score"
                    return {score_key: 0.5, "reasoning": "Parse error - using default"}
                    
                except Exception as e:
                    print(f"Warning: Could not parse JSON response for {song_name}: {e}")
                    score_key = "genre_match_score" if case_type == "case1" else \
                              "artist_consistency_score" if case_type == "case2" else \
                              "overall_quality_score"
                    return {score_key: 0.5, "reasoning": "Parse error"}
                
        except Exception as e:
            print(f"Error evaluating {song_name}: {e}")
            score_key = "genre_match_score" if case_type == "case1" else \
                      "artist_consistency_score" if case_type == "case2" else \
                      "overall_quality_score"
            return {score_key: 0.0, "reasoning": f"Error: {e}"}

class MusicRecommendationEvaluator:
    def __init__(self, mistral_api_key: str):
        """Initialize the evaluation system"""
        self.llm_judge = MistralLLMJudge(mistral_api_key)
        self.spotify_system = SpotifyRecommendationSystem()
    
    def evaluate_recommendations(self, artist_name: str, genre: str = None, 
                               num_recommendations: int = 8) -> Dict:
        """
        Evaluate recommendations across cases and calculate metrics
        Uses actual Spotify genres from the artist
        """
        print(f"\nStarting evaluation for {artist_name}")
        print("=" * 60)
        
        # Get artist ID first
        artist_id = self.spotify_system.get_artist_id(artist_name)
        if not artist_id:
            print("Artist not found")
            return {}
        
        # Get available Spotify genres for the artist
        available_genres = self.spotify_system.get_artist_genres(artist_id)
        
        # Show available genres and let user select
        if available_genres:
            print(f"Available Spotify genres for {artist_name}:")
            for idx, g in enumerate(available_genres, 1):
                print(f"{idx}. {g}")
            
            try:
                choice = input(f"\nSelect genre number (1-{len(available_genres)}): ").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_genres):
                        genre = available_genres[idx]
                    else:
                        print("Invalid selection. Using first available genre.")
                        genre = available_genres[0]
                else:
                    print("Invalid input. Using first available genre.")
                    genre = available_genres[0]
            except:
                genre = available_genres[0]
        else:
            # No Spotify genres available, use custom genres
            print(f"No Spotify genres available for {artist_name}")
            genre = input("Enter a genre (e.g., 'sad', 'pop', 'romantic'): ").strip()
            if not genre:
                genre = "pop"  # default
        
        print(f"Selected genre: {genre}")
        
        # Get recommendations using the selected genre
        recommendations = self.spotify_system.recommend_by_genre(
            artist_name, genre, num_recommendations
        )
        
        if not recommendations:
            print("No recommendations to evaluate")
            return {}
        
        # Evaluate each case with focus on artist + genre matching
        results = {
            "case1": [],  # Genre matching
            "case2": [],  # Artist consistency
            "case3": []   # Combined relevance
        }
        
        print(f"\nEvaluating {len(recommendations)} recommendations...")
        
        for i, (track, similarity_score) in enumerate(recommendations):
            song_name = track.get('name', 'Unknown')
            track_artists = [artist['name'] for artist in track.get('artists', [])]
            track_artist = ', '.join(track_artists)
            
            print(f"\nEvaluating: {song_name} by {track_artist}")
            
            # Case 1
            print("  - Checking genre match...")
            case1_eval = self.llm_judge.evaluate_song(
                song_name, track_artist, genre, artist_name, "case1"
            )
            time.sleep(1)
            
            # Case 2
            print("  - Checking artist consistency...")
            case2_eval = self.llm_judge.evaluate_song(
                song_name, track_artist, genre, artist_name, "case2"
            )
            time.sleep(1)
            
            # Case 3
            print("  - Checking overall relevance...")
            case3_eval = self.llm_judge.evaluate_song(
                song_name, track_artist, genre, artist_name, "case3"
            )
            time.sleep(1)
            
            # Store results
            results["case1"].append({
                "song": song_name,
                "artist": track_artist,
                "target_artist": artist_name,
                "target_genre": genre,
                "similarity_score": similarity_score,
                "llm_score": case1_eval.get("genre_match_score", case1_eval.get("score", 0.0)),
                "reasoning": case1_eval.get("reasoning", "")
            })
            
            results["case2"].append({
                "song": song_name,
                "artist": track_artist,
                "target_artist": artist_name,
                "target_genre": genre,
                "similarity_score": similarity_score,
                "llm_score": case2_eval.get("artist_consistency_score", case2_eval.get("score", 0.0)),
                "reasoning": case2_eval.get("reasoning", "")
            })
            
            results["case3"].append({
                "song": song_name,
                "artist": track_artist,
                "target_artist": artist_name,
                "target_genre": genre,
                "similarity_score": similarity_score,
                "llm_score": case3_eval.get("overall_quality_score", case3_eval.get("score", 0.0)),
                "reasoning": case3_eval.get("reasoning", "")
            })
        
        return results
    
    def calculate_precision_recall(self, scores: List[float], threshold: float = 0.6) -> Tuple[float, float]:
        """Calculate precision and recall"""
        relevant = [1 if score >= threshold else 0 for score in scores]
        predicted = [1] * len(scores)  # All items are "recommended"
        
        if sum(predicted) == 0:
            precision = 0.0
        else:
            precision = sum(relevant) / sum(predicted)
        
        recall = sum(relevant) / len(relevant) if len(relevant) > 0 else 0.0
        
        return precision, recall
    
    def calculate_map(self, scores: List[float], threshold: float = 0.6) -> float:
        """Calculate Mean Average Precision"""
        relevant = [1 if score >= threshold else 0 for score in scores]
        
        if sum(relevant) == 0:
            return 0.0
        
        precision_at_k = []
        relevant_count = 0
        
        for i, rel in enumerate(relevant):
            if rel == 1:
                relevant_count += 1
                precision_at_k.append(relevant_count / (i + 1))
        
        return sum(precision_at_k) / len(precision_at_k) if precision_at_k else 0.0
    
    def calculate_ndcg(self, scores: List[float], k: Optional[int] = None) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        if k is None:
            k = len(scores)
        
        scores = scores[:k]
        
        dcg = scores[0] if scores else 0.0
        for i in range(1, len(scores)):
            dcg += scores[i] / np.log2(i + 1)
        
        ideal_scores = sorted(scores, reverse=True)
        idcg = ideal_scores[0] if ideal_scores else 0.0
        for i in range(1, len(ideal_scores)):
            idcg += ideal_scores[i] / np.log2(i + 1)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def print_detailed_results(self, results: Dict, artist_name: str, genre: str):
        """Print comprehensive evaluation results"""
        print(f"\nEVALUATION RESULTS: {artist_name} - {genre}")
        print("=" * 80)
        
        for case_name, case_results in results.items():
            case_titles = {
                "case1": "Case 1: Genre Matching",
                "case2": "Case 2: Artist Consistency", 
                "case3": "Case 3: Overall Quality"
            }
            
            print(f"\n{case_titles[case_name]}")
            print("-" * 50)
            
            llm_scores = [r["llm_score"] for r in case_results]
            similarity_scores = [r["similarity_score"] for r in case_results]
            
            precision, recall = self.calculate_precision_recall(llm_scores)
            map_score = self.calculate_map(llm_scores)
            ndcg_score = self.calculate_ndcg(llm_scores)
            
            print(f"Metrics (LLM Judge):")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   MAP: {map_score:.3f}")
            print(f"   NDCG: {ndcg_score:.3f}")
            print(f"   Avg LLM Score: {np.mean(llm_scores):.3f}")
            print(f"   Avg Similarity Score: {np.mean(similarity_scores):.3f}")
            
            print(f"\nIndividual Song Evaluations:")
            for i, result in enumerate(case_results, 1):
                print(f"{i:2d}. {result['song']} - {result['artist']}")
                print(f"    LLM Score: {result['llm_score']:.3f} | Similarity: {result['similarity_score']:.3f}")
                print(f"    Reasoning: {result['reasoning']}")
                print()


def main():
    """Main evaluation function"""
    mistral_api_key = "ey4fgxAQqsB0DeYOCOTJkO1GbMRgwkaD"
    evaluator = MusicRecommendationEvaluator(mistral_api_key)
    
    print("Music Recommendation Evaluation System")
    print("=" * 50)
    
    artist_name = input("Enter artist name: ").strip()
    if not artist_name:
        print("Please enter a valid artist name")
        return
    
    try:
        num_recs = int(input("Number of recommendations to evaluate (default 5): ").strip() or "5")
    except ValueError:
        num_recs = 5
    
    try:
        results = evaluator.evaluate_recommendations(artist_name, None, num_recs)
        
        if results:
            if results.get("case1"):
                actual_genre = results["case1"][0]["target_genre"]
                evaluator.print_detailed_results(results, artist_name, actual_genre)
                
                print("\nEVALUATION SUMMARY")
                print("=" * 50)
                print(f"Artist: {artist_name}")
                print(f"Genre: {actual_genre}")
                print(f"Songs Evaluated: {len(results['case1'])}")
                print()
                
                for case in ["case1", "case2", "case3"]:
                    if case in results:
                        scores = [r["llm_score"] for r in results[case]]
                        avg_score = np.mean(scores)
                        case_names = {
                            "case1": "Genre Match", 
                            "case2": "Artist Consistency",
                            "case3": "Overall Quality"
                        }
                        status = "PASS" if avg_score >= 0.6 else "WARNING" if avg_score >= 0.4 else "FAIL"
                        print(f"{status} {case_names[case]}: {avg_score:.3f}")
        else:
            print("No results to display")
            
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
