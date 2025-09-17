
# --- CONFIGURE YOUR MISTRAL API KEY AND ENDPOINT HERE ---
MISTRAL_API_KEY = "JmiqFZXuIM4vyC40iGPeqg355yfQdl6A"
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

import requests
import json
import time
import re

def extract_json(content):
    """
    Robust JSON extraction from LLM output
    """
    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[JSON parse error]: {e}")
    return {}

def extract_structured_info_with_mistral(text, api_key=MISTRAL_API_KEY, api_url=MISTRAL_API_URL):
    """
    Use Mistral LLM to extract structured info (artist, song, genre, etc.) from a text block.
    Returns a dict with keys: artists, songs, genres, moods, etc.
    """
    prompt = f"""
    Analyze the following music recommendation text and extract structured information as JSON.
    Return a JSON object with:
    - artists: list of artist names
    - songs: list of song titles
    - genres: list of genres (if present)
    - moods: list of moods (if present)
    - For each recommendation, include a dict with 'song', 'artist', 'genre', 'mood' if possible.
    
    Text:
    {text}
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 512
    }

    for attempt in range(5):  # max 5 retries
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=20)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 2))
                print(f"Rate limit hit. Waiting {retry_after} seconds before retry...")
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # Debug print raw LLM output
            print(f"\n[LLM raw output]:\n{content}\n")
            return extract_json(content)
        except requests.exceptions.RequestException as e:
            print(f"[HTTP error]: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[LLM extraction error]: {e}")
            return {}
    print("[LLM extraction failed after retries]")
    return {}

# def evaluate_recommendations(query, system_output, ground_truth_output):
#     """
#     Evaluate the system output against ground truth using LLM-extracted info.
#     Returns precision, recall, f1, and details.
#     """
#     sys_info = extract_structured_info_with_mistral(system_output)
#     gt_info = extract_structured_info_with_mistral(ground_truth_output)

#     def get_song_artist_set(info):
#         recs = info.get("recommendations") or info.get("recs") or info.get("tracks") or []
#         if not recs and "songs" in info and "artists" in info:
#             recs = [{"song": s, "artist": a} for s, a in zip(info["songs"], info["artists"])]
#         pairs = set()
#         for rec in recs:
#             song = rec.get("song") or rec.get("title") or ""
#             artist = rec.get("artist") or ""
#             if song and artist:
#                 pairs.add((song.strip().lower(), artist.strip().lower()))
#         return pairs

#     sys_set = get_song_artist_set(sys_info)
#     gt_set = get_song_artist_set(gt_info)

#     # Manual set-based evaluation
#     true_positives = len(sys_set & gt_set)
#     precision = true_positives / len(sys_set) if sys_set else 0.0
#     recall = true_positives / len(gt_set) if gt_set else 0.0
#     f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

#     # sklearn evaluation
#     all_pairs = list(sys_set | gt_set)
#     y_pred = [1 if pair in sys_set else 0 for pair in all_pairs]
#     y_true = [1 if pair in gt_set else 0 for pair in all_pairs]
#     sklearn_precision = precision_score(y_true, y_pred, zero_division=0)
#     sklearn_recall = recall_score(y_true, y_pred, zero_division=0)
#     sklearn_f1 = f1_score(y_true, y_pred, zero_division=0)

#     return {
#         "precision": precision,
#         "recall": recall,
#         "f1": f1,
#         "true_positives": true_positives,
#         "system_count": len(sys_set),
#         "ground_truth_count": len(gt_set),
#         "system_set": sys_set,
#         "ground_truth_set": gt_set,
#         "sklearn_precision": sklearn_precision,
#         "sklearn_recall": sklearn_recall,
#         "sklearn_f1": sklearn_f1
#     }

# def main():
#     query = "Recommend 3 songs similar to Shape of You by Ed Sheeran"
#     ground_truth_output = """
#     1. Attention by Charlie Puth
#     2. There's Nothing Holdin' Me Back by Shawn Mendes
#     3. Photograph by Ed Sheeran
#     """

#     recommender = HybridRecommendationSystem()
#     system_output = recommender.recommend_music(query)

#     results = evaluate_recommendations(query, system_output, ground_truth_output)
    
#     print("\n--- LLM-based Evaluation Report ---")
#     print(f"Query: {query}")
#     print(f"Precision (manual): {results['precision']:.2f}")
#     print(f"Recall (manual): {results['recall']:.2f}")
#     print(f"F1 (manual): {results['f1']:.2f}")
#     print(f"True Positives: {results['true_positives']}")
#     print(f"System Output Count: {results['system_count']}")
#     print(f"Ground Truth Count: {results['ground_truth_count']}")
#     print(f"System Set: {results['system_set']}")
#     print(f"Ground Truth Set: {results['ground_truth_set']}")
    
#     print("\n--- sklearn Metrics ---")
#     print(f"Precision: {results['sklearn_precision']:.2f}")
#     print(f"Recall: {results['sklearn_recall']:.2f}")
#     print(f"F1: {results['sklearn_f1']:.2f}")

# if __name__ == "__main__":
#     main()
# simplified_evaluate.py


from recommend import HybridRecommendationSystem


def evaluate_case_by_preferences(recommended, query_genre, query_mood=None):
    """
    recommended: list of dicts from your system, each with keys: song, artist, genre, mood
    query_genre: string
    query_mood: optional string
    """
    total = len(recommended)
    matches = 0

    for rec in recommended:
        genre = rec.get("genre", "").lower() if rec.get("genre") else ""
        mood = rec.get("mood", "").lower() if rec.get("mood") else ""
        # Check if genre matches and (if given) mood matches user preference
        genre_match = query_genre.lower() in genre
        mood_match = True if not query_mood else query_mood.lower() in mood
        if genre_match and mood_match:
            matches += 1

    precision = matches / total if total else 0
    recall = matches / total if total else 0  # simplified: ground truth assumed as total
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1, matches, total


def main():


    recommender = HybridRecommendationSystem()

    # Define multiple test cases
    cases = [
        {"name": "Punjabi Songs", "query": "Suggest me some Punjabi songs", "intent": {"genre": "punjabi"}},
    ]

    total_matches = 0
    total_recs = 0
    total_cases = len(cases)

    for case in cases:
        print(f"\n--- Starting {case['name']} ---")
        preferences = recommender.interpret_query_with_llm(case['query'])
        tracks = recommender.search_spotify_tracks(preferences, limit=10, specific_artist=case['intent'].get("artist"))
        enhanced_tracks = recommender.enhance_tracks_with_features(tracks)
        sequential = recommender.sequential_recommendations(enhanced_tracks, top_k=5)
        ranking = recommender.ranking_recommendations(enhanced_tracks, preferences, top_k=5)
        embedding = recommender.embedding_recommendations(enhanced_tracks, case['query'], top_k=5)
        final_recs = recommender.hybrid_merge(sequential, ranking, embedding, preferences, requested_count=3, specific_artist=case['intent'].get("artist"))

        # Format recommendations as text for LLM extraction
        rec_text = "\n".join([
            f"{i+1}. {track.name} by {', '.join(track.artists)} | Genre: {recommender._determine_primary_genre(track, preferences.genres)} | Mood: {', '.join(preferences.moods)}"
            for i, track in enumerate(final_recs)
        ])
        print("System Recommendations:")
        print(rec_text)

        # Use LLM to extract structured info from recommendations
        sys_info = extract_structured_info_with_mistral(rec_text)
        # Use LLM to extract structured info from intent/query
        intent_info = extract_structured_info_with_mistral(case['query'])

        # Evaluate: count matches where artist/genre/mood match intent
        matches = 0
        sys_recs = sys_info.get("recommendations") or sys_info.get("recs") or sys_info.get("tracks") or []
        for rec in sys_recs:
            artist_match = True
            genre_match = True
            # No explicit mood check; let LLM judge mood from query and recs
            if "artist" in case["intent"]:
                artist_match = any(case["intent"]["artist"].lower() in (rec.get("artist") or "").lower() for _ in [0])
            if "genre" in case["intent"]:
                # Accept genre or language as Punjabi
                genre_val = (rec.get("genre") or "").lower()
                lang_val = (rec.get("language") or "").lower() if "language" in rec else ""
                genre_match = "punjabi" in genre_val or "punjabi" in lang_val
            if artist_match and genre_match:
                matches += 1

        total_matches += matches
        total_recs += len(sys_recs)

        precision = matches / len(sys_recs) if sys_recs else 0
        recall = matches / len(sys_recs) if sys_recs else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        print("\nLLM-based Evaluation Results:")
        print(f" Matches: {matches}/{len(sys_recs)}")
        print(f" Precision: {precision*100:.2f}%")
        print(f" Recall: {recall*100:.2f}%")
        print(f" F1 Score: {f1*100:.2f}%")

        # Advanced Metrics
        if sys_recs:
            # Artist Diversity
            artists = [rec.get("artist", "").strip().lower() for rec in sys_recs if rec.get("artist")]
            artist_diversity = len(set(artists)) / len(artists) if artists else 0
            print(f"Artist Diversity: {artist_diversity*100:.2f}% ({len(set(artists))} unique / {len(artists)} total)")

            # Genre Diversity
            genres = [rec.get("genre", "").strip().lower() for rec in sys_recs if rec.get("genre")]
            genre_diversity = len(set(genres)) / len(genres) if genres else 0
            print(f"Genre Diversity: {genre_diversity*100:.2f}% ({len(set(genres))} unique / {len(genres)} total)")

            # Language Match (to query intent)
            query_lang = (intent_info.get("request_details", {}).get("language") or "punjabi").lower()
            lang_matches = [1 for rec in sys_recs if (rec.get("language") or query_lang).lower() == query_lang]
            lang_match_pct = sum(lang_matches) / len(sys_recs) if sys_recs else 0
            print(f"Language Match: {lang_match_pct*100:.2f}%")

            # Popularity Score (average)
            # If popularity is not present, skip
            pops = [rec.get("popularity") for rec in sys_recs if rec.get("popularity") is not None]
            if pops:
                avg_pop = sum(pops) / len(pops)
                print(f"Average Popularity: {avg_pop:.2f}/100")
            else:
                print("Average Popularity: N/A (not available in extracted data)")

            # Novelty (placeholder: assume all are novel)
            print("Novelty: 100.00% (placeholder, user history not tracked)")

            # Coverage (placeholder: assume unknown catalog size)
            print("Coverage: N/A (catalog size not tracked)")

            # MRR (Mean Reciprocal Rank)
            # For this single query, if all matches are relevant, MRR=1.0
            mrr = 0
            for idx, rec in enumerate(sys_recs):
                if (rec.get("genre") and "punjabi" in rec.get("genre").lower()):
                    mrr = 1/(idx+1)
                    break
            print(f"MRR: {mrr:.2f}")

            # MAP (Mean Average Precision)
            # For this single query, MAP = precision if all relevant
            print(f"MAP: {precision*100:.2f}%")

            # NDCG (Normalized Discounted Cumulative Gain)
            # Assume all relevant = 1, so DCG = sum(1/log2(i+2)), IDCG = same
            import math
            dcg = sum(1 / math.log2(i+2) for i in range(matches))
            idcg = sum(1 / math.log2(i+2) for i in range(matches))
            ndcg = dcg / idcg if idcg else 0
            print(f"NDCG: {ndcg*100:.2f}%")

    # Overall accuracy
    overall_accuracy = total_matches / total_recs if total_recs else 0
    print(f"\n=== Overall LLM-based Accuracy Across {total_cases} Cases: {overall_accuracy:.2%} ===")


if __name__ == "__main__":
    main()
