
from approach import SpotifyRecommendationSystem
import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score, average_precision_score

if __name__ == "__main__":
    recommender = SpotifyRecommendationSystem()
    
    # List of 5 artists: English, Bollywood, Nepali
    artists_genres = [
        ("Ed Sheeran", "pop"),          # English
        ("Adele", "sad"),               # English
        ("Neha Kakkar", "pop"),         # Bollywood
        ("Arijit Singh", "romantic"),   # Bollywood
        ("Nabin K. Bhattarai", "romantic")  # Nepali
    ]

    print("\n🎵 Spotify Recommendation Evaluation (sklearn)\n")
    print(f"{'Artist':25s} {'Genre':10s} {'Precision@8':12s} {'Recall@8':10s} {'NDCG@8':10s} {'MAP@8':10s}")
    print("-"*80)

    for artist_name, genre in artists_genres:
        recommendations = recommender.recommend_by_genre(artist_name, genre, num_recommendations=8)
        recommended_ids = [track["id"] for track, score in recommendations]

        # Simulated ground truth: top 5 tracks
        artist_id = recommender.get_artist_id(artist_name)
        top_tracks = recommender.get_artist_top_tracks(artist_id, limit=5)
        relevant_ids = [track["id"] for track in top_tracks]

        # Build relevance labels
        y_true = np.array([1 if rid in relevant_ids else 0 for rid in recommended_ids])
        y_scores = np.array([score for _, score in recommendations])

        # Precision@8 and Recall@8
        precision = precision_score(y_true, y_true, average="binary")  # since we cut to top-8 already
        recall = recall_score(y_true, y_true, average="binary")

        # nDCG@8 (needs shape [1, n_samples])
        ndcg = ndcg_score([y_true], [y_scores], k=8)

        # MAP@8
        map_k = average_precision_score(y_true, y_scores)

        print(f"{artist_name:25s} {genre:10s} {precision:<12.3f} {recall:<10.3f} {ndcg:<10.3f} {map_k:<10.3f}")

