from approach import SpotifyRecommendationSystem
import numpy as np

def precision_at_k(recommended, relevant, k=8):
    recommended_k = recommended[:k]
    hits = sum(1 for r in recommended_k if r in relevant)
    return hits / k

def recall_at_k(recommended, relevant, k=8):
    recommended_k = recommended[:k]
    hits = sum(1 for r in recommended_k if r in relevant)
    return hits / len(relevant) if relevant else 0

def ndcg_at_k(recommended, relevant, k=8):
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, track in enumerate(recommended_k):
        if track in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0

def average_precision(recommended, relevant, k=8):
    recommended_k = recommended[:k]
    hits = 0
    sum_precisions = 0.0
    for i, track in enumerate(recommended_k, 1):
        if track in relevant:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / min(len(relevant), k) if relevant else 0

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

    print("\n🎵 Spotify Recommendation Evaluation\n")
    print(f"{'Artist':25s} {'Genre':10s} {'Precision@8':12s} {'Recall@8':10s} {'NDCG@8':10s} {'MAP@8':10s}")
    print("-"*80)

    for artist_name, genre in artists_genres:
        recommendations = recommender.recommend_by_genre(artist_name, genre, num_recommendations=8)
        recommended_ids = [track["id"] for track, _ in recommendations]
        top_tracks = recommender.get_artist_top_tracks(recommender.get_artist_id(artist_name), limit=5)
        relevant_ids = [track["id"] for track in top_tracks]  # simulate ground truth

        precision = precision_at_k(recommended_ids, relevant_ids, k=8)
        recall = recall_at_k(recommended_ids, relevant_ids, k=8)
        ndcg = ndcg_at_k(recommended_ids, relevant_ids, k=8)
        map_k = average_precision(recommended_ids, relevant_ids, k=8)

        print(f"{artist_name:25s} {genre:10s} {precision:<12.3f} {recall:<10.3f} {ndcg:<10.3f} {map_k:<10.3f}")
