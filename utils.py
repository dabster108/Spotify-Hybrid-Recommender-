import re

def clean_text(text: str) -> str:
    """
    Preprocess user input:
    - Lowercase
    - Remove special characters
    - Strip extra spaces
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Keep letters, numbers, spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def extract_keywords(text: str, stopwords: list = None) -> list:
    """
    Extract important words from text, ignoring common stopwords
    """
    if stopwords is None:
        stopwords = ['i', 'want', 'to', 'listen', 'play', 'give', 'me', 'song', 'music']
    words = text.split()
    keywords = [w for w in words if w not in stopwords]
    return keywords

def calculate_hybrid_score(song, llm_analysis, content_weight=0.7, popularity_weight=0.3, regional_boost=0.5):
    """
    Calculate hybrid recommendation score, now with regional boosting.
    
    Args:
        song: Dictionary containing song info including 'popularity' and 'is_regional'
        llm_analysis: Dictionary with LLM extracted info
        content_weight: Weight for content-based relevance
        popularity_weight: Weight for popularity score
        regional_boost: Additional score for regional tracks
    
    Returns:
        Float: Combined hybrid score (0.0 to 1.0)
    """
    relevance_score = 0.5
    
    # Keyword matching
    if llm_analysis.get('keywords'):
        song_text = f"{song.get('name', '')} {song.get('artist', '')}".lower()
        keyword_matches = sum(1 for keyword in llm_analysis['keywords'] if keyword.lower() in song_text)
        if keyword_matches > 0:
            relevance_score += min(0.4, keyword_matches * 0.15) # Increased boost for keywords
    
    # Popularity score (normalized)
    popularity_score = song.get('popularity', 0) / 100.0
    
    # Hybrid score calculation
    hybrid_score = (content_weight * relevance_score) + (popularity_weight * popularity_score)
    
    # Apply regional boost
    if song.get('is_regional'):
        hybrid_score += regional_boost
        
    return min(1.0, hybrid_score)

def format_recommendations(recs: list) -> str:
    """
    Convert list of song dictionaries into a readable string with clickable links
    Enhanced to show hybrid scores and featured artist indicators
    """
    if not recs:
        return "No recommendations found."
    
    formatted = "🎵 Here are your music recommendations:\n\n"
    for idx, song in enumerate(recs, 1):
        name = song.get('name', 'Unknown')
        artist = song.get('artist', 'Unknown Artist')
        url = song.get('spotify_url', '')
        popularity = song.get('popularity', 0)
        hybrid_score = song.get('hybrid_score')
        is_featured = song.get('is_featured', False)
        
        # Add feature indicator
        feature_indicator = " (Featured)" if is_featured else ""
        
        formatted += f"{idx}. **{name}** by {artist}{feature_indicator}"
        
        # Show scores for debugging/transparency
        if popularity > 0:
            formatted += f" (Pop: {popularity}"
            if hybrid_score is not None:
                formatted += f", Score: {hybrid_score:.2f}"
            formatted += ")"
        
        formatted += f"\n   🎧 Listen: {url}\n\n"
    
    return formatted
