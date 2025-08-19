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

def calculate_hybrid_score(song, llm_analysis, content_weight=0.6, popularity_weight=0.4):
    """
    Calculate hybrid recommendation score combining relevance and popularity
    
    Args:
        song: Dictionary containing song info including 'popularity'
        llm_analysis: Dictionary with LLM extracted info (song, artist, keywords, etc.)
        content_weight: Weight for content-based relevance (default 0.6)
        popularity_weight: Weight for popularity score (default 0.4)
    
    Returns:
        Float: Combined hybrid score (0.0 to 1.0)
    """
    # Base relevance score
    relevance_score = 0.5
    
    # Boost for exact song match
    if (llm_analysis.get('song') and 
        song.get('name', '').lower() in llm_analysis['song'].lower()):
        relevance_score += 0.3
    
    # Boost for exact artist match  
    if (llm_analysis.get('artist') and 
        song.get('artist', '').lower() in llm_analysis['artist'].lower()):
        relevance_score += 0.3
    
    # Boost for keyword matches in song name or artist
    if llm_analysis.get('keywords'):
        song_text = f"{song.get('name', '')} {song.get('artist', '')}".lower()
        keyword_matches = sum(1 for keyword in llm_analysis['keywords'] 
                            if keyword.lower() in song_text)
        if keyword_matches > 0:
            relevance_score += min(0.2, keyword_matches * 0.1)
    
    # Normalize popularity (0-100) to 0-1 scale
    popularity_score = song.get('popularity', 0) / 100.0
    
    # Calculate weighted hybrid score
    hybrid_score = (content_weight * relevance_score) + (popularity_weight * popularity_score)
    
    return min(1.0, hybrid_score)  # Cap at 1.0

def format_recommendations(recs: list) -> str:
    """
    Convert list of song dictionaries into a readable string with clickable links
    Enhanced to show hybrid scores if available
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
        
        formatted += f"{idx}. **{name}** by {artist}"
        
        # Show scores for debugging/transparency
        if popularity > 0:
            formatted += f" (Pop: {popularity}"
            if hybrid_score is not None:
                formatted += f", Score: {hybrid_score:.2f}"
            formatted += ")"
        
        formatted += f"\n   🎧 Listen: {url}\n\n"
    
    return formatted
