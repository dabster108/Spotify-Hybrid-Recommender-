from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Approach_alternative.approach import SpotifyRecommendationSystem
from Recommend_System.recommend import HybridRecommendationSystem

# Initialize FastAPI app
app = FastAPI(
    title="Spotify Recommendation API",
    description="API for music recommendations using two different approaches: Basic Genre-based and Hybrid AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation systems
basic_system = SpotifyRecommendationSystem()
hybrid_system = HybridRecommendationSystem()

# Store user sessions for the basic approach (multi-step process)
user_sessions = {}

# ==================== Pydantic Models ====================

class ArtistGenreRequest(BaseModel):
    artist_name: str = Field(..., description="Name of the artist to search for")

class ArtistGenreResponse(BaseModel):
    artist_id: str
    artist_name: str
    genres: List[str]
    message: str

class BasicRecommendationRequest(BaseModel):
    artist_name: str = Field(..., description="Name of the artist")
    genre: str = Field(..., description="Genre for recommendations")
    num_recommendations: int = Field(default=8, ge=1, le=20, description="Number of recommendations (1-20)")

class BasicRecommendationResponse(BaseModel):
    recommendations: List[Dict]
    genre: str
    artist_name: str
    count: int

class HybridRecommendationRequest(BaseModel):
    query: str = Field(..., description="Natural language music query")
    existing_songs: Optional[List[str]] = Field(default=None, description="List of songs already known/listened to")

class TrackRecommendation(BaseModel):
    name: str
    artists: str
    album: str
    genre: str
    language: str
    popularity: int
    duration: str
    preview_url: Optional[str]
    external_url: str

class HybridRecommendationResponse(BaseModel):
    query: str
    strategy: str
    language: Optional[str] = None
    genres: List[str]
    mood: Optional[str] = None
    count: int
    recommendations: List[TrackRecommendation]
    quality_metrics: Dict[str, Any]
    analysis_summary: Dict[str, Any]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# ==================== Health Check ====================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Spotify Recommendation API is running!",
        "version": "1.0.0",
        "endpoints": {
            "basic_approach": {
                "step_1": "/api/basic/artist-genres",
                "step_2": "/api/basic/recommend"
            },
            "hybrid_approach": "/api/hybrid/recommend"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

# ==================== BASIC APPROACH (Multi-Step) ====================

@app.post("/api/basic/artist-genres", response_model=ArtistGenreResponse)
async def get_artist_genres(request: ArtistGenreRequest):
    """
    Step 1: Get available genres for an artist
    
    This is the first step in the basic recommendation approach.
    User provides an artist name, and we return available genres.
    """
    try:
        artist_name = request.artist_name.strip()
        
        # Get artist ID
        artist_id = basic_system.get_artist_id(artist_name)
        
        if not artist_id:
            raise HTTPException(
                status_code=404, 
                detail=f"Artist '{artist_name}' not found on Spotify"
            )
        
        # Get artist genres
        genres = basic_system.get_artist_genres(artist_id)
        
        if not genres:
            return ArtistGenreResponse(
                artist_id=artist_id,
                artist_name=artist_name,
                genres=[],
                message="No Spotify genres available. You can try custom genres like 'sad', 'pop', or 'romantic'"
            )
        
        return ArtistGenreResponse(
            artist_id=artist_id,
            artist_name=artist_name,
            genres=genres,
            message=f"Found {len(genres)} genres. Choose one for recommendations, or use custom genres like 'sad', 'pop', 'romantic'"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching artist genres: {str(e)}")

@app.post("/api/basic/recommend", response_model=BasicRecommendationResponse)
async def get_basic_recommendations(request: BasicRecommendationRequest):
    """
    Step 2: Get recommendations based on artist and genre
    
    After selecting a genre from Step 1, use this endpoint to get recommendations.
    Supports both Spotify genres and custom genres (sad, pop, romantic).
    """
    try:
        artist_name = request.artist_name.strip()
        genre = request.genre.strip()
        num_recommendations = request.num_recommendations
        
        # Get recommendations
        recommendations = basic_system.recommend_by_genre(
            artist_name=artist_name,
            genre=genre,
            num_recommendations=num_recommendations
        )
        
        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for {artist_name} in {genre} genre"
            )
        
        # Format recommendations
        formatted_recommendations = []
        for track, similarity in recommendations:
            formatted_recommendations.append({
                "name": track.get("name", "Unknown"),
                "artists": ", ".join([artist["name"] for artist in track.get("artists", [])]),
                "album": track.get("album", {}).get("name", "Unknown"),
                "popularity": track.get("popularity", 0),
                "similarity_score": round(similarity, 3),
                "preview_url": track.get("preview_url"),
                "external_url": track.get("external_urls", {}).get("spotify")
            })
        
        return BasicRecommendationResponse(
            recommendations=formatted_recommendations,
            genre=genre,
            artist_name=artist_name,
            count=len(formatted_recommendations)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# ==================== HYBRID APPROACH (Single-Step) ====================

def parse_hybrid_output(markdown_output: str, query: str) -> Dict:
    """Parse the markdown output from hybrid system into structured JSON"""
    import re
    
    # Initialize the result structure
    result = {
        "query": query,
        "strategy": "",
        "language": None,
        "genres": [],
        "mood": None,
        "count": 0,
        "recommendations": [],
        "quality_metrics": {},
        "analysis_summary": {}
    }
    
    # Extract strategy
    strategy_match = re.search(r'Strategy\*\*:\s*([^\n]+)', markdown_output)
    if strategy_match:
        result["strategy"] = strategy_match.group(1).strip()
    
    # Extract language
    lang_match = re.search(r'Language/Culture\*\*:\s*([^\n]+)', markdown_output)
    if lang_match:
        result["language"] = lang_match.group(1).strip()
    
    # Extract genres
    genres_match = re.search(r'Genres\*\*:\s*([^\n]+)', markdown_output)
    if genres_match:
        genres_text = genres_match.group(1).strip()
        result["genres"] = [g.strip() for g in genres_text.split(',')]
    
    # Extract mood
    mood_match = re.search(r'Mood\*\*:\s*[^a-zA-Z]*([^\n]+)', markdown_output)
    if mood_match:
        result["mood"] = mood_match.group(1).strip()
    
    # Extract recommendations
    # Pattern to match each recommendation block
    rec_pattern = r'\*\*(\d+)\.\s*([^\n]+)\*\*\s*\n\s*🎤\s*\*\*Artist\*\*:\s*([^\n]+)\s*\n\s*🎸\s*\*\*Genre\*\*:\s*([^\n]+)\s*\n\s*🌍\s*\*\*Language\*\*:\s*([^\n]+)\s*\n\s*💿\s*\*\*Album\*\*:\s*([^\n]+)\s*\n\s*📈\s*\*\*Popularity\*\*:\s*[^(]*\((\d+)/100[^\)]*\)\s*\n\s*⏱️\s*\*\*Duration\*\*:\s*([^\n]+)\s*\n\s*🔗\s*\*\*Listen\*\*:\s*\[(?:Spotify|Preview on Spotify)\]\(([^\)]+)\)'
    
    matches = re.finditer(rec_pattern, markdown_output)
    
    for match in matches:
        track = {
            "name": match.group(2).strip(),
            "artists": match.group(3).strip(),
            "album": match.group(6).strip(),
            "genre": match.group(4).strip(),
            "language": match.group(5).strip(),
            "popularity": int(match.group(7)),
            "duration": match.group(8).strip(),
            "preview_url": match.group(9).strip() if match.group(9) else None,
            "external_url": match.group(9).strip()
        }
        result["recommendations"].append(track)
    
    result["count"] = len(result["recommendations"])
    
    # Extract quality metrics
    metrics_section = re.search(r'Advanced Quality Analysis:\*\*(.+?)(?:💡|$)', markdown_output, re.DOTALL)
    if metrics_section:
        metrics_text = metrics_section.group(1)
        
        # Artist diversity
        diversity_match = re.search(r'Artist Diversity\*\*:\s*(\d+)[^\(]*\(([^\)]+)\)', metrics_text)
        if diversity_match:
            result["quality_metrics"]["artist_diversity"] = {
                "count": int(diversity_match.group(1)),
                "score": diversity_match.group(2).strip()
            }
        
        # Quality distribution
        quality_match = re.search(r'Quality Distribution\*\*:\s*(\d+)\s*premium,\s*(\d+)\s*popular,\s*(\d+)\s*emerging', metrics_text)
        if quality_match:
            result["quality_metrics"]["quality_distribution"] = {
                "premium": int(quality_match.group(1)),
                "popular": int(quality_match.group(2)),
                "emerging": int(quality_match.group(3))
            }
        
        # Premium rate
        premium_match = re.search(r'Premium Rate\*\*:\s*(\d+)%', metrics_text)
        if premium_match:
            result["quality_metrics"]["premium_rate"] = int(premium_match.group(1))
        
        # Cultural accuracy
        cultural_match = re.search(r'Cultural Accuracy\*\*:\s*(\d+)/(\d+)[^\(]*\((\d+)%\)', metrics_text)
        if cultural_match:
            result["quality_metrics"]["cultural_accuracy"] = {
                "matches": int(cultural_match.group(1)),
                "total": int(cultural_match.group(2)),
                "percentage": int(cultural_match.group(3))
            }
        
        # Average popularity
        avg_pop_match = re.search(r'Average Popularity\*\*:\s*(\d+)/100', metrics_text)
        if avg_pop_match:
            result["quality_metrics"]["average_popularity"] = int(avg_pop_match.group(1))
        
        # Freshness
        freshness_match = re.search(r'Freshness\*\*:\s*(\d+)%\s*recent releases', metrics_text)
        if freshness_match:
            result["quality_metrics"]["freshness_percentage"] = int(freshness_match.group(1))
    
    # Analysis summary
    result["analysis_summary"] = {
        "total_tracks_found": result["count"],
        "primary_language": result["language"],
        "primary_genres": result["genres"],
        "mood": result["mood"]
    }
    
    return result

@app.post("/api/hybrid/recommend", response_model=HybridRecommendationResponse)
async def get_hybrid_recommendations(request: HybridRecommendationRequest):
    """
    Get AI-powered hybrid recommendations from natural language query
    
    This uses advanced AI with LLM interpretation, cultural awareness,
    and multiple recommendation algorithms (sequential, ranking, embedding).
    
    Returns structured JSON with parsed recommendations and quality metrics.
    
    Examples:
    - "beautiful nepali songs"
    - "sad bollywood songs"
    - "energetic workout music"
    - "romantic evening songs"
    - "songs similar to Shape of You"
    """
    try:
        query = request.query.strip()
        existing_songs = request.existing_songs or []
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if it's a music query
        if not hybrid_system.is_music_query(query):
            raise HTTPException(
                status_code=400, 
                detail="Query does not appear to be music-related. Please ask for music recommendations."
            )
        
        # Get hybrid recommendations (markdown format)
        markdown_result = hybrid_system.get_hybrid_recommendations(
            query=query,
            existing_songs=existing_songs if existing_songs else None
        )
        
        # Parse the markdown output into structured JSON
        parsed_result = parse_hybrid_output(markdown_result, query)
        
        return HybridRecommendationResponse(**parsed_result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# ==================== Utility Endpoints ====================

@app.get("/api/info")
async def get_api_info():
    """Get information about the API and how to use it"""
    return {
        "api_name": "Spotify Recommendation API",
        "version": "1.0.0",
        "approaches": {
            "basic": {
                "name": "Basic Genre-Based Recommendations",
                "description": "Two-step process: First get artist genres, then get recommendations",
                "steps": [
                    {
                        "step": 1,
                        "endpoint": "POST /api/basic/artist-genres",
                        "input": {"artist_name": "string"},
                        "output": "List of available genres for the artist"
                    },
                    {
                        "step": 2,
                        "endpoint": "POST /api/basic/recommend",
                        "input": {
                            "artist_name": "string",
                            "genre": "string (from step 1)",
                            "num_recommendations": "integer (1-20)"
                        },
                        "output": "List of recommended tracks with similarity scores"
                    }
                ],
                "example_flow": [
                    "1. POST /api/basic/artist-genres with {'artist_name': 'Arijit Singh'}",
                    "2. Review returned genres: ['bollywood', 'indian pop', ...]",
                    "3. POST /api/basic/recommend with {'artist_name': 'Arijit Singh', 'genre': 'bollywood', 'num_recommendations': 8}"
                ]
            },
            "hybrid": {
                "name": "Hybrid AI Recommendations",
                "description": "Single-step natural language query with AI-powered interpretation",
                "endpoint": "POST /api/hybrid/recommend",
                "input": {
                    "query": "Natural language music request",
                    "existing_songs": "Optional list of songs to exclude"
                },
                "features": [
                    "LLM-powered query interpretation",
                    "Cultural awareness (Nepali, Hindi, Korean, etc.)",
                    "Mood and activity context detection",
                    "Multi-algorithm hybrid approach",
                    "Strict language filtering"
                ],
                "example_queries": [
                    "beautiful nepali songs",
                    "sad bollywood music",
                    "energetic workout songs",
                    "romantic evening music",
                    "songs similar to Shape of You",
                    "relaxing study music"
                ]
            }
        }
    }

# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
