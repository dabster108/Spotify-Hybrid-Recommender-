# Spotify Recommendation API

FastAPI application providing two music recommendation approaches:
1. **Basic Genre-Based** (Multi-step: Artist → Genres → Recommendations)
2. **Hybrid AI** (Single-step: Natural language query → AI recommendations)

## Installation

1. Install dependencies:
```bash
cd fastapi
pip install -r requirements.txt
```

2. Make sure your `.env` file is configured in the parent directory with:
```
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

## Running the API

Start the server:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## Interactive Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Health Check
- `GET /` - Root endpoint with API info
- `GET /health` - Health check
- `GET /api/info` - Detailed API information

### Basic Approach (Two-Step Process)

#### Step 1: Get Artist Genres
```bash
POST /api/basic/artist-genres
```

**Request:**
```json
{
  "artist_name": "Arijit Singh"
}
```

**Response:**
```json
{
  "artist_id": "4YRxDV8wJFPHPTeXepOstw",
  "artist_name": "Arijit Singh",
  "genres": ["bollywood", "indian pop", "modern bollywood"],
  "message": "Found 3 genres. Choose one for recommendations, or use custom genres like 'sad', 'pop', 'romantic'"
}
```

#### Step 2: Get Recommendations
```bash
POST /api/basic/recommend
```

**Request:**
```json
{
  "artist_name": "Arijit Singh",
  "genre": "bollywood",
  "num_recommendations": 8
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "name": "Tum Hi Ho",
      "artists": "Arijit Singh",
      "album": "Aashiqui 2",
      "popularity": 75,
      "similarity_score": 0.923,
      "preview_url": "https://...",
      "external_url": "https://open.spotify.com/track/..."
    }
  ],
  "genre": "bollywood",
  "artist_name": "Arijit Singh",
  "count": 8
}
```

### Hybrid Approach (Single-Step)

```bash
POST /api/hybrid/recommend
```

**Request:**
```json
{
  "query": "beautiful nepali songs",
  "existing_songs": []
}
```

**Response:**
```json
{
  "result": "🎵 **Enhanced Music Recommendations**\n============================================================\n\n📊 **AI Analysis Summary:**\n   🎯 **Strategy**: Hybrid AI Recommendations (3 diverse tracks)...",
  "formatted_output": true
}
```

## Example Usage

### Python Example

```python
import requests

# Basic Approach
# Step 1: Get genres
response = requests.post(
    "http://localhost:8000/api/basic/artist-genres",
    json={"artist_name": "Arijit Singh"}
)
data = response.json()
print(f"Available genres: {data['genres']}")

# Step 2: Get recommendations
response = requests.post(
    "http://localhost:8000/api/basic/recommend",
    json={
        "artist_name": "Arijit Singh",
        "genre": "bollywood",
        "num_recommendations": 5
    }
)
recommendations = response.json()
print(f"Got {recommendations['count']} recommendations")

# Hybrid Approach
response = requests.post(
    "http://localhost:8000/api/hybrid/recommend",
    json={"query": "beautiful nepali songs"}
)
result = response.json()
print(result['result'])
```

### cURL Examples

**Basic Approach - Step 1:**
```bash
curl -X POST "http://localhost:8000/api/basic/artist-genres" \
  -H "Content-Type: application/json" \
  -d '{"artist_name": "Arijit Singh"}'
```

**Basic Approach - Step 2:**
```bash
curl -X POST "http://localhost:8000/api/basic/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "artist_name": "Arijit Singh",
    "genre": "bollywood",
    "num_recommendations": 8
  }'
```

**Hybrid Approach:**
```bash
curl -X POST "http://localhost:8000/api/hybrid/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "beautiful nepali songs"
  }'
```

## Example Queries for Hybrid Approach

- `"beautiful nepali songs"`
- `"sad bollywood music"`
- `"energetic workout songs"`
- `"romantic evening music"`
- `"songs similar to Shape of You"`
- `"relaxing study music"`
- `"party dance music"`
- `"korean pop songs"`

## Features

### Basic Approach
- ✅ Two-step process for controlled recommendations
- ✅ Artist genre discovery
- ✅ Similarity-based recommendations
- ✅ Custom genre support (sad, pop, romantic)
- ✅ Configurable number of recommendations

### Hybrid Approach
- ✅ Natural language query understanding
- ✅ LLM-powered interpretation (Groq API)
- ✅ Cultural awareness (Nepali, Hindi, Korean, etc.)
- ✅ Mood and activity detection
- ✅ Multi-algorithm hybrid (Sequential + Ranking + Embedding)
- ✅ Strict language filtering
- ✅ Quality scoring and metrics

## Error Handling

The API returns standard HTTP error codes:
- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found (artist/tracks not found)
- `500` - Internal Server Error

Error response format:
```json
{
  "detail": "Error message describing what went wrong"
}
```

## Notes

- The Basic approach requires two API calls (get genres, then recommend)
- The Hybrid approach is single-call but more computationally intensive
- Both approaches use real Spotify data
- Language filtering is strict in Hybrid approach (Nepali songs won't include Hindi/English)
