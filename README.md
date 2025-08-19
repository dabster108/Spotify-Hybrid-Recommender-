# 🎵 Enhanced Spotify Hybrid Music Recommendation System

A sophisticated music recommendation chatbot that combines **Groq LLM** semantic understanding with **Spotify API** to provide personalized music recommendations using an advanced hybrid approach with popularity-based ranking.

## 🚀 Features

- **Advanced LLM Semantic Understanding**: Uses Groq API to extract song/artist names, music preferences, genres, moods, and cultural context
- **Multi-Strategy Content-Based Filtering**: Intelligent search with song detection, genre matching, and cultural context
- **Popularity-Based Ranking**: Leverages Spotify's popularity scores for collaborative filtering without user data
- **Enhanced Hybrid Fusion**: Sophisticated scoring system combining relevance and popularity (60% content + 40% popularity)
- **Cultural Music Support**: Special handling for Hindi, Bollywood, regional music requests
- **Interactive Chat Interface**: User-friendly command-line chatbot with detailed feedback
- **Robust Error Handling**: Graceful fallbacks when APIs are unavailable

## 🏗️ Enhanced Architecture

The system follows a sophisticated hybrid recommendation approach:

```
User Input → Text Preprocessing → LLM Semantic Analysis (Extract song/artist/genres/mood)
                                               ↓
Content-Based Filtering (5 search strategies) → Popularity-Based Ranking
                                               ↓
Lightweight Collaborative Filtering (Popular tracks) → Hybrid Score Calculation
                                               ↓
Final Recommendations (Sorted by combined relevance + popularity score)
```

### Hybrid Flow Details:

1. **Content-Based Filtering**:
   - LLM extracts keywords, genres, mood, song/artist names, cultural context
   - 5-tier search strategy: Direct song/artist → Artist-only → Genre/cultural → Keywords → Fallback
   - Uses Spotify search API with intelligent query building

2. **Popularity-Based Ranking**:
   - Each song gets Spotify's popularity score (0-100)
   - Acts as lightweight collaborative filtering without user database
   - Higher popular songs get boosted in final ranking

3. **Hybrid Score Calculation**:
   - **Relevance Score**: Based on LLM analysis matches (song, artist, keywords)
   - **Popularity Score**: Normalized Spotify popularity (0-1 scale)
   - **Final Score**: 60% relevance + 40% popularity
   - Ensures both relevant and popular songs in recommendations

## 📁 Project Structure

```
SpotifyRecommendation/
├── main.py               # Interactive chatbot interface
├── model.py              # Enhanced hybrid recommendation logic
├── spotify_api.py        # Spotify API integration with advanced search
├── config.py             # Configuration and environment variables
├── utils.py              # Text processing, hybrid scoring, formatting
├── test_hybrid.py        # Test suite for hybrid system
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (API keys)
└── README.md             # Documentation
```

## 🛠️ Installation

### 1. Clone or Set Up the Project

Make sure you have Python 3.7+ installed.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up API Keys

Create a `.env` file in the root directory (use `.env.example` as template):

```env
# Groq API Key (get from https://console.groq.com/)
GROQ_API_KEY=your_groq_api_key_here

# Spotify API Credentials (get from https://developer.spotify.com/)
SPOTIFY_CLIENT_ID=your_spotify_client_id_here
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret_here
```

#### Getting API Keys:

**Groq API Key:**
1. Visit [https://console.groq.com/](https://console.groq.com/)
2. Sign up/login and create an API key
3. Copy the key to your `.env` file

**Spotify API Credentials:**
1. Visit [https://developer.spotify.com/](https://developer.spotify.com/)
2. Login and go to Dashboard
3. Create a new app
4. Copy Client ID and Client Secret to your `.env` file

## 🎮 Usage

### Running the Chatbot

```bash
python main.py
```

### Example Interactions

```
🎧 You: I want to listen to relaxing pop music
🎵 Here are your music recommendations:

1. **Blinding Lights** by The Weeknd (Popularity: 95)
   🔗 https://open.spotify.com/track/...

2. **Watermelon Sugar** by Harry Styles (Popularity: 89)
   🔗 https://open.spotify.com/track/...
   
...
```

### Sample User Inputs:
- "I want to listen to relaxing pop music"
- "Give me some upbeat rock songs"
- "I'm in the mood for chill electronic music"
- "Play some energetic hip-hop tracks"
- "I need calming classical music for studying"

## 🏗️ File Responsibilities

### `main.py`
- Chatbot interface and user interaction
- Input validation and error handling
- API key verification
- Formats and displays recommendations

### `model.py`
- **`hybrid_recommend()`**: Main function combining all approaches
- **`llm_semantic_analysis()`**: Uses Groq to extract music preferences
- **`content_based_filtering()`**: Gets songs based on keywords/genres
- **`collaborative_filtering()`**: Provides popular track recommendations
- **`preprocess_input()`**: Cleans and prepares user input

### `spotify_api.py`
- Spotify authentication and client initialization
- **`get_recommendations()`**: Fetches songs based on genres/tracks
- **`get_top_tracks()`**: Gets popular songs for collaborative filtering
- **`find_closest_genre()`**: Maps user inputs to valid Spotify genres
- Error handling for API calls

### `utils.py`
- **`clean_text()`**: Text preprocessing (lowercase, remove special chars)
- **`extract_keywords()`**: Extracts relevant keywords from text
- **`format_recommendations()`**: Formats song lists for display

### `config.py`
- Loads environment variables from `.env`
- Exports API keys and configuration

## 🎯 Key Features

### 1. Smart Genre Mapping
The system automatically maps user inputs to valid Spotify genres:
- "chill" → "ambient"
- "edm" → "electronic"  
- "rap" → "hip-hop"
- "rnb" → "r-n-b"

### 2. Robust Error Handling
- Graceful fallbacks when APIs fail
- Input validation and sanitization
- Missing API key detection

### 3. Modular Design
- Easy to extend with new recommendation strategies
- Clean separation of concerns
- Configurable recommendation weights

### 4. Hybrid Scoring
- Content-based recommendations: 70% weight
- Collaborative filtering: 30% weight
- Easily adjustable for different use cases

## 🔧 Customization

### Adding New Recommendation Strategies

1. Create a new function in `model.py`:
```python
def your_new_strategy(user_input, limit=5):
    # Your recommendation logic here
    return recommendations
```

2. Integrate into `hybrid_recommend()`:
```python
new_recs = your_new_strategy(user_text, limit)
final_recs.extend(new_recs)
```

### Adjusting Recommendation Weights

In `model.py`, modify the hybrid fusion weights:
```python
weight_cbf = 0.8  # Increase for more content-based focus
weight_cf = 0.2   # Decrease collaborative filtering influence
```

## 🚨 Troubleshooting

### Common Issues:

1. **"Missing API keys"**: Make sure your `.env` file exists and contains valid keys
2. **"No recommendations found"**: Check your internet connection and API key validity
3. **"Invalid genre"**: The system will automatically map to closest valid genre
4. **Import errors**: Run `pip install -r requirements.txt`

### Debug Mode:

Add print statements in `model.py` to see intermediate results:
```python
print(f"LLM keywords: {llm_keywords}")
print(f"CBF results: {len(cbf_recs)}")
```

## 📝 Dependencies

- **spotipy**: Spotify Web API wrapper
- **groq**: Groq LLM API client
- **python-dotenv**: Environment variable management
- **nltk**: Natural language processing utilities

## 🔮 Future Enhancements

- [ ] User profile persistence
- [ ] Machine learning-based collaborative filtering
- [ ] Music mood analysis from audio features
- [ ] Playlist generation
- [ ] Web interface with Flask/FastAPI
- [ ] Database integration for user preferences
- [ ] Real-time recommendation updates
- [ ] Integration with more music APIs

## 📄 License

This project is open source. Feel free to modify and distribute as needed.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**Happy Listening! 🎵**