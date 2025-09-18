# Spotify Recommendation — Enhanced Evaluation

A small project to generate and evaluate music recommendations using a hybrid recommender and an LLM-based evaluation pipeline (Mistral). This repository includes the recommender, caching utilities, and an enhanced evaluation harness that runs structured tests and uses the Mistral API to analyze recommendation quality.

## Features
- Hybrid recommendation system (local heuristics + external metadata)
- Structured evaluation harness with test cases
- LLM-assisted analysis using Mistral (JSON output expected)
- Language-detection rules that avoid guessing — marks "Unknown" if ambiguous
- Test-suite-style scoring: genre match, artist diversity, mood, language, overall quality

## Language handling rules (important)
- Do NOT infer language from genre or album name alone.
- Prefer reliable metadata sources: Spotify track/artist metadata or explicit language fields, or external language-detection applied to lyrics/title/artist metadata.
- If metadata is missing or ambiguous, set Language to `Unknown`.
- Follow examples:
  - "Muskurayera" by Sushant KC → Language: Nepali (NOT Hindi).
  - "Tum Hi Ho" by Arijit Singh → Language: Hindi.
- Output format for each recommended track (exact structured format used by the system):
  
  **Title**  
  Artist:  
  Genre:  
  Language:  
  Album:  
  Popularity:  
  Duration:  
  Spotify:  

## Requirements
- macOS (development targeted for Mac)
- Python 3.10+
- Recommended Python packages (install via pip):
  - requests
- (Optional) virtualenv / venv for isolated environment

Create a `requirements.txt` with at least:
```
requests
```

## Installation
1. Clone the repo:
   - git clone <repo-url> "or open the project folder in VS Code"
2. Create and activate a venv (macOS):
   - python3 -m venv .venv
   - source .venv/bin/activate
3. Install dependencies:
   - pip install -r requirements.txt

## Configuration
- Do NOT commit API keys to the repository.
- The evaluation harness (enhanced_evaluate.py) expects a Mistral API key and endpoint. Remove any hard-coded keys from code and set them via environment variables or a local config:
  - export MISTRAL_API_KEY="your_api_key_here"
  - export MISTRAL_API_URL="https://api.mistral.ai/v1/chat/completions"
- If the recommender uses Spotify APIs, set Spotify credentials separately (client id, secret, refresh token) and follow the recommend module's configuration.

## Usage
- Run full evaluation (will call Mistral and the recommender; avoid flooding the API):
  - python3 enhanced_evaluate.py
- The script will:
  - Run a set of predefined test cases
  - Request recommendations from the HybridRecommendationSystem
  - Use the MistralEvaluator to extract structured recommendation info and quality metrics
  - Print per-test and final aggregated reports

## Safety & Best Practices
- Rate limiting: The evaluator includes delays and retry logic. Keep Mistral rate limits in mind.
- Secrets: Keep API keys out of the code. Use environment variables or a .env file (and add .env to .gitignore).
- LLM output: The pipeline expects strict JSON from Mistral; fallback parsers exist but are less reliable.

## Development notes
- Main evaluation entry: `enhanced_evaluate.py`
- Recommender entrypoint: `recommend.py` (HybridRecommendationSystem)
- Caching utilities: `cache.py`
- Add unit tests for language-detection and recommendation formatting to increase robustness.

## Contributing
- Open an issue for bugs or feature requests.
- Follow the existing style and add tests for new behavior.
- Ensure API keys are never committed.

## License
- Add a license file appropriate for your needs (e.g., MIT).

## Example
A single recommended track must be printed/stored in the exact structured format shown above. Example (illustrative only — follow metadata sources for language):

**Tum Hi Ho**  
Artist: Arijit Singh  
Genre: Bollywood  
Language: Hindi  
Album: Aashiqui 2  
Popularity: 92  
Duration: 04:21  
Spotify: https://open.spotify.com/track/...
