#!/usr/bin/env python3
"""
Test script for the enhanced hybrid recommendation system
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if required API keys are present
required_keys = ['GROQ_API_KEY', 'SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET']
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    print("❌ Error: Missing required environment variables:")
    for key in missing_keys:
        print(f"   - {key}")
    print("\nPlease make sure your .env file contains all required API keys.")
    sys.exit(1)

from model import hybrid_recommend
from utils import format_recommendations

def test_hybrid_system():
    """Test the hybrid recommendation system with various queries"""
    
    test_queries = [
        "I want to listen to relaxing pop music",
        "Give me some upbeat rock songs",
        "Play something like Shape of You by Ed Sheeran", 
        "I'm in the mood for chill electronic music",
        "Hindi classical music",
        "Bollywood songs",
        "Something by Taylor Swift",
        "Energetic dance music for workout"
    ]
    
    print("🧪 Testing Enhanced Hybrid Recommendation System")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)
        
        try:
            recommendations = hybrid_recommend(query, limit=3)
            
            if recommendations:
                print(f"✅ Got {len(recommendations)} recommendations:")
                for j, rec in enumerate(recommendations, 1):
                    hybrid_score = rec.get('hybrid_score', 0)
                    popularity = rec.get('popularity', 0)
                    print(f"   {j}. {rec['name']} by {rec['artist']}")
                    print(f"      Score: {hybrid_score:.2f}, Popularity: {popularity}")
            else:
                print("❌ No recommendations returned")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()

def test_individual_components():
    """Test individual components of the system"""
    from model import llm_semantic_analysis, content_based_filtering, collaborative_filtering
    
    print("\n🔬 Testing Individual Components")
    print("=" * 50)
    
    # Test LLM analysis
    print("\n1. Testing LLM Semantic Analysis")
    test_text = "I want to listen to something like Blinding Lights by The Weeknd"
    try:
        analysis = llm_semantic_analysis(test_text)
        print(f"   Query: {test_text}")
        print(f"   Extracted song: {analysis.get('song')}")
        print(f"   Extracted artist: {analysis.get('artist')}")
        print(f"   Genres: {analysis.get('genres')}")
        print(f"   Keywords: {analysis.get('keywords')}")
    except Exception as e:
        print(f"   ❌ LLM Analysis Error: {e}")
    
    # Test Content-Based Filtering
    print("\n2. Testing Content-Based Filtering")
    try:
        content_recs = content_based_filtering(test_text, analysis, limit=3)
        print(f"   Got {len(content_recs)} content-based recommendations")
        for rec in content_recs[:2]:
            print(f"   - {rec['name']} by {rec['artist']} (Pop: {rec.get('popularity')})")
    except Exception as e:
        print(f"   ❌ Content-Based Error: {e}")
    
    # Test Collaborative Filtering
    print("\n3. Testing Collaborative Filtering (Popularity-Based)")
    try:
        cf_recs = collaborative_filtering(limit=3)
        print(f"   Got {len(cf_recs)} popularity-based recommendations")
        for rec in cf_recs[:2]:
            print(f"   - {rec['name']} by {rec['artist']} (Pop: {rec.get('popularity')})")
    except Exception as e:
        print(f"   ❌ Collaborative Filtering Error: {e}")

if __name__ == "__main__":
    print("🎵 Hybrid Recommendation System Test Suite")
    print("=" * 60)
    
    # Test the main hybrid system
    test_hybrid_system()
    
    # Test individual components
    test_individual_components()
    
    print("\n✅ Test suite completed!")
