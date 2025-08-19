#!/usr/bin/env python3
"""
Simple test script to verify the recommendation system works
"""

import os
import sys
from dotenv import load_dotenv

def test_api_keys():
    """Test if API keys are properly configured and valid"""
    load_dotenv()
    
    required_keys = ['GROQ_API_KEY', 'SPOTIFY_CLIENT_ID', 'SPOTIFY_CLIENT_SECRET']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print("❌ Missing API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        return False
        
    # Test Spotify credentials
    print("\nTesting Spotify credentials...")
    try:
        from spotify_api import get_spotify_client
        sp = get_spotify_client()
        if sp:
            # Try a simple API call
            sp.search(q="test", limit=1)
            print("✅ Spotify credentials verified")
        else:
            print("❌ Failed to initialize Spotify client")
            return False
    except Exception as e:
        print(f"❌ Spotify API error: {e}")
        return False
    
    # Test Groq API key
    print("\nTesting Groq API key...")
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        print("✅ Groq API key verified")
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return False
    
    print("\n✅ All API keys are configured and working")
    return True

def test_search_functionality():
    """Test the search functionality with various queries"""
    from spotify_api import search_track
    
    test_queries = [
        ("Bohemian Rhapsody", "Queen"),  # Exact match
        ("Shape of You", "Ed Sheeran"),   # Popular song
        ("Yesterday", "The Beatles"),      # Classic song
        ("nonexistentsongxyz123", None)   # Should return None for invalid query
    ]
    
    print("\nTesting search functionality...")
    for query, expected_artist in test_queries:
        try:
            result = search_track(query)
            if result:
                track_info = result
                if track_info.get('artists'):
                    actual_artist = track_info['artists'][0]['name']
                    if expected_artist is None:
                        print(f"❌ Query '{query}' should return None but got a result")
                        continue
                    elif actual_artist and actual_artist.lower() == expected_artist.lower():
                        print(f"✅ Successfully found '{query}' by {actual_artist}")
                    else:
                        print(f"⚠️ Query '{query}' returned {actual_artist} instead of {expected_artist}")
                else:
                    print(f"❌ Query '{query}' returned a track without artist information")
            else:
                if expected_artist is None:
                    print(f"✅ Correctly returned None for invalid query '{query}'")
                else:
                    print(f"❌ Failed to find '{query}' by {expected_artist}")
        except Exception as e:
            print(f"❌ Error testing search for '{query}': {e}")
    
    return True

def test_recommendation_functionality():
    """Test the recommendation functionality"""
    from spotify_api import get_recommendations
    from model import hybrid_recommend
    
    print("\nTesting recommendation functionality...")
    
    # Test basic recommendation
    seed_tracks = ["spotify:track:4cOdK2wGLETKBW3PvgPWqT"]  # "Never Gonna Give You Up"
    try:
        spotify_recs = get_recommendations(seed_tracks)
        if spotify_recs and len(spotify_recs) > 0:
            print(f"✅ Successfully got {len(spotify_recs)} Spotify recommendations")
        else:
            print("❌ Failed to get Spotify recommendations")
    except Exception as e:
        print(f"❌ Error getting Spotify recommendations: {e}")
    
    # Test hybrid recommendations
    test_queries = [
        "I want upbeat pop songs for working out",
        "Relaxing classical music for studying"
    ]
    
    for query in test_queries:
        try:
            recommendations = hybrid_recommend(query)
            if recommendations and len(recommendations) > 0:
                print(f"✅ Hybrid recommendations successful for query: '{query}'")
                print(f"   Got {len(recommendations)} recommendations")
            else:
                print(f"❌ No hybrid recommendations for query: '{query}'")
        except Exception as e:
            print(f"❌ Error getting hybrid recommendations for '{query}': {e}")
    
    return True

if __name__ == "__main__":
    test_api_keys()
    test_search_functionality()
    test_recommendation_functionality()

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from model import hybrid_recommend
        from utils import format_recommendations
        from spotify_api import get_spotify_client
        from groq import Groq
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Try running: pip install -r requirements.txt")
        return False

def test_recommendation():
    """Test the recommendation system with specific test cases"""
    try:
        from model import hybrid_recommend
        from utils import format_recommendations
        from spotify_api import get_spotify_client
        
        # First test Spotify connection
        print("🔍 Testing Spotify connection...")
        sp = get_spotify_client()
        if not sp:
            print("❌ Failed to connect to Spotify API")
            return False
            
        # Test basic search functionality
        print("🔍 Testing basic search...")
        try:
            results = sp.search(q="artist:Taylor Swift", type='track', limit=1)
            if not results['tracks']['items']:
                print("❌ Spotify search returned no results")
                return False
            print("✅ Spotify search working")
        except Exception as e:
            print(f"❌ Spotify search failed: {e}")
            return False
        
        # Test recommendation system
        print("\n🔍 Testing recommendation system...")
        test_queries = [
            "Play something by Taylor Swift",  # Artist-based
            "I want relaxing piano music",     # Genre-based
        ]
        
        success = False
        for query in test_queries:
            print(f"\nTrying query: '{query}'")
            recommendations = hybrid_recommend(query, limit=2)
            
            if recommendations and len(recommendations) > 0:
                print("✅ Got recommendations!")
                print("\nSample recommendation:")
                print(format_recommendations([recommendations[0]]))
                success = True
                break
        
        if success:
            print("\n✅ Recommendation system working!")
            return True
        else:
            print("\n❌ No recommendations returned for any test query")
            return False
            
    except Exception as e:
        print(f"\n❌ Error testing recommendations: {e}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Running Spotify Recommendation System Tests")
    print("=" * 50)
    
    tests = [
        ("API Keys", test_api_keys),
        ("Imports", test_imports),
        ("Recommendations", test_recommendation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your system is ready to use.")
        print("Run 'python main.py' to start the chatbot.")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
