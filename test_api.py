#!/usr/bin/env python3
"""Test script for Groq API integration."""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_groq_api():
    """Test Groq API connection and functionality."""
    try:
        from Recommend_System.config import config
        
        print("🤖 Testing Groq API Integration")
        print("=" * 40)
        
        # Check if API key is configured
        if not config.groq_api_key:
            print("❌ Groq API key not found in environment")
            print("💡 Please set GROQ_API_KEY in your .env file")
            return False
        
        if not config.groq_api_key.startswith('gsk_'):
            print("❌ Invalid Groq API key format")
            print("💡 Groq API keys should start with 'gsk_'")
            return False
        
        print(f"✅ API Key configured: {config.groq_api_key[:10]}...")
        
        # Test API connection
        import requests
        
        headers = {
            'Authorization': f'Bearer {config.groq_api_key}',
            'Content-Type': 'application/json'
        }
        
        # Simple test payload
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Hello from Groq!' if you can hear me."
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        print("🔄 Testing API connection...")
        response = requests.post(
            config.groq_api_url,
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            message = result['choices'][0]['message']['content']
            print(f"✅ API Response: {message}")
            print("🎉 Groq API is working correctly!")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
        print("💡 Check your internet connection")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_recommendation_system():
    """Test the recommendation system with Groq integration."""
    try:
        print("\n🎵 Testing Recommendation System")
        print("=" * 40)
        
        from Recommend_System.config import config
        config.print_status()
        
        # Test if we can import the main classes
        from Recommend_System.recommend import HybridMusicRecommender
        print("✅ Successfully imported HybridMusicRecommender")
        
        # Test basic initialization (without API calls)
        recommender = HybridMusicRecommender()
        print("✅ Successfully initialized recommender")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing recommendation system: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Spotify Recommendation System - API Tests")
    print("=" * 50)
    
    success = True
    
    # Test Groq API
    if not test_groq_api():
        success = False
    
    # Test recommendation system
    if not test_recommendation_system():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Your system is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Run setup.py first if you haven't already")

if __name__ == "__main__":
    main()