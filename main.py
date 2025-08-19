#!/usr/bin/env python3
"""
Spotify Music Recommendation Chatbot
A hybrid recommendation system using Groq LLM and Spotify API
"""

import os
import sys
from dotenv import load_dotenv
from model import hybrid_recommend
from utils import format_recommendations

def main():
    """Main chatbot interface"""
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
    
    # Test API connections
    from spotify_api import get_spotify_client
    sp = get_spotify_client()
    if not sp:
        print("❌ Failed to initialize Spotify client. Please check your credentials and try again.")
        sys.exit(1)
    
    from groq import Groq
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # Test LLM with a simple query
        client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
    except Exception as e:
        print(f"❌ Failed to initialize Groq client: {e}")
        print("Please check your Groq API key and try again.")
        sys.exit(1)
    
    # Welcome message
    print("🎵 Welcome to the Spotify Music Recommendation Chatbot! 🎵")
    print("=" * 60)
    print("Tell me what kind of music you'd like to listen to, and I'll")
    print("recommend some great songs for you!")
    print()
    print("Examples:")
    print("  • 'I want to listen to relaxing pop music'")
    print("  • 'Give me some upbeat rock songs'")
    print("  • 'I'm in the mood for chill electronic music'")
    print("  • 'Play something like Shape of You by Ed Sheeran'")
    print()
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    print()
    
    # Chat loop
    retry_count = 0
    max_retries = 3
    
    while True:
        try:
            # Get user input
            user_input = input("🎧 You: ").strip()
            
            # Input validation
            if not user_input:
                print("Please enter a music request.")
                continue
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'stop']:
                print("\n👋 Thanks for using the Spotify Recommendation Bot!")
                print("Happy listening! 🎵")
                break
            
            # Skip empty inputs
            if not user_input:
                print("Please tell me what kind of music you're looking for!")
                continue
            
            # Show loading message
            print("\n🔍 Analyzing your request and finding recommendations...")
            
            # Get recommendations using hybrid approach
            recommendations = hybrid_recommend(user_input, limit=5)
            
            # Format and display results
            if recommendations:
                print("\n" + format_recommendations(recommendations))
            else:
                print("\n❌ Sorry, I couldn't find any recommendations.")
                print("Try rephrasing your request or check your API keys.")
            
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the Spotify Recommendation Bot!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again with a different request.")
            continue

if __name__ == "__main__":
    main()
