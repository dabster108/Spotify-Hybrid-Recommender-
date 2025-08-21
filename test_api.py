#!/usr/bin/env python3
"""
Simple test script to check Spotify API response for artist searches
"""

import os
import sys
from dotenv import load_dotenv
from spotify_api_fixed import get_spotify_client

def test_artist_search(artist_name):
    """Test what the Spotify API returns for an artist search"""
    load_dotenv()
    
    sp = get_spotify_client()
    if not sp:
        print("❌ Failed to initialize Spotify client")
        return
    
    print(f"🔍 Testing API search for: {artist_name}")
    print("=" * 50)
    
    # Test different search queries
    search_queries = [
        f"artist:{artist_name}",
        f"artist:\"{artist_name}\"",
        f"{artist_name}",
        f"{artist_name} songs"
    ]
    
    for query in search_queries:
        print(f"\n📍 Search query: {query}")
        try:
            results = sp.search(q=query, type='track', limit=5)
            tracks = results['tracks']['items']
            
            if tracks:
                print(f"✅ Found {len(tracks)} tracks:")
                for i, track in enumerate(tracks, 1):
                    artist_names = [artist['name'] for artist in track['artists']]
                    print(f"   {i}. '{track['name']}' by {', '.join(artist_names)}")
                    print(f"      Main artist: {track['artists'][0]['name']}")
                    print(f"      Popularity: {track.get('popularity', 'N/A')}")
            else:
                print("❌ No tracks found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        artist_name = " ".join(sys.argv[1:])
    else:
        artist_name = "playboi carti"
    
    test_artist_search(artist_name)
