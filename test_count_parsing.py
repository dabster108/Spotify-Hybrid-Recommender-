#!/usr/bin/env python3

# Quick test for count-based parsing functionality
import sys
import os

# Add the helo directory to the path
helo_path = os.path.join(os.path.dirname(__file__), 'helo')
sys.path.insert(0, helo_path)

# Import from the helo/main.py file
import main
HybridMusicSystem = main.HybridMusicSystem

def test_count_parsing():
    """Test the count-based artist request parsing."""
    system = HybridMusicSystem()
    
    test_queries = [
        "3 songs by Bipul Chettri",
        "5 tracks by Taylor Swift", 
        "play 2 Ed Sheeran songs",
        "just 4 BTS tracks",
        "recommend 6 songs by Ariana Grande",
        "songs by Bipul Chettri",  # No count
        "relaxing nepali music"     # No artist
    ]
    
    print("🎯 Testing Count-Based Artist Request Parsing")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n🔍 Testing: '{query}'")
        try:
            clean_query, existing_songs, specific_artist, requested_count = system.parse_input_songs(query)
            print(f"   ✅ Clean query: '{clean_query}'")
            print(f"   ✅ Existing songs: {existing_songs}")
            print(f"   ✅ Specific artist: {specific_artist}")
            print(f"   ✅ Requested count: {requested_count}")
            
            # Test if the parsing worked correctly
            if "by" in query.lower() and any(word in query.lower() for word in ["songs", "tracks", "music"]):
                if specific_artist is None:
                    print(f"   ❌ Expected artist detection but got None")
                else:
                    print(f"   ✅ Successfully detected artist request")
                    
            if any(char.isdigit() for char in query):
                if requested_count is None:
                    print(f"   ⚠️  Contains number but no count detected")
                else:
                    print(f"   ✅ Successfully detected count request")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_count_parsing()
