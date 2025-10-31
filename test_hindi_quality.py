#!/usr/bin/env python3
"""Quick test for Hindi sad songs recommendations"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_hindi_recommendations():
    """Test Hindi sad songs specifically"""
    print("🎵 Testing Hindi Sad Songs Recommendations")
    print("=" * 50)
    
    try:
        # Import and test
        from Recommend_System.recommend import HybridRecommendationSystem
        
        # Initialize system
        system = HybridRecommendationSystem()
        
        # Test query
        query = "some hindi sad songs"
        print(f"Testing query: '{query}'")
        
        result = system.recommend_music(query)
        
        if result and "System: ❌ No tracks found" not in result:
            print("✅ Recommendations generated!")
            
            # Check if we're getting better quality results
            if "Hot Bollywood" in result or "Hey Ghansham" in result:
                print("⚠️  Still getting low-quality results")
            else:
                print("✅ Quality seems improved!")
                
            # Show first few lines
            lines = result.split('\n')[:15]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ No recommendations generated")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_hindi_recommendations()