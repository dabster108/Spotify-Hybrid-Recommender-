#!/usr/bin/env python3
"""
Debug script to test the hybrid_recommend function directly
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Test the hybrid_recommend function directly
try:
    from model_fixed import hybrid_recommend
    print("🔍 Testing hybrid_recommend function...")
    
    test_query = "songs by playboi carti"
    print(f"Query: {test_query}")
    
    result = hybrid_recommend(test_query)
    print(f"Result: {result}")
    
    if result:
        print("\n✅ Recommendations found:")
        for i, track in enumerate(result, 1):
            print(f"{i}. {track}")
    else:
        print("❌ No recommendations returned")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
