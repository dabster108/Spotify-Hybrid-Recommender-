#!/usr/bin/env python3
"""Setup script for Spotify Hybrid Recommendation System."""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_path = Path(".env")
    example_path = Path(".env.example")
    
    if env_path.exists():
        print("✅ .env file already exists")
        return True
    
    if not example_path.exists():
        print("❌ .env.example not found")
        return False
    
    # Copy example to .env
    env_content = example_path.read_text()
    env_path.write_text(env_content)
    print("✅ Created .env file from .env.example")
    print("📝 Please edit .env and add your actual API keys")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'requests', 'numpy', 'groq'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("💡 Install with: pip install " + " ".join(missing))
        return False
    
    print("✅ All required packages are installed")
    return True

def check_api_keys():
    """Check if API keys are properly configured."""
    # Import here to avoid circular imports
    sys.path.insert(0, str(Path(__file__).parent))
    
    try:
        from Recommend_System.config import config
        
        validation = config.validate_api_keys()
        
        print("\n🔑 API Key Status:")
        all_valid = True
        
        for service, valid in validation.items():
            status = "✅" if valid else "❌"
            print(f"  {service.title()}: {status}")
            if not valid and service != 'mistral':  # Mistral is optional
                all_valid = False
        
        if not all_valid:
            print("\n💡 API Key Setup Instructions:")
            if not validation.get('spotify'):
                print("  📱 Spotify API:")
                print("    1. Go to https://developer.spotify.com/dashboard/applications")
                print("    2. Create a new app")
                print("    3. Copy Client ID and Client Secret to .env")
            
            if not validation.get('groq'):
                print("  🤖 Groq API:")
                print("    1. Go to https://console.groq.com/keys")
                print("    2. Create a new API key")
                print("    3. Copy the key to .env (should start with 'gsk_')")
        
        return all_valid
        
    except Exception as e:
        print(f"❌ Error checking API keys: {e}")
        return False

def main():
    """Main setup function."""
    print("🎵 Spotify Hybrid Recommendation System Setup")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = True
    
    # Step 1: Create .env file
    print("\n1. Setting up environment file...")
    if not create_env_file():
        success = False
    
    # Step 2: Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        success = False
    
    # Step 3: Check API keys
    print("\n3. Checking API keys...")
    if not check_api_keys():
        success = False
    
    # Summary
    print("\n" + "=" * 50)
    if success:
        print("🎉 Setup completed successfully!")
        print("🚀 You can now run the recommendation system")
    else:
        print("⚠️  Setup completed with warnings")
        print("📝 Please address the issues above before running the system")
    
    print("\n📚 Next steps:")
    print("  1. Edit .env with your API keys")
    print("  2. Run: python -c 'from Recommend_System import config; config.print_status()'")
    print("  3. Test the system with your first recommendation!")

if __name__ == "__main__":
    main()