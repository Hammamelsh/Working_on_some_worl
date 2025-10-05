#!/usr/bin/env python3
"""
Simple runner script for the unified app
"""

import subprocess
import sys
import os

def main():
    """Run the unified Streamlit app"""
    
    # Check if we're in the right directory
    if not os.path.exists('unified_app.py'):
        print("❌ Error: unified_app.py not found in current directory")
        print("💡 Please run this script from the stdf-data-factory directory")
        return 1
    
    # Check environment variables
    missing_vars = []
    if not os.getenv("GROQ_API_KEY"):
        missing_vars.append("GROQ_API_KEY")
    if not os.getenv("XAI_API_KEY"):
        missing_vars.append("XAI_API_KEY")
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   • {var}")
        print("\n💡 Please set these environment variables:")
        print(f"   export GROQ_API_KEY='your_groq_api_key'")
        print(f"   export XAI_API_KEY='your_xai_api_key'")
        return 1
    
    print("🚀 Starting Data Hunting Agent...")
    print("✅ Environment variables configured")
    print("🌐 App will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run streamlit
        result = subprocess.run([
            "streamlit", "run", "unified_app.py",
            "--server.headless", "false"
        ], check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
        return 0
    except FileNotFoundError:
        print("❌ Error: streamlit command not found")
        print("💡 Please install streamlit: pip install streamlit")
        return 1

if __name__ == "__main__":
    sys.exit(main())
