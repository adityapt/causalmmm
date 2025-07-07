#!/usr/bin/env python3
"""
Start script for Causal MMM Streamlit Application
"""
import subprocess
import sys
import os

def main():
    print("Starting Causal MMM Streamlit Application...")
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
    os.environ["STREAMLIT_SERVER_PORT"] = "5000"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Run Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port=5000",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStreamlit application stopped.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "__main__":
    main()