"""
Advanced RAG System - Streamlit Application Entry Point
Run with: streamlit run app.py
"""

import sys
import os
import subprocess
import time
import threading
import requests

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_api_server():
    """Check if FastAPI server is running."""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the FastAPI server in background."""
    if not check_api_server():
        print("🚀 Starting FastAPI server...")
        
        # Get the absolute path to the current directory
        current_dir = os.path.abspath(os.path.dirname(__file__))
        
        # Get the python executable path
        python_exe = sys.executable
        print(f"Using Python: {python_exe}")
        print(f"Working directory: {current_dir}")
        
        # Check if uvicorn is available
        try:
            result = subprocess.run([python_exe, "-m", "uvicorn", "--help"], 
                                  capture_output=True, timeout=5)
            if result.returncode != 0:
                print("❌ uvicorn not found. Installing...")
                subprocess.run([python_exe, "-m", "pip", "install", "uvicorn"], check=True)
        except Exception as e:
            print(f"Error checking uvicorn: {e}")
            return
        
        # Start the FastAPI server
        try:
            process = subprocess.Popen([
                python_exe, "-m", "uvicorn", 
                "main:app", 
                "--host", "0.0.0.0", 
                "--port", "8000", 
                "--reload"
            ], cwd=current_dir, 
               stdout=subprocess.DEVNULL, 
               stderr=subprocess.DEVNULL,
               env=os.environ.copy())
            
            print(f"FastAPI server started with PID: {process.pid}")
            
        except Exception as e:
            print(f"❌ Failed to start FastAPI server: {e}")
            print("Please start manually with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload")
            return
        
        # Wait for server to start
        print("⏳ Waiting for API server to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_api_server():
                print("✅ API server is ready!")
                break
            time.sleep(1)
            if i % 5 == 0:
                print(f"   Still waiting... ({i+1}/30 seconds)")
        else:
            print("❌ API server failed to start within 30 seconds")
    else:
        print("✅ API server is already running!")

if __name__ == "__main__":
    # Start API server if not running
    start_api_server()
    
    # Import and run the Streamlit UI
    from frontend.streamlit_ui import main
    main()