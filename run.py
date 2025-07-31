#!/usr/bin/env python3
"""
Stock Sentiment Analytics - Startup Script
Run either the dashboard or data collection stream.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit', 'pandas', 'plotly', 'transformers', 
        'torch', 'yfinance', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def check_env_file():
    """Check if .env file exists"""
    if not os.path.exists('.env'):
        print("⚠️  No .env file found. Creating from template...")
        if os.path.exists('env_template.txt'):
            with open('env_template.txt', 'r') as template:
                with open('.env', 'w') as env_file:
                    env_file.write(template.read())
            print("✅ Created .env file from template")
            print("📝 Please edit .env file with your API keys")
        else:
            print("❌ No env_template.txt found")
            return False
    else:
        print("✅ .env file found")
    
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    print("🚀 Starting Stock Sentiment Analytics Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        return False
    
    return True

def run_data_stream():
    """Run the data collection stream"""
    print("🔄 Starting data collection stream...")
    print("📈 Collecting sentiment data from Reddit and news sources")
    print("⏹️  Press Ctrl+C to stop the stream")
    
    try:
        subprocess.run([sys.executable, "stream_sentiment.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Data stream stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running data stream: {e}")
        return False
    
    return True

def run_both():
    """Run both dashboard and data stream in separate processes"""
    print("🚀 Starting both dashboard and data stream...")
    print("📊 Dashboard will be available at: http://localhost:8501")
    print("📈 Data stream will collect sentiment data in background")
    print("⏹️  Press Ctrl+C to stop both")
    
    try:
        # Start data stream in background
        stream_process = subprocess.Popen([sys.executable, "stream_sentiment.py"])
        
        # Start dashboard
        dashboard_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
        
        # Wait for either process to finish
        try:
            stream_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping both processes...")
            stream_process.terminate()
            dashboard_process.terminate()
            stream_process.wait()
            dashboard_process.wait()
            print("✅ Both processes stopped")
            
    except Exception as e:
        print(f"❌ Error running both processes: {e}")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Stock Sentiment Analytics")
    parser.add_argument(
        'mode',
        choices=['dashboard', 'stream', 'both'],
        help='Mode to run: dashboard (UI only), stream (data collection only), or both'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip dependency and environment checks'
    )
    
    args = parser.parse_args()
    
    print("📈 Stock Sentiment Analytics")
    print("=" * 40)
    
    # Run checks unless skipped
    if not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        
        if not check_env_file():
            print("⚠️  Continuing without proper environment setup...")
    
    # Run selected mode
    if args.mode == 'dashboard':
        run_dashboard()
    elif args.mode == 'stream':
        run_data_stream()
    elif args.mode == 'both':
        run_both()

if __name__ == "__main__":
    main() 