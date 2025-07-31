#!/usr/bin/env python3
"""
Test script to verify Stock Sentiment Analytics setup
"""

import sys
import os
import importlib
import sqlite3
from datetime import datetime

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    required_modules = [
        'streamlit',
        'pandas',
        'plotly',
        'transformers',
        'torch',
        'yfinance',
        'dotenv',
        'sqlite3'
    ]
    
    failed_imports = []
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All imports successful")
        return True

def test_database():
    """Test database creation and operations"""
    print("\n🔍 Testing database...")
    
    try:
        # Test database connection
        conn = sqlite3.connect('test_sentiment_data.db')
        cursor = conn.cursor()
        
        # Create test table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                sentiment_label TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL
            )
        ''')
        
        # Insert test data
        test_data = {
            'ticker': 'TEST',
            'sentiment_score': 0.75,
            'sentiment_label': 'positive',
            'source': 'test',
            'text': 'This is a test sentiment entry'
        }
        
        cursor.execute('''
            INSERT INTO test_sentiment_data 
            (ticker, sentiment_score, sentiment_label, source, text)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            test_data['ticker'],
            test_data['sentiment_score'],
            test_data['sentiment_label'],
            test_data['source'],
            test_data['text']
        ))
        
        # Query test data
        cursor.execute('SELECT * FROM test_sentiment_data WHERE ticker = ?', ('TEST',))
        result = cursor.fetchone()
        
        conn.commit()
        conn.close()
        
        # Clean up test database
        os.remove('test_sentiment_data.db')
        
        if result:
            print("✅ Database operations successful")
            return True
        else:
            print("❌ Database query failed")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\n🔍 Testing sentiment analysis...")
    
    try:
        # Import our modules
        from utils import extract_tickers, clean_text
        from stream_sentiment import SentimentAnalyzer
        
        # Test text processing
        test_text = "AAPL and TSLA are performing well today! Great earnings from both companies."
        cleaned_text = clean_text(test_text)
        tickers = extract_tickers(test_text)
        
        print(f"  ✅ Text cleaning: {len(cleaned_text)} characters")
        print(f"  ✅ Ticker extraction: {tickers}")
        
        # Test sentiment analyzer (this might take a moment to load the model)
        print("  ⏳ Loading sentiment model...")
        analyzer = SentimentAnalyzer()
        
        # Test sentiment analysis
        sentiment_result = analyzer.analyze_sentiment("This is a positive financial statement.")
        print(f"  ✅ Sentiment analysis: {sentiment_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False

def test_dashboard_components():
    """Test dashboard components"""
    print("\n🔍 Testing dashboard components...")
    
    try:
        # Import dashboard functions
        from dashboard import create_sample_data, plot_sentiment_trends
        
        # Test sample data creation
        sample_df = create_sample_data()
        print(f"  ✅ Sample data created: {len(sample_df)} rows")
        
        # Test plotting function
        if not sample_df.empty:
            fig = plot_sentiment_trends(sample_df, 'AAPL')
            print("  ✅ Plot generation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import DEFAULT_TICKERS, REDDIT_SUBREDDITS, STREAM_INTERVAL
        
        print(f"  ✅ Default tickers: {len(DEFAULT_TICKERS)} tickers")
        print(f"  ✅ Reddit subreddits: {len(REDDIT_SUBREDDITS)} subreddits")
        print(f"  ✅ Stream interval: {STREAM_INTERVAL} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Stock Sentiment Analytics - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Database", test_database),
        ("Configuration", test_configuration),
        ("Dashboard Components", test_dashboard_components),
        ("Sentiment Analysis", test_sentiment_analysis)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your setup is ready to go.")
        print("\n🚀 You can now run:")
        print("  python run.py dashboard    # Start dashboard")
        print("  python run.py stream       # Start data collection")
        print("  python run.py both         # Start both")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("💡 Make sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 