"""
Configuration settings for Stock Sentiment Analytics
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_PATH = os.getenv('DATABASE_PATH', 'data/sentiment_data.db')

# API Configuration
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'StockSentimentBot/1.0')

# Model Configuration
MODEL_NAME = os.getenv('MODEL_NAME', 'ProsusAI/finbert')
MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', '512'))
DEVICE = os.getenv('DEVICE', 'auto')  # 'auto', 'cpu', 'cuda'

# Data Collection Configuration
DEFAULT_TICKERS = [
    'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX',
    'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'NFLX', 'DIS'
]

REDDIT_SUBREDDITS = [
    'wallstreetbets', 'investing', 'stocks', 'StockMarket',
    'investments', 'SecurityAnalysis'
]

# Stream Configuration
STREAM_INTERVAL = int(os.getenv('STREAM_INTERVAL', '300'))  # 5 minutes
REDDIT_LIMIT = int(os.getenv('REDDIT_LIMIT', '100'))
NEWS_LIMIT = int(os.getenv('NEWS_LIMIT', '50'))

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'logs/sentiment_analysis.log')

# Dashboard Configuration
DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', '8501'))
DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', 'localhost')

# Sentiment Analysis Configuration
SENTIMENT_THRESHOLDS = {
    'positive': 0.6,
    'negative': 0.4,
    'neutral': 0.5
}

# Stock Data Configuration
STOCK_UPDATE_INTERVAL = int(os.getenv('STOCK_UPDATE_INTERVAL', '60'))  # 1 minute
YAHOO_FINANCE_TIMEOUT = int(os.getenv('YAHOO_FINANCE_TIMEOUT', '10'))

# Development Configuration
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
SAMPLE_DATA = os.getenv('SAMPLE_DATA', 'True').lower() == 'true'

# File Paths
DATA_DIR = os.getenv('DATA_DIR', 'data')
MODELS_DIR = os.getenv('MODELS_DIR', 'models')
LOGS_DIR = os.getenv('LOGS_DIR', 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True) 