#!/usr/bin/env python3
"""
Stock Sentiment Analytics - Data Collection and Sentiment Analysis
Real-time sentiment analysis of stock mentions from Reddit and news sources.
"""

import asyncio
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
import os
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import time
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial text"""
    
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the FinBERT model and tokenizer"""
        try:
            logger.info("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            # Fallback to a simpler model
            self.pipeline = pipeline("sentiment-analysis")
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of financial text"""
        try:
            if not text or len(text.strip()) < 10:
                return {'label': 'neutral', 'score': 0.5}
            
            # Truncate text if too long
            if len(text) > 512:
                text = text[:512]
            
            result = self.pipeline(text)[0]
            
            # Map FinBERT labels to our format
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral'
            }
            
            return {
                'label': label_mapping.get(result['label'], 'neutral'),
                'score': result['score']
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'label': 'neutral', 'score': 0.5}

class DataCollector:
    """Collect data from various sources (Reddit, News APIs)"""
    
    def __init__(self):
        self.reddit_client = None
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self._setup_reddit()
    
    def _setup_reddit(self):
        """Setup Reddit client (placeholder for now)"""
        try:
            # This would be implemented with PRAW
            # For now, we'll use sample data
            logger.info("Reddit client setup (placeholder)")
        except Exception as e:
            logger.error(f"Error setting up Reddit client: {e}")
    
    async def collect_reddit_data(self, subreddits=None, limit=100):
        """Collect posts from Reddit subreddits"""
        if subreddits is None:
            subreddits = ['wallstreetbets', 'investing', 'stocks']
        
        # Placeholder for Reddit data collection
        # In a real implementation, this would use PRAW to collect posts
        sample_posts = [
            {
                'title': 'AAPL earnings beat expectations',
                'text': 'Apple just reported amazing earnings! Stock is going to the moon!',
                'score': 150,
                'created_utc': time.time(),
                'subreddit': 'wallstreetbets'
            },
            {
                'title': 'TSLA stock analysis',
                'text': 'Tesla fundamentals look strong despite market volatility',
                'score': 89,
                'created_utc': time.time() - 3600,
                'subreddit': 'investing'
            },
            {
                'title': 'NVDA earnings disappointment',
                'text': 'NVIDIA missed earnings estimates. Stock is tanking.',
                'score': 45,
                'created_utc': time.time() - 7200,
                'subreddit': 'stocks'
            }
        ]
        
        return sample_posts
    
    async def collect_news_data(self, tickers=None, limit=50):
        """Collect news headlines from NewsAPI"""
        if tickers is None:
            tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']
        
        # Placeholder for news data collection
        # In a real implementation, this would use NewsAPI
        sample_news = [
            {
                'title': 'Apple Reports Record Quarterly Revenue',
                'description': 'Apple Inc. reported record quarterly revenue driven by strong iPhone sales',
                'publishedAt': datetime.now().isoformat(),
                'source': 'Reuters'
            },
            {
                'title': 'Tesla Stock Rises on New Model Announcement',
                'description': 'Tesla shares gained 5% after announcing new electric vehicle model',
                'publishedAt': datetime.now().isoformat(),
                'source': 'Bloomberg'
            }
        ]
        
        return sample_news

class DatabaseManager:
    """Manage SQLite database operations"""
    
    def __init__(self, db_path='data/sentiment_data.db'):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sentiment_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                sentiment_score REAL NOT NULL,
                sentiment_label TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                title TEXT,
                score INTEGER,
                subreddit TEXT,
                created_at DATETIME
            )
        ''')
        
        # Create stock_data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ticker TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER,
                change REAL,
                change_percent REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def save_sentiment_data(self, data_list):
        """Save sentiment analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for data in data_list:
            cursor.execute('''
                INSERT INTO sentiment_data 
                (ticker, sentiment_score, sentiment_label, source, text, title, score, subreddit, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['ticker'],
                data['sentiment_score'],
                data['sentiment_label'],
                data['source'],
                data['text'],
                data.get('title', ''),
                data.get('score', 0),
                data.get('subreddit', ''),
                data.get('created_at', datetime.now())
            ))
        
        conn.commit()
        conn.close()
    
    def save_stock_data(self, ticker, price_data):
        """Save stock price data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO stock_data (ticker, price, volume, change, change_percent)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            ticker,
            price_data['price'],
            price_data.get('volume', 0),
            price_data.get('change', 0),
            price_data.get('change_percent', 0)
        ))
        
        conn.commit()
        conn.close()

class StockDataCollector:
    """Collect real-time stock price data"""
    
    def __init__(self):
        self.tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX']
    
    def get_stock_data(self, ticker):
        """Get current stock data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = stock.history(period="1d")
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                previous_price = hist['Open'].iloc[0]
                change = current_price - previous_price
                change_percent = (change / previous_price) * 100
                
                return {
                    'price': current_price,
                    'volume': hist['Volume'].iloc[-1],
                    'change': change,
                    'change_percent': change_percent
                }
        except Exception as e:
            logger.error(f"Error getting stock data for {ticker}: {e}")
        
        return None

class SentimentStreamProcessor:
    """Main processor for streaming sentiment analysis"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data_collector = DataCollector()
        self.db_manager = DatabaseManager()
        self.stock_collector = StockDataCollector()
        self.running = False
    
    def extract_tickers(self, text):
        """Extract stock tickers from text"""
        import re
        # Simple regex to find stock tickers (1-5 capital letters)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        tickers = re.findall(ticker_pattern, text)
        
        # Filter out common words that aren't tickers
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'YOU', 'ALL', 'NEW', 'TOP', 'BEST', 'GOOD', 'BIG'}
        tickers = [ticker for ticker in tickers if ticker not in common_words]
        
        return list(set(tickers))
    
    async def process_reddit_data(self):
        """Process Reddit data and perform sentiment analysis"""
        posts = await self.data_collector.collect_reddit_data()
        sentiment_data = []
        
        for post in posts:
            # Extract tickers from post
            text = f"{post['title']} {post['text']}"
            tickers = self.extract_tickers(text)
            
            if tickers:
                # Analyze sentiment
                sentiment = self.sentiment_analyzer.analyze_sentiment(text)
                
                for ticker in tickers:
                    sentiment_data.append({
                        'ticker': ticker,
                        'sentiment_score': sentiment['score'],
                        'sentiment_label': sentiment['label'],
                        'source': 'reddit',
                        'text': text[:500],  # Truncate for storage
                        'title': post['title'],
                        'score': post['score'],
                        'subreddit': post['subreddit'],
                        'created_at': datetime.fromtimestamp(post['created_utc'])
                    })
        
        if sentiment_data:
            self.db_manager.save_sentiment_data(sentiment_data)
            logger.info(f"Processed {len(sentiment_data)} Reddit sentiment records")
    
    async def process_news_data(self):
        """Process news data and perform sentiment analysis"""
        news_items = await self.data_collector.collect_news_data()
        sentiment_data = []
        
        for news in news_items:
            # Extract tickers from news
            text = f"{news['title']} {news['description']}"
            tickers = self.extract_tickers(text)
            
            if tickers:
                # Analyze sentiment
                sentiment = self.sentiment_analyzer.analyze_sentiment(text)
                
                for ticker in tickers:
                    sentiment_data.append({
                        'ticker': ticker,
                        'sentiment_score': sentiment['score'],
                        'sentiment_label': sentiment['label'],
                        'source': 'news',
                        'text': text[:500],  # Truncate for storage
                        'title': news['title'],
                        'created_at': datetime.fromisoformat(news['publishedAt'])
                    })
        
        if sentiment_data:
            self.db_manager.save_sentiment_data(sentiment_data)
            logger.info(f"Processed {len(sentiment_data)} news sentiment records")
    
    async def collect_stock_data(self):
        """Collect current stock price data"""
        for ticker in self.stock_collector.tickers:
            stock_data = self.stock_collector.get_stock_data(ticker)
            if stock_data:
                self.db_manager.save_stock_data(ticker, stock_data)
                logger.info(f"Collected stock data for {ticker}: ${stock_data['price']:.2f}")
    
    async def run_stream(self, interval=300):  # 5 minutes default
        """Run the sentiment analysis stream"""
        self.running = True
        logger.info("Starting sentiment analysis stream...")
        
        while self.running:
            try:
                # Process Reddit data
                await self.process_reddit_data()
                
                # Process news data
                await self.process_news_data()
                
                # Collect stock data
                await self.collect_stock_data()
                
                logger.info(f"Stream cycle completed. Waiting {interval} seconds...")
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Stream interrupted by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in stream cycle: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

async def main():
    """Main function to run the sentiment analysis stream"""
    processor = SentimentStreamProcessor()
    
    # Run initial data collection
    logger.info("Running initial data collection...")
    await processor.process_reddit_data()
    await processor.process_news_data()
    await processor.collect_stock_data()
    
    # Start the stream
    await processor.run_stream()

if __name__ == "__main__":
    asyncio.run(main()) 