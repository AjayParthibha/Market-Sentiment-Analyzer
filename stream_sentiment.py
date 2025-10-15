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
import praw
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
        self.use_real_data = os.getenv('USE_REAL_DATA', 'false').lower() == 'true'
        self._setup_reddit()

    def _setup_reddit(self):
        """Setup Reddit client with PRAW"""
        try:
            if not self.use_real_data:
                logger.info("Using sample Reddit data (set USE_REAL_DATA=true for real data)")
                return

            client_id = os.getenv('REDDIT_CLIENT_ID')
            client_secret = os.getenv('REDDIT_CLIENT_SECRET')
            user_agent = os.getenv('REDDIT_USER_AGENT', 'StockSentimentBot/1.0')

            if not client_id or not client_secret:
                logger.warning("Reddit credentials not found. Using sample data. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env")
                self.use_real_data = False
                return

            self.reddit_client = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            logger.info("Reddit client setup successful")
        except ImportError:
            logger.error("PRAW not installed. Using sample data. Install with: pip install praw")
            self.use_real_data = False
        except Exception as e:
            logger.error(f"Error setting up Reddit client: {e}. Using sample data.")
            self.use_real_data = False
    
    async def collect_reddit_data(self, subreddits=None, limit=100):
        """Collect posts from Reddit subreddits"""
        if subreddits is None:
            subreddits = ['wallstreetbets', 'investing', 'stocks']

        # Return sample data if not using real API
        if not self.use_real_data or self.reddit_client is None:
            return self._get_sample_reddit_posts()

        # Collect real Reddit data
        posts = []
        try:
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    # Get hot posts from each subreddit
                    for submission in subreddit.hot(limit=limit // len(subreddits)):
                        # Skip stickied posts and posts without content
                        if submission.stickied or (not submission.selftext and not submission.title):
                            continue

                        posts.append({
                            'title': submission.title,
                            'text': submission.selftext if submission.selftext else submission.title,
                            'score': submission.score,
                            'created_utc': submission.created_utc,
                            'subreddit': subreddit_name,
                            'url': submission.url,
                            'num_comments': submission.num_comments
                        })

                    logger.info(f"Collected {len([p for p in posts if p['subreddit'] == subreddit_name])} posts from r/{subreddit_name}")

                except Exception as e:
                    logger.error(f"Error collecting from r/{subreddit_name}: {e}")
                    continue

            if not posts:
                logger.warning("No Reddit posts collected, using sample data")
                return self._get_sample_reddit_posts()

            return posts

        except Exception as e:
            logger.error(f"Error in Reddit data collection: {e}")
            return self._get_sample_reddit_posts()

    def _get_sample_reddit_posts(self):
        """Return sample Reddit posts for testing"""
        return [
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
    
    async def collect_news_data(self, tickers=None, limit=50):
        """Collect news headlines from NewsAPI"""
        if tickers is None:
            tickers = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL']

        # Return sample data if not using real API
        if not self.use_real_data or not self.news_api_key:
            if not self.news_api_key:
                logger.info("NewsAPI key not found. Using sample data. Set NEWS_API_KEY in .env")
            return self._get_sample_news()

        # Collect real news data from NewsAPI
        news_items = []
        try:
            import requests

            base_url = "https://newsapi.org/v2/everything"

            for ticker in tickers[:5]:  # Limit to 5 tickers to avoid rate limits
                try:
                    # Search for news about each ticker
                    params = {
                        'q': f'{ticker} stock OR {ticker} shares',
                        'apiKey': self.news_api_key,
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': min(limit // len(tickers), 20),  # Limit per ticker
                        'from': (datetime.now() - timedelta(days=7)).isoformat()
                    }

                    response = requests.get(base_url, params=params, timeout=10)

                    if response.status_code == 200:
                        data = response.json()
                        articles = data.get('articles', [])

                        for article in articles:
                            if article.get('title') and article.get('description'):
                                news_items.append({
                                    'title': article['title'],
                                    'description': article['description'],
                                    'publishedAt': article.get('publishedAt', datetime.now().isoformat()),
                                    'source': article.get('source', {}).get('name', 'Unknown'),
                                    'url': article.get('url', '')
                                })

                        logger.info(f"Collected {len(articles)} news articles for {ticker}")

                    elif response.status_code == 426:
                        logger.warning("NewsAPI requires HTTPS upgrade. Using sample data.")
                        return self._get_sample_news()
                    elif response.status_code == 429:
                        logger.warning("NewsAPI rate limit reached. Using cached/sample data.")
                        break
                    else:
                        logger.error(f"NewsAPI error {response.status_code}: {response.text}")

                    # Rate limiting - NewsAPI free tier: 100 requests/day
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Error collecting news for {ticker}: {e}")
                    continue

            if not news_items:
                logger.warning("No news articles collected, using sample data")
                return self._get_sample_news()

            return news_items

        except Exception as e:
            logger.error(f"Error in news data collection: {e}")
            return self._get_sample_news()

    def _get_sample_news(self):
        """Return sample news for testing"""
        return [
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
        self.last_request_time = 0
        self.min_request_interval = 10  # Minimum seconds between requests (increased)
        self.failed_tickers = set()  # Track tickers that consistently fail
        self.max_retries = 2

    def get_stock_data(self, ticker):
        """Get current stock data for a ticker"""
        # Skip tickers that have failed multiple times
        if ticker in self.failed_tickers:
            return None

        for attempt in range(self.max_retries):
            try:
                # Rate limiting - ensure minimum interval between requests
                current_time = time.time()
                time_since_last = current_time - self.last_request_time
                if time_since_last < self.min_request_interval:
                    time.sleep(self.min_request_interval - time_since_last)

                # Create ticker with session for better reliability
                stock = yf.Ticker(ticker)

                # Try different periods in order of preference
                periods = ["1d", "5d", "1mo"]
                hist = None

                for period in periods:
                    try:
                        hist = stock.history(period=period)
                        if not hist.empty:
                            break
                    except:
                        continue

                if hist is not None and not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    previous_price = hist['Close'].iloc[-2]
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100

                    self.last_request_time = time.time()

                    logger.info(f"Successfully collected data for {ticker}: ${current_price:.2f}")

                    return {
                        'price': current_price,
                        'volume': int(hist['Volume'].iloc[-1]),
                        'change': change,
                        'change_percent': change_percent
                    }
                elif attempt < self.max_retries - 1:
                    # Wait before retry
                    logger.debug(f"Retrying {ticker} (attempt {attempt + 2}/{self.max_retries})")
                    time.sleep(3)
                else:
                    logger.warning(f"No historical data available for {ticker} after {self.max_retries} attempts")
                    self.failed_tickers.add(ticker)

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.debug(f"Error getting stock data for {ticker}, retrying: {e}")
                    time.sleep(3)
                else:
                    logger.warning(f"Failed to get stock data for {ticker} after {self.max_retries} attempts: {e}")
                    self.failed_tickers.add(ticker)

        return None

class SentimentStreamProcessor:
    """Main processor for streaming sentiment analysis"""

    def __init__(self, collect_stock_prices=True):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.data_collector = DataCollector()
        self.db_manager = DatabaseManager()
        self.stock_collector = StockDataCollector() if collect_stock_prices else None
        self.running = False
        self.collect_stock_prices = collect_stock_prices
    
    def extract_tickers(self, text):
        """Extract stock tickers from text"""
        import re

        # Common stock tickers to look for (most actively traded)
        known_tickers = {
            'AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NFLX', 'AMD',
            'INTC', 'CSCO', 'ADBE', 'AVGO', 'TXN', 'QCOM', 'COST', 'CMCSA', 'PEP', 'TMUS',
            'ABNB', 'PYPL', 'CHTR', 'INTU', 'ISRG', 'BKNG', 'REGN', 'GILD', 'ADP', 'VRTX',
            'SBUX', 'MDLZ', 'LRCX', 'MU', 'MELI', 'ASML', 'AMAT', 'CSX', 'ADI', 'KLAC',
            'SPY', 'QQQ', 'DIA', 'IWM', 'VOO', 'VTI', 'ARKK', 'GME', 'AMC', 'BB', 'NOK',
            'PLTR', 'NIO', 'LCID', 'RIVN', 'F', 'GM', 'T', 'VZ', 'DIS', 'NFLX', 'WMT', 'TGT',
            'BA', 'CAT', 'DE', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK', 'V', 'MA',
            'COIN', 'SQ', 'HOOD', 'SOFI', 'UPST', 'AFRM'
        }

        # Extended common words to filter out
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'YOU', 'ALL', 'NEW', 'TOP', 'BEST', 'GOOD', 'BIG',
            'BUT', 'NOT', 'CAN', 'WAS', 'WILL', 'HAS', 'HAVE', 'HAD', 'MORE', 'BEEN', 'BEEN',
            'THAN', 'SOME', 'WHAT', 'WHEN', 'WHERE', 'WHO', 'WHICH', 'THERE', 'THEIR', 'THEY',
            'THIS', 'THAT', 'THESE', 'THOSE', 'FROM', 'INTO', 'OUT', 'UP', 'DOWN', 'OVER',
            'UNDER', 'AGAIN', 'FURTHER', 'THEN', 'ONCE', 'HERE', 'VERY', 'MUCH', 'MANY',
            'ANY', 'BOTH', 'EACH', 'FEW', 'OTHER', 'SUCH', 'ONLY', 'OWN', 'SAME', 'SO',
            'JUST', 'NOW', 'ALSO', 'WELL', 'EVEN', 'BACK', 'WAY', 'COULD', 'WOULD', 'SHOULD',
            'MIGHT', 'MUST', 'MAY', 'NEED', 'MAKE', 'TAKE', 'SEE', 'GET', 'COME', 'GO',
            'WANT', 'KNOW', 'THINK', 'LOOK', 'USE', 'FIND', 'GIVE', 'TELL', 'WORK', 'CALL'
        }

        # Find potential tickers (1-5 capital letters)
        ticker_pattern = r'\b[A-Z]{1,5}\b'
        potential_tickers = re.findall(ticker_pattern, text)

        # Filter to only known tickers or very likely patterns
        tickers = []
        for ticker in potential_tickers:
            # Include if it's in our known list
            if ticker in known_tickers:
                tickers.append(ticker)
            # Or if it's mentioned with $ sign (e.g., $AAPL)
            elif f'${ticker}' in text:
                tickers.append(ticker)
            # Skip if it's a common word
            elif ticker in common_words:
                continue

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
        if not self.collect_stock_prices or self.stock_collector is None:
            logger.info("Stock price collection is disabled, skipping...")
            return

        logger.info("Starting stock price collection...")
        success_count = 0
        for ticker in self.stock_collector.tickers:
            stock_data = self.stock_collector.get_stock_data(ticker)
            if stock_data:
                self.db_manager.save_stock_data(ticker, stock_data)
                success_count += 1
            # Rate limiting is handled in StockDataCollector.get_stock_data()

        if success_count > 0:
            logger.info(f"Successfully collected stock data for {success_count}/{len(self.stock_collector.tickers)} tickers")
        else:
            logger.warning("Failed to collect stock data for all tickers. Yahoo Finance may be rate limiting.")
    
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
    # Check if stock price collection should be disabled
    collect_stock_prices = os.getenv('COLLECT_STOCK_PRICES', 'true').lower() == 'true'

    if not collect_stock_prices:
        logger.info("Stock price collection is DISABLED via environment variable")

    processor = SentimentStreamProcessor(collect_stock_prices=collect_stock_prices)

    # Run initial data collection
    logger.info("Running initial data collection...")
    await processor.process_reddit_data()
    await processor.process_news_data()

    # Only collect stock data if enabled
    if collect_stock_prices:
        await processor.collect_stock_data()

    # Start the stream
    await processor.run_stream(interval=int(os.getenv('STREAM_INTERVAL', '300')))

if __name__ == "__main__":
    asyncio.run(main()) 