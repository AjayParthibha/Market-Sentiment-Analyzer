"""
Utility functions for Stock Sentiment Analytics
"""

import re
import pandas as pd
from datetime import datetime, timedelta
import sqlite3
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def extract_tickers(text: str) -> List[str]:
    """
    Extract stock tickers from text using regex patterns.
    
    Args:
        text: Input text to search for tickers
        
    Returns:
        List of found ticker symbols
    """
    # Common words that might be mistaken for tickers
    common_words = {
        'THE', 'AND', 'FOR', 'ARE', 'YOU', 'ALL', 'NEW', 'TOP', 'BEST', 'GOOD', 'BIG',
        'ONE', 'TWO', 'GET', 'SEE', 'NOW', 'DAY', 'WAY', 'MAY', 'CAN', 'WILL', 'HAS',
        'HAD', 'HER', 'HIS', 'ITS', 'OUR', 'THEY', 'THEM', 'THIS', 'THAT', 'WITH',
        'FROM', 'INTO', 'DURING', 'BEFORE', 'AFTER', 'ABOVE', 'BELOW', 'BETWEEN',
        'AMONG', 'AGAINST', 'TOWARD', 'TOWARDS', 'UPON', 'WITHIN', 'WITHOUT'
    }
    
    # Pattern for stock tickers (1-5 capital letters)
    ticker_pattern = r'\b[A-Z]{1,5}\b'
    tickers = re.findall(ticker_pattern, text)
    
    # Filter out common words and duplicates
    tickers = [ticker for ticker in tickers if ticker not in common_words]
    return list(set(tickers))

def clean_text(text: str) -> str:
    """
    Clean and normalize text for sentiment analysis.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def calculate_sentiment_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate sentiment metrics from a DataFrame.
    
    Args:
        df: DataFrame with sentiment data
        
    Returns:
        Dictionary with calculated metrics
    """
    if df.empty:
        return {
            'total_mentions': 0,
            'avg_sentiment': 0.0,
            'positive_pct': 0.0,
            'negative_pct': 0.0,
            'neutral_pct': 0.0,
            'sentiment_trend': 'neutral'
        }
    
    metrics = {
        'total_mentions': len(df),
        'avg_sentiment': df['sentiment_score'].mean(),
        'positive_pct': (df['sentiment_label'] == 'positive').mean() * 100,
        'negative_pct': (df['sentiment_label'] == 'negative').mean() * 100,
        'neutral_pct': (df['sentiment_label'] == 'neutral').mean() * 100
    }
    
    # Determine sentiment trend
    if metrics['positive_pct'] > metrics['negative_pct']:
        metrics['sentiment_trend'] = 'positive'
    elif metrics['negative_pct'] > metrics['positive_pct']:
        metrics['sentiment_trend'] = 'negative'
    else:
        metrics['sentiment_trend'] = 'neutral'
    
    return metrics

def get_time_based_data(df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
    """
    Filter data to last N hours.
    
    Args:
        df: DataFrame with timestamp column
        hours: Number of hours to look back
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    return df[df['timestamp'] >= cutoff_time]

def aggregate_sentiment_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sentiment data by ticker.
    
    Args:
        df: DataFrame with sentiment data
        
    Returns:
        Aggregated DataFrame
    """
    if df.empty:
        return pd.DataFrame()
    
    agg_data = df.groupby('ticker').agg({
        'sentiment_score': ['mean', 'count', 'std'],
        'sentiment_label': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Flatten column names
    agg_data.columns = ['ticker', 'avg_sentiment', 'mention_count', 'sentiment_std', 'sentiment_distribution']
    
    return agg_data

def get_database_connection(db_path: str = 'sentiment_data.db') -> sqlite3.Connection:
    """
    Get a database connection.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        SQLite connection
    """
    return sqlite3.connect(db_path)

def load_sentiment_data(db_path: str = 'sentiment_data.db', 
                       ticker: Optional[str] = None,
                       hours: Optional[int] = None) -> pd.DataFrame:
    """
    Load sentiment data from database.
    
    Args:
        db_path: Path to database
        ticker: Filter by specific ticker
        hours: Filter by time range (last N hours)
        
    Returns:
        DataFrame with sentiment data
    """
    try:
        conn = get_database_connection(db_path)
        
        query = "SELECT * FROM sentiment_data"
        conditions = []
        
        if ticker:
            conditions.append(f"ticker = '{ticker}'")
        
        if hours:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            conditions.append(f"timestamp >= '{cutoff_time.isoformat()}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp column
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading sentiment data: {e}")
        return pd.DataFrame()

def load_stock_data(db_path: str = 'sentiment_data.db',
                   ticker: Optional[str] = None,
                   hours: Optional[int] = None) -> pd.DataFrame:
    """
    Load stock price data from database.
    
    Args:
        db_path: Path to database
        ticker: Filter by specific ticker
        hours: Filter by time range (last N hours)
        
    Returns:
        DataFrame with stock data
    """
    try:
        conn = get_database_connection(db_path)
        
        query = "SELECT * FROM stock_data"
        conditions = []
        
        if ticker:
            conditions.append(f"ticker = '{ticker}'")
        
        if hours:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            conditions.append(f"timestamp >= '{cutoff_time.isoformat()}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp column
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading stock data: {e}")
        return pd.DataFrame()

def format_sentiment_score(score: float) -> str:
    """
    Format sentiment score for display.
    
    Args:
        score: Sentiment score (0-1)
        
    Returns:
        Formatted string
    """
    return f"{score:.3f}"

def get_sentiment_color(label: str) -> str:
    """
    Get color for sentiment label.
    
    Args:
        label: Sentiment label
        
    Returns:
        CSS color string
    """
    colors = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#6c757d'
    }
    return colors.get(label, '#6c757d')

def validate_ticker(ticker: str) -> bool:
    """
    Validate if a string is a valid stock ticker.
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Basic validation: 1-5 capital letters
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, ticker))

def get_trend_arrow(trend: str) -> str:
    """
    Get arrow emoji for trend direction.
    
    Args:
        trend: Trend direction ('positive', 'negative', 'neutral')
        
    Returns:
        Arrow emoji
    """
    arrows = {
        'positive': '📈',
        'negative': '📉',
        'neutral': '➡️'
    }
    return arrows.get(trend, '➡️') 