"""
Database connection utilities
Supports both SQLite (local) and PostgreSQL (production)
"""
import os
import sqlite3
from urllib.parse import urlparse


def get_database_url():
    """Get database URL from environment or default to SQLite"""
    return os.getenv('DATABASE_URL', 'sqlite:///data/sentiment_data.db')


def get_connection():
    """Get database connection based on DATABASE_URL"""
    db_url = get_database_url()

    if db_url.startswith('sqlite'):
        db_path = db_url.replace('sqlite:///', '')
        return sqlite3.connect(db_path)

    elif db_url.startswith('postgres'):
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            # Railway provides postgres:// but psycopg2 needs postgresql://
            if db_url.startswith('postgres://'):
                db_url = db_url.replace('postgres://', 'postgresql://', 1)

            return psycopg2.connect(db_url, cursor_factory=RealDictCursor)
        except ImportError:
            raise ImportError("psycopg2 is required for PostgreSQL. Install with: pip install psycopg2-binary")

    else:
        raise ValueError(f"Unsupported database URL: {db_url}")


def is_postgres():
    """Check if using PostgreSQL"""
    return get_database_url().startswith('postgres')


def execute_query(query, params=None):
    """Execute a query and return results"""
    conn = get_connection()
    cursor = conn.cursor()

    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)

    results = cursor.fetchall() if cursor.description else None
    conn.commit()
    conn.close()

    return results
