import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import os
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Stock Sentiment Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #4da6ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        color: #e0e0e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4da6ff;
        margin-bottom: 1rem;
    }
    .metric-card strong {
        color: #ffffff;
    }
    .metric-card small {
        color: #b0b0b0;
    }
    .metric-card em {
        color: #c0c0c0;
    }
    .sentiment-positive {
        color: #4ade80;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #f87171;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #a3a3a3;
        font-weight: bold;
    }
    /* Ensure dark theme for plotly charts */
    .js-plotly-plot {
        background-color: #0e1117 !important;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """Create sample data for demonstration"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='h')
    
    # Sample sentiment data
    sentiment_data = []
    for date in dates:
        sentiment_data.append({
            'timestamp': date,
            'ticker': 'AAPL',
            'sentiment_score': 0.6 + 0.3 * (date.hour % 24) / 24,  # Varying sentiment
            'sentiment_label': 'positive' if (date.hour % 24) > 12 else 'negative',
            'source': 'reddit',
            'text': f'Sample post about AAPL at {date.strftime("%H:%M")}'
        })
        sentiment_data.append({
            'timestamp': date,
            'ticker': 'TSLA',
            'sentiment_score': 0.4 + 0.4 * (date.hour % 24) / 24,
            'sentiment_label': 'neutral' if (date.hour % 24) > 8 else 'positive',
            'source': 'news',
            'text': f'Sample news about TSLA at {date.strftime("%H:%M")}'
        })
    
    return pd.DataFrame(sentiment_data)

def load_data():
    """Load data from database or create sample data"""
    try:
        # Try to connect to database
        conn = sqlite3.connect('data/sentiment_data.db')
        sentiment_df = pd.read_sql_query(
            "SELECT * FROM sentiment_data",
            conn,
            parse_dates=['timestamp']
        )
        conn.close()

        # Ensure timestamp is datetime type
        if not pd.api.types.is_datetime64_any_dtype(sentiment_df['timestamp']):
            sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])

        return sentiment_df
    except Exception as e:
        # Return sample data if database doesn't exist or has errors
        return create_sample_data()

def plot_sentiment_trends(df, ticker='AAPL'):
    """Plot sentiment trends over time"""
    ticker_data = df[df['ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('timestamp')

    fig = go.Figure()

    # Sentiment score line
    fig.add_trace(go.Scatter(
        x=ticker_data['timestamp'],
        y=ticker_data['sentiment_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='#4da6ff', width=2),
        marker=dict(size=6)
    ))

    # Add sentiment labels as colors
    colors = []
    for label in ticker_data['sentiment_label']:
        if label == 'positive':
            colors.append('#4ade80')
        elif label == 'negative':
            colors.append('#f87171')
        else:
            colors.append('#a3a3a3')

    fig.update_layout(
        title=f'Sentiment Trends for {ticker}',
        xaxis_title='Time',
        yaxis_title='Sentiment Score',
        height=400,
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#e0e0e0')
    )

    return fig

def plot_sentiment_distribution(df):
    """Plot sentiment distribution by ticker"""
    sentiment_counts = df.groupby(['ticker', 'sentiment_label']).size().reset_index(name='count')

    fig = px.bar(
        sentiment_counts,
        x='ticker',
        y='count',
        color='sentiment_label',
        color_discrete_map={
            'positive': '#4ade80',
            'negative': '#f87171',
            'neutral': '#a3a3a3'
        },
        title='Sentiment Distribution by Ticker'
    )

    fig.update_layout(
        height=400,
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#e0e0e0')
    )
    return fig

def plot_source_distribution(df):
    """Plot data source distribution"""
    source_counts = df['source'].value_counts()

    fig = px.pie(
        values=source_counts.values,
        names=source_counts.index,
        title='Data Sources Distribution'
    )

    fig.update_layout(
        height=400,
        template='plotly_dark',
        paper_bgcolor='#0e1117',
        font=dict(color='#e0e0e0')
    )
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Sentiment Analytics</h1>', unsafe_allow_html=True)

    # Load data
    df = load_data()

    # Sidebar
    st.sidebar.header("📊 Dashboard Controls")

    # Auto-refresh controls
    st.sidebar.subheader("🔄 Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)

    refresh_interval = 30
    if auto_refresh:
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (seconds)",
            min_value=5,
            max_value=300,
            value=30,
            step=5
        )

        # Show last update time
        st.sidebar.info(f"🕒 Last updated: {datetime.now().strftime('%H:%M:%S')}")
        st.sidebar.info(f"⏰ Auto-refreshing every {refresh_interval}s")

    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()

    st.sidebar.markdown("---")
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=7), datetime.now()),
        max_value=datetime.now()
    )
    
    # Ticker filter
    available_tickers = df['ticker'].unique()
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        options=available_tickers,
        index=0
    )
    
    # Filter data based on selections
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[
            (df['timestamp'].dt.date >= start_date) &
            (df['timestamp'].dt.date <= end_date) &
            (df['ticker'] == selected_ticker)
        ]
    else:
        filtered_df = df[df['ticker'] == selected_ticker]
    
    # Main content
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Total Mentions",
            value=len(filtered_df),
            delta=len(filtered_df) - len(df[df['ticker'] == selected_ticker]) + 10
        )
    
    with col2:
        avg_sentiment = filtered_df['sentiment_score'].mean()
        st.metric(
            label="Average Sentiment",
            value=f"{avg_sentiment:.3f}",
            delta=f"{avg_sentiment - 0.5:.3f}"
        )
    
    with col3:
        positive_pct = (filtered_df['sentiment_label'] == 'positive').mean() * 100
        st.metric(
            label="Positive Sentiment %",
            value=f"{positive_pct:.1f}%",
            delta=f"{positive_pct - 50:.1f}%"
        )
    
    # Charts
    st.subheader("📈 Sentiment Trends")
    sentiment_fig = plot_sentiment_trends(filtered_df, selected_ticker)
    st.plotly_chart(sentiment_fig, use_container_width=True)
    
    # Distribution charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sentiment Distribution")
        dist_fig = plot_sentiment_distribution(df)
        st.plotly_chart(dist_fig, use_container_width=True)
    
    with col2:
        st.subheader("📰 Data Sources")
        source_fig = plot_source_distribution(df)
        st.plotly_chart(source_fig, use_container_width=True)
    
    # Recent mentions table
    st.subheader("🔄 Recent Mentions")
    recent_mentions = filtered_df.sort_values('timestamp', ascending=False).head(10)
    
    if not recent_mentions.empty:
        for _, row in recent_mentions.iterrows():
            sentiment_class = f"sentiment-{row['sentiment_label']}"
            # Format timestamp safely
            timestamp_str = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M')
            st.markdown(f"""
            <div class="metric-card">
                <strong>{row['ticker']}</strong> - {timestamp_str}
                <br>
                <span class="{sentiment_class}">Sentiment: {row['sentiment_label'].title()} ({row['sentiment_score']:.3f})</span>
                <br>
                <small>Source: {row['source'].title()}</small>
                <br>
                <em>{row['text'][:100]}...</em>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent mentions found for the selected criteria.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Stock Sentiment Analytics Dashboard | Real-time sentiment analysis powered by FinBERT</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Trigger auto-refresh if enabled
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main() 