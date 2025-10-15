# Project Structure

```
market-sentiment-analyzer/
├── 📁 Core Application Files
│   ├── dashboard.py              # Streamlit dashboard frontend
│   ├── stream_sentiment.py       # Backend data collection and sentiment analysis
│   ├── utils.py                  # Utility functions and helpers
│   ├── config.py                 # Configuration settings
│   ├── run.py                    # Startup script for easy execution
│   └── test_setup.py             # Comprehensive test suite
│
├── 📁 Data & Logs
│   ├── data/
│   │   └── sentiment_data.db     # SQLite database for sentiment and stock data
│   └── logs/
│       └── sentiment_analysis.log # Application logs
│
├── 📁 Configuration
│   ├── requirements.txt          # Python dependencies
│   ├── env_template.txt          # Environment variables template
│   ├── .cursorrules              # Cursor IDE configuration
│   ├── .vscode/settings.json     # VS Code/Cursor workspace settings
│   └── .gitignore               # Git ignore patterns
│
├── 📁 Virtual Environment
│   └── venv/                    # Python virtual environment
│
└── 📁 Documentation
    ├── README.md                # Main project documentation
    └── PROJECT_STRUCTURE.md     # This file
```

## File Organization

### **Core Application Files**
- **`dashboard.py`** - Beautiful Streamlit dashboard with interactive charts and filters
- **`stream_sentiment.py`** - Backend system for data collection and FinBERT sentiment analysis
- **`utils.py`** - Helper functions for data processing, ticker extraction, and analysis
- **`config.py`** - Centralized configuration management with environment variables
- **`run.py`** - Easy startup script with multiple modes (dashboard, stream, both)
- **`test_setup.py`** - Comprehensive test suite to verify all components

### **Data & Logs**
- **`data/sentiment_data.db`** - SQLite database storing sentiment analysis results and stock data
- **`logs/sentiment_analysis.log`** - Application logs for debugging and monitoring

### **Configuration**
- **`requirements.txt`** - All Python dependencies with specific versions
- **`env_template.txt`** - Template for environment variables (API keys, settings)
- **`.cursorrules`** - Tells Cursor IDE to use the virtual environment
- **`.vscode/settings.json`** - Workspace settings for Python interpreter and linting
- **`.gitignore`** - Prevents committing sensitive files and build artifacts

### **Virtual Environment**
- **`venv/`** - Isolated Python environment with all dependencies installed

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Test setup
python test_setup.py

# Run dashboard
python run.py dashboard

# Run data collection
python run.py stream

# Run both
python run.py both
```

## Environment Setup

1. **Copy environment template:**
   ```bash
   cp env_template.txt .env
   ```

2. **Edit .env with your API keys:**
   ```bash
   nano .env
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## IDE Configuration

The project includes configuration files for:
- **Cursor IDE** (`.cursorrules`)
- **VS Code** (`.vscode/settings.json`)

These ensure the IDE automatically uses the virtual environment and provides proper linting and formatting. 