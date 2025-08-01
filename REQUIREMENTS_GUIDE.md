# Python Requirements Management Guide

## Commands to Update Requirements

### **1. Generate Complete Requirements List**
```bash
# Activate virtual environment first
source venv/bin/activate

# Generate complete requirements with exact versions
pip freeze > requirements.txt
```

### **2. Generate Clean Requirements (Recommended)**
```bash
# Generate requirements with only direct dependencies
pip list --format=freeze > requirements.txt
```

### **3. Update Specific Package**
```bash
# Install new package
pip install package_name

# Update requirements
pip freeze > requirements.txt
```

### **4. Install from Requirements**
```bash
# Install all packages from requirements.txt
pip install -r requirements.txt

# Install with upgrade
pip install -r requirements.txt --upgrade
```

### **5. Check for Outdated Packages**
```bash
# List outdated packages
pip list --outdated

# Update specific package
pip install --upgrade package_name
```

## Requirements File Structure

### **Current Clean Requirements:**
```txt
# Core dependencies
streamlit==1.47.1
pandas==2.3.1
plotly==5.18.0
numpy==1.26.4

# Machine Learning & NLP
transformers==4.38.2
torch==2.2.1
tokenizers==0.15.2

# Data Collection
yfinance==0.2.36
praw==7.7.1
requests==2.32.4

# Database & Utilities
python-dotenv==1.0.1
SQLAlchemy==2.0.27
asyncio-mqtt==0.16.1

# Additional dependencies
altair==5.5.0
protobuf==6.31.1
pyarrow==21.0.0
python-dateutil==2.9.0.post0
pytz==2025.2
tornado==6.5.1
watchdog==6.0.0
```

## Best Practices

### **1. Always Use Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### **2. Keep Requirements Organized**
- Group related packages together
- Add comments for clarity
- Use exact versions for reproducibility

### **3. Regular Maintenance**
```bash
# Check for security vulnerabilities
pip-audit

# Update all packages (use with caution)
pip install --upgrade -r requirements.txt

# Clean up unused packages
pip-autoremove
```

### **4. Development vs Production**
```bash
# Development requirements (include dev tools)
pip install pytest black flake8
pip freeze > requirements-dev.txt

# Production requirements (core only)
pip freeze > requirements.txt
```

## Troubleshooting

### **Common Issues:**

1. **Version Conflicts:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

2. **Missing Dependencies:**
   ```bash
   pip install -r requirements.txt --no-deps
   pip install package_name
   ```

3. **Virtual Environment Issues:**
   ```bash
   # Recreate virtual environment
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### **Package Management Commands:**

```bash
# List installed packages
pip list

# Show package info
pip show package_name

# Uninstall package
pip uninstall package_name

# Check package dependencies
pip show package_name | grep Requires
```

## Project-Specific Commands

For this Stock Sentiment Analytics project:

```bash
# Activate environment
source venv/bin/activate

# Update requirements
pip freeze > requirements.txt

# Install new package
pip install new_package
pip freeze > requirements.txt

# Test setup
python test_setup.py

# Run application
python run.py dashboard
```

## Version Pinning Strategy

### **Exact Versions (Recommended for Production):**
```txt
streamlit==1.47.1
pandas==2.3.1
```

### **Minimum Versions (Development):**
```txt
streamlit>=1.47.0
pandas>=2.3.0
```

### **Compatible Releases:**
```txt
streamlit~=1.47.0
pandas~=2.3.0
```

## Security Considerations

1. **Regular Updates:**
   ```bash
   pip list --outdated
   pip install --upgrade package_name
   ```

2. **Security Audits:**
   ```bash
   pip-audit
   pip-audit --fix
   ```

3. **Vulnerability Checks:**
   ```bash
   safety check
   ```

This guide ensures your project dependencies are properly managed and reproducible across different environments. 