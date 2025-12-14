# 🚦 Traffic Pollution Dashboard

A comprehensive web application for analyzing correlations between traffic congestion and air pollution in Indian cities. Built with Streamlit, this dashboard provides real-time insights into how traffic patterns affect air quality in Delhi, Bengaluru, and Chennai.

## ✨ Features

### 📊 Real-time Analysis
- **Traffic-Pollution Correlations**: Statistical analysis using Pearson and Spearman correlation coefficients
- **Peak Hour Analysis**: Identify when pollution levels spike during high traffic periods
- **Automated Insights**: AI-generated insights and recommendations based on data patterns

### 📈 Interactive Visualizations
- **Time Series Charts**: Traffic congestion vs air quality over time
- **Scatter Plots**: Correlation analysis with trend lines
- **Bar Charts**: Peak vs non-peak pollution comparisons
- **Heatmaps**: Correlation matrices across different pollutants

### 🏙️ Multi-City Support
- **Delhi**: National Capital Region analysis
- **Bengaluru**: Silicon Valley of India insights
- **Chennai**: Detroit of India patterns

### 🔧 Advanced Features
- **Data Caching**: Intelligent caching for improved performance
- **Property-Based Testing**: Comprehensive test suite with 57+ tests
- **Modular Architecture**: Clean separation of concerns
- **Error Handling**: Robust error handling and graceful degradation

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd traffic-pollution-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (optional - uses mock data by default)
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

### Using the Deployment Script

For easier deployment, use the included deployment script:

```bash
# Run locally
python deploy.py local

# Deploy with Docker
python deploy.py docker

# Run tests only
python deploy.py test

# Get cloud deployment instructions
python deploy.py cloud
```

## 📖 Usage Guide

### Basic Workflow

1. **Select City**: Choose from Delhi, Bengaluru, or Chennai in the sidebar
2. **Set Date Range**: Pick start and end dates (up to 30 days)
3. **Configure Analysis**: Enable/disable correlation analysis, peak hour analysis, and time series
4. **Load Data**: Click "Load Data" to fetch and process data
5. **Explore Results**: View automated insights, interactive charts, and raw data

### Understanding the Results

#### Correlation Analysis
- **Strong Correlation (|r| > 0.7)**: Clear relationship between traffic and pollution
- **Moderate Correlation (0.3 < |r| < 0.7)**: Noticeable relationship
- **Weak Correlation (|r| < 0.3)**: Limited relationship

#### Peak Hour Analysis
- Identifies hours with highest traffic congestion
- Shows pollution level increases during peak periods
- Provides percentage changes and health context

#### Automated Insights
- Statistical significance testing
- Health impact assessments
- Actionable recommendations for city planners

## 🏗️ Architecture

### Project Structure
```
traffic_pollution_dashboard/
├── config/           # Configuration and settings
├── data/            # API clients and data models
├── services/        # Business logic (correlation, caching, processing)
├── visualization/   # Chart factory and plotting utilities
tests/               # Comprehensive test suite
app.py              # Main Streamlit application
deploy.py           # Deployment automation script
```

### Key Components

#### Data Layer
- **TrafficDataClient**: Fetches traffic congestion data
- **PollutionDataClient**: Retrieves air quality measurements
- **DataProcessor**: Cleans, aligns, and normalizes datasets

#### Analysis Layer
- **CorrelationEngine**: Statistical correlation analysis
- **InsightGenerator**: Automated insight generation
- **DataCache**: Performance optimization through caching

#### Visualization Layer
- **ChartFactory**: Creates consistent, interactive Plotly charts
- **Streamlit Dashboard**: User interface and interaction handling

## 🧪 Testing

The project includes comprehensive testing with 57+ test cases:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=traffic_pollution_dashboard

# Run specific test categories
pytest tests/test_correlation_engine.py  # Correlation analysis tests
pytest tests/test_data_cache.py          # Caching behavior tests
```

### Test Categories
- **Property-Based Tests**: Using Hypothesis for robust testing across input ranges
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Caching and optimization validation

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Docker Deployment
```bash
docker build -t traffic-pollution-dashboard .
docker run -p 8501:8501 traffic-pollution-dashboard
```

### Cloud Deployment

#### Streamlit Cloud
1. Push code to GitHub
2. Connect repository at [share.streamlit.io](https://share.streamlit.io/)
3. Deploy automatically

#### Heroku
```bash
git push heroku main
```

#### AWS/GCP/Azure
Use the provided Docker configuration for container deployment.

## 🔧 Configuration

### Environment Variables
```bash
# API Configuration (optional - uses mock data if not provided)
TRAFFIC_API_KEY=your_traffic_api_key_here
TRAFFIC_API_BASE_URL=https://api.traffic-provider.com

POLLUTION_API_KEY=your_pollution_api_key_here  
POLLUTION_API_BASE_URL=https://api.openweathermap.org/data/2.5

# Supported Cities
SUPPORTED_CITIES=Delhi,Bengaluru,Chennai

# Performance Settings
API_RATE_LIMIT_PER_MINUTE=60
API_TIMEOUT_SECONDS=30
```

## 📊 Data Sources

### Traffic Data
- **Primary**: Google Maps Traffic API / TomTom Traffic API
- **Metrics**: Congestion level (0-100%), average speed, delay index
- **Frequency**: Hourly updates

### Pollution Data  
- **Primary**: OpenWeatherMap Air Pollution API / WAQI API
- **Metrics**: AQI, PM2.5, PM10, NO2, CO levels
- **Frequency**: Hourly measurements

### Mock Data
For development and testing, the system generates realistic mock data that simulates:
- Daily traffic patterns with morning and evening peaks
- Seasonal pollution variations
- City-specific characteristics
- Realistic correlations between traffic and pollution

## 📈 Performance

### Optimization Features
- **Intelligent Caching**: 15-minute TTL for API responses
- **Data Compression**: Efficient DataFrame storage
- **Lazy Loading**: Charts loaded on demand
- **Connection Pooling**: Optimized HTTP requests

### Benchmarks
- **Dashboard Load Time**: < 3 seconds
- **Data Processing**: < 2 seconds for 30-day datasets
- **Cache Hit Rate**: > 80% for repeated queries
- **Memory Usage**: < 100MB for typical workloads

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements.txt`
4. Run tests: `pytest`
5. Submit a pull request

### Code Standards
- **Formatting**: Black code formatter
- **Type Hints**: MyPy type checking
- **Testing**: Minimum 80% test coverage
- **Documentation**: Comprehensive docstrings

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenWeatherMap**: Air pollution data API
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **Pandas**: Data manipulation and analysis
- **SciPy**: Statistical analysis functions

---

**Built with ❤️ for cleaner, smarter cities**