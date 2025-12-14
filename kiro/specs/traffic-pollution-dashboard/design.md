# Design Document

## Overview

The Traffic-Pollution Dashboard is a Streamlit-based web application that integrates traffic congestion and air pollution data from public APIs to provide meaningful insights about environmental impacts in Indian cities. The system follows a modular architecture with separate components for data fetching, processing, correlation analysis, and visualization.

## Architecture

The system uses a layered architecture pattern:

```
┌─────────────────────────────────────────┐
│           Streamlit UI Layer            │
├─────────────────────────────────────────┤
│         Visualization Layer             │
│    (Plotly Charts & Components)         │
├─────────────────────────────────────────┤
│        Business Logic Layer            │
│  (Correlation Engine & Insight Gen)     │
├─────────────────────────────────────────┤
│         Data Processing Layer           │
│    (Cleaning, Alignment, Caching)      │
├─────────────────────────────────────────┤
│          Data Access Layer              │
│      (API Clients & Adapters)          │
└─────────────────────────────────────────┘
```

**Key Architectural Principles:**
- Separation of concerns between data, business logic, and presentation
- Dependency injection for API clients to enable testing
- Caching layer to minimize API calls and improve performance
- Error boundaries to handle API failures gracefully

## Components and Interfaces

### 1. API Integration Layer

**TrafficDataClient**
```python
class TrafficDataClient:
    def fetch_traffic_data(self, city: str, date_range: DateRange) -> pd.DataFrame
    def validate_response(self, response: dict) -> bool
    def handle_rate_limiting(self) -> None
```

**PollutionDataClient**
```python
class PollutionDataClient:
    def fetch_pollution_data(self, city: str, date_range: DateRange) -> pd.DataFrame
    def get_aqi_metrics(self, city: str) -> Dict[str, float]
    def handle_api_errors(self, error: Exception) -> None
```

**Recommended APIs:**
- **Traffic Data**: Google Maps Traffic API or TomTom Traffic API for real-time congestion levels
- **Air Pollution**: OpenWeatherMap Air Pollution API or WAQI (World Air Quality Index) API for AQI, PM2.5, PM10 data
- **Backup**: Government APIs like CPCB (Central Pollution Control Board) for Indian cities

### 2. Data Processing Layer

**DataProcessor**
```python
class DataProcessor:
    def clean_traffic_data(self, raw_data: pd.DataFrame) -> pd.DataFrame
    def clean_pollution_data(self, raw_data: pd.DataFrame) -> pd.DataFrame
    def align_datasets(self, traffic_df: pd.DataFrame, pollution_df: pd.DataFrame) -> pd.DataFrame
    def normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame
```

**DataCache**
```python
class DataCache:
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]
    def store_data(self, key: str, data: pd.DataFrame, ttl: int) -> None
    def invalidate_cache(self, pattern: str) -> None
```

### 3. Analysis Layer

**CorrelationEngine**
```python
class CorrelationEngine:
    def calculate_pearson_correlation(self, traffic_data: pd.Series, pollution_data: pd.Series) -> float
    def calculate_significance(self, correlation: float, sample_size: int) -> float
    def identify_peak_hours(self, traffic_data: pd.DataFrame) -> List[int]
    def analyze_pollution_during_peaks(self, pollution_data: pd.DataFrame, peak_hours: List[int]) -> Dict
```

**InsightGenerator**
```python
class InsightGenerator:
    def generate_correlation_insights(self, correlation: float, significance: float) -> str
    def generate_peak_hour_insights(self, peak_analysis: Dict) -> str
    def format_percentage_change(self, baseline: float, peak: float) -> str
```

### 4. Visualization Layer

**ChartFactory**
```python
class ChartFactory:
    def create_traffic_pollution_line_chart(self, data: pd.DataFrame) -> plotly.graph_objects.Figure
    def create_peak_hour_bar_chart(self, peak_data: Dict) -> plotly.graph_objects.Figure
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> plotly.graph_objects.Figure
```

## Data Models

### Traffic Data Schema
```python
@dataclass
class TrafficData:
    timestamp: datetime
    city: str
    congestion_level: float  # 0-100 scale
    average_speed: float     # km/h
    delay_index: float       # relative to free-flow
    road_segment: str        # optional
```

### Pollution Data Schema
```python
@dataclass
class PollutionData:
    timestamp: datetime
    city: str
    aqi: int                 # Air Quality Index
    pm25: float             # PM2.5 μg/m³
    pm10: float             # PM10 μg/m³
    no2: float              # Nitrogen Dioxide
    co: float               # Carbon Monoxide
    station_id: str         # monitoring station
```

### Correlation Result Schema
```python
@dataclass
class CorrelationResult:
    traffic_pollution_correlation: float
    statistical_significance: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    peak_hour_analysis: Dict[str, float]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Based on the prework analysis, I'll focus on the most critical properties that provide unique validation value:

**Property 1: Data fetching triggers for city selection**
*For any* valid city selection (Delhi, Bengaluru, Chennai), both traffic and pollution data fetching functions should be called
**Validates: Requirements 1.2**

**Property 2: Correlation metrics display with valid data**
*For any* successfully retrieved traffic and pollution dataset, correlation metrics between traffic congestion and PM2.5 levels should be displayed
**Validates: Requirements 1.3**

**Property 3: Graceful error handling for missing data**
*For any* city selection that results in no available data, the system should display an informative message and maintain stability without crashing
**Validates: Requirements 1.4**

**Property 4: Chart generation with data presence**
*For any* loaded dataset containing traffic and pollution data, both line charts (traffic vs AQI) and bar charts (PM2.5 during peak hours) should be generated
**Validates: Requirements 2.1, 2.2**

**Property 5: DataFrame structure consistency**
*For any* traffic data fetch operation, the returned DataFrame should contain timestamp and congestion_level columns with appropriate data types
**Validates: Requirements 3.1**

**Property 6: Pollution data completeness**
*For any* pollution data fetch operation, the returned DataFrame should contain AQI, PM2.5, and PM10 columns with numeric values
**Validates: Requirements 3.2**

**Property 7: Data alignment by timestamp**
*For any* pair of traffic and pollution datasets, the merge operation should align records by timestamp and handle timezone differences appropriately
**Validates: Requirements 3.5**

**Property 8: Correlation calculation validity**
*For any* valid traffic and pollution dataset pair, the Pearson correlation coefficient should be calculated and fall within the valid range [-1, 1]
**Validates: Requirements 4.1**

**Property 9: Date range filtering accuracy**
*For any* selected date range, the filtered dataset should contain only records within the specified start and end dates
**Validates: Requirements 5.2**

**Property 10: API response validation**
*For any* API response received, the system should validate the data schema and handle malformed responses without system failure
**Validates: Requirements 6.2**

**Property 11: Caching behavior for large datasets**
*For any* large dataset processing operation, subsequent identical requests should use cached data and avoid redundant API calls
**Validates: Requirements 7.3**

## Error Handling

The system implements comprehensive error handling at multiple levels:

**API Level Errors:**
- Network timeouts and connection failures
- Rate limiting and quota exceeded responses
- Invalid API keys or authentication failures
- Malformed or unexpected response formats

**Data Processing Errors:**
- Missing or null values in datasets
- Timezone conversion failures
- Data type mismatches during merging
- Insufficient data for correlation analysis

**UI Level Errors:**
- Invalid date range selections
- City selection with no available data
- Chart rendering failures with malformed data
- User input validation errors

**Error Recovery Strategies:**
- Graceful degradation with cached data when APIs fail
- User-friendly error messages with suggested actions
- Automatic retry mechanisms with exponential backoff
- Fallback to alternative data sources when available

## Testing Strategy

The testing approach combines unit testing and property-based testing to ensure comprehensive coverage:

**Unit Testing Framework:** pytest
**Property-Based Testing Framework:** Hypothesis

**Unit Testing Focus:**
- Specific API integration examples with known responses
- Edge cases like empty datasets or single data points
- Error handling scenarios with mocked API failures
- UI component rendering with sample data
- Date range validation with boundary conditions

**Property-Based Testing Focus:**
- Data processing functions with randomly generated datasets
- Correlation calculations across various data distributions
- Chart generation with diverse data shapes and sizes
- Caching behavior with different access patterns
- API response validation with generated malformed data

**Testing Configuration:**
- Property-based tests configured to run minimum 100 iterations
- Each property test tagged with format: **Feature: traffic-pollution-dashboard, Property {number}: {property_text}**
- Mock external APIs for consistent testing environment
- Separate test data fixtures for Indian cities
- Performance benchmarks for dashboard loading and updates

**Integration Testing:**
- End-to-end workflows from city selection to insight display
- API integration with rate limiting and error scenarios
- Data pipeline testing with real API responses (in staging)
- Cross-browser compatibility for Streamlit dashboard

## Performance Considerations

**Data Caching Strategy:**
- Redis-based caching for API responses with TTL of 15 minutes
- In-memory caching for processed datasets during user sessions
- Lazy loading of charts to improve initial page load times

**API Optimization:**
- Batch API requests where possible to reduce round trips
- Implement request queuing to respect rate limits
- Use compression for large data transfers
- Connection pooling for HTTP requests

**Frontend Performance:**
- Streamlit session state for maintaining user selections
- Plotly chart optimization with data sampling for large datasets
- Progressive loading of dashboard components
- Debounced user input to prevent excessive API calls

**Scalability Considerations:**
- Horizontal scaling capability through stateless design
- Database connection pooling for multi-user scenarios
- CDN integration for static assets
- Load balancing for high-traffic scenarios

## Security Considerations

**API Security:**
- Secure storage of API keys using environment variables
- Rate limiting implementation to prevent abuse
- Input validation and sanitization for all user inputs
- HTTPS enforcement for all external API communications

**Data Privacy:**
- No storage of personally identifiable information
- Anonymized logging that excludes sensitive data
- Secure session management in Streamlit
- Regular security audits of dependencies

## Deployment Architecture

**Development Environment:**
- Local Streamlit development server
- Docker containerization for consistent environments
- Environment-specific configuration management

**Production Deployment:**
- Cloud deployment on AWS/GCP/Azure
- Container orchestration with Docker Compose or Kubernetes
- Automated CI/CD pipeline with GitHub Actions
- Health monitoring and alerting systems

**Monitoring and Observability:**
- Application performance monitoring (APM)
- API response time and error rate tracking
- User interaction analytics for dashboard usage
- System resource utilization monitoring