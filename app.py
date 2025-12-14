"""
Main Streamlit application entry point for the Traffic Pollution Dashboard.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
from typing import Optional

# Import dashboard components
from traffic_pollution_dashboard.config.settings import settings
from traffic_pollution_dashboard.data.traffic_client import TrafficDataClient
from traffic_pollution_dashboard.data.pollution_client import PollutionDataClient
from traffic_pollution_dashboard.services.data_processor import DataProcessor
from traffic_pollution_dashboard.services.correlation_engine import CorrelationEngine
from traffic_pollution_dashboard.services.insight_generator import InsightGenerator
from traffic_pollution_dashboard.visualization.chart_factory import ChartFactory
from traffic_pollution_dashboard.data.models import DateRange


def initialize_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'aligned_data' not in st.session_state:
        st.session_state.aligned_data = None


def create_sidebar():
    """Create the sidebar with controls and city selection."""
    st.sidebar.header("🏙️ City Selection")
    
    # City selector
    selected_city = st.sidebar.selectbox(
        "Choose a city:",
        options=settings.SUPPORTED_CITIES,
        index=0,
        help="Select an Indian city to analyze traffic-pollution correlations"
    )
    
    st.sidebar.header("📅 Date Range")
    
    # Date range picker
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=7),
            max_value=date.today(),
            help="Select the start date for analysis"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            max_value=date.today(),
            help="Select the end date for analysis"
        )
    
    # Validate date range
    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date")
        return None, None, None
    
    if (end_date - start_date).days > 30:
        st.sidebar.warning("Date range limited to 30 days for performance")
        end_date = start_date + timedelta(days=30)
    
    # Analysis options
    st.sidebar.header("⚙️ Analysis Options")
    
    show_correlation = st.sidebar.checkbox("Show Correlation Analysis", value=True)
    show_peak_analysis = st.sidebar.checkbox("Show Peak Hour Analysis", value=True)
    show_time_series = st.sidebar.checkbox("Show Time Series Charts", value=True)
    
    analysis_options = {
        'correlation': show_correlation,
        'peak_analysis': show_peak_analysis,
        'time_series': show_time_series
    }
    
    # Load data button
    load_data = st.sidebar.button("🔄 Load Data", type="primary")
    
    return selected_city, (start_date, end_date), analysis_options, load_data


def load_and_process_data(city: str, date_range: tuple) -> Optional[pd.DataFrame]:
    """Load and process traffic and pollution data."""
    try:
        with st.spinner(f"Loading data for {city}..."):
            # Initialize clients and processors
            traffic_client = TrafficDataClient()
            pollution_client = PollutionDataClient()
            processor = DataProcessor()
            
            # Create date range object
            start_datetime = datetime.combine(date_range[0], datetime.min.time())
            end_datetime = datetime.combine(date_range[1], datetime.max.time())
            date_range_obj = DateRange(start_date=start_datetime, end_date=end_datetime)
            
            # Fetch data
            st.info("Fetching traffic data...")
            traffic_data = traffic_client.fetch_traffic_data(city, date_range_obj)
            
            st.info("Fetching pollution data...")
            pollution_data = pollution_client.fetch_pollution_data(city, date_range_obj)
            
            # Process and align data
            st.info("Processing and aligning data...")
            cleaned_traffic = processor.clean_traffic_data(traffic_data)
            cleaned_pollution = processor.clean_pollution_data(pollution_data)
            aligned_data = processor.align_datasets(cleaned_traffic, cleaned_pollution)
            
            if aligned_data.empty:
                st.error("No overlapping data found between traffic and pollution datasets")
                return None
            
            st.success(f"Successfully loaded {len(aligned_data)} data points")
            return aligned_data
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def perform_analysis(aligned_data: pd.DataFrame) -> dict:
    """Perform correlation and peak hour analysis."""
    try:
        with st.spinner("Performing correlation analysis..."):
            engine = CorrelationEngine()
            results = engine.perform_comprehensive_analysis(aligned_data)
            return results
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return {}


def display_overview(city: str, aligned_data: pd.DataFrame, analysis_results: dict):
    """Display overview section with key metrics."""
    st.header("📊 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Data Points",
            f"{len(aligned_data):,}",
            help="Total number of aligned data points"
        )
    
    with col2:
        if analysis_results and 'correlations' in analysis_results:
            aqi_corr = analysis_results['correlations'].get('aqi', {}).get('pearson', {}).get('correlation', 0)
            st.metric(
                "Traffic-AQI Correlation",
                f"{aqi_corr:.3f}",
                help="Pearson correlation between traffic and AQI"
            )
        else:
            st.metric("Traffic-AQI Correlation", "N/A")
    
    with col3:
        avg_traffic = aligned_data['congestion_level'].mean()
        st.metric(
            "Avg Traffic Congestion",
            f"{avg_traffic:.1f}%",
            help="Average traffic congestion level"
        )
    
    with col4:
        avg_aqi = aligned_data['aqi'].mean()
        st.metric(
            "Avg Air Quality Index",
            f"{avg_aqi:.0f}",
            help="Average Air Quality Index"
        )


def display_insights(analysis_results: dict):
    """Display automated insights."""
    st.header("🧠 Automated Insights")
    
    if not analysis_results:
        st.info("No analysis results available")
        return
    
    generator = InsightGenerator()
    summary = generator.generate_comprehensive_summary(analysis_results)
    
    # Overview insight
    st.subheader("Overview")
    st.write(summary['overview'])
    
    # Correlation insights
    if summary['correlations']:
        st.subheader("Correlation Analysis")
        st.write(summary['correlations'])
    
    # Peak hour insights
    if summary['peak_hours']:
        st.subheader("Peak Hour Analysis")
        st.write(summary['peak_hours'])
    
    # Recommendations
    if summary['recommendations']:
        st.subheader("Recommendations")
        st.info(summary['recommendations'])
    
    # Data quality
    if summary['data_quality']:
        st.subheader("Data Quality Assessment")
        st.write(summary['data_quality'])


def display_charts(aligned_data: pd.DataFrame, analysis_results: dict, 
                  city: str, analysis_options: dict):
    """Display various charts and visualizations."""
    factory = ChartFactory()
    
    st.header("📈 Visualizations")
    
    # Time series chart
    if analysis_options['time_series']:
        st.subheader("Traffic vs Air Quality Over Time")
        line_chart = factory.create_traffic_pollution_line_chart(aligned_data, city)
        st.plotly_chart(line_chart, use_container_width=True)
    
    # Correlation analysis
    if analysis_options['correlation'] and analysis_results:
        st.subheader("Correlation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot
            scatter_plot = factory.create_scatter_plot(
                aligned_data, 'congestion_level', 'aqi', city=city
            )
            st.plotly_chart(scatter_plot, use_container_width=True)
        
        with col2:
            # Correlation summary
            if 'correlations' in analysis_results:
                corr_summary = factory.create_correlation_summary_chart(
                    analysis_results['correlations']
                )
                st.plotly_chart(corr_summary, use_container_width=True)
    
    # Peak hour analysis
    if analysis_options['peak_analysis'] and analysis_results:
        if 'peak_analysis' in analysis_results and analysis_results['peak_analysis']:
            st.subheader("Peak Hour Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Peak hour bar chart
                bar_chart = factory.create_peak_hour_bar_chart(
                    analysis_results['peak_analysis']
                )
                st.plotly_chart(bar_chart, use_container_width=True)
            
            with col2:
                # Hourly pattern
                hourly_chart = factory.create_hourly_pattern_chart(
                    aligned_data, 'congestion_level', city
                )
                st.plotly_chart(hourly_chart, use_container_width=True)
    
    # Additional metrics comparison
    st.subheader("Multi-Metric Comparison")
    metrics_chart = factory.create_time_series_comparison(
        aligned_data, ['congestion_level', 'aqi', 'pm25', 'pm10'], city
    )
    st.plotly_chart(metrics_chart, use_container_width=True)


def display_data_table(aligned_data: pd.DataFrame):
    """Display data table with filtering options."""
    st.header("📋 Data Table")
    
    # Data filtering options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_traffic = st.slider(
            "Min Traffic Congestion (%)",
            min_value=0,
            max_value=100,
            value=0
        )
    
    with col2:
        max_aqi = st.slider(
            "Max AQI",
            min_value=int(aligned_data['aqi'].min()),
            max_value=int(aligned_data['aqi'].max()),
            value=int(aligned_data['aqi'].max())
        )
    
    with col3:
        show_rows = st.selectbox(
            "Rows to show",
            options=[10, 25, 50, 100],
            index=1
        )
    
    # Filter data
    filtered_data = aligned_data[
        (aligned_data['congestion_level'] >= min_traffic) &
        (aligned_data['aqi'] <= max_aqi)
    ].head(show_rows)
    
    # Display table
    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True
    )
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"traffic_pollution_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Traffic Pollution Dashboard",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Main title
    st.title("🚦 Traffic Pollution Dashboard")
    st.markdown("**Correlating traffic congestion with air pollution in Indian cities**")
    
    # Validate configuration
    if not settings.validate_config():
        st.error("⚠️ Configuration incomplete. Please check your API keys in the .env file.")
        st.info("Copy .env.example to .env and add your API keys to get started.")
        st.info("For development, the dashboard will use mock data.")
    
    # Create sidebar
    sidebar_result = create_sidebar()
    if sidebar_result is None:
        return
    
    selected_city, date_range, analysis_options, load_data = sidebar_result
    
    # Load data when button is clicked
    if load_data:
        aligned_data = load_and_process_data(selected_city, date_range)
        if aligned_data is not None:
            st.session_state.aligned_data = aligned_data
            st.session_state.data_loaded = True
            
            # Perform analysis
            analysis_results = perform_analysis(aligned_data)
            st.session_state.analysis_results = analysis_results
    
    # Display dashboard content if data is loaded
    if st.session_state.data_loaded and st.session_state.aligned_data is not None:
        aligned_data = st.session_state.aligned_data
        analysis_results = st.session_state.analysis_results or {}
        
        # Display overview
        display_overview(selected_city, aligned_data, analysis_results)
        
        # Display insights
        display_insights(analysis_results)
        
        # Display charts
        display_charts(aligned_data, analysis_results, selected_city, analysis_options)
        
        # Display data table
        with st.expander("📋 View Raw Data", expanded=False):
            display_data_table(aligned_data)
    
    else:
        # Welcome message
        st.info("👈 Select a city and date range from the sidebar, then click 'Load Data' to begin analysis.")
        
        # Show feature overview
        st.header("🌟 Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Real-time Analysis")
            st.write("Analyze correlations between traffic congestion and air pollution in real-time")
        
        with col2:
            st.subheader("🏙️ Multiple Cities")
            st.write("Support for Delhi, Bengaluru, and Chennai with city-specific insights")
        
        with col3:
            st.subheader("📈 Interactive Charts")
            st.write("Interactive Plotly charts with hover details and zoom capabilities")
        
        # Show supported cities
        st.header("🏙️ Supported Cities")
        cities_col1, cities_col2, cities_col3 = st.columns(3)
        
        with cities_col1:
            st.info("📍 **Delhi**\nNational Capital Region")
        
        with cities_col2:
            st.info("📍 **Bengaluru**\nSilicon Valley of India")
        
        with cities_col3:
            st.info("📍 **Chennai**\nDetroit of India")


if __name__ == "__main__":
    main()