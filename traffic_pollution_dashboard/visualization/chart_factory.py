"""
Chart factory for creating Plotly visualizations for traffic-pollution dashboard.

This module provides the ChartFactory class that creates consistent, interactive
charts for displaying traffic and pollution data correlations.
"""

import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ChartFactory:
    """
    Factory for creating consistent Plotly charts for traffic-pollution analysis.
    
    Provides methods for creating line charts, bar charts, scatter plots, heatmaps,
    and other visualizations with consistent styling and interactivity.
    """
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize the chart factory.
        
        Args:
            theme: Plotly theme to use for charts
        """
        self.theme = theme
        self.logger = logging.getLogger(__name__)
        
        # Define consistent color palette
        self.colors = {
            'traffic': '#FF6B6B',      # Red for traffic
            'aqi': '#4ECDC4',          # Teal for AQI
            'pm25': '#45B7D1',         # Blue for PM2.5
            'pm10': '#96CEB4',         # Green for PM10
            'no2': '#FFEAA7',          # Yellow for NO2
            'co': '#DDA0DD',           # Purple for CO
            'correlation': '#FF7F50',   # Coral for correlations
            'peak_hours': '#FFB347',    # Orange for peak hours
            'background': '#F8F9FA',    # Light gray background
            'grid': '#E9ECEF'          # Grid lines
        }
        
        # Chart configuration
        self.config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'traffic_pollution_chart',
                'height': 600,
                'width': 1000,
                'scale': 2
            }
        }
    
    def create_traffic_pollution_line_chart(self, data: pd.DataFrame, 
                                          city: str = None) -> go.Figure:
        """
        Create a line chart showing traffic congestion levels versus AQI over time.
        
        Args:
            data: DataFrame with timestamp, congestion_level, and aqi columns
            city: City name for the chart title
            
        Returns:
            Plotly Figure object
        """
        if data.empty:
            return self._create_empty_chart("No data available for line chart")
        
        required_columns = ['timestamp', 'congestion_level', 'aqi']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return self._create_error_chart(f"Missing columns: {missing_columns}")
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=[f"Traffic Congestion vs Air Quality{' - ' + city if city else ''}"]
        )
        
        # Add traffic congestion line
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['congestion_level'],
                mode='lines+markers',
                name='Traffic Congestion (%)',
                line=dict(color=self.colors['traffic'], width=2),
                marker=dict(size=4),
                hovertemplate='<b>Traffic Congestion</b><br>' +
                             'Time: %{x}<br>' +
                             'Congestion: %{y:.1f}%<br>' +
                             '<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add AQI line
        fig.add_trace(
            go.Scatter(
                x=data['timestamp'],
                y=data['aqi'],
                mode='lines+markers',
                name='Air Quality Index',
                line=dict(color=self.colors['aqi'], width=2),
                marker=dict(size=4),
                hovertemplate='<b>Air Quality Index</b><br>' +
                             'Time: %{x}<br>' +
                             'AQI: %{y:.0f}<br>' +
                             '<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            template=self.theme,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Update y-axes
        fig.update_yaxes(
            title_text="Traffic Congestion (%)",
            secondary_y=False,
            range=[0, 100]
        )
        fig.update_yaxes(
            title_text="Air Quality Index",
            secondary_y=True
        )
        
        # Update x-axis
        fig.update_xaxes(title_text="Time")
        
        logger.info(f"Created traffic-pollution line chart with {len(data)} data points")
        return fig
    
    def create_peak_hour_bar_chart(self, peak_data: Dict[str, float]) -> go.Figure:
        """
        Create a bar chart of PM2.5 levels during peak traffic hours.
        
        Args:
            peak_data: Dictionary with peak hour analysis data
            
        Returns:
            Plotly Figure object
        """
        if not peak_data:
            return self._create_empty_chart("No peak hour data available")
        
        # Extract data for chart
        categories = []
        values = []
        colors = []
        
        # AQI comparison
        if 'aqi_peak_avg' in peak_data and 'aqi_non_peak_avg' in peak_data:
            categories.extend(['AQI (Peak Hours)', 'AQI (Non-Peak Hours)'])
            values.extend([peak_data['aqi_peak_avg'], peak_data['aqi_non_peak_avg']])
            colors.extend([self.colors['aqi'], self.colors['aqi']])
        
        # PM2.5 comparison
        if 'pm25_peak_avg' in peak_data and 'pm25_non_peak_avg' in peak_data:
            categories.extend(['PM2.5 (Peak Hours)', 'PM2.5 (Non-Peak Hours)'])
            values.extend([peak_data['pm25_peak_avg'], peak_data['pm25_non_peak_avg']])
            colors.extend([self.colors['pm25'], self.colors['pm25']])
        
        # PM10 comparison
        if 'pm10_peak_avg' in peak_data and 'pm10_non_peak_avg' in peak_data:
            categories.extend(['PM10 (Peak Hours)', 'PM10 (Non-Peak Hours)'])
            values.extend([peak_data['pm10_peak_avg'], peak_data['pm10_non_peak_avg']])
            colors.extend([self.colors['pm10'], self.colors['pm10']])
        
        if not categories:
            return self._create_empty_chart("No valid peak hour data for chart")
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=colors,
                text=[f'{val:.1f}' for val in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>' +
                             'Value: %{y:.1f}<br>' +
                             '<extra></extra>'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title="Pollution Levels: Peak vs Non-Peak Traffic Hours",
            template=self.theme,
            xaxis_title="Measurement Type",
            yaxis_title="Concentration",
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        logger.info(f"Created peak hour bar chart with {len(categories)} categories")
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """
        Create a heatmap showing correlations between different metrics.
        
        Args:
            correlation_matrix: DataFrame with correlation coefficients
            
        Returns:
            Plotly Figure object
        """
        if correlation_matrix.empty:
            return self._create_empty_chart("No correlation data available")
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 12},
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlation: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title="Traffic-Pollution Correlation Matrix",
            template=self.theme,
            xaxis_title="Metrics",
            yaxis_title="Metrics",
            margin=dict(l=100, r=50, t=80, b=100)
        )
        
        logger.info(f"Created correlation heatmap with {correlation_matrix.shape} matrix")
        return fig
    
    def create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str,
                          color_col: str = None, city: str = None) -> go.Figure:
        """
        Create a scatter plot for correlation analysis.
        
        Args:
            data: DataFrame with data points
            x_col: Column name for x-axis
            y_col: Column name for y-axis
            color_col: Optional column for color coding
            city: City name for title
            
        Returns:
            Plotly Figure object
        """
        if data.empty or x_col not in data.columns or y_col not in data.columns:
            return self._create_empty_chart("Invalid data for scatter plot")
        
        # Create scatter plot
        if color_col and color_col in data.columns:
            fig = px.scatter(
                data, 
                x=x_col, 
                y=y_col, 
                color=color_col,
                template=self.theme,
                title=f"{y_col} vs {x_col}{' - ' + city if city else ''}",
                hover_data=[color_col]
            )
        else:
            fig = go.Figure(data=go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='markers',
                marker=dict(
                    color=self.colors['correlation'],
                    size=6,
                    opacity=0.7
                ),
                hovertemplate=f'<b>{x_col}</b>: %{{x:.1f}}<br>' +
                             f'<b>{y_col}</b>: %{{y:.1f}}<br>' +
                             '<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"{y_col} vs {x_col}{' - ' + city if city else ''}",
                template=self.theme,
                xaxis_title=x_col,
                yaxis_title=y_col
            )
        
        # Add trend line
        if len(data) > 2:
            z = np.polyfit(data[x_col].dropna(), data[y_col].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(data[x_col].min(), data[x_col].max(), 100)
            y_trend = p(x_trend)
            
            fig.add_trace(go.Scatter(
                x=x_trend,
                y=y_trend,
                mode='lines',
                name='Trend Line',
                line=dict(color='red', dash='dash', width=2),
                hoverinfo='skip'
            ))
        
        logger.info(f"Created scatter plot for {x_col} vs {y_col} with {len(data)} points")
        return fig
    
    def create_time_series_comparison(self, data: pd.DataFrame, 
                                    metrics: List[str], city: str = None) -> go.Figure:
        """
        Create a multi-line time series chart comparing different metrics.
        
        Args:
            data: DataFrame with timestamp and metric columns
            metrics: List of metric column names to plot
            city: City name for title
            
        Returns:
            Plotly Figure object
        """
        if data.empty or 'timestamp' not in data.columns:
            return self._create_empty_chart("No time series data available")
        
        available_metrics = [m for m in metrics if m in data.columns]
        if not available_metrics:
            return self._create_empty_chart("No valid metrics found in data")
        
        fig = go.Figure()
        
        # Add trace for each metric
        for metric in available_metrics:
            color = self.colors.get(metric, '#1f77b4')
            
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=data[metric],
                mode='lines+markers',
                name=metric.upper(),
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=f'<b>{metric.upper()}</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: %{y:.1f}<br>' +
                             '<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"Time Series Comparison{' - ' + city if city else ''}",
            template=self.theme,
            xaxis_title="Time",
            yaxis_title="Values",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        logger.info(f"Created time series comparison with {len(available_metrics)} metrics")
        return fig
    
    def create_hourly_pattern_chart(self, data: pd.DataFrame, 
                                  metric: str, city: str = None) -> go.Figure:
        """
        Create a chart showing hourly patterns of a specific metric.
        
        Args:
            data: DataFrame with timestamp and metric columns
            metric: Metric column name to analyze
            city: City name for title
            
        Returns:
            Plotly Figure object
        """
        if data.empty or 'timestamp' not in data.columns or metric not in data.columns:
            return self._create_empty_chart("Invalid data for hourly pattern chart")
        
        # Extract hour and calculate hourly averages
        data_copy = data.copy()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        hourly_avg = data_copy.groupby('hour')[metric].agg(['mean', 'std']).reset_index()
        
        # Create bar chart with error bars
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=hourly_avg['hour'],
            y=hourly_avg['mean'],
            error_y=dict(
                type='data',
                array=hourly_avg['std'],
                visible=True
            ),
            marker_color=self.colors.get(metric, '#1f77b4'),
            name=f'Average {metric.upper()}',
            hovertemplate='<b>Hour %{x}:00</b><br>' +
                         f'{metric.upper()}: %{{y:.1f}} ± %{{error_y.array:.1f}}<br>' +
                         '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Hourly Pattern - {metric.upper()}{' - ' + city if city else ''}",
            template=self.theme,
            xaxis_title="Hour of Day",
            yaxis_title=f"{metric.upper()} Level",
            xaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=2
            ),
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        logger.info(f"Created hourly pattern chart for {metric}")
        return fig
    
    def create_correlation_summary_chart(self, correlations: Dict[str, Dict]) -> go.Figure:
        """
        Create a summary chart of all correlations.
        
        Args:
            correlations: Dictionary with correlation results for different pollutants
            
        Returns:
            Plotly Figure object
        """
        if not correlations:
            return self._create_empty_chart("No correlation data available")
        
        pollutants = []
        pearson_values = []
        spearman_values = []
        significance = []
        
        for pollutant, corr_data in correlations.items():
            if corr_data and 'pearson' in corr_data and 'spearman' in corr_data:
                pollutants.append(pollutant.upper())
                pearson_values.append(corr_data['pearson']['correlation'])
                spearman_values.append(corr_data['spearman']['correlation'])
                significance.append('Significant' if corr_data['pearson']['is_significant'] else 'Not Significant')
        
        if not pollutants:
            return self._create_empty_chart("No valid correlation data")
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Pearson Correlation',
            x=pollutants,
            y=pearson_values,
            marker_color=self.colors['correlation'],
            text=[f'{val:.3f}' for val in pearson_values],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Spearman Correlation',
            x=pollutants,
            y=spearman_values,
            marker_color=self.colors['peak_hours'],
            text=[f'{val:.3f}' for val in spearman_values],
            textposition='auto'
        ))
        
        # Update layout
        fig.update_layout(
            title="Traffic-Pollution Correlation Summary",
            template=self.theme,
            xaxis_title="Pollutant",
            yaxis_title="Correlation Coefficient",
            yaxis=dict(range=[-1, 1]),
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Add horizontal lines for reference
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_hline(y=0.3, line_dash="dot", line_color="green", opacity=0.5, 
                     annotation_text="Moderate correlation")
        fig.add_hline(y=-0.3, line_dash="dot", line_color="green", opacity=0.5)
        
        logger.info(f"Created correlation summary chart with {len(pollutants)} pollutants")
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            template=self.theme,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig
    
    def _create_error_chart(self, error_message: str) -> go.Figure:
        """Create an error chart with error message."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            template=self.theme,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig