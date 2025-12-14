"""
Automated insight generation service for traffic-pollution correlation analysis.

This module provides the InsightGenerator class that creates human-readable
insights and analysis from correlation data and statistical results.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class InsightGenerator:
    """
    Generator for human-readable insights from traffic-pollution correlation analysis.
    
    Converts statistical results into meaningful, actionable insights for
    city planners, researchers, and the general public.
    """
    
    def __init__(self):
        """Initialize the insight generator."""
        self.logger = logging.getLogger(__name__)
    
    def generate_correlation_insights(self, correlation: float, significance: float, 
                                    sample_size: int, method: str = 'pearson') -> str:
        """
        Generate human-readable insights from correlation analysis.
        
        Args:
            correlation: Correlation coefficient (-1 to 1)
            significance: P-value of the correlation
            sample_size: Number of data points used
            method: Correlation method used ('pearson' or 'spearman')
            
        Returns:
            Human-readable insight string
        """
        # Determine correlation strength
        abs_corr = abs(correlation)
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.3:
            strength = "moderate"
        elif abs_corr >= 0.1:
            strength = "weak"
        else:
            strength = "negligible"
        
        # Determine direction
        direction = "positive" if correlation > 0 else "negative"
        
        # Determine significance
        is_significant = significance < 0.05
        confidence_level = "high" if significance < 0.01 else "moderate" if significance < 0.05 else "low"
        
        # Generate base insight
        if abs_corr < 0.1:
            insight = f"There is {strength} correlation between traffic congestion and air pollution levels"
        else:
            insight = f"There is a {strength} {direction} correlation between traffic congestion and air pollution levels"
        
        # Add correlation value
        insight += f" (r = {correlation:.3f})"
        
        # Add significance information
        if is_significant:
            insight += f". This relationship is statistically significant with {confidence_level} confidence"
        else:
            insight += f". However, this relationship is not statistically significant"
        
        # Add sample size context
        if sample_size < 30:
            insight += f" (based on {sample_size} data points - small sample)"
        elif sample_size < 100:
            insight += f" (based on {sample_size} data points)"
        else:
            insight += f" (based on {sample_size} data points - large sample)"
        
        # Add interpretation
        if is_significant and abs_corr >= 0.3:
            if correlation > 0:
                insight += ". This suggests that higher traffic congestion is associated with increased air pollution levels"
            else:
                insight += ". This suggests that higher traffic congestion is associated with decreased air pollution levels"
        elif is_significant and abs_corr >= 0.1:
            insight += ". While statistically significant, the relationship is relatively weak"
        else:
            insight += ". No strong evidence of a meaningful relationship was found"
        
        insight += "."
        
        logger.info(f"Generated correlation insight for r={correlation:.3f}, p={significance:.3f}")
        return insight
    
    def generate_peak_hour_insights(self, peak_analysis: Dict[str, Any]) -> str:
        """
        Generate insights about pollution levels during peak traffic hours.
        
        Args:
            peak_analysis: Dictionary with peak hour analysis results
            
        Returns:
            Human-readable insight about peak hour pollution
        """
        if not peak_analysis or 'peak_hours_count' not in peak_analysis:
            return "Insufficient data available for peak hour analysis."
        
        peak_hours_count = peak_analysis.get('peak_hours_count', 0)
        
        if peak_hours_count == 0:
            return "No clear peak traffic hours were identified in the data."
        
        insights = []
        
        # Analyze AQI changes
        if 'aqi_percent_increase' in peak_analysis:
            aqi_increase = peak_analysis['aqi_percent_increase']
            aqi_peak = peak_analysis.get('aqi_peak_avg', 0)
            aqi_non_peak = peak_analysis.get('aqi_non_peak_avg', 0)
            
            if aqi_increase > 20:
                insights.append(f"Air Quality Index (AQI) increases significantly by {aqi_increase:.1f}% during peak traffic hours")
            elif aqi_increase > 10:
                insights.append(f"Air Quality Index (AQI) increases moderately by {aqi_increase:.1f}% during peak traffic hours")
            elif aqi_increase > 0:
                insights.append(f"Air Quality Index (AQI) increases slightly by {aqi_increase:.1f}% during peak traffic hours")
            else:
                insights.append(f"Air Quality Index (AQI) does not show significant increase during peak traffic hours")
            
            # Add absolute values for context
            insights.append(f"Average AQI: {aqi_peak:.0f} during peaks vs {aqi_non_peak:.0f} during non-peak hours")
        
        # Analyze PM2.5 changes
        if 'pm25_percent_increase' in peak_analysis:
            pm25_increase = peak_analysis['pm25_percent_increase']
            pm25_peak = peak_analysis.get('pm25_peak_avg', 0)
            pm25_non_peak = peak_analysis.get('pm25_non_peak_avg', 0)
            
            if pm25_increase > 25:
                insights.append(f"PM2.5 levels show a substantial {pm25_increase:.1f}% increase during peak hours")
            elif pm25_increase > 15:
                insights.append(f"PM2.5 levels show a notable {pm25_increase:.1f}% increase during peak hours")
            elif pm25_increase > 5:
                insights.append(f"PM2.5 levels show a modest {pm25_increase:.1f}% increase during peak hours")
            
            # Health context for PM2.5
            if pm25_peak > 75:
                insights.append(f"Peak hour PM2.5 levels ({pm25_peak:.1f} μg/m³) exceed WHO guidelines")
            elif pm25_peak > 35:
                insights.append(f"Peak hour PM2.5 levels ({pm25_peak:.1f} μg/m³) are above recommended levels")
        
        # Analyze PM10 changes
        if 'pm10_percent_increase' in peak_analysis:
            pm10_increase = peak_analysis['pm10_percent_increase']
            if pm10_increase > 20:
                insights.append(f"PM10 particulate matter increases by {pm10_increase:.1f}% during peak traffic periods")
        
        # Peak hours information
        insights.append(f"Analysis based on {peak_hours_count} identified peak traffic hours")
        
        # Combine insights
        if len(insights) == 0:
            return "Peak hour analysis did not reveal significant pollution patterns."
        
        main_insight = ". ".join(insights) + "."
        
        logger.info(f"Generated peak hour insights for {peak_hours_count} peak hours")
        return main_insight
    
    def format_percentage_change(self, baseline: float, peak: float) -> str:
        """
        Format percentage change between baseline and peak values.
        
        Args:
            baseline: Baseline value
            peak: Peak value
            
        Returns:
            Formatted percentage change string
        """
        if baseline == 0:
            return "N/A (baseline is zero)"
        
        percent_change = ((peak - baseline) / baseline) * 100
        
        if abs(percent_change) < 0.1:
            return "no significant change"
        elif percent_change > 0:
            return f"{percent_change:.1f}% increase"
        else:
            return f"{abs(percent_change):.1f}% decrease"
    
    def generate_comprehensive_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate a comprehensive summary of all analysis results.
        
        Args:
            analysis_results: Complete analysis results from correlation engine
            
        Returns:
            Dictionary with categorized insights
        """
        summary = {
            'overview': '',
            'correlations': '',
            'peak_hours': '',
            'recommendations': '',
            'data_quality': ''
        }
        
        sample_size = analysis_results.get('sample_size', 0)
        
        # Overview
        summary['overview'] = self._generate_overview_insight(analysis_results)
        
        # Correlation insights
        correlations = analysis_results.get('correlations', {})
        correlation_insights = []
        
        for pollutant, corr_data in correlations.items():
            if corr_data and 'pearson' in corr_data:
                pearson = corr_data['pearson']
                insight = self.generate_correlation_insights(
                    pearson['correlation'],
                    pearson['p_value'],
                    pearson['sample_size']
                )
                correlation_insights.append(f"{pollutant.upper()}: {insight}")
        
        summary['correlations'] = " ".join(correlation_insights) if correlation_insights else "No correlation data available."
        
        # Peak hour insights
        peak_analysis = analysis_results.get('peak_analysis', {})
        summary['peak_hours'] = self.generate_peak_hour_insights(peak_analysis)
        
        # Recommendations
        summary['recommendations'] = self._generate_recommendations(analysis_results)
        
        # Data quality assessment
        summary['data_quality'] = self._assess_data_quality(analysis_results)
        
        logger.info("Generated comprehensive summary")
        return summary
    
    def _generate_overview_insight(self, analysis_results: Dict[str, Any]) -> str:
        """Generate overview insight from analysis results."""
        sample_size = analysis_results.get('sample_size', 0)
        correlations = analysis_results.get('correlations', {})
        
        if sample_size == 0:
            return "No data available for analysis."
        
        # Find strongest correlation
        strongest_corr = 0
        strongest_pollutant = None
        
        for pollutant, corr_data in correlations.items():
            if corr_data and 'pearson' in corr_data:
                corr_value = abs(corr_data['pearson']['correlation'])
                if corr_value > abs(strongest_corr):
                    strongest_corr = corr_data['pearson']['correlation']
                    strongest_pollutant = pollutant
        
        if strongest_pollutant and abs(strongest_corr) > 0.1:
            direction = "positive" if strongest_corr > 0 else "negative"
            return (f"Analysis of {sample_size} data points reveals a {direction} relationship "
                   f"between traffic congestion and {strongest_pollutant.upper()} levels "
                   f"(r = {strongest_corr:.3f}).")
        else:
            return (f"Analysis of {sample_size} data points shows no strong correlation "
                   f"between traffic congestion and air pollution levels.")
    
    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> str:
        """Generate actionable recommendations based on analysis."""
        correlations = analysis_results.get('correlations', {})
        peak_analysis = analysis_results.get('peak_analysis', {})
        
        recommendations = []
        
        # Check for significant correlations
        significant_correlations = []
        for pollutant, corr_data in correlations.items():
            if (corr_data and 'pearson' in corr_data and 
                corr_data['pearson']['is_significant'] and 
                abs(corr_data['pearson']['correlation']) > 0.3):
                significant_correlations.append(pollutant)
        
        if significant_correlations:
            recommendations.append(
                "Traffic management strategies should be prioritized as they show significant "
                "impact on air quality"
            )
            
            if 'pm25' in significant_correlations:
                recommendations.append(
                    "Focus on reducing PM2.5 emissions from vehicles as this shows strong "
                    "correlation with traffic patterns"
                )
        
        # Peak hour recommendations
        aqi_increase = peak_analysis.get('aqi_percent_increase', 0)
        if aqi_increase > 15:
            recommendations.append(
                "Consider implementing peak-hour traffic restrictions or promoting "
                "alternative transportation during high-congestion periods"
            )
        
        if aqi_increase > 25:
            recommendations.append(
                "Air quality alerts should be issued during peak traffic hours to "
                "protect public health"
            )
        
        # General recommendations
        if not recommendations:
            recommendations.append(
                "Continue monitoring traffic and air quality patterns to identify "
                "emerging trends and intervention opportunities"
            )
        
        return ". ".join(recommendations) + "."
    
    def _assess_data_quality(self, analysis_results: Dict[str, Any]) -> str:
        """Assess and report on data quality."""
        sample_size = analysis_results.get('sample_size', 0)
        
        if sample_size < 30:
            return (f"Data quality: Limited sample size ({sample_size} points). "
                   f"Results should be interpreted with caution.")
        elif sample_size < 100:
            return (f"Data quality: Moderate sample size ({sample_size} points). "
                   f"Results provide reasonable confidence.")
        else:
            return (f"Data quality: Large sample size ({sample_size} points). "
                   f"Results provide high confidence in findings.")
    
    def generate_time_based_insights(self, rolling_correlations: List[float], 
                                   time_periods: List[str]) -> str:
        """
        Generate insights about how correlations change over time.
        
        Args:
            rolling_correlations: List of correlation values over time
            time_periods: List of time period labels
            
        Returns:
            Time-based insight string
        """
        if not rolling_correlations or len(rolling_correlations) < 2:
            return "Insufficient data for time-based correlation analysis."
        
        # Calculate trend
        correlations = np.array(rolling_correlations)
        valid_correlations = correlations[~np.isnan(correlations)]
        
        if len(valid_correlations) < 2:
            return "Insufficient valid data for time-based analysis."
        
        # Simple trend analysis
        first_half = valid_correlations[:len(valid_correlations)//2]
        second_half = valid_correlations[len(valid_correlations)//2:]
        
        first_avg = np.mean(first_half)
        second_avg = np.mean(second_half)
        
        trend_change = second_avg - first_avg
        
        if abs(trend_change) < 0.05:
            trend_desc = "remained relatively stable"
        elif trend_change > 0:
            trend_desc = f"strengthened by {trend_change:.3f}"
        else:
            trend_desc = f"weakened by {abs(trend_change):.3f}"
        
        max_corr = np.max(valid_correlations)
        min_corr = np.min(valid_correlations)
        variability = max_corr - min_corr
        
        insight = (f"Over the analysis period, the traffic-pollution correlation has {trend_desc}. "
                  f"Correlation values ranged from {min_corr:.3f} to {max_corr:.3f}, "
                  f"showing {'high' if variability > 0.3 else 'moderate' if variability > 0.1 else 'low'} variability.")
        
        return insight