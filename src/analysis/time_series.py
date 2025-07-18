"""
Time series analysis utilities for mental health data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    """Analyze time series patterns in mental health data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the time series analyzer.
        
        Args:
            data: Mental health dataframe with time series data
        """
        self.data = data.copy()
        self.ensure_datetime()
    
    def ensure_datetime(self):
        """Ensure year column is in datetime format."""
        if 'year' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['year'], format='%Y')
    
    def calculate_trends(self, countries: Optional[List[str]] = None, 
                        metric: str = 'depression_prevalence') -> pd.DataFrame:
        """
        Calculate trend statistics for specified countries and metric.
        
        Args:
            countries: List of countries to analyze (all if None)
            metric: Mental health metric to analyze
            
        Returns:
            pd.DataFrame: Trend statistics by country
        """
        if countries is None:
            countries = self.data['country'].unique()
        
        if metric not in self.data.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        trend_results = []
        
        for country in countries:
            country_data = self.data[self.data['country'] == country].copy()
            
            if len(country_data) < 3:  # Need at least 3 points for trend
                continue
            
            country_data = country_data.sort_values('year')
            x = country_data['year'].values
            y = country_data[metric].dropna().values
            
            if len(y) < 3:
                continue
            
            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x[:len(y)], y
            )
            
            # Calculate additional statistics
            start_value = y[0] if len(y) > 0 else np.nan
            end_value = y[-1] if len(y) > 0 else np.nan
            percent_change = ((end_value - start_value) / start_value * 100) if start_value != 0 else np.nan
            
            # Detect trend direction
            if abs(slope) < 0.01:
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
            
            trend_results.append({
                'country': country,
                'metric': metric,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_direction': trend_direction,
                'start_value': start_value,
                'end_value': end_value,
                'percent_change': percent_change,
                'data_points': len(y)
            })
        
        return pd.DataFrame(trend_results)
    
    def detect_seasonality(self, country: str, metric: str) -> Dict:
        """
        Detect seasonal patterns in mental health data.
        
        Args:
            country: Country to analyze
            metric: Mental health metric to analyze
            
        Returns:
            Dict: Seasonality analysis results
        """
        country_data = self.data[self.data['country'] == country].copy()
        
        if len(country_data) < 12:  # Need sufficient data for seasonality
            return {'has_seasonality': False, 'reason': 'Insufficient data'}
        
        country_data = country_data.sort_values('year')
        values = country_data[metric].dropna().values
        
        if len(values) < 12:
            return {'has_seasonality': False, 'reason': 'Insufficient non-null values'}
        
        # Since we have yearly data, check for cyclic patterns
        # Use autocorrelation to detect cycles
        from scipy.stats import pearsonr
        
        autocorr_results = []
        for lag in range(1, min(10, len(values)//2)):
            if len(values) > lag:
                corr, p_val = pearsonr(values[:-lag], values[lag:])
                autocorr_results.append({'lag': lag, 'correlation': corr, 'p_value': p_val})
        
        # Look for significant autocorrelations
        significant_lags = [r for r in autocorr_results if r['p_value'] < 0.05 and abs(r['correlation']) > 0.3]
        
        return {
            'has_seasonality': len(significant_lags) > 0,
            'autocorrelations': autocorr_results,
            'significant_lags': significant_lags
        }
    
    def analyze_volatility(self, countries: Optional[List[str]] = None,
                          metric: str = 'depression_prevalence') -> pd.DataFrame:
        """
        Analyze volatility in mental health metrics.
        
        Args:
            countries: List of countries to analyze
            metric: Mental health metric to analyze
            
        Returns:
            pd.DataFrame: Volatility statistics
        """
        if countries is None:
            countries = self.data['country'].unique()
        
        volatility_results = []
        
        for country in countries:
            country_data = self.data[self.data['country'] == country].copy()
            country_data = country_data.sort_values('year')
            
            values = country_data[metric].dropna().values
            
            if len(values) < 3:
                continue
            
            # Calculate year-over-year changes
            yoy_changes = np.diff(values) / values[:-1] * 100
            
            volatility_stats = {
                'country': country,
                'metric': metric,
                'volatility_std': np.std(yoy_changes),
                'volatility_mean': np.mean(np.abs(yoy_changes)),
                'max_increase': np.max(yoy_changes) if len(yoy_changes) > 0 else np.nan,
                'max_decrease': np.min(yoy_changes) if len(yoy_changes) > 0 else np.nan,
                'coefficient_of_variation': np.std(values) / np.mean(values) * 100
            }
            
            volatility_results.append(volatility_stats)
        
        return pd.DataFrame(volatility_results)
    
    def smooth_time_series(self, country: str, metric: str, 
                          method: str = 'savgol') -> pd.DataFrame:
        """
        Apply smoothing to time series data.
        
        Args:
            country: Country to analyze
            metric: Mental health metric to smooth
            method: Smoothing method ('savgol', 'rolling', 'ewm')
            
        Returns:
            pd.DataFrame: Data with smoothed values
        """
        country_data = self.data[self.data['country'] == country].copy()
        country_data = country_data.sort_values('year')
        
        values = country_data[metric].values
        
        if method == 'savgol' and len(values) >= 5:
            # Savitzky-Golay filter
            window_length = min(5, len(values) if len(values) % 2 == 1 else len(values) - 1)
            if window_length >= 3:
                smoothed = savgol_filter(values, window_length, 2)
                country_data[f'{metric}_smoothed'] = smoothed
        
        elif method == 'rolling':
            # Rolling average
            window = min(3, len(values) // 3)
            country_data[f'{metric}_smoothed'] = country_data[metric].rolling(
                window=window, center=True, min_periods=1
            ).mean()
        
        elif method == 'ewm':
            # Exponential weighted moving average
            country_data[f'{metric}_smoothed'] = country_data[metric].ewm(
                span=3, adjust=False
            ).mean()
        
        return country_data
    
    def compare_countries(self, countries: List[str], 
                         metric: str = 'depression_prevalence') -> pd.DataFrame:
        """
        Compare time series between countries.
        
        Args:
            countries: List of countries to compare
            metric: Mental health metric to compare
            
        Returns:
            pd.DataFrame: Comparison statistics
        """
        comparison_data = []
        
        for country in countries:
            country_data = self.data[self.data['country'] == country]
            
            values = country_data[metric].dropna()
            
            if len(values) == 0:
                continue
            
            stats_dict = {
                'country': country,
                'metric': metric,
                'mean': values.mean(),
                'median': values.median(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'latest_value': values.iloc[-1] if len(values) > 0 else np.nan,
                'first_value': values.iloc[0] if len(values) > 0 else np.nan,
                'data_points': len(values)
            }
            
            comparison_data.append(stats_dict)
        
        return pd.DataFrame(comparison_data)
    
    def identify_outliers(self, metric: str = 'depression_prevalence',
                         method: str = 'iqr') -> pd.DataFrame:
        """
        Identify outliers in the time series data.
        
        Args:
            metric: Mental health metric to analyze
            method: Outlier detection method ('iqr', 'zscore')
            
        Returns:
            pd.DataFrame: Data points identified as outliers
        """
        outliers = []
        
        for country in self.data['country'].unique():
            country_data = self.data[self.data['country'] == country].copy()
            values = country_data[metric].dropna()
            
            if len(values) < 4:  # Need sufficient data
                continue
            
            if method == 'iqr':
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (values < lower_bound) | (values > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(values))
                outlier_mask = z_scores > 2  # 2 standard deviations
            
            outlier_data = country_data[country_data[metric].isin(values[outlier_mask])]
            
            for _, row in outlier_data.iterrows():
                outliers.append({
                    'country': row['country'],
                    'year': row['year'],
                    'metric': metric,
                    'value': row[metric],
                    'method': method
                })
        
        return pd.DataFrame(outliers)
    
    def analyze_trend(self, x_values, y_values) -> Dict:
        """
        Analyze trend in time series data.
        
        Args:
            x_values: Time values (years)
            y_values: Data values (prevalence)
            
        Returns:
            Dict: Trend analysis results
        """
        if len(x_values) != len(y_values) or len(x_values) < 3:
            return {
                'slope': 0,
                'r_squared': 0,
                'p_value': 1,
                'trend_direction': 'insufficient_data',
                'is_significant': False
            }
        
        # Remove any NaN values
        mask = ~(np.isnan(x_values) | np.isnan(y_values))
        x_clean = np.array(x_values)[mask]
        y_clean = np.array(y_values)[mask]
        
        if len(x_clean) < 3:
            return {
                'slope': 0,
                'r_squared': 0,
                'p_value': 1,
                'trend_direction': 'insufficient_data',
                'is_significant': False
            }
        
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'trend_direction': trend_direction,
            'is_significant': p_value < 0.05
        }
    
    def calculate_correlations(self, metrics: List[str]) -> pd.DataFrame:
        """
        Calculate correlations between different mental health metrics.
        
        Args:
            metrics: List of mental health metrics to correlate
            
        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Filter data to only include specified metrics
        available_metrics = [m for m in metrics if m in self.data.columns]
        
        if len(available_metrics) < 2:
            raise ValueError("Need at least 2 available metrics for correlation")
        
        correlation_data = self.data[available_metrics].corr()
        
        return correlation_data

def perform_time_series_analysis(data: pd.DataFrame) -> Dict:
    """
    Perform comprehensive time series analysis.
    
    Args:
        data: Mental health dataframe
        
    Returns:
        Dict: Complete analysis results
    """
    analyzer = TimeSeriesAnalyzer(data)
    
    results = {}
    
    # Get available metrics
    metrics = [col for col in data.columns if 'prevalence' in col]
    countries = data['country'].unique()[:10]  # Analyze top 10 countries
    
    # Calculate trends for main metrics
    for metric in metrics[:2]:  # Analyze first 2 metrics
        if metric in data.columns:
            trends = analyzer.calculate_trends(countries, metric)
            results[f'{metric}_trends'] = trends
    
    # Volatility analysis
    if metrics:
        volatility = analyzer.analyze_volatility(countries, metrics[0])
        results['volatility_analysis'] = volatility
    
    # Correlation analysis
    if len(metrics) > 1:
        correlations = analyzer.calculate_correlations(metrics)
        results['metric_correlations'] = correlations
    
    # Outlier detection
    if metrics:
        outliers = analyzer.identify_outliers(metrics[0])
        results['outliers'] = outliers
    
    return results
