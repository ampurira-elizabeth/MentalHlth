"""
Statistical analysis utilities for mental health data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import normaltest, shapiro, levene, ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Perform statistical tests and analysis on mental health data."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the statistical analyzer.
        
        Args:
            data: Mental health dataframe
        """
        self.data = data.copy()
    
    def test_normality(self, metric: str, country: Optional[str] = None) -> Dict:
        """
        Test normality of data distribution.
        
        Args:
            metric: Mental health metric to test
            country: Specific country (all countries if None)
            
        Returns:
            Dict: Normality test results
        """
        if country:
            test_data = self.data[self.data['country'] == country][metric].dropna()
            data_label = f"{metric} - {country}"
        else:
            test_data = self.data[metric].dropna()
            data_label = metric
        
        if len(test_data) < 8:
            return {
                'data_label': data_label,
                'sample_size': len(test_data),
                'error': 'Insufficient data for normality testing (need >= 8 samples)'
            }
        
        results = {
            'data_label': data_label,
            'sample_size': len(test_data),
            'mean': test_data.mean(),
            'std': test_data.std(),
            'skewness': stats.skew(test_data),
            'kurtosis': stats.kurtosis(test_data)
        }
        
        # Shapiro-Wilk test (for smaller samples)
        if len(test_data) <= 5000:
            shapiro_stat, shapiro_p = shapiro(test_data)
            results['shapiro_test'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # D'Agostino's normality test
        if len(test_data) >= 20:
            dagostino_stat, dagostino_p = normaltest(test_data)
            results['dagostino_test'] = {
                'statistic': dagostino_stat,
                'p_value': dagostino_p,
                'is_normal': dagostino_p > 0.05
            }
        
        return results
    
    def compare_groups(self, metric: str, group1_countries: List[str], 
                      group2_countries: List[str], 
                      group_names: Tuple[str, str] = ('Group 1', 'Group 2')) -> Dict:
        """
        Compare mental health metrics between two groups of countries.
        
        Args:
            metric: Mental health metric to compare
            group1_countries: List of countries in group 1
            group2_countries: List of countries in group 2
            group_names: Names for the groups
            
        Returns:
            Dict: Statistical comparison results
        """
        # Extract data for each group
        group1_data = self.data[
            self.data['country'].isin(group1_countries)
        ][metric].dropna()
        
        group2_data = self.data[
            self.data['country'].isin(group2_countries)
        ][metric].dropna()
        
        if len(group1_data) == 0 or len(group2_data) == 0:
            return {
                'error': 'One or both groups have no data',
                'group1_size': len(group1_data),
                'group2_size': len(group2_data)
            }
        
        results = {
            'metric': metric,
            'group1_name': group_names[0],
            'group2_name': group_names[1],
            'group1_size': len(group1_data),
            'group2_size': len(group2_data),
            'group1_stats': {
                'mean': group1_data.mean(),
                'median': group1_data.median(),
                'std': group1_data.std(),
                'min': group1_data.min(),
                'max': group1_data.max()
            },
            'group2_stats': {
                'mean': group2_data.mean(),
                'median': group2_data.median(),
                'std': group2_data.std(),
                'min': group2_data.min(),
                'max': group2_data.max()
            }
        }
        
        # Test for equal variances (Levene's test)
        if len(group1_data) >= 3 and len(group2_data) >= 3:
            levene_stat, levene_p = levene(group1_data, group2_data)
            equal_variances = levene_p > 0.05
            
            results['levene_test'] = {
                'statistic': levene_stat,
                'p_value': levene_p,
                'equal_variances': equal_variances
            }
        else:
            equal_variances = True
        
        # Independent t-test
        if len(group1_data) >= 2 and len(group2_data) >= 2:
            t_stat, t_p = ttest_ind(group1_data, group2_data, equal_var=equal_variances)
            
            results['t_test'] = {
                'statistic': t_stat,
                'p_value': t_p,
                'significant': t_p < 0.05,
                'effect_size': abs(group1_data.mean() - group2_data.mean()) / np.sqrt(
                    ((len(group1_data) - 1) * group1_data.var() + 
                     (len(group2_data) - 1) * group2_data.var()) / 
                    (len(group1_data) + len(group2_data) - 2)
                )
            }
        
        # Mann-Whitney U test (non-parametric alternative)
        if len(group1_data) >= 1 and len(group2_data) >= 1:
            u_stat, u_p = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            
            results['mann_whitney_test'] = {
                'statistic': u_stat,
                'p_value': u_p,
                'significant': u_p < 0.05
            }
        
        return results
    
    def regional_analysis(self, metric: str) -> Dict:
        """
        Perform statistical analysis by region.
        
        Args:
            metric: Mental health metric to analyze
            
        Returns:
            Dict: Regional analysis results
        """
        if 'region' not in self.data.columns:
            return {'error': 'Region column not found in data'}
        
        regions = self.data['region'].unique()
        regional_stats = {}
        
        for region in regions:
            if pd.isna(region):
                continue
                
            region_data = self.data[self.data['region'] == region][metric].dropna()
            
            if len(region_data) == 0:
                continue
            
            regional_stats[region] = {
                'count': len(region_data),
                'mean': region_data.mean(),
                'median': region_data.median(),
                'std': region_data.std(),
                'min': region_data.min(),
                'max': region_data.max(),
                'q25': region_data.quantile(0.25),
                'q75': region_data.quantile(0.75)
            }
        
        # ANOVA test if we have multiple regions with data
        regions_with_data = [r for r in regions if not pd.isna(r) and 
                           len(self.data[self.data['region'] == r][metric].dropna()) > 0]
        
        anova_result = None
        if len(regions_with_data) >= 2:
            try:
                groups = [self.data[self.data['region'] == r][metric].dropna() 
                         for r in regions_with_data]
                
                # Only include groups with at least 2 data points
                groups = [g for g in groups if len(g) >= 2]
                
                if len(groups) >= 2:
                    f_stat, f_p = stats.f_oneway(*groups)
                    anova_result = {
                        'f_statistic': f_stat,
                        'p_value': f_p,
                        'significant': f_p < 0.05,
                        'groups_tested': len(groups)
                    }
            except Exception as e:
                anova_result = {'error': f'ANOVA failed: {str(e)}'}
        
        return {
            'metric': metric,
            'regional_statistics': regional_stats,
            'anova_test': anova_result
        }
    
    def temporal_analysis(self, metric: str) -> Dict:
        """
        Analyze temporal patterns in mental health data.
        
        Args:
            metric: Mental health metric to analyze
            
        Returns:
            Dict: Temporal analysis results
        """
        results = {}
        
        # Analyze by decade
        if 'decade' in self.data.columns:
            decade_stats = {}
            for decade in sorted(self.data['decade'].unique()):
                if pd.isna(decade):
                    continue
                    
                decade_data = self.data[self.data['decade'] == decade][metric].dropna()
                
                if len(decade_data) > 0:
                    decade_stats[int(decade)] = {
                        'count': len(decade_data),
                        'mean': decade_data.mean(),
                        'median': decade_data.median(),
                        'std': decade_data.std()
                    }
            
            results['decade_analysis'] = decade_stats
        
        # Year-over-year correlation
        yearly_means = self.data.groupby('year')[metric].mean()
        
        if len(yearly_means) >= 3:
            # Calculate correlation with time
            years = yearly_means.index.values
            values = yearly_means.values
            
            correlation, p_value = stats.pearsonr(years, values)
            
            results['temporal_correlation'] = {
                'correlation_with_time': correlation,
                'p_value': p_value,
                'trend_direction': 'increasing' if correlation > 0 else 'decreasing',
                'significant': p_value < 0.05
            }
        
        return results
    
    def correlation_analysis(self, metrics: List[str]) -> Dict:
        """
        Perform comprehensive correlation analysis.
        
        Args:
            metrics: List of mental health metrics to analyze
            
        Returns:
            Dict: Correlation analysis results
        """
        available_metrics = [m for m in metrics if m in self.data.columns]
        
        if len(available_metrics) < 2:
            return {'error': 'Need at least 2 available metrics for correlation analysis'}
        
        # Calculate correlation matrix
        correlation_matrix = self.data[available_metrics].corr()
        
        # Calculate p-values for correlations
        n = len(self.data[available_metrics].dropna())
        p_values = np.zeros((len(available_metrics), len(available_metrics)))
        
        for i, metric1 in enumerate(available_metrics):
            for j, metric2 in enumerate(available_metrics):
                if i != j:
                    data1 = self.data[metric1].dropna()
                    data2 = self.data[metric2].dropna()
                    
                    # Find common indices
                    common_idx = data1.index.intersection(data2.index)
                    
                    if len(common_idx) >= 3:
                        _, p_val = stats.pearsonr(data1[common_idx], data2[common_idx])
                        p_values[i, j] = p_val
                    else:
                        p_values[i, j] = np.nan
                else:
                    p_values[i, j] = 0  # Perfect correlation with itself
        
        p_value_matrix = pd.DataFrame(p_values, 
                                     index=available_metrics, 
                                     columns=available_metrics)
        
        # Find significant correlations
        significant_correlations = []
        for i, metric1 in enumerate(available_metrics):
            for j, metric2 in enumerate(available_metrics):
                if i < j:  # Avoid duplicates
                    corr_val = correlation_matrix.iloc[i, j]
                    p_val = p_value_matrix.iloc[i, j]
                    
                    if not np.isnan(p_val) and p_val < 0.05:
                        significant_correlations.append({
                            'metric1': metric1,
                            'metric2': metric2,
                            'correlation': corr_val,
                            'p_value': p_val,
                            'strength': self._interpret_correlation(abs(corr_val))
                        })
        
        return {
            'correlation_matrix': correlation_matrix,
            'p_value_matrix': p_value_matrix,
            'significant_correlations': significant_correlations,
            'sample_size': n
        }
    
    def _interpret_correlation(self, corr_value: float) -> str:
        """
        Interpret correlation strength.
        
        Args:
            corr_value: Absolute correlation value
            
        Returns:
            str: Interpretation of correlation strength
        """
        if corr_value >= 0.7:
            return 'strong'
        elif corr_value >= 0.5:
            return 'moderate'
        elif corr_value >= 0.3:
            return 'weak'
        else:
            return 'very weak'
    
    def outlier_analysis(self, metric: str) -> Dict:
        """
        Comprehensive outlier analysis.
        
        Args:
            metric: Mental health metric to analyze
            
        Returns:
            Dict: Outlier analysis results
        """
        data_series = self.data[metric].dropna()
        
        if len(data_series) < 4:
            return {'error': 'Insufficient data for outlier analysis'}
        
        # IQR method
        Q1 = data_series.quantile(0.25)
        Q3 = data_series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = data_series[(data_series < lower_bound) | (data_series > upper_bound)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(data_series))
        zscore_outliers = data_series[z_scores > 2]
        
        # Modified Z-score method (using median)
        median = data_series.median()
        mad = np.median(np.abs(data_series - median))
        modified_z_scores = 0.6745 * (data_series - median) / mad
        modified_zscore_outliers = data_series[np.abs(modified_z_scores) > 3.5]
        
        return {
            'metric': metric,
            'total_data_points': len(data_series),
            'iqr_method': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(iqr_outliers),
                'outlier_percentage': len(iqr_outliers) / len(data_series) * 100,
                'outlier_values': iqr_outliers.tolist()
            },
            'zscore_method': {
                'outlier_count': len(zscore_outliers),
                'outlier_percentage': len(zscore_outliers) / len(data_series) * 100,
                'outlier_values': zscore_outliers.tolist()
            },
            'modified_zscore_method': {
                'outlier_count': len(modified_zscore_outliers),
                'outlier_percentage': len(modified_zscore_outliers) / len(data_series) * 100,
                'outlier_values': modified_zscore_outliers.tolist()
            }
        }

def perform_statistical_analysis(data: pd.DataFrame) -> Dict:
    """
    Perform comprehensive statistical analysis.
    
    Args:
        data: Mental health dataframe
        
    Returns:
        Dict: Complete statistical analysis results
    """
    analyzer = StatisticalAnalyzer(data)
    
    results = {}
    
    # Get available metrics
    metrics = [col for col in data.columns if 'prevalence' in col]
    
    if not metrics:
        return {'error': 'No prevalence metrics found in data'}
    
    # Normality tests for each metric
    results['normality_tests'] = {}
    for metric in metrics[:3]:  # Test first 3 metrics
        results['normality_tests'][metric] = analyzer.test_normality(metric)
    
    # Regional analysis
    if 'region' in data.columns and metrics:
        results['regional_analysis'] = analyzer.regional_analysis(metrics[0])
    
    # Temporal analysis
    if metrics:
        results['temporal_analysis'] = analyzer.temporal_analysis(metrics[0])
    
    # Correlation analysis
    if len(metrics) > 1:
        results['correlation_analysis'] = analyzer.correlation_analysis(metrics)
    
    # Outlier analysis
    if metrics:
        results['outlier_analysis'] = analyzer.outlier_analysis(metrics[0])
    
    return results
