"""
Forecasting models for mental health trends.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced forecasting libraries
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class ForecastingModel:
    """Create forecasting models for mental health trends."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the forecasting model.
        
        Args:
            data: Mental health dataframe with time series data
        """
        self.data = data.copy()
        self.models = {}
        self.forecasts = {}
    
    def prepare_data(self, country: str, metric: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for forecasting.
        
        Args:
            country: Country to forecast
            metric: Mental health metric to forecast
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Years and values for modeling
        """
        country_data = self.data[self.data['country'] == country].copy()
        country_data = country_data.sort_values('year')
        
        # Remove missing values
        country_data = country_data.dropna(subset=[metric])
        
        if len(country_data) < 3:
            raise ValueError(f"Insufficient data for {country} - {metric}")
        
        years = country_data['year'].values
        values = country_data[metric].values
        
        return years, values
    
    def linear_trend_forecast(self, country: str, metric: str, 
                            forecast_years: int = 5) -> Dict:
        """
        Create linear trend forecast.
        
        Args:
            country: Country to forecast
            metric: Mental health metric to forecast
            forecast_years: Number of years to forecast
            
        Returns:
            Dict: Forecast results
        """
        try:
            years, values = self.prepare_data(country, metric)
            
            if SKLEARN_AVAILABLE:
                # Use sklearn for linear regression
                X = years.reshape(-1, 1)
                y = values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Generate forecasts
                future_years = np.arange(years.max() + 1, years.max() + 1 + forecast_years)
                future_X = future_years.reshape(-1, 1)
                forecast_values = model.predict(future_X)
                
                # Calculate metrics
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                return {
                    'model_type': 'linear_trend',
                    'country': country,
                    'metric': metric,
                    'historical_years': years.tolist(),
                    'historical_values': values.tolist(),
                    'forecast_years': future_years.tolist(),
                    'forecast_values': forecast_values.tolist(),
                    'model_metrics': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'slope': model.coef_[0],
                        'intercept': model.intercept_
                    }
                }
            else:
                # Simple linear regression without sklearn
                from scipy.stats import linregress
                
                slope, intercept, r_value, p_value, std_err = linregress(years, values)
                
                # Generate forecasts
                future_years = np.arange(years.max() + 1, years.max() + 1 + forecast_years)
                forecast_values = slope * future_years + intercept
                
                # Calculate simple metrics
                y_pred = slope * years + intercept
                mse = np.mean((values - y_pred) ** 2)
                mae = np.mean(np.abs(values - y_pred))
                
                return {
                    'model_type': 'linear_trend',
                    'country': country,
                    'metric': metric,
                    'historical_years': years.tolist(),
                    'historical_values': values.tolist(),
                    'forecast_years': future_years.tolist(),
                    'forecast_values': forecast_values.tolist(),
                    'model_metrics': {
                        'mse': mse,
                        'mae': mae,
                        'r2': r_value**2,
                        'slope': slope,
                        'intercept': intercept,
                        'p_value': p_value
                    }
                }
                
        except Exception as e:
            return {
                'error': f"Linear trend forecasting failed: {str(e)}",
                'country': country,
                'metric': metric
            }
    
    def polynomial_forecast(self, country: str, metric: str, 
                          degree: int = 2, forecast_years: int = 5) -> Dict:
        """
        Create polynomial trend forecast.
        
        Args:
            country: Country to forecast
            metric: Mental health metric to forecast
            degree: Polynomial degree
            forecast_years: Number of years to forecast
            
        Returns:
            Dict: Forecast results
        """
        if not SKLEARN_AVAILABLE:
            return {
                'error': 'Sklearn not available for polynomial forecasting',
                'country': country,
                'metric': metric
            }
        
        try:
            years, values = self.prepare_data(country, metric)
            
            if len(years) <= degree:
                return {
                    'error': f'Need more data points than polynomial degree ({degree})',
                    'country': country,
                    'metric': metric
                }
            
            # Prepare polynomial features
            poly_features = PolynomialFeatures(degree=degree)
            X = years.reshape(-1, 1)
            X_poly = poly_features.fit_transform(X)
            
            # Fit model
            model = LinearRegression()
            model.fit(X_poly, values)
            
            # Generate forecasts
            future_years = np.arange(years.max() + 1, years.max() + 1 + forecast_years)
            future_X = future_years.reshape(-1, 1)
            future_X_poly = poly_features.transform(future_X)
            forecast_values = model.predict(future_X_poly)
            
            # Calculate metrics
            y_pred = model.predict(X_poly)
            mse = mean_squared_error(values, y_pred)
            mae = mean_absolute_error(values, y_pred)
            r2 = r2_score(values, y_pred)
            
            return {
                'model_type': f'polynomial_degree_{degree}',
                'country': country,
                'metric': metric,
                'historical_years': years.tolist(),
                'historical_values': values.tolist(),
                'forecast_years': future_years.tolist(),
                'forecast_values': forecast_values.tolist(),
                'model_metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'degree': degree
                }
            }
            
        except Exception as e:
            return {
                'error': f"Polynomial forecasting failed: {str(e)}",
                'country': country,
                'metric': metric
            }
    
    def exponential_smoothing_forecast(self, country: str, metric: str,
                                     alpha: float = 0.3, 
                                     forecast_years: int = 5) -> Dict:
        """
        Create exponential smoothing forecast.
        
        Args:
            country: Country to forecast
            metric: Mental health metric to forecast
            alpha: Smoothing parameter
            forecast_years: Number of years to forecast
            
        Returns:
            Dict: Forecast results
        """
        try:
            years, values = self.prepare_data(country, metric)
            
            # Simple exponential smoothing
            smoothed = np.zeros_like(values)
            smoothed[0] = values[0]
            
            for i in range(1, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
            
            # Forecast using last smoothed value
            last_smoothed = smoothed[-1]
            forecast_values = np.full(forecast_years, last_smoothed)
            
            # Add slight trend if present
            if len(values) >= 3:
                recent_trend = (values[-1] - values[-3]) / 2
                for i in range(forecast_years):
                    forecast_values[i] = last_smoothed + recent_trend * (i + 1)
            
            # Generate future years
            future_years = np.arange(years.max() + 1, years.max() + 1 + forecast_years)
            
            # Calculate metrics
            mse = np.mean((values[1:] - smoothed[1:]) ** 2)
            mae = np.mean(np.abs(values[1:] - smoothed[1:]))
            
            return {
                'model_type': 'exponential_smoothing',
                'country': country,
                'metric': metric,
                'historical_years': years.tolist(),
                'historical_values': values.tolist(),
                'smoothed_values': smoothed.tolist(),
                'forecast_years': future_years.tolist(),
                'forecast_values': forecast_values.tolist(),
                'model_metrics': {
                    'mse': mse,
                    'mae': mae,
                    'alpha': alpha
                }
            }
            
        except Exception as e:
            return {
                'error': f"Exponential smoothing failed: {str(e)}",
                'country': country,
                'metric': metric
            }
    
    def moving_average_forecast(self, country: str, metric: str,
                              window: int = 3, forecast_years: int = 5) -> Dict:
        """
        Create moving average forecast.
        
        Args:
            country: Country to forecast
            metric: Mental health metric to forecast
            window: Moving average window size
            forecast_years: Number of years to forecast
            
        Returns:
            Dict: Forecast results
        """
        try:
            years, values = self.prepare_data(country, metric)
            
            if len(values) < window:
                return {
                    'error': f'Need at least {window} data points for moving average',
                    'country': country,
                    'metric': metric
                }
            
            # Calculate moving average
            moving_avg = []
            for i in range(len(values)):
                if i < window - 1:
                    moving_avg.append(np.mean(values[:i+1]))
                else:
                    moving_avg.append(np.mean(values[i-window+1:i+1]))
            
            # Forecast using last window average
            last_avg = np.mean(values[-window:])
            forecast_values = np.full(forecast_years, last_avg)
            
            # Generate future years
            future_years = np.arange(years.max() + 1, years.max() + 1 + forecast_years)
            
            # Calculate metrics
            mse = np.mean((values - moving_avg) ** 2)
            mae = np.mean(np.abs(values - moving_avg))
            
            return {
                'model_type': f'moving_average_window_{window}',
                'country': country,
                'metric': metric,
                'historical_years': years.tolist(),
                'historical_values': values.tolist(),
                'moving_average': moving_avg,
                'forecast_years': future_years.tolist(),
                'forecast_values': forecast_values.tolist(),
                'model_metrics': {
                    'mse': mse,
                    'mae': mae,
                    'window': window
                }
            }
            
        except Exception as e:
            return {
                'error': f"Moving average forecasting failed: {str(e)}",
                'country': country,
                'metric': metric
            }
    
    def ensemble_forecast(self, country: str, metric: str,
                         forecast_years: int = 5) -> Dict:
        """
        Create ensemble forecast combining multiple methods.
        
        Args:
            country: Country to forecast
            metric: Mental health metric to forecast
            forecast_years: Number of years to forecast
            
        Returns:
            Dict: Ensemble forecast results
        """
        try:
            # Get individual forecasts
            forecasts = []
            
            # Linear trend
            linear_result = self.linear_trend_forecast(country, metric, forecast_years)
            if 'error' not in linear_result:
                forecasts.append(linear_result)
            
            # Exponential smoothing
            exp_result = self.exponential_smoothing_forecast(country, metric, forecast_years)
            if 'error' not in exp_result:
                forecasts.append(exp_result)
            
            # Moving average
            ma_result = self.moving_average_forecast(country, metric, forecast_years)
            if 'error' not in ma_result:
                forecasts.append(ma_result)
            
            if len(forecasts) == 0:
                return {
                    'error': 'No forecasting methods succeeded',
                    'country': country,
                    'metric': metric
                }
            
            # Combine forecasts (simple average)
            forecast_arrays = [np.array(f['forecast_values']) for f in forecasts]
            ensemble_forecast = np.mean(forecast_arrays, axis=0)
            
            # Calculate weights based on model performance (R²)
            weights = []
            for forecast in forecasts:
                r2 = forecast.get('model_metrics', {}).get('r2', 0)
                weights.append(max(r2, 0))  # Ensure non-negative weights
            
            # Weighted average if weights are available
            if sum(weights) > 0:
                weights = np.array(weights) / sum(weights)  # Normalize
                weighted_forecast = np.average(forecast_arrays, axis=0, weights=weights)
            else:
                weighted_forecast = ensemble_forecast
            
            years, values = self.prepare_data(country, metric)
            future_years = np.arange(years.max() + 1, years.max() + 1 + forecast_years)
            
            return {
                'model_type': 'ensemble',
                'country': country,
                'metric': metric,
                'historical_years': years.tolist(),
                'historical_values': values.tolist(),
                'forecast_years': future_years.tolist(),
                'simple_average_forecast': ensemble_forecast.tolist(),
                'weighted_average_forecast': weighted_forecast.tolist(),
                'individual_forecasts': forecasts,
                'model_weights': weights.tolist() if len(weights) > 0 else None
            }
            
        except Exception as e:
            return {
                'error': f"Ensemble forecasting failed: {str(e)}",
                'country': country,
                'metric': metric
            }
    
    def forecast_multiple_countries(self, countries: List[str], metric: str,
                                  method: str = 'linear', 
                                  forecast_years: int = 5) -> Dict:
        """
        Forecast for multiple countries.
        
        Args:
            countries: List of countries to forecast
            metric: Mental health metric to forecast
            method: Forecasting method ('linear', 'ensemble', 'exponential')
            forecast_years: Number of years to forecast
            
        Returns:
            Dict: Multiple country forecasts
        """
        results = {}
        
        for country in countries:
            try:
                if method == 'linear':
                    result = self.linear_trend_forecast(country, metric, forecast_years)
                elif method == 'ensemble':
                    result = self.ensemble_forecast(country, metric, forecast_years)
                elif method == 'exponential':
                    result = self.exponential_smoothing_forecast(country, metric, forecast_years)
                else:
                    result = {'error': f'Unknown method: {method}'}
                
                results[country] = result
                
            except Exception as e:
                results[country] = {
                    'error': f'Forecasting failed for {country}: {str(e)}'
                }
        
        return {
            'method': method,
            'metric': metric,
            'forecast_years': forecast_years,
            'country_forecasts': results
        }

def create_forecasts(data: pd.DataFrame, countries: List[str] = None,
                    metrics: List[str] = None) -> Dict:
    """
    Create comprehensive forecasts for mental health data.
    
    Args:
        data: Mental health dataframe
        countries: List of countries to forecast (top 5 if None)
        metrics: List of metrics to forecast (first prevalence metric if None)
        
    Returns:
        Dict: Comprehensive forecast results
    """
    forecaster = ForecastingModel(data)
    
    # Default to top countries by data availability
    if countries is None:
        country_counts = data['country'].value_counts()
        countries = country_counts.head(5).index.tolist()
    
    # Default to first prevalence metric
    if metrics is None:
        prevalence_cols = [col for col in data.columns if 'prevalence' in col]
        metrics = [prevalence_cols[0]] if prevalence_cols else []
    
    if not metrics:
        return {'error': 'No prevalence metrics found in data'}
    
    results = {}
    
    for metric in metrics[:2]:  # Limit to first 2 metrics
        results[metric] = {}
        
        # Linear forecasts
        linear_results = forecaster.forecast_multiple_countries(
            countries, metric, 'linear'
        )
        results[metric]['linear'] = linear_results
        
        # Ensemble forecasts for first few countries
        ensemble_results = forecaster.forecast_multiple_countries(
            countries[:3], metric, 'ensemble'
        )
        results[metric]['ensemble'] = ensemble_results
    
    return results
