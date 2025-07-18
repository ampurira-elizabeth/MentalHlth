"""
Static plotting utilities using matplotlib and seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class StaticPlotter:
    """Create static visualizations for mental health data."""
    
    def __init__(self, data: pd.DataFrame, output_dir: str = "visualizations"):
        """
        Initialize the static plotter.
        
        Args:
            data: Mental health dataframe
            output_dir: Directory to save plots
        """
        self.data = data.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def plot_time_series(self, countries: List[str], metric: str,
                        save: bool = True, filename: Optional[str] = None) -> plt.Figure:
        """
        Create time series plot for multiple countries.
        
        Args:
            countries: List of countries to plot
            metric: Mental health metric to plot
            save: Whether to save the plot
            filename: Custom filename for saving
            
        Returns:
            plt.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = sns.color_palette("husl", len(countries))
        
        for i, country in enumerate(countries):
            country_data = self.data[self.data['country'] == country].copy()
            country_data = country_data.sort_values('year')
            
            if len(country_data) > 0:
                ax.plot(country_data['year'], country_data[metric], 
                       marker='o', linewidth=2, markersize=4,
                       label=country, color=colors[i])
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{metric.replace("_", " ").title()} Trends Over Time', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f"time_series_{metric}_{len(countries)}_countries.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_country_comparison(self, countries: List[str], metrics: List[str],
                              year: Optional[int] = None,
                              save: bool = True) -> plt.Figure:
        """
        Create comparison plot between countries for multiple metrics.
        
        Args:
            countries: List of countries to compare
            metrics: List of metrics to compare
            year: Specific year to compare (latest if None)
            save: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        # Get data for comparison
        if year is None:
            # Use the latest year with data for each country
            comparison_data = []
            for country in countries:
                country_data = self.data[self.data['country'] == country]
                if len(country_data) > 0:
                    latest_data = country_data.loc[country_data['year'].idxmax()]
                    comparison_data.append(latest_data)
        else:
            comparison_data = self.data[
                (self.data['country'].isin(countries)) & 
                (self.data['year'] == year)
            ]
        
        if len(comparison_data) == 0:
            print("No data available for comparison")
            return None
        
        df_compare = pd.DataFrame(comparison_data)
        
        # Create subplot for each metric
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 8))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in df_compare.columns:
                # Sort countries by metric value
                metric_data = df_compare[['country', metric]].dropna()
                metric_data = metric_data.sort_values(metric, ascending=True)
                
                bars = axes[i].barh(metric_data['country'], metric_data[metric],
                                   color=sns.color_palette("viridis", len(metric_data)))
                
                axes[i].set_xlabel(metric.replace('_', ' ').title())
                axes[i].set_title(f'{metric.replace("_", " ").title()}\n({year if year else "Latest Year"})')
                
                # Add value labels on bars
                for j, bar in enumerate(bars):
                    width = bar.get_width()
                    axes[i].text(width + max(metric_data[metric]) * 0.01, 
                               bar.get_y() + bar.get_height()/2,
                               f'{width:.2f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save:
            filename = f"country_comparison_{year if year else 'latest'}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_correlation_matrix(self, metrics: List[str], 
                               save: bool = True) -> plt.Figure:
        """
        Create correlation matrix heatmap.
        
        Args:
            metrics: List of metrics to correlate
            save: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        # Calculate correlation matrix
        available_metrics = [m for m in metrics if m in self.data.columns]
        
        if len(available_metrics) < 2:
            print("Need at least 2 metrics for correlation matrix")
            return None
        
        corr_matrix = self.data[available_metrics].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.3f', ax=ax)
        
        ax.set_title('Mental Health Metrics Correlation Matrix', 
                    fontsize=14, fontweight='bold')
        
        # Clean up labels
        labels = [label.get_text().replace('_', ' ').title() for label in ax.get_xticklabels()]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "correlation_matrix.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_distribution(self, metric: str, by_region: bool = False,
                         save: bool = True) -> plt.Figure:
        """
        Create distribution plots for a metric.
        
        Args:
            metric: Mental health metric to plot
            by_region: Whether to split by region
            save: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        data_to_plot = self.data[metric].dropna()
        
        # Histogram
        axes[0, 0].hist(data_to_plot, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title(f'Distribution of {metric.replace("_", " ").title()}')
        axes[0, 0].set_xlabel(metric.replace('_', ' ').title())
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        if by_region and 'region' in self.data.columns:
            region_data = []
            region_labels = []
            for region in self.data['region'].unique():
                if pd.notna(region):
                    region_values = self.data[self.data['region'] == region][metric].dropna()
                    if len(region_values) > 0:
                        region_data.append(region_values)
                        region_labels.append(region)
            
            if region_data:
                axes[0, 1].boxplot(region_data, labels=region_labels)
                axes[0, 1].set_title(f'{metric.replace("_", " ").title()} by Region')
                axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].boxplot([data_to_plot])
            axes[0, 1].set_title(f'{metric.replace("_", " ").title()} Box Plot')
            axes[0, 1].set_xticklabels(['All Data'])
        
        # Q-Q plot for normality check
        from scipy import stats
        stats.probplot(data_to_plot, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Check)')
        
        # Violin plot by region
        if by_region and 'region' in self.data.columns:
            sns.violinplot(data=self.data, x='region', y=metric, ax=axes[1, 1])
            axes[1, 1].set_title(f'{metric.replace("_", " ").title()} Distribution by Region')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            sns.violinplot(y=data_to_plot, ax=axes[1, 1])
            axes[1, 1].set_title(f'{metric.replace("_", " ").title()} Violin Plot')
        
        plt.tight_layout()
        
        if save:
            filename = f"distribution_{metric}{'_by_region' if by_region else ''}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_trends_analysis(self, trend_results: pd.DataFrame,
                           save: bool = True) -> plt.Figure:
        """
        Create visualization of trend analysis results.
        
        Args:
            trend_results: DataFrame with trend analysis results
            save: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Slope distribution
        axes[0, 0].hist(trend_results['slope'], bins=20, alpha=0.7, 
                       color='lightgreen', edgecolor='black')
        axes[0, 0].set_title('Distribution of Trend Slopes')
        axes[0, 0].set_xlabel('Slope (change per year)')
        axes[0, 0].set_ylabel('Number of Countries')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # R-squared vs Slope
        scatter = axes[0, 1].scatter(trend_results['slope'], trend_results['r_squared'],
                                   c=trend_results['data_points'], cmap='viridis',
                                   alpha=0.7, s=50)
        axes[0, 1].set_xlabel('Slope')
        axes[0, 1].set_ylabel('R-squared')
        axes[0, 1].set_title('Trend Quality vs. Slope')
        plt.colorbar(scatter, ax=axes[0, 1], label='Data Points')
        
        # Trend direction pie chart
        trend_counts = trend_results['trend_direction'].value_counts()
        axes[1, 0].pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%',
                      colors=['lightcoral', 'lightblue', 'lightgreen'])
        axes[1, 0].set_title('Distribution of Trend Directions')
        
        # Percent change distribution
        axes[1, 1].hist(trend_results['percent_change'].dropna(), bins=20, 
                       alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_title('Distribution of Percent Changes')
        axes[1, 1].set_xlabel('Percent Change (%)')
        axes[1, 1].set_ylabel('Number of Countries')
        axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "trends_analysis.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_forecast_results(self, forecast_results: Dict, country: str,
                            save: bool = True) -> plt.Figure:
        """
        Plot forecasting results for a country.
        
        Args:
            forecast_results: Forecast results dictionary
            country: Country name
            save: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        if 'error' in forecast_results:
            print(f"Error in forecast results: {forecast_results['error']}")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Historical data
        hist_years = forecast_results['historical_years']
        hist_values = forecast_results['historical_values']
        
        ax.plot(hist_years, hist_values, 'o-', linewidth=2, markersize=6,
               label='Historical Data', color='blue')
        
        # Forecast data
        forecast_years = forecast_results['forecast_years']
        
        if 'forecast_values' in forecast_results:
            forecast_values = forecast_results['forecast_values']
            ax.plot(forecast_years, forecast_values, 's--', linewidth=2, markersize=6,
                   label='Forecast', color='red', alpha=0.8)
        
        # If ensemble forecast, plot multiple forecasts
        if forecast_results.get('model_type') == 'ensemble':
            if 'simple_average_forecast' in forecast_results:
                ax.plot(forecast_years, forecast_results['simple_average_forecast'],
                       '^--', linewidth=2, markersize=4,
                       label='Simple Average', color='green', alpha=0.6)
            
            if 'weighted_average_forecast' in forecast_results:
                ax.plot(forecast_years, forecast_results['weighted_average_forecast'],
                       'd--', linewidth=2, markersize=4,
                       label='Weighted Average', color='orange', alpha=0.6)
        
        # Add smoothed line if available
        if 'smoothed_values' in forecast_results:
            ax.plot(hist_years, forecast_results['smoothed_values'],
                   '-', linewidth=1, alpha=0.6,
                   label='Smoothed', color='gray')
        
        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(forecast_results.get('metric', 'Value').replace('_', ' ').title(), 
                     fontsize=12)
        ax.set_title(f'{forecast_results.get("metric", "Metric").replace("_", " ").title()} '
                    f'Forecast for {country}\n'
                    f'Model: {forecast_results.get("model_type", "Unknown")}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add vertical line to separate historical and forecast
        if hist_years and forecast_years:
            separation_year = max(hist_years) + 0.5
            ax.axvline(x=separation_year, color='gray', linestyle=':', alpha=0.5)
            ax.text(separation_year, ax.get_ylim()[1] * 0.9, 'Forecast →', 
                   rotation=0, ha='center', va='center', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            model_type = forecast_results.get('model_type', 'unknown')
            metric = forecast_results.get('metric', 'metric')
            filename = f"forecast_{country}_{metric}_{model_type}.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_summary_dashboard(self, metrics: List[str], countries: List[str],
                               save: bool = True) -> plt.Figure:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            metrics: List of metrics to include
            countries: List of countries to highlight
            save: Whether to save the plot
            
        Returns:
            plt.Figure: The created figure
        """
        fig = plt.figure(figsize=(20, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Time series for main metric (top left, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if metrics and len(countries) > 0:
            for i, country in enumerate(countries[:5]):  # Top 5 countries
                country_data = self.data[self.data['country'] == country].sort_values('year')
                if len(country_data) > 0 and metrics[0] in country_data.columns:
                    ax1.plot(country_data['year'], country_data[metrics[0]], 
                           'o-', label=country, linewidth=2, markersize=4)
            
            ax1.set_title(f'{metrics[0].replace("_", " ").title()} Trends', fontweight='bold')
            ax1.set_xlabel('Year')
            ax1.set_ylabel(metrics[0].replace('_', ' ').title())
            ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax1.grid(True, alpha=0.3)
        
        # 2. Country comparison bar chart (top right, spanning 2 columns)
        ax2 = fig.add_subplot(gs[0, 2:])
        if metrics and len(countries) > 0:
            latest_data = []
            for country in countries:
                country_data = self.data[self.data['country'] == country]
                if len(country_data) > 0:
                    latest = country_data.loc[country_data['year'].idxmax()]
                    latest_data.append({'country': country, 'value': latest.get(metrics[0], 0)})
            
            if latest_data:
                df_latest = pd.DataFrame(latest_data).sort_values('value')
                bars = ax2.barh(df_latest['country'], df_latest['value'],
                               color=sns.color_palette("viridis", len(df_latest)))
                ax2.set_title(f'Latest {metrics[0].replace("_", " ").title()} by Country', 
                             fontweight='bold')
                ax2.set_xlabel(metrics[0].replace('_', ' ').title())
        
        # 3. Correlation heatmap (middle left, spanning 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        available_metrics = [m for m in metrics if m in self.data.columns]
        if len(available_metrics) >= 2:
            corr_matrix = self.data[available_metrics].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       ax=ax3, cbar_kws={"shrink": .8}, fmt='.2f')
            ax3.set_title('Metrics Correlation Matrix', fontweight='bold')
        
        # 4. Distribution plot (middle right, spanning 2 columns)
        ax4 = fig.add_subplot(gs[1, 2:])
        if metrics:
            data_to_plot = self.data[metrics[0]].dropna()
            ax4.hist(data_to_plot, bins=25, alpha=0.7, color='lightblue', edgecolor='black')
            ax4.set_title(f'{metrics[0].replace("_", " ").title()} Distribution', 
                         fontweight='bold')
            ax4.set_xlabel(metrics[0].replace('_', ' ').title())
            ax4.set_ylabel('Frequency')
        
        # 5. Regional analysis (bottom left)
        ax5 = fig.add_subplot(gs[2, :2])
        if 'region' in self.data.columns and metrics:
            regional_means = self.data.groupby('region')[metrics[0]].mean().sort_values()
            bars = ax5.bar(range(len(regional_means)), regional_means.values,
                          color=sns.color_palette("Set2", len(regional_means)))
            ax5.set_xticks(range(len(regional_means)))
            ax5.set_xticklabels(regional_means.index, rotation=45, ha='right')
            ax5.set_title(f'Average {metrics[0].replace("_", " ").title()} by Region', 
                         fontweight='bold')
            ax5.set_ylabel(metrics[0].replace('_', ' ').title())
        
        # 6. Time trend summary (bottom right)
        ax6 = fig.add_subplot(gs[2, 2:])
        yearly_means = self.data.groupby('year')[metrics[0]].mean()
        if len(yearly_means) > 1:
            ax6.plot(yearly_means.index, yearly_means.values, 'o-', 
                    linewidth=3, markersize=6, color='red')
            ax6.set_title(f'Global Average {metrics[0].replace("_", " ").title()} Trend', 
                         fontweight='bold')
            ax6.set_xlabel('Year')
            ax6.set_ylabel(f'Average {metrics[0].replace("_", " ").title()}')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Mental Health Analysis Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        if save:
            plt.savefig(self.output_dir / "summary_dashboard.png", 
                       dpi=300, bbox_inches='tight')
        
        return fig

def create_all_static_plots(data: pd.DataFrame, output_dir: str = "visualizations") -> None:
    """
    Create all static plots for the mental health analysis.
    
    Args:
        data: Mental health dataframe
        output_dir: Directory to save plots
    """
    plotter = StaticPlotter(data, output_dir)
    
    # Get available metrics and countries
    metrics = [col for col in data.columns if 'prevalence' in col]
    countries = data['country'].value_counts().head(10).index.tolist()
    
    print("Creating static visualizations...")
    
    try:
        # Time series plots
        if metrics and countries:
            plotter.plot_time_series(countries[:5], metrics[0])
            print("✓ Time series plot created")
        
        # Country comparison
        if metrics and countries:
            plotter.plot_country_comparison(countries[:8], metrics[:2])
            print("✓ Country comparison plot created")
        
        # Correlation matrix
        if len(metrics) >= 2:
            plotter.plot_correlation_matrix(metrics)
            print("✓ Correlation matrix created")
        
        # Distribution plots
        if metrics:
            plotter.plot_distribution(metrics[0], by_region=True)
            print("✓ Distribution plots created")
        
        # Summary dashboard
        if metrics and countries:
            plotter.create_summary_dashboard(metrics, countries)
            print("✓ Summary dashboard created")
        
        print(f"\nAll plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()
