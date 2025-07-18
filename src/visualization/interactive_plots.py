"""
Interactive plotting utilities using plotly.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class InteractivePlotter:
    """Create interactive visualizations for mental health data."""
    
    def __init__(self, data: pd.DataFrame, output_dir: str = "visualizations"):
        """
        Initialize the interactive plotter.
        
        Args:
            data: Mental health dataframe
            output_dir: Directory to save plots
        """
        self.data = data.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set default template
        self.template = "plotly_white"
    
    def create_interactive_time_series(self, countries: List[str], metric: str,
                                     save: bool = True, filename: Optional[str] = None) -> go.Figure:
        """
        Create interactive time series plot.
        
        Args:
            countries: List of countries to plot
            metric: Mental health metric to plot
            save: Whether to save the plot
            filename: Custom filename for saving
            
        Returns:
            go.Figure: Plotly figure object
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, country in enumerate(countries):
            country_data = self.data[self.data['country'] == country].copy()
            country_data = country_data.sort_values('year')
            
            if len(country_data) > 0:
                fig.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data[metric],
                    mode='lines+markers',
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=6),
                    hovertemplate=f'<b>{country}</b><br>' +
                                f'Year: %{{x}}<br>' +
                                f'{metric.replace("_", " ").title()}: %{{y:.2f}}<br>' +
                                '<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(
                text=f'{metric.replace("_", " ").title()} Trends Over Time',
                x=0.5,
                font=dict(size=20)
            ),
            xaxis_title='Year',
            yaxis_title=metric.replace('_', ' ').title(),
            template=self.template,
            hovermode='x unified',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            width=1000,
            height=600
        )
        
        if save:
            if filename is None:
                filename = f"interactive_time_series_{metric}_{len(countries)}_countries.html"
            pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        
        return fig
    
    def create_choropleth_map(self, metric: str, year: Optional[int] = None,
                            save: bool = True) -> go.Figure:
        """
        Create choropleth map for global mental health data.
        
        Args:
            metric: Mental health metric to map
            year: Specific year to map (latest if None)
            save: Whether to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Prepare data for mapping
        if year is None:
            # Use latest available data for each country
            map_data = self.data.loc[self.data.groupby('country')['year'].idxmax()]
        else:
            map_data = self.data[self.data['year'] == year]
        
        # Create the choropleth map
        fig = go.Figure(data=go.Choropleth(
            locations=map_data['country'],
            z=map_data[metric],
            locationmode='country names',
            colorscale='RdYlBu_r',
            hovertemplate='<b>%{location}</b><br>' +
                         f'{metric.replace("_", " ").title()}: %{{z:.2f}}<br>' +
                         '<extra></extra>',
            colorbar_title=metric.replace('_', ' ').title()
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Global {metric.replace("_", " ").title()} Distribution' +
                     (f' ({year})' if year else ' (Latest Data)'),
                x=0.5,
                font=dict(size=18)
            ),
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            template=self.template,
            width=1200,
            height=700
        )
        
        if save:
            filename = f"choropleth_{metric}_{year if year else 'latest'}.html"
            pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        
        return fig
    
    def create_scatter_matrix(self, metrics: List[str], color_by: str = 'region',
                            save: bool = True) -> go.Figure:
        """
        Create interactive scatter matrix.
        
        Args:
            metrics: List of metrics for scatter matrix
            color_by: Column to color by
            save: Whether to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Filter to available metrics
        available_metrics = [m for m in metrics if m in self.data.columns]
        
        if len(available_metrics) < 2:
            print("Need at least 2 metrics for scatter matrix")
            return None
        
        # Prepare data
        plot_data = self.data[available_metrics + [color_by]].dropna()
        
        # Clean column names for display
        clean_names = [m.replace('_', ' ').title() for m in available_metrics]
        plot_data.columns = clean_names + [color_by]
        
        # Create scatter matrix
        fig = px.scatter_matrix(
            plot_data,
            dimensions=clean_names,
            color=color_by,
            title='Mental Health Metrics Scatter Matrix',
            template=self.template,
            width=1000,
            height=1000
        )
        
        fig.update_traces(diagonal_visible=False)
        
        if save:
            filename = "scatter_matrix_mental_health.html"
            pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        
        return fig
    
    def create_animated_timeline(self, countries: List[str], metric: str,
                               save: bool = True) -> go.Figure:
        """
        Create animated timeline showing changes over years.
        
        Args:
            countries: List of countries to include
            metric: Mental health metric to animate
            save: Whether to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Prepare data for animation
        anim_data = self.data[self.data['country'].isin(countries)].copy()
        anim_data = anim_data.sort_values(['year', 'country'])
        
        # Create animated scatter plot
        fig = px.scatter(
            anim_data,
            x='year',
            y=metric,
            color='country',
            size='population' if 'population' in anim_data.columns else None,
            animation_frame='year',
            animation_group='country',
            title=f'{metric.replace("_", " ").title()} Evolution Over Time',
            template=self.template,
            width=1000,
            height=600,
            range_y=[anim_data[metric].min() * 0.9, anim_data[metric].max() * 1.1]
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Year',
            yaxis_title=metric.replace('_', ' ').title(),
            title_x=0.5
        )
        
        # Update animation settings
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 1000
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 500
        
        if save:
            filename = f"animated_timeline_{metric}.html"
            pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        
        return fig
    
    def create_correlation_heatmap(self, metrics: List[str], 
                                 save: bool = True) -> go.Figure:
        """
        Create interactive correlation heatmap.
        
        Args:
            metrics: List of metrics to correlate
            save: Whether to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Calculate correlation matrix
        available_metrics = [m for m in metrics if m in self.data.columns]
        
        if len(available_metrics) < 2:
            print("Need at least 2 metrics for correlation heatmap")
            return None
        
        corr_matrix = self.data[available_metrics].corr()
        
        # Clean labels
        clean_labels = [m.replace('_', ' ').title() for m in available_metrics]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=clean_labels,
            y=clean_labels,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Mental Health Metrics Correlation Matrix',
                x=0.5,
                font=dict(size=18)
            ),
            template=self.template,
            width=800,
            height=800,
            xaxis=dict(side='bottom'),
            yaxis=dict(autorange='reversed')
        )
        
        if save:
            filename = "interactive_correlation_heatmap.html"
            pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        
        return fig
    
    def create_box_plot_comparison(self, metric: str, group_by: str = 'region',
                                 save: bool = True) -> go.Figure:
        """
        Create interactive box plot for group comparison.
        
        Args:
            metric: Mental health metric to plot
            group_by: Column to group by
            save: Whether to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        if group_by not in self.data.columns:
            print(f"Column '{group_by}' not found in data")
            return None
        
        # Create box plot
        fig = px.box(
            self.data,
            x=group_by,
            y=metric,
            title=f'{metric.replace("_", " ").title()} Distribution by {group_by.title()}',
            template=self.template,
            width=1000,
            height=600
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=group_by.replace('_', ' ').title(),
            yaxis_title=metric.replace('_', ' ').title(),
            title_x=0.5
        )
        
        # Add individual points
        fig.update_traces(boxpoints='outliers')
        
        if save:
            filename = f"box_plot_{metric}_by_{group_by}.html"
            pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        
        return fig
    
    def create_forecast_visualization(self, forecast_results: Dict,
                                    save: bool = True) -> go.Figure:
        """
        Create interactive forecast visualization.
        
        Args:
            forecast_results: Forecast results dictionary
            save: Whether to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        if 'error' in forecast_results:
            print(f"Error in forecast results: {forecast_results['error']}")
            return None
        
        fig = go.Figure()
        
        # Historical data
        hist_years = forecast_results['historical_years']
        hist_values = forecast_results['historical_values']
        
        fig.add_trace(go.Scatter(
            x=hist_years,
            y=hist_values,
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='Year: %{x}<br>Value: %{y:.2f}<extra></extra>'
        ))
        
        # Forecast data
        forecast_years = forecast_results['forecast_years']
        
        if 'forecast_values' in forecast_results:
            forecast_values = forecast_results['forecast_values']
            fig.add_trace(go.Scatter(
                x=forecast_years,
                y=forecast_values,
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=8, symbol='square'),
                hovertemplate='Year: %{x}<br>Forecast: %{y:.2f}<extra></extra>'
            ))
        
        # Add confidence intervals if available
        if 'confidence_intervals' in forecast_results:
            ci = forecast_results['confidence_intervals']
            fig.add_trace(go.Scatter(
                x=forecast_years + forecast_years[::-1],
                y=ci['upper'] + ci['lower'][::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence Interval',
                hoverinfo='skip'
            ))
        
        # Add vertical line to separate historical and forecast
        if hist_years and forecast_years:
            separation_year = max(hist_years) + 0.5
            fig.add_vline(
                x=separation_year,
                line_dash="dot",
                line_color="gray",
                annotation_text="Forecast →",
                annotation_position="top"
            )
        
        # Update layout
        country = forecast_results.get('country', 'Unknown')
        metric = forecast_results.get('metric', 'Metric')
        model_type = forecast_results.get('model_type', 'Unknown')
        
        fig.update_layout(
            title=dict(
                text=f'{metric.replace("_", " ").title()} Forecast for {country}<br>' +
                     f'<sub>Model: {model_type.replace("_", " ").title()}</sub>',
                x=0.5,
                font=dict(size=18)
            ),
            xaxis_title='Year',
            yaxis_title=metric.replace('_', ' ').title(),
            template=self.template,
            hovermode='x unified',
            width=1000,
            height=600,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        if save:
            filename = f"interactive_forecast_{country}_{metric}_{model_type}.html"
            # Clean filename
            filename = filename.replace(' ', '_').replace('(', '').replace(')', '')
            pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        
        return fig
    
    def create_dashboard_overview(self, metrics: List[str], countries: List[str],
                                save: bool = True) -> go.Figure:
        """
        Create a comprehensive interactive dashboard overview.
        
        Args:
            metrics: List of metrics to include
            countries: List of countries to highlight
            save: Whether to save the plot
            
        Returns:
            go.Figure: Plotly figure object
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'{metrics[0].replace("_", " ").title()} Time Series',
                'Country Comparison (Latest)',
                'Regional Distribution',
                'Correlation Analysis'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Time series (top left)
        colors = px.colors.qualitative.Set1
        for i, country in enumerate(countries[:5]):
            country_data = self.data[self.data['country'] == country].sort_values('year')
            if len(country_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=country_data['year'],
                        y=country_data[metrics[0]],
                        mode='lines+markers',
                        name=country,
                        line=dict(color=colors[i % len(colors)]),
                        showlegend=True
                    ),
                    row=1, col=1
                )
        
        # 2. Country comparison (top right)
        latest_data = []
        for country in countries:
            country_data = self.data[self.data['country'] == country]
            if len(country_data) > 0:
                latest = country_data.loc[country_data['year'].idxmax()]
                latest_data.append({
                    'country': country, 
                    'value': latest.get(metrics[0], 0)
                })
        
        if latest_data:
            df_latest = pd.DataFrame(latest_data).sort_values('value')
            fig.add_trace(
                go.Bar(
                    x=df_latest['value'],
                    y=df_latest['country'],
                    orientation='h',
                    name='Latest Values',
                    showlegend=False,
                    marker_color='lightblue'
                ),
                row=1, col=2
            )
        
        # 3. Regional distribution (bottom left)
        if 'region' in self.data.columns:
            regional_data = self.data.groupby('region')[metrics[0]].mean().reset_index()
            fig.add_trace(
                go.Bar(
                    x=regional_data['region'],
                    y=regional_data[metrics[0]],
                    name='Regional Average',
                    showlegend=False,
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )
        
        # 4. Correlation heatmap (bottom right)
        available_metrics = [m for m in metrics[:3] if m in self.data.columns]
        if len(available_metrics) >= 2:
            corr_matrix = self.data[available_metrics].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=[m.replace('_', ' ').title() for m in available_metrics],
                    y=[m.replace('_', ' ').title() for m in available_metrics],
                    colorscale='RdBu',
                    zmid=0,
                    showscale=False,
                    name='Correlation'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text='Mental Health Analysis Dashboard',
                x=0.5,
                font=dict(size=20)
            ),
            template=self.template,
            height=800,
            width=1200
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text=metrics[0].replace('_', ' ').title(), row=1, col=1)
        
        fig.update_xaxes(title_text=metrics[0].replace('_', ' ').title(), row=1, col=2)
        fig.update_yaxes(title_text="Country", row=1, col=2)
        
        if save:
            filename = "interactive_dashboard_overview.html"
            pyo.plot(fig, filename=str(self.output_dir / filename), auto_open=False)
        
        return fig

def create_all_interactive_plots(data: pd.DataFrame, output_dir: str = "visualizations") -> None:
    """
    Create all interactive plots for the mental health analysis.
    
    Args:
        data: Mental health dataframe
        output_dir: Directory to save plots
    """
    plotter = InteractivePlotter(data, output_dir)
    
    # Get available metrics and countries
    metrics = [col for col in data.columns if 'prevalence' in col]
    countries = data['country'].value_counts().head(10).index.tolist()
    
    print("Creating interactive visualizations...")
    
    try:
        # Interactive time series
        if metrics and countries:
            plotter.create_interactive_time_series(countries[:5], metrics[0])
            print("✓ Interactive time series created")
        
        # Choropleth map
        if metrics:
            plotter.create_choropleth_map(metrics[0])
            print("✓ Choropleth map created")
        
        # Scatter matrix
        if len(metrics) >= 2:
            plotter.create_scatter_matrix(metrics[:3])
            print("✓ Scatter matrix created")
        
        # Correlation heatmap
        if len(metrics) >= 2:
            plotter.create_correlation_heatmap(metrics)
            print("✓ Interactive correlation heatmap created")
        
        # Box plot comparison
        if metrics and 'region' in data.columns:
            plotter.create_box_plot_comparison(metrics[0], 'region')
            print("✓ Box plot comparison created")
        
        # Dashboard overview
        if metrics and countries:
            plotter.create_dashboard_overview(metrics, countries)
            print("✓ Interactive dashboard overview created")
        
        print(f"\nAll interactive plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error creating interactive plots: {e}")
        import traceback
        traceback.print_exc()
