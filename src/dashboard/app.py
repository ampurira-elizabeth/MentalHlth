"""
Interactive Dashboard for Mental Health Analysis
Uses Dash/Plotly for interactive web-based visualizations
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import DataPreprocessor
from visualization.interactive_plots import InteractivePlotter
from analysis.time_series import TimeSeriesAnalyzer

class MentalHealthDashboard:
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.app = dash.Dash(__name__)
        self.data = None
        self.preprocessor = DataPreprocessor(data_dir)
        self.plotter = None  # Will initialize after loading data
        self.analyzer = None  # Will initialize after loading data
        
        # Load and preprocess data
        self.load_data()
        self.setup_layout()
        self.setup_callbacks()
    
    def load_data(self):
        """Load and preprocess data for dashboard"""
        try:
            # Load raw data
            raw_path = self.data_dir / "raw" / "mental_health_prevalence.csv"
            if raw_path.exists():
                self.data = pd.read_csv(raw_path)
                print(f"✓ Loaded {len(self.data)} records")
            else:
                # Create sample data if real data not available
                self.data = self.create_sample_data()
                print("⚠️ Using sample data - run data download first for real data")
            
            # Initialize plotter and analyzer after data is loaded
            self.plotter = InteractivePlotter(self.data)
            self.analyzer = TimeSeriesAnalyzer(self.data)
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = self.create_sample_data()
            self.plotter = InteractivePlotter(self.data)
            self.analyzer = TimeSeriesAnalyzer(self.data)
    
    def create_sample_data(self):
        """Create sample mental health data for demonstration"""
        np.random.seed(42)
        
        countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Japan', 
                    'Canada', 'Australia', 'Brazil', 'India', 'China']
        years = list(range(1990, 2024))
        
        data = []
        for country in countries:
            base_depression = np.random.uniform(3, 8)
            base_anxiety = np.random.uniform(2, 6)
            base_bipolar = np.random.uniform(0.5, 1.5)
            
            for year in years:
                # Add some trend and noise
                trend_factor = (year - 1990) * 0.02
                noise = np.random.normal(0, 0.3)
                
                data.append({
                    'Entity': country,
                    'Year': year,
                    'Depression_prevalence': max(0, base_depression + trend_factor + noise),
                    'Anxiety_prevalence': max(0, base_anxiety + trend_factor * 0.8 + noise),
                    'Bipolar_prevalence': max(0, base_bipolar + trend_factor * 0.3 + noise * 0.5),
                    'Population': np.random.randint(10000000, 1400000000)
                })
        
        return pd.DataFrame(data)
    
    def setup_layout(self):
        """Setup dashboard layout"""
        # Get available metrics
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        prevalence_cols = [col for col in numeric_cols if 'prevalence' in col.lower()]
        
        countries = sorted(self.data['Entity'].unique()) if 'Entity' in self.data.columns else []
        years = sorted([int(year) for year in self.data['Year'].unique()]) if 'Year' in self.data.columns else []
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Mental Health Global Trends Dashboard", 
                       className="dashboard-title"),
                html.P("Interactive analysis of global mental health disorder prevalence",
                      className="dashboard-subtitle")
            ], className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Select Mental Health Metric:"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[{'label': col.replace('_', ' ').title(), 'value': col} 
                                for col in prevalence_cols],
                        value=prevalence_cols[0] if prevalence_cols else None,
                        className="dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Select Countries:"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} for country in countries],
                        value=countries[:5] if len(countries) >= 5 else countries,
                        multi=True,
                        className="dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Year Range:"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=min(years) if years else 1990,
                        max=max(years) if years else 2023,
                        value=[min(years) if years else 1990, max(years) if years else 2023],
                        marks={int(year): str(int(year)) for year in years[::5]} if years else {},
                        tooltip={"placement": "bottom", "always_visible": True},
                        className="slider"
                    )
                ], className="control-item")
            ], className="controls"),
            
            # Main Content
            html.Div([
                # Time Series Plot
                html.Div([
                    dcc.Graph(id='time-series-plot')
                ], className="chart-container"),
                
                # Geographic Map
                html.Div([
                    dcc.Graph(id='world-map')
                ], className="chart-container"),
                
                # Statistics Panel
                html.Div([
                    html.H3("Key Statistics"),
                    html.Div(id='statistics-panel')
                ], className="stats-panel")
            ], className="main-content"),
            
            # Footer
            html.Div([
                html.P("Mental Health Analysis Dashboard • Data visualization for global trends"),
                html.P("Built with Dash and Plotly")
            ], className="footer")
        ])
    
    def setup_callbacks(self):
        """Setup interactive callbacks"""
        
        @self.app.callback(
            [Output('time-series-plot', 'figure'),
             Output('world-map', 'figure'),
             Output('statistics-panel', 'children')],
            [Input('metric-dropdown', 'value'),
             Input('country-dropdown', 'value'),
             Input('year-slider', 'value')]
        )
        def update_dashboard(selected_metric, selected_countries, year_range):
            if not selected_metric or not selected_countries:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="Please select metric and countries")
                return empty_fig, empty_fig, "No data selected"
            
            # Filter data
            filtered_data = self.data[
                (self.data['Entity'].isin(selected_countries)) &
                (self.data['Year'] >= year_range[0]) &
                (self.data['Year'] <= year_range[1])
            ].copy()
            
            if filtered_data.empty:
                empty_fig = go.Figure()
                empty_fig.update_layout(title="No data available for selection")
                return empty_fig, empty_fig, "No data available"
            
            # Create time series plot
            time_series_fig = px.line(
                filtered_data,
                x='Year',
                y=selected_metric,
                color='Entity',
                title=f'{selected_metric.replace("_", " ").title()} Over Time',
                markers=True
            )
            time_series_fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Prevalence (%)",
                legend_title="Country",
                hovermode='x unified'
            )
            
            # Create world map (latest year data)
            latest_year = filtered_data['Year'].max()
            map_data = filtered_data[filtered_data['Year'] == latest_year]
            
            world_map_fig = px.choropleth(
                map_data,
                locations='Entity',
                locationmode='country names',
                color=selected_metric,
                title=f'{selected_metric.replace("_", " ").title()} - {latest_year}',
                color_continuous_scale='Reds'
            )
            world_map_fig.update_layout(
                geo=dict(showframe=False, showcoastlines=True)
            )
            
            # Generate statistics
            stats = self.generate_statistics(filtered_data, selected_metric)
            
            return time_series_fig, world_map_fig, stats
    
    def generate_statistics(self, data, metric):
        """Generate key statistics for the selected data"""
        if data.empty or metric not in data.columns:
            return html.Div("No statistics available")
        
        # Calculate statistics
        latest_year = data['Year'].max()
        earliest_year = data['Year'].min()
        
        latest_data = data[data['Year'] == latest_year][metric]
        earliest_data = data[data['Year'] == earliest_year][metric]
        
        avg_latest = latest_data.mean()
        avg_earliest = earliest_data.mean()
        change = ((avg_latest - avg_earliest) / avg_earliest) * 100 if avg_earliest > 0 else 0
        
        highest_country = data.loc[data[metric].idxmax(), 'Entity'] if not data.empty else "N/A"
        highest_value = data[metric].max()
        
        stats_content = html.Div([
            html.Div([
                html.H4(f"{avg_latest:.2f}%"),
                html.P("Average Latest Year")
            ], className="stat-item"),
            
            html.Div([
                html.H4(f"{change:+.1f}%"),
                html.P(f"Change ({earliest_year}-{latest_year})")
            ], className="stat-item"),
            
            html.Div([
                html.H4(f"{highest_value:.2f}%"),
                html.P(f"Highest: {highest_country}")
            ], className="stat-item"),
            
            html.Div([
                html.H4(f"{len(data['Entity'].unique())}"),
                html.P("Countries Analyzed")
            ], className="stat-item")
        ], className="stats-grid")
        
        return stats_content
    
    def run(self, host='127.0.0.1', port=8050, debug=True):
        """Run the dashboard"""
        print(f"🚀 Starting Mental Health Dashboard...")
        print(f"📊 Dashboard will be available at: http://{host}:{port}")
        print(f"📈 Loaded data: {len(self.data)} records")
        print(f"🌍 Countries: {self.data['Entity'].nunique()}")
        print(f"📅 Years: {self.data['Year'].min()}-{self.data['Year'].max()}")
        
        self.app.run(host=host, port=port, debug=debug)

# Custom CSS styles
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

def create_app():
    """Create and return dashboard app"""
    dashboard = MentalHealthDashboard()
    return dashboard

if __name__ == '__main__':
    # Run dashboard
    dashboard = MentalHealthDashboard()
    dashboard.run()
