"""
Dash dashboard components for mental health analysis.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class DashboardComponents:
    """Components for the mental health dashboard."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize dashboard components.
        
        Args:
            data: Processed mental health dataframe
        """
        self.data = data.copy()
        self.countries = sorted(data['country'].unique()) if 'country' in data.columns else []
        self.years = sorted(data['year'].unique()) if 'year' in data.columns else []
        self.metrics = [col for col in data.columns if 'prevalence' in col]
        self.regions = sorted(data['region'].unique()) if 'region' in data.columns else []
    
    def create_layout(self) -> html.Div:
        """
        Create the main dashboard layout.
        
        Returns:
            html.Div: Dashboard layout
        """
        return html.Div([
            # Header
            html.Div([
                html.H1("Mental Health Global Trends Analysis", 
                       className="text-center text-primary mb-4"),
                html.P("Interactive dashboard for exploring global mental health data",
                      className="text-center text-muted")
            ], className="container-fluid py-3 bg-light"),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Select Countries:", className="form-label"),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} 
                                for country in self.countries],
                        value=self.countries[:5] if len(self.countries) >= 5 else self.countries,
                        multi=True,
                        className="mb-3"
                    )
                ], className="col-md-6"),
                
                html.Div([
                    html.Label("Select Metric:", className="form-label"),
                    dcc.Dropdown(
                        id='metric-dropdown',
                        options=[{'label': metric.replace('_', ' ').title(), 'value': metric} 
                                for metric in self.metrics],
                        value=self.metrics[0] if self.metrics else None,
                        className="mb-3"
                    )
                ], className="col-md-6")
            ], className="row container-fluid"),
            
            # Year slider
            html.Div([
                html.Div([
                    html.Label("Year Range:", className="form-label"),
                    dcc.RangeSlider(
                        id='year-slider',
                        min=min(self.years) if self.years else 1990,
                        max=max(self.years) if self.years else 2020,
                        value=[min(self.years) if self.years else 1990, 
                              max(self.years) if self.years else 2020],
                        marks={year: str(year) for year in self.years[::5]} if self.years else {},
                        step=1,
                        className="mb-4"
                    )
                ], className="col-12")
            ], className="row container-fluid"),
            
            # Main content tabs
            html.Div([
                dcc.Tabs(id="main-tabs", value="time-series", children=[
                    dcc.Tab(label="Time Series", value="time-series"),
                    dcc.Tab(label="Geographic View", value="geographic"),
                    dcc.Tab(label="Comparisons", value="comparisons"),
                    dcc.Tab(label="Statistics", value="statistics")
                ])
            ], className="container-fluid"),
            
            # Content area
            html.Div(id="tab-content", className="container-fluid mt-4")
        ])
    
    def create_time_series_tab(self) -> html.Div:
        """Create time series tab content."""
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id="time-series-plot")
                ], className="col-12")
            ], className="row"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="trend-analysis-plot")
                ], className="col-md-6"),
                
                html.Div([
                    dcc.Graph(id="country-ranking-plot")
                ], className="col-md-6")
            ], className="row mt-4")
        ])
    
    def create_geographic_tab(self) -> html.Div:
        """Create geographic view tab content."""
        return html.Div([
            html.Div([
                html.Div([
                    html.Label("Select Year for Map:", className="form-label"),
                    dcc.Dropdown(
                        id='map-year-dropdown',
                        options=[{'label': str(year), 'value': year} 
                                for year in sorted(self.years, reverse=True)],
                        value=max(self.years) if self.years else None,
                        className="mb-3"
                    )
                ], className="col-md-4"),
            ], className="row"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="choropleth-map")
                ], className="col-12")
            ], className="row"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="regional-comparison")
                ], className="col-12")
            ], className="row mt-4")
        ])
    
    def create_comparison_tab(self) -> html.Div:
        """Create comparison tab content."""
        return html.Div([
            html.Div([
                html.Div([
                    dcc.Graph(id="scatter-plot")
                ], className="col-md-6"),
                
                html.Div([
                    dcc.Graph(id="correlation-heatmap")
                ], className="col-md-6")
            ], className="row"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="distribution-comparison")
                ], className="col-12")
            ], className="row mt-4")
        ])
    
    def create_statistics_tab(self) -> html.Div:
        """Create statistics tab content."""
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Summary Statistics", className="text-center"),
                    html.Div(id="summary-stats")
                ], className="col-md-6"),
                
                html.Div([
                    html.H4("Top/Bottom Countries", className="text-center"),
                    html.Div(id="country-rankings")
                ], className="col-md-6")
            ], className="row"),
            
            html.Div([
                html.Div([
                    dcc.Graph(id="box-plot-comparison")
                ], className="col-12")
            ], className="row mt-4")
        ])

# Callback functions for interactivity
def register_callbacks(app: dash.Dash, components: DashboardComponents):
    """Register all dashboard callbacks."""
    
    @app.callback(
        Output('tab-content', 'children'),
        Input('main-tabs', 'value')
    )
    def render_tab_content(active_tab):
        if active_tab == 'time-series':
            return components.create_time_series_tab()
        elif active_tab == 'geographic':
            return components.create_geographic_tab()
        elif active_tab == 'comparisons':
            return components.create_comparison_tab()
        elif active_tab == 'statistics':
            return components.create_statistics_tab()
        return html.Div("Select a tab")
    
    @app.callback(
        Output('time-series-plot', 'figure'),
        [Input('country-dropdown', 'value'),
         Input('metric-dropdown', 'value'),
         Input('year-slider', 'value')]
    )
    def update_time_series(selected_countries, selected_metric, year_range):
        if not selected_countries or not selected_metric:
            return go.Figure()
        
        # Filter data
        filtered_data = components.data[
            (components.data['country'].isin(selected_countries)) &
            (components.data['year'] >= year_range[0]) &
            (components.data['year'] <= year_range[1])
        ]
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for i, country in enumerate(selected_countries):
            country_data = filtered_data[filtered_data['country'] == country]
            if len(country_data) > 0:
                fig.add_trace(go.Scatter(
                    x=country_data['year'],
                    y=country_data[selected_metric],
                    mode='lines+markers',
                    name=country,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title=f'{selected_metric.replace("_", " ").title()} Trends Over Time',
            xaxis_title='Year',
            yaxis_title=selected_metric.replace('_', ' ').title(),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('choropleth-map', 'figure'),
        [Input('metric-dropdown', 'value'),
         Input('map-year-dropdown', 'value')]
    )
    def update_choropleth(selected_metric, selected_year):
        if not selected_metric or not selected_year:
            return go.Figure()
        
        # Filter data for selected year
        year_data = components.data[components.data['year'] == selected_year]
        
        fig = go.Figure(data=go.Choropleth(
            locations=year_data['country'],
            z=year_data[selected_metric],
            locationmode='country names',
            colorscale='Viridis',
            hovertemplate='<b>%{location}</b><br>' +
                         f'{selected_metric.replace("_", " ").title()}: %{{z:.2f}}<br>' +
                         '<extra></extra>',
            colorbar_title=selected_metric.replace('_', ' ').title()
        ))
        
        fig.update_layout(
            title=f'Global {selected_metric.replace("_", " ").title()} Distribution ({selected_year})',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular'
            ),
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('correlation-heatmap', 'figure'),
        Input('metric-dropdown', 'value')
    )
    def update_correlation_heatmap(selected_metric):
        if not components.metrics or len(components.metrics) < 2:
            return go.Figure()
        
        # Calculate correlation matrix
        corr_matrix = components.data[components.metrics].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[m.replace('_', ' ').title() for m in components.metrics],
            y=[m.replace('_', ' ').title() for m in components.metrics],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Mental Health Metrics Correlation Matrix',
            template='plotly_white'
        )
        
        return fig
    
    @app.callback(
        Output('summary-stats', 'children'),
        [Input('metric-dropdown', 'value'),
         Input('country-dropdown', 'value')]
    )
    def update_summary_stats(selected_metric, selected_countries):
        if not selected_metric:
            return "Select a metric to view statistics"
        
        # Filter data if countries are selected
        if selected_countries:
            data_subset = components.data[components.data['country'].isin(selected_countries)]
        else:
            data_subset = components.data
        
        stats = data_subset[selected_metric].describe()
        
        return html.Div([
            html.P(f"Count: {stats['count']:.0f}"),
            html.P(f"Mean: {stats['mean']:.2f}"),
            html.P(f"Std: {stats['std']:.2f}"),
            html.P(f"Min: {stats['min']:.2f}"),
            html.P(f"Max: {stats['max']:.2f}"),
        ])

def create_mental_health_dashboard(data: pd.DataFrame, port: int = 8050) -> dash.Dash:
    """
    Create and configure the mental health dashboard.
    
    Args:
        data: Processed mental health dataframe
        port: Port to run the dashboard on
        
    Returns:
        dash.Dash: Configured Dash application
    """
    # Initialize Dash app
    app = dash.Dash(__name__, external_stylesheets=[
        'https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css'
    ])
    
    # Create components
    components = DashboardComponents(data)
    
    # Set layout
    app.layout = components.create_layout()
    
    # Register callbacks
    register_callbacks(app, components)
    
    return app
