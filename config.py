# Mental Health Analysis Project Configuration

# Project metadata
PROJECT_NAME = "Mental Health Global Trends Analysis"
PROJECT_VERSION = "1.0.0"
PROJECT_DESCRIPTION = "Comprehensive analysis of global mental health disorder prevalence using interactive visualizations and time series analysis"

# Data configuration
DATA_SOURCES = {
    "primary": "https://lnkd.in/gcyE-85A",
    "backup": "sample_data_generator",
    "formats": ["csv", "json", "xlsx"]
}

# Analysis parameters
ANALYSIS_CONFIG = {
    "primary_metrics": [
        "Depression_prevalence",
        "Anxiety_prevalence", 
        "Bipolar_prevalence",
        "Eating_disorders_prevalence"
    ],
    "time_range": {
        "start_year": 1990,
        "end_year": 2023
    },
    "forecasting": {
        "horizon": 5,  # years
        "methods": ["linear", "exponential_smoothing", "arima"]
    },
    "statistical_tests": {
        "significance_level": 0.05,
        "confidence_interval": 0.95
    }
}

# Visualization settings
VISUALIZATION_CONFIG = {
    "default_style": "seaborn-v0_8-whitegrid",
    "color_schemes": {
        "primary": "Blues",
        "secondary": "Reds", 
        "categorical": "Set1"
    },
    "figure_sizes": {
        "standard": (12, 8),
        "wide": (15, 8),
        "tall": (10, 12),
        "dashboard": (14, 10)
    },
    "dpi": 300,
    "formats": ["png", "pdf", "svg", "html"]
}

# Dashboard configuration
DASHBOARD_CONFIG = {
    "host": "127.0.0.1",
    "port": 8050,
    "debug": True,
    "theme": "plotly_white",
    "update_interval": 1000  # milliseconds
}

# Regional classifications
REGIONAL_MAPPING = {
    "North America": ["United States", "Canada", "Mexico"],
    "Europe": ["United Kingdom", "Germany", "France", "Italy", "Spain", "Netherlands", "Russia"],
    "Asia": ["Japan", "China", "India", "South Korea", "Indonesia"],
    "Oceania": ["Australia", "New Zealand"],
    "South America": ["Brazil", "Argentina", "Chile", "Colombia"],
    "Africa": ["South Africa", "Nigeria", "Egypt", "Kenya"]
}

# File paths
PATHS = {
    "data": {
        "raw": "data/raw/",
        "processed": "data/processed/",
        "external": "data/external/"
    },
    "outputs": {
        "visualizations": "visualizations/",
        "reports": "reports/",
        "models": "models/"
    },
    "notebooks": "notebooks/",
    "src": "src/"
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    "min_data_completeness": 0.8,  # 80%
    "max_missing_percentage": 0.2,  # 20%
    "outlier_threshold": 3,  # standard deviations
    "min_countries_per_region": 3,
    "min_years_for_trend": 5
}

# Export settings
EXPORT_CONFIG = {
    "include_metadata": True,
    "compression": "gzip",
    "date_format": "%Y-%m-%d",
    "decimal_places": 4
}
