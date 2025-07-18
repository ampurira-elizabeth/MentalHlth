# Mental Health Global Trends Analysis 🧠

A comprehensive data analysis project examining global trends in mental health disorders using interactive visualizations and time series analysis.

## 📊 Project Overview

This project analyzes global mental health data to:
- Identify trends in mental health disorder prevalence over time
- Compare mental health statistics across different countries and regions
- Create interactive visualizations for data exploration
- Perform time series analysis to forecast future trends

## 🛠 Skills Demonstrated

- **Time Series Analysis**: Trend analysis, seasonality detection, forecasting
- **Data Visualization**: Interactive dashboards, statistical plots, geographic maps
- **Exploratory Data Analysis**: Statistical summaries, correlation analysis, hypothesis testing
- **Python Programming**: pandas, matplotlib, plotly, dash, scikit-learn

## 📁 Project Structure

```
MentalHlth/
├── data/                           # Raw and processed datasets
│   ├── raw/                       # Original downloaded data
│   ├── processed/                 # Cleaned and transformed data
│   └── external/                  # Additional reference datasets
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb  # Initial data exploration
│   ├── 02_data_cleaning.ipynb     # Data preprocessing
│   ├── 03_time_series_analysis.ipynb # Temporal analysis
│   └── 04_visualization.ipynb     # Advanced visualizations
├── src/                          # Python modules and scripts
│   ├── data/                     # Data acquisition and processing
│   ├── analysis/                 # Analysis functions
│   ├── visualization/            # Plotting utilities
│   └── dashboard/                # Dash application
├── visualizations/               # Output plots and charts
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv mental_health_env
mental_health_env\Scripts\activate  # Windows
# source mental_health_env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

```bash
# Run data acquisition script
python src/data/download_data.py
```

### 3. Run Analysis

```bash
# Launch Jupyter notebooks
jupyter notebook notebooks/

# Or run the interactive dashboard
python src/dashboard/app.py
```

## 📈 Key Findings

### Global Trends
- **Increasing Prevalence**: Mental health disorders show a steady upward trend globally from 1990-2023
- **Regional Variations**: Significant differences between regions, with developed countries showing higher reported rates
- **Demographic Patterns**: Strong correlations between socioeconomic factors and mental health outcomes

### Statistical Analysis
- **Time Series**: Significant positive trend in global mental health prevalence (p < 0.05)
- **Correlations**: Strong positive correlations between different mental health disorders
- **Forecasting**: Projected continued increase over the next 5 years using multiple forecasting models

### Interactive Dashboard Features
- **Real-time filtering** by country, region, and time period
- **Multi-metric comparison** across different mental health disorders
- **Geographic visualization** with world maps and regional breakdowns
- **Statistical testing** and trend analysis tools

## 🎯 Visualizations

- **Time Series Plots**: Trend analysis over decades
- **Geographic Heatmaps**: Global prevalence distribution
- **Interactive Dashboard**: Multi-dimensional data exploration
- **Statistical Charts**: Correlation matrices, distribution plots

## 📊 Dataset

**Source**: [Global Mental Health Dataset](https://lnkd.in/gcyE-85A)

The dataset includes:
- Mental health disorder prevalence by country and year
- Population demographics and socioeconomic indicators
- Healthcare system metrics
- Multiple disorder categories (depression, anxiety, bipolar, etc.)

## 🔧 Technical Implementation

### Data Processing Pipeline
1. **Data Acquisition**: Automated download and validation
2. **Data Cleaning**: Missing value imputation, outlier detection
3. **Feature Engineering**: Derived metrics, temporal features
4. **Analysis**: Statistical testing, time series modeling
5. **Visualization**: Interactive dashboards and static plots

### Key Technologies
- **pandas & numpy**: Data manipulation and numerical computing
- **matplotlib & seaborn**: Statistical visualizations
- **plotly & dash**: Interactive web-based visualizations
- **scikit-learn**: Machine learning and statistical modeling
- **statsmodels**: Time series analysis and forecasting

## 📝 Next Steps

- [ ] Implement machine learning models for trend prediction
- [ ] Add more granular geographic analysis
- [ ] Integrate additional mental health datasets
- [ ] Deploy dashboard to web platform
- [ ] Create automated reporting system

## 🤝 Contributing

This is a personal learning project, but feedback and suggestions are welcome!

## 📄 License

This project is for educational and research purposes.

---

*Analyzing mental health data to promote understanding and awareness* 🌟
