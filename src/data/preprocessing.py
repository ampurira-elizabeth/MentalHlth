"""
Data preprocessing and cleaning utilities for mental health analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Clean and preprocess mental health datasets."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data preprocessor.
        
        Args:
            data_dir: Directory containing the data
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.external_dir = self.data_dir / "external"
        
        # Ensure processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load the raw mental health data.
        
        Returns:
            pd.DataFrame: Raw mental health data
        """
        filepath = self.raw_dir / "mental_health_prevalence.csv"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Raw data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        logger.info(f"Loaded raw data with shape: {df.shape}")
        
        return df
    
    def clean_mental_health_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the mental health dataset.
        
        Args:
            df: Raw mental health dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Make a copy to avoid modifying original
        cleaned_df = df.copy()
        
        # Standardize column names
        column_mapping = {
            'Entity': 'country',
            'Year': 'year'
        }
        
        # Find prevalence columns and standardize their names
        prevalence_cols = [col for col in cleaned_df.columns if 'prevalence' in col.lower()]
        
        for col in prevalence_cols:
            if 'depression' in col.lower():
                column_mapping[col] = 'depression_prevalence'
            elif 'anxiety' in col.lower():
                column_mapping[col] = 'anxiety_prevalence'
            elif 'bipolar' in col.lower():
                column_mapping[col] = 'bipolar_prevalence'
            elif 'schizophrenia' in col.lower():
                column_mapping[col] = 'schizophrenia_prevalence'
        
        cleaned_df = cleaned_df.rename(columns=column_mapping)
        
        # Remove rows with invalid country names (codes, regions, etc.)
        invalid_entities = [
            'World', 'OWID_WRL', 'High-income countries', 'Low-income countries',
            'Middle-income countries', 'Upper-middle-income countries',
            'Lower-middle-income countries'
        ]
        
        cleaned_df = cleaned_df[~cleaned_df['country'].isin(invalid_entities)]
        
        # Filter for reasonable year range
        cleaned_df = cleaned_df[(cleaned_df['year'] >= 1990) & (cleaned_df['year'] <= 2025)]
        
        # Handle missing values
        numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        
        # Remove rows where all prevalence values are missing
        prevalence_columns = [col for col in numeric_cols if 'prevalence' in col]
        if prevalence_columns:
            cleaned_df = cleaned_df.dropna(subset=prevalence_columns, how='all')
        
        # Fill remaining missing values with interpolation where possible
        for col in prevalence_columns:
            if col in cleaned_df.columns:
                # Group by country and interpolate missing values
                cleaned_df[col] = cleaned_df.groupby('country')[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
        
        # Remove outliers (values beyond 3 standard deviations)
        for col in prevalence_columns:
            if col in cleaned_df.columns:
                q1 = cleaned_df[col].quantile(0.25)
                q3 = cleaned_df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Cap outliers instead of removing them
                cleaned_df[col] = cleaned_df[col].clip(lower=max(0, lower_bound), 
                                                     upper=upper_bound)
        
        # Sort by country and year
        cleaned_df = cleaned_df.sort_values(['country', 'year']).reset_index(drop=True)
        
        logger.info(f"Cleaned data shape: {cleaned_df.shape}")
        logger.info(f"Countries in dataset: {cleaned_df['country'].nunique()}")
        logger.info(f"Year range: {cleaned_df['year'].min()} - {cleaned_df['year'].max()}")
        
        return cleaned_df
    
    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to the dataset.
        
        Args:
            df: Cleaned mental health dataframe
            
        Returns:
            pd.DataFrame: Dataframe with additional features
        """
        enhanced_df = df.copy()
        
        # Calculate total mental health burden
        prevalence_cols = [col for col in df.columns if 'prevalence' in col]
        if len(prevalence_cols) > 1:
            enhanced_df['total_mental_health_prevalence'] = enhanced_df[prevalence_cols].sum(axis=1)
        
        # Add time-based features
        enhanced_df['decade'] = (enhanced_df['year'] // 10) * 10
        enhanced_df['years_since_1990'] = enhanced_df['year'] - 1990
        
        # Calculate year-over-year changes
        for col in prevalence_cols:
            if col in enhanced_df.columns:
                enhanced_df[f'{col}_change'] = enhanced_df.groupby('country')[col].pct_change()
        
        # Add regional groupings (simplified)
        region_mapping = {
            'United States': 'North America',
            'Canada': 'North America',
            'Mexico': 'North America',
            'Brazil': 'South America',
            'Argentina': 'South America',
            'Chile': 'South America',
            'United Kingdom': 'Europe',
            'Germany': 'Europe',
            'France': 'Europe',
            'Italy': 'Europe',
            'Spain': 'Europe',
            'Netherlands': 'Europe',
            'Sweden': 'Europe',
            'Norway': 'Europe',
            'China': 'Asia',
            'Japan': 'Asia',
            'India': 'Asia',
            'South Korea': 'Asia',
            'Thailand': 'Asia',
            'Australia': 'Oceania',
            'New Zealand': 'Oceania'
        }
        
        enhanced_df['region'] = enhanced_df['country'].map(region_mapping)
        enhanced_df['region'] = enhanced_df['region'].fillna('Other')
        
        return enhanced_df
    
    def load_external_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load external datasets (population, GDP, etc.).
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of external datasets
        """
        external_data = {}
        
        # Load population data
        pop_file = self.external_dir / "population_data.csv"
        if pop_file.exists():
            pop_df = pd.read_csv(pop_file)
            # Standardize column names
            if 'Entity' in pop_df.columns:
                pop_df = pop_df.rename(columns={'Entity': 'country', 'Year': 'year'})
            external_data['population'] = pop_df
        
        # Load GDP data
        gdp_file = self.external_dir / "gdp_data.csv"
        if gdp_file.exists():
            gdp_df = pd.read_csv(gdp_file)
            # Standardize column names
            if 'Entity' in gdp_df.columns:
                gdp_df = gdp_df.rename(columns={'Entity': 'country', 'Year': 'year'})
            external_data['gdp'] = gdp_df
        
        return external_data
    
    def merge_datasets(self, mental_health_df: pd.DataFrame, 
                      external_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge mental health data with external datasets.
        
        Args:
            mental_health_df: Main mental health dataframe
            external_data: Dictionary of external datasets
            
        Returns:
            pd.DataFrame: Merged dataframe
        """
        merged_df = mental_health_df.copy()
        
        # Merge with population data
        if 'population' in external_data:
            pop_df = external_data['population'][['country', 'year', 'Population']]
            merged_df = merged_df.merge(pop_df, on=['country', 'year'], how='left')
            merged_df = merged_df.rename(columns={'Population': 'population'})
        
        # Merge with GDP data
        if 'gdp' in external_data:
            gdp_df = external_data['gdp']
            gdp_col = [col for col in gdp_df.columns if 'GDP' in col or 'gdp' in col.lower()]
            if gdp_col:
                gdp_df = gdp_df[['country', 'year', gdp_col[0]]]
                gdp_df = gdp_df.rename(columns={gdp_col[0]: 'gdp_per_capita'})
                merged_df = merged_df.merge(gdp_df, on=['country', 'year'], how='left')
        
        return merged_df
    
    def process_all_data(self) -> pd.DataFrame:
        """
        Complete data processing pipeline.
        
        Returns:
            pd.DataFrame: Fully processed dataframe
        """
        logger.info("Starting data processing pipeline...")
        
        # Load and clean main dataset
        raw_df = self.load_raw_data()
        cleaned_df = self.clean_mental_health_data(raw_df)
        
        # Add derived features
        enhanced_df = self.add_derived_features(cleaned_df)
        
        # Load and merge external data
        external_data = self.load_external_data()
        final_df = self.merge_datasets(enhanced_df, external_data)
        
        # Save processed data
        output_path = self.processed_dir / "mental_health_processed.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to: {output_path}")
        
        # Create summary statistics
        self.create_data_summary(final_df)
        
        return final_df
    
    def create_data_summary(self, df: pd.DataFrame) -> None:
        """
        Create and save data summary statistics.
        
        Args:
            df: Processed dataframe
        """
        summary = {
            'shape': df.shape,
            'countries': df['country'].nunique(),
            'years_range': f"{df['year'].min()} - {df['year'].max()}",
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict()
        }
        
        # Save summary as JSON
        import json
        summary_path = self.processed_dir / "data_summary.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean the summary for JSON serialization
        clean_summary = {}
        for key, value in summary.items():
            if isinstance(value, dict):
                clean_summary[key] = {k: convert_numpy(v) for k, v in value.items()}
            else:
                clean_summary[key] = convert_numpy(value)
        
        with open(summary_path, 'w') as f:
            json.dump(clean_summary, f, indent=2)
        
        logger.info(f"Data summary saved to: {summary_path}")
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Assess data quality and completeness.
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Dict: Data quality metrics
        """
        total_records = len(df)
        total_columns = len(df.columns)
        
        # Missing data analysis
        missing_data = df.isnull().sum()
        missing_percentage = (missing_data.sum() / (total_records * total_columns)) * 100
        
        # Missing data by column
        missing_by_column = {}
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_records) * 100
            missing_by_column[col] = {
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            }
        
        # Duplicate records
        duplicate_count = df.duplicated().sum()
        
        # Data types
        data_types = df.dtypes.to_dict()
        
        return {
            'total_records': total_records,
            'total_columns': total_columns,
            'missing_percentage': missing_percentage,
            'missing_by_column': missing_by_column,
            'duplicate_count': duplicate_count,
            'data_types': {str(k): str(v) for k, v in data_types.items()}
        }

def main():
    """Main function to run data preprocessing."""
    preprocessor = DataPreprocessor()
    
    try:
        processed_df = preprocessor.process_all_data()
        logger.info("Data preprocessing completed successfully!")
        logger.info(f"Final dataset shape: {processed_df.shape}")
        return 0
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
