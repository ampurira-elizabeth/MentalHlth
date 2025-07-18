#!/usr/bin/env python3
"""
Mental Health Analysis Project Runner

This script provides a convenient way to run the complete mental health analysis pipeline
or individual components of the project.

Usage:
    python run_analysis.py [command]

Commands:
    setup       - Install dependencies and set up environment
    download    - Download and prepare data
    explore     - Run data exploration analysis
    clean       - Run data cleaning and preprocessing
    analyze     - Run time series analysis
    visualize   - Generate visualizations
    dashboard   - Launch interactive dashboard
    all         - Run complete pipeline (download -> analyze -> visualize)
    help        - Show this help message

Examples:
    python run_analysis.py all         # Run complete analysis
    python run_analysis.py dashboard   # Launch dashboard only
    python run_analysis.py download    # Download data only
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
import time

class MentalHealthAnalysisRunner:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.notebooks_dir = self.project_root / "notebooks"
        self.data_dir = self.project_root / "data"
        
        # Add src to Python path
        sys.path.insert(0, str(self.src_dir))
    
    def setup(self):
        """Set up the project environment"""
        print("🔧 Setting up Mental Health Analysis Project...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required")
            return False
        
        print(f"✓ Python {sys.version.split()[0]} detected")
        
        # Install dependencies
        print("📦 Installing dependencies...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                         check=True, cwd=self.project_root)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            return False
        
        # Create directories
        for dir_path in [self.data_dir / "raw", self.data_dir / "processed", 
                        self.project_root / "visualizations"]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("✅ Project setup completed successfully!")
        return True
    
    def download_data(self):
        """Download and prepare mental health data"""
        print("📥 Downloading mental health data...")
        
        try:
            from data.download_data import DataDownloader
            
            downloader = DataDownloader(data_dir=str(self.data_dir))
            success = downloader.download_all_data()
            
            if success:
                print("✅ Data download completed successfully!")
                
                # Show data info
                info = downloader.get_data_info()
                print(f"\nDownloaded datasets:")
                for category, files in info.items():
                    print(f"  {category}: {len(files)} files")
                return True
            else:
                print("⚠️ Data download failed, will use sample data")
                return True  # Continue with sample data
                
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            print("⚠️ Will proceed with sample data")
            return True
    
    def run_exploration(self):
        """Run data exploration notebook"""
        print("🔍 Running data exploration...")
        
        notebook_path = self.notebooks_dir / "01_data_exploration.ipynb"
        if not notebook_path.exists():
            print(f"❌ Notebook not found: {notebook_path}")
            return False
        
        try:
            cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--execute", 
                   "--to", "notebook", "--inplace", str(notebook_path)]
            subprocess.run(cmd, check=True, cwd=self.project_root)
            print("✅ Data exploration completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Data exploration failed: {e}")
            return False
    
    def run_cleaning(self):
        """Run data cleaning and preprocessing"""
        print("🧹 Running data cleaning and preprocessing...")
        
        try:
            from data.preprocessing import DataPreprocessor
            
            preprocessor = DataPreprocessor(data_dir=str(self.data_dir))
            
            # Load and process data
            raw_data = preprocessor.load_raw_data()
            if raw_data is not None:
                print(f"✓ Raw data loaded: {raw_data.shape}")
                
                # Clean data
                cleaned_data = preprocessor.handle_missing_values(raw_data)
                cleaned_data = preprocessor.standardize_data(cleaned_data)
                cleaned_data = preprocessor.create_features(cleaned_data)
                cleaned_data = preprocessor.add_regional_info(cleaned_data)
                
                # Save cleaned data
                output_path = self.data_dir / "processed" / "mental_health_cleaned.csv"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cleaned_data.to_csv(output_path, index=False)
                
                print(f"✅ Data cleaning completed! Saved to: {output_path}")
                print(f"   Final dataset: {cleaned_data.shape}")
                return True
            else:
                print("❌ Failed to load raw data")
                return False
                
        except Exception as e:
            print(f"❌ Data cleaning failed: {e}")
            return False
    
    def run_analysis(self):
        """Run time series analysis"""
        print("📈 Running time series analysis...")
        
        try:
            from analysis.time_series import TimeSeriesAnalyzer
            import pandas as pd
            import json
            
            # Load cleaned data
            data_path = self.data_dir / "processed" / "mental_health_cleaned.csv"
            if not data_path.exists():
                print("❌ Cleaned data not found. Please run cleaning first.")
                return False
            
            data = pd.read_csv(data_path)
            print(f"✓ Loaded cleaned data: {data.shape}")
            
            # Initialize analyzer
            analyzer = TimeSeriesAnalyzer()
            
            # Analyze global trends
            prevalence_cols = [col for col in data.columns if 'prevalence' in col.lower()]
            if not prevalence_cols:
                print("❌ No prevalence columns found")
                return False
            
            main_metric = prevalence_cols[0]
            global_trends = data.groupby('Year')[main_metric].agg(['mean', 'std', 'count']).reset_index()
            
            # Perform trend analysis
            trend_analysis = analyzer.analyze_trend(global_trends['Year'], global_trends['mean'])
            
            # Save results
            results = {
                'global_trend': trend_analysis,
                'metrics_analyzed': prevalence_cols,
                'data_summary': {
                    'total_records': len(data),
                    'countries': data['Entity'].nunique(),
                    'year_range': [int(data['Year'].min()), int(data['Year'].max())]
                }
            }
            
            results_path = self.data_dir / "processed" / "time_series_analysis_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"✅ Time series analysis completed!")
            print(f"   Global trend: {trend_analysis['trend_direction']}")
            print(f"   Significance: {'Yes' if trend_analysis['is_significant'] else 'No'}")
            return True
            
        except Exception as e:
            print(f"❌ Time series analysis failed: {e}")
            return False
    
    def run_visualization(self):
        """Generate visualizations"""
        print("🎨 Generating visualizations...")
        
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            import plotly.express as px
            
            # Load data
            data_path = self.data_dir / "processed" / "mental_health_cleaned.csv"
            if not data_path.exists():
                print("❌ Cleaned data not found. Please run cleaning first.")
                return False
            
            data = pd.read_csv(data_path)
            prevalence_cols = [col for col in data.columns if 'prevalence' in col.lower()]
            
            if not prevalence_cols:
                print("❌ No prevalence columns found")
                return False
            
            main_metric = prevalence_cols[0]
            latest_year = data['Year'].max()
            
            # Create visualizations directory
            viz_dir = self.project_root / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Generate key visualizations
            print("  📊 Creating global trend plot...")
            
            # Global trend
            global_trend = data.groupby('Year')[main_metric].mean()
            plt.figure(figsize=(12, 8))
            plt.plot(global_trend.index, global_trend.values, 'o-', linewidth=2, markersize=6)
            
            # Add trend line
            import numpy as np
            z = np.polyfit(global_trend.index, global_trend.values, 1)
            plt.plot(global_trend.index, np.poly1d(z)(global_trend.index), '--r', linewidth=2)
            
            plt.xlabel('Year')
            plt.ylabel(f'{main_metric.replace("_", " ").title()} (%)')
            plt.title(f'Global {main_metric.replace("_", " ").title()} Trend')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / 'global_trend.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Regional comparison (if available)
            if 'Region' in data.columns:
                print("  🌍 Creating regional comparison...")
                latest_data = data[data['Year'] == latest_year]
                regional_avg = latest_data.groupby('Region')[main_metric].mean().sort_values()
                
                plt.figure(figsize=(10, 6))
                bars = plt.barh(regional_avg.index, regional_avg.values, color='lightcoral')
                plt.xlabel(f'{main_metric.replace("_", " ").title()} (%)')
                plt.title(f'Regional Comparison - {main_metric.replace("_", " ").title()} ({latest_year})')
                
                # Add value labels
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                           f'{width:.2f}%', ha='left', va='center')
                
                plt.tight_layout()
                plt.savefig(viz_dir / 'regional_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Interactive world map
            print("  🗺️ Creating interactive world map...")
            latest_data = data[data['Year'] == latest_year]
            
            fig = px.choropleth(
                latest_data,
                locations='Entity',
                locationmode='country names',
                color=main_metric,
                hover_name='Entity',
                title=f'Global {main_metric.replace("_", " ").title()} - {latest_year}'
            )
            fig.write_html(str(viz_dir / 'world_map.html'))
            
            print(f"✅ Visualizations generated successfully!")
            print(f"   Saved to: {viz_dir.absolute()}")
            return True
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
            return False
    
    def launch_dashboard(self):
        """Launch interactive dashboard"""
        print("🚀 Launching interactive dashboard...")
        
        try:
            # Import and run dashboard
            dashboard_path = self.src_dir / "dashboard" / "app.py"
            if not dashboard_path.exists():
                print(f"❌ Dashboard not found: {dashboard_path}")
                return False
            
            print("📊 Starting dashboard server...")
            print("🌐 Dashboard will be available at: http://localhost:8050")
            print("⏹️ Press Ctrl+C to stop the dashboard")
            
            # Run dashboard
            subprocess.run([sys.executable, str(dashboard_path)], cwd=self.project_root)
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️ Dashboard stopped by user")
            return True
        except Exception as e:
            print(f"❌ Dashboard launch failed: {e}")
            return False
    
    def run_all(self):
        """Run complete analysis pipeline"""
        print("🎯 Running complete Mental Health Analysis pipeline...")
        print("=" * 60)
        
        steps = [
            ("Setup", self.setup),
            ("Data Download", self.download_data),
            ("Data Cleaning", self.run_cleaning),
            ("Time Series Analysis", self.run_analysis),
            ("Visualization", self.run_visualization)
        ]
        
        for step_name, step_func in steps:
            print(f"\n🔄 Step: {step_name}")
            start_time = time.time()
            
            success = step_func()
            elapsed = time.time() - start_time
            
            if success:
                print(f"✅ {step_name} completed in {elapsed:.1f}s")
            else:
                print(f"❌ {step_name} failed after {elapsed:.1f}s")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 Complete analysis pipeline finished successfully!")
        print("\n📋 What you can do next:")
        print("   🚀 Launch dashboard: python run_analysis.py dashboard")
        print("   📓 Open notebooks: jupyter notebook notebooks/")
        print("   📊 View visualizations: open visualizations/")
        print("   📁 Check processed data: data/processed/")
        
        return True
    
    def show_help(self):
        """Show help message"""
        print(__doc__)

def main():
    runner = MentalHealthAnalysisRunner()
    
    parser = argparse.ArgumentParser(description="Mental Health Analysis Project Runner")
    parser.add_argument('command', nargs='?', default='help',
                       choices=['setup', 'download', 'explore', 'clean', 'analyze', 
                               'visualize', 'dashboard', 'all', 'help'],
                       help='Command to run')
    
    args = parser.parse_args()
    
    # Command mapping
    commands = {
        'setup': runner.setup,
        'download': runner.download_data,
        'explore': runner.run_exploration,
        'clean': runner.run_cleaning,
        'analyze': runner.run_analysis,
        'visualize': runner.run_visualization,
        'dashboard': runner.launch_dashboard,
        'all': runner.run_all,
        'help': runner.show_help
    }
    
    if args.command in commands:
        try:
            success = commands[args.command]()
            if success:
                print(f"\n✅ Command '{args.command}' completed successfully!")
            else:
                print(f"\n❌ Command '{args.command}' failed!")
                sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n⏹️ Command '{args.command}' interrupted by user")
        except Exception as e:
            print(f"\n💥 Unexpected error in '{args.command}': {e}")
            sys.exit(1)
    else:
        runner.show_help()

if __name__ == "__main__":
    main()
