"""
Mental Health Analysis Project - Final Setup and Validation

This script performs final setup steps and validates that all components 
of the Mental Health Analysis project are properly configured and functional.
"""

import sys
import os
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def check_dependencies():
    """Check if all required packages are available"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 
        'dash', 'scipy', 'statsmodels', 'sklearn', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def validate_project_structure():
    """Validate project directory structure"""
    project_root = Path(__file__).parent
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/external",
        "notebooks",
        "src/data",
        "src/analysis",
        "src/visualization",
        "src/dashboard",
        "visualizations"
    ]
    
    required_files = [
        "requirements.txt",
        "README.md",
        "config.py",
        "run_analysis.py",
        "src/data/__init__.py",
        "src/data/download_data.py",
        "src/data/preprocessing.py",
        "src/analysis/__init__.py",
        "src/analysis/time_series.py",
        "src/analysis/statistical_tests.py",
        "src/visualization/__init__.py",
        "src/visualization/static_plots.py",
        "src/visualization/interactive_plots.py",
        "src/dashboard/__init__.py",
        "src/dashboard/app.py",
        "notebooks/01_data_exploration.ipynb"
    ]
    
    print("\n📁 Validating project structure...")
    
    # Check directories
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ Directory: {dir_path}")
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"🔧 Created: {dir_path}")
    
    # Check files
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ File: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path}")
    
    if missing_files:
        print(f"\n⚠️ Warning: {len(missing_files)} files are missing")
        return False
    
    return True

def test_imports():
    """Test importing custom modules"""
    print("\n🔧 Testing custom module imports...")
    
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    # Add src to path
    sys.path.insert(0, str(src_dir))
    
    modules_to_test = [
        ("data.download_data", "DataDownloader"),
        ("data.preprocessing", "DataPreprocessor"),
        ("analysis.time_series", "TimeSeriesAnalyzer"),
        ("analysis.statistical_tests", "StatisticalAnalyzer"),
        ("visualization.static_plots", "StaticPlotter"),
        ("visualization.interactive_plots", "InteractivePlotter"),
        ("dashboard.app", "MentalHealthDashboard")
    ]
    
    failed_imports = []
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {module_name}.{class_name}")
        except Exception as e:
            failed_imports.append((module_name, class_name, str(e)))
            print(f"❌ {module_name}.{class_name} - {e}")
    
    if failed_imports:
        print(f"\n⚠️ {len(failed_imports)} import failures detected")
        return False
    
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("\n📊 Creating sample data for testing...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample mental health data
        np.random.seed(42)
        
        countries = ['United States', 'United Kingdom', 'Germany', 'France', 'Japan', 
                    'Canada', 'Australia', 'Brazil', 'India', 'China']
        years = list(range(1990, 2024))
        
        sample_data = []
        for country in countries:
            base_depression = np.random.uniform(3, 8)
            base_anxiety = np.random.uniform(2, 6)
            
            for year in years:
                trend_factor = (year - 1990) * 0.025
                noise = np.random.normal(0, 0.3)
                
                sample_data.append({
                    'Entity': country,
                    'Year': year,
                    'Depression_prevalence': max(0, base_depression + trend_factor + noise),
                    'Anxiety_prevalence': max(0, base_anxiety + trend_factor * 0.8 + noise),
                    'Population': np.random.randint(10000000, 1400000000)
                })
        
        df = pd.DataFrame(sample_data)
        
        # Save sample data
        project_root = Path(__file__).parent
        data_dir = project_root / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = data_dir / "mental_health_prevalence.csv"
        df.to_csv(output_path, index=False)
        
        print(f"✅ Sample data created: {output_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Countries: {df['Entity'].nunique()}")
        print(f"   Years: {df['Year'].min()}-{df['Year'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Running quick functionality test...")
    
    try:
        # Test data processing
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        from data.preprocessing import DataPreprocessor
        import pandas as pd
        
        # Load sample data
        data_path = Path(__file__).parent / "data" / "raw" / "mental_health_prevalence.csv"
        
        if data_path.exists():
            data = pd.read_csv(data_path)
            print(f"✅ Data loading: {data.shape}")
            
            # Test preprocessing
            preprocessor = DataPreprocessor()
            quality_report = preprocessor.assess_data_quality(data)
            print(f"✅ Data quality assessment: {quality_report['total_records']} records")
            
            # Test basic analysis
            from analysis.time_series import TimeSeriesAnalyzer
            analyzer = TimeSeriesAnalyzer(data)
            
            global_trend = data.groupby('Year')['Depression_prevalence'].mean()
            trend_result = analyzer.analyze_trend(global_trend.index, global_trend.values)
            print(f"✅ Trend analysis: {trend_result['trend_direction']}")
            
            return True
        else:
            print("❌ Sample data not found")
            return False
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def main():
    """Main setup and validation function"""
    print("🚀 Mental Health Analysis Project - Setup & Validation")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Project Structure", validate_project_structure),
        ("Module Imports", test_imports),
        ("Sample Data", create_sample_data),
        ("Quick Test", run_quick_test)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}:")
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed with error: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 Project setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run complete analysis: python run_analysis.py all")
        print("   2. Launch dashboard: python run_analysis.py dashboard")
        print("   3. Open notebooks: jupyter notebook notebooks/")
        print("   4. Check documentation: open README.md")
        
        print("\n🔧 Quick commands:")
        print("   python run_analysis.py help     # Show all available commands")
        print("   python run_analysis.py setup    # Install dependencies")
        print("   python run_analysis.py download # Download real data")
        print("   python run_analysis.py all      # Run full pipeline")
        
    else:
        print("❌ Setup validation failed!")
        print("   Please check the errors above and fix them before proceeding.")
        print("   You may need to install missing dependencies or check file paths.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
