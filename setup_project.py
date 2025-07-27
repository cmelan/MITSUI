#!/usr/bin/env python3
"""
MITSUI&CO. Commodity Prediction Challenge - Project Setup Script

This script helps set up the project environment and checks dependencies.
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible."""
    print("=== Python Version Check ===")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_dependencies():
    """Check if required packages are installed."""
    print("\n=== Dependency Check ===")
    
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn',
        'xgboost', 'lightgbm', 'catboost', 'jupyter', 'optuna', 'shap', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies are installed")
        return True

def create_directories():
    """Create necessary directories if they don't exist."""
    print("\n=== Directory Structure ===")
    
    directories = ['data', 'models', 'outputs', 'notebooks']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def check_data_files():
    """Check if competition data files are present."""
    print("\n=== Data Files Check ===")
    
    required_files = [
        'data/train.csv',
        'data/test.csv', 
        'data/train_labels.csv',
        'data/target_pairs.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({file_size:,} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing data files: {len(missing_files)}")
        print("Please download the competition data and place it in the data/ directory")
        print("You can find the data at: https://kaggle.com/competitions/mitsui-commodity-prediction-challenge")
        return False
    else:
        print("\n✅ All data files are present")
        return True

def test_imports():
    """Test if custom modules can be imported."""
    print("\n=== Custom Module Test ===")
    
    try:
        sys.path.append('src')
        
        modules = [
            'data_processing',
            'models', 
            'ensemble',
            'feature_selection',
            'cv',
            'tuning'
        ]
        
        for module in modules:
            try:
                importlib.import_module(module)
                print(f"✅ {module}")
            except ImportError as e:
                print(f"❌ {module} - {e}")
                return False
        
        print("\n✅ All custom modules can be imported")
        return True
        
    except Exception as e:
        print(f"❌ Error testing imports: {e}")
        return False

def generate_sample_data():
    """Generate sample data for testing if real data is not available."""
    print("\n=== Sample Data Generation ===")
    
    if not os.path.exists('data/train.csv'):
        print("Creating sample data for testing...")
        
        import pandas as pd
        import numpy as np
        
        # Generate sample dates
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Generate sample features
        np.random.seed(42)
        sample_data = {
            'date_id': dates,
            'LME_Aluminum': np.random.randn(1000).cumsum() + 2000,
            'LME_Copper': np.random.randn(1000).cumsum() + 8000,
            'LME_Zinc': np.random.randn(1000).cumsum() + 2500,
            'JPX_Nikkei': np.random.randn(1000).cumsum() + 25000,
            'JPX_Topix': np.random.randn(1000).cumsum() + 1800,
            'US_SP500': np.random.randn(1000).cumsum() + 4000,
            'US_Nasdaq': np.random.randn(1000).cumsum() + 12000,
            'Forex_USDJPY': np.random.randn(1000).cumsum() + 110,
            'Forex_EURUSD': np.random.randn(1000).cumsum() + 1.1
        }
        
        # Create train data
        train_df = pd.DataFrame(sample_data)
        train_df.to_csv('data/train.csv', index=False)
        print("✅ Created sample train.csv")
        
        # Create test data (last 200 days)
        test_dates = pd.date_range('2023-01-01', periods=200, freq='D')
        test_data = {k: np.random.randn(200).cumsum() + v[0] for k, v in sample_data.items() if k != 'date_id'}
        test_data['date_id'] = test_dates
        test_df = pd.DataFrame(test_data)
        test_df.to_csv('data/test.csv', index=False)
        print("✅ Created sample test.csv")
        
        # Create train labels with multiple targets
        labels_data = {'date_id': dates}
        
        # Create 4 target columns to match our sample target_pairs
        for i in range(4):
            labels_data[f'target_{i}'] = np.random.randn(1000).cumsum()
        
        labels_df = pd.DataFrame(labels_data)
        labels_df.to_csv('data/train_labels.csv', index=False)
        print("✅ Created sample train_labels.csv")
        
                # Create target pairs with proper structure
        pairs_data = {
            'target': ['target_0', 'target_1', 'target_2', 'target_3'],
            'lag': [1, 1, 2, 1],
            'pair': [
                'feature_1',
                'feature_1 - feature_2', 
                'feature_2 - feature_3',
                'feature_3'
            ]
        }
        pairs_df = pd.DataFrame(pairs_data)
        pairs_df.to_csv('data/target_pairs.csv', index=False)
        print("✅ Created sample target_pairs.csv")
        
        print("\n📝 Sample data created for testing purposes")
        print("Replace with actual competition data when available")
        
        return True
    else:
        print("✅ Real data files already exist")
        return True

def main():
    """Main setup function."""
    print("🚀 MITSUI&CO. Commodity Prediction Challenge - Project Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 To install dependencies, run:")
        print("pip install -r requirements.txt")
        return False
    
    # Create directories
    create_directories()
    
    # Test custom modules
    if not test_imports():
        print("\n❌ Custom module import failed")
        return False
    
    # Check data files
    data_available = check_data_files()
    
    if not data_available:
        # Generate sample data for testing
        generate_sample_data()
    
    print("\n" + "=" * 60)
    print("✅ Project setup completed successfully!")
    
    print("\n📋 Next Steps:")
    print("1. If you have real competition data, replace the sample files in data/")
    print("2. Run the EDA notebook: jupyter notebook notebooks/eda.ipynb")
    print("3. Run the baseline model: jupyter notebook notebooks/baseline_model.ipynb")
    print("4. Explore the source code in src/ directory")
    print("5. Start building your models!")
    
    print("\n🔗 Competition Link:")
    print("https://kaggle.com/competitions/mitsui-commodity-prediction-challenge")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 