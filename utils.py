import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(filepath=None):
    """Load traffic dataset from CSV file"""
    if filepath is None:
        filepath = "attached_assets/traffic_weather_speed_dataset_1754508149737.csv"
    
    try:
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully: {len(df)} records")
        return df
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None

def preprocess_data(df):
    """Preprocess traffic data for machine learning"""
    if df is None or df.empty:
        raise ValueError("Dataset is empty or None")
    
    # Create a copy of the dataframe
    data = df.copy()
    
    # Convert timestamp to datetime features
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
    data['hour'] = data['datetime'].dt.hour
    data['day_of_week'] = data['datetime'].dt.dayofweek
    data['month'] = data['datetime'].dt.month
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    
    # Rush hour indicator
    data['is_rush_hour'] = ((data['hour'].between(7, 9)) | 
                           (data['hour'].between(17, 19))).astype(int)
    
    # Encode categorical variables
    le_location = LabelEncoder()
    le_weather = LabelEncoder()
    
    data['location_encoded'] = le_location.fit_transform(data['location'])
    data['weather_encoded'] = le_weather.fit_transform(data['weather'])
    
    # Select features for ML
    feature_columns = ['vehicle_count', 'location_encoded', 'weather_encoded', 
                      'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour']
    
    X = data[feature_columns]
    y = data['average_speed_kmph']  # Target variable
    
    return X, y

def create_time_series_features(df, target_col='average_speed_kmph'):
    """Create time series features for LSTM model"""
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp').copy()
    
    # Create lagged features
    for lag in [1, 2, 3, 6, 12, 24]:
        df_sorted[f'{target_col}_lag_{lag}'] = df_sorted[target_col].shift(lag)
    
    # Rolling statistics
    for window in [3, 6, 12]:
        df_sorted[f'{target_col}_rolling_mean_{window}'] = df_sorted[target_col].rolling(window=window).mean()
        df_sorted[f'{target_col}_rolling_std_{window}'] = df_sorted[target_col].rolling(window=window).std()
    
    # Remove rows with NaN values
    df_sorted = df_sorted.dropna()
    
    return df_sorted

def calculate_traffic_metrics(df):
    """Calculate various traffic metrics"""
    metrics = {}
    
    if df is None or df.empty:
        return metrics
    
    # Basic statistics
    metrics['total_records'] = len(df)
    metrics['avg_speed'] = df['average_speed_kmph'].mean()
    metrics['avg_vehicle_count'] = df['vehicle_count'].mean()
    metrics['speed_std'] = df['average_speed_kmph'].std()
    metrics['vehicle_count_std'] = df['vehicle_count'].std()
    
    # Time-based metrics
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    metrics['date_range'] = {
        'start': df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
        'end': df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Location metrics
    metrics['unique_locations'] = df['location'].nunique()
    metrics['location_stats'] = df.groupby('location').agg({
        'vehicle_count': ['mean', 'std'],
        'average_speed_kmph': ['mean', 'std']
    }).round(2).to_dict()
    
    # Weather impact
    metrics['weather_distribution'] = df['weather'].value_counts().to_dict()
    metrics['weather_impact'] = df.groupby('weather')['average_speed_kmph'].mean().round(2).to_dict()
    
    # Traffic patterns
    df['hour'] = df['datetime'].dt.hour
    metrics['hourly_patterns'] = df.groupby('hour').agg({
        'vehicle_count': 'mean',
        'average_speed_kmph': 'mean'
    }).round(2).to_dict()
    
    # Congestion analysis
    df['congestion_score'] = (df['vehicle_count'] / df['vehicle_count'].max()) * 0.6 + \
                            ((100 - df['average_speed_kmph']) / 100) * 0.4
    
    metrics['congestion_levels'] = {
        'low': (df['congestion_score'] < 0.33).sum(),
        'medium': ((df['congestion_score'] >= 0.33) & (df['congestion_score'] < 0.66)).sum(),
        'high': (df['congestion_score'] >= 0.66).sum()
    }
    
    return metrics

def validate_data_quality(df):
    """Validate data quality and return report"""
    quality_report = {}
    
    if df is None:
        quality_report['status'] = 'error'
        quality_report['message'] = 'Dataset is None'
        return quality_report
    
    # Check for missing values
    missing_values = df.isnull().sum()
    quality_report['missing_values'] = missing_values.to_dict()
    quality_report['missing_percentage'] = (missing_values / len(df) * 100).round(2).to_dict()
    
    # Check data types
    quality_report['data_types'] = df.dtypes.to_dict()
    
    # Check for duplicates
    quality_report['duplicate_rows'] = df.duplicated().sum()
    
    # Check value ranges
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    quality_report['value_ranges'] = {}
    
    for col in numeric_columns:
        quality_report['value_ranges'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    # Check for outliers (using IQR method)
    quality_report['outliers'] = {}
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        quality_report['outliers'][col] = outliers
    
    # Overall quality score
    total_issues = sum(missing_values) + quality_report['duplicate_rows'] + sum(quality_report['outliers'].values())
    quality_report['quality_score'] = max(0, min(100, 100 - (total_issues / len(df)) * 100))
    
    quality_report['status'] = 'good' if quality_report['quality_score'] > 80 else 'needs_attention'
    
    return quality_report

def save_model_results(results, filepath="model_results.json"):
    """Save model training results to file"""
    try:
        # Convert numpy types to Python native types for JSON serialization
        serializable_results = {}
        
        for model_name, result in results.items():
            serializable_results[model_name] = {}
            for key, value in result.items():
                if key == 'model':
                    continue  # Skip model object
                elif key == 'confusion_matrix':
                    serializable_results[model_name][key] = value.tolist()
                elif key == 'predictions':
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, (np.int64, np.float64)):
                    serializable_results[model_name][key] = float(value)
                else:
                    serializable_results[model_name][key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Model results saved to {filepath}")
        return True
    
    except Exception as e:
        print(f"Error saving model results: {str(e)}")
        return False

def load_model_results(filepath="model_results.json"):
    """Load model training results from file"""
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        print(f"Model results loaded from {filepath}")
        return results
    
    except FileNotFoundError:
        print(f"Results file not found: {filepath}")
        return {}
    except Exception as e:
        print(f"Error loading model results: {str(e)}")
        return {}

def generate_sample_predictions(df, n_samples=100):
    """Generate sample predictions for demonstration"""
    if df is None or df.empty:
        return []
    
    # Select random samples
    sample_data = df.sample(min(n_samples, len(df)))
    
    predictions = []
    for _, row in sample_data.iterrows():
        # Simulate prediction based on vehicle count and weather
        base_speed = 50
        
        # Weather impact
        weather_impact = {
            'Sunny': 1.0,
            'Cloudy': 0.95,
            'Rainy': 0.8,
            'Snowy': 0.7,
            'Foggy': 0.75
        }.get(row['weather'], 1.0)
        
        # Vehicle count impact (inverse relationship)
        if row['vehicle_count'] > 100:
            count_impact = 0.7
        elif row['vehicle_count'] > 50:
            count_impact = 0.85
        else:
            count_impact = 1.0
        
        # Add some randomness
        random_factor = np.random.normal(1.0, 0.1)
        
        predicted_speed = base_speed * weather_impact * count_impact * random_factor
        predicted_speed = max(10, min(120, predicted_speed))  # Clamp values
        
        predictions.append({
            'actual': row['average_speed_kmph'],
            'predicted': round(predicted_speed, 2),
            'vehicle_count': row['vehicle_count'],
            'weather': row['weather'],
            'location': row['location']
        })
    
    return predictions

def format_timestamp(timestamp):
    """Format Unix timestamp to readable datetime"""
    try:
        if isinstance(timestamp, str):
            timestamp = float(timestamp)
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, TypeError):
        return "Invalid timestamp"

def calculate_model_comparison(results):
    """Calculate model comparison metrics"""
    if not results:
        return {}
    
    comparison = {}
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for metric in metrics:
        comparison[metric] = {}
        values = [result.get(metric, 0) for result in results.values()]
        
        if values:
            comparison[metric]['best'] = max(values)
            comparison[metric]['worst'] = min(values)
            comparison[metric]['average'] = np.mean(values)
            comparison[metric]['std'] = np.std(values)
            
            # Find best model for this metric
            best_model = max(results.keys(), key=lambda k: results[k].get(metric, 0))
            comparison[metric]['best_model'] = best_model
    
    return comparison

def create_feature_importance_data(feature_names, importance_scores):
    """Create feature importance data for visualization"""
    if len(feature_names) != len(importance_scores):
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    return importance_df

def export_results_to_csv(results, filepath="traffic_analysis_results.csv"):
    """Export analysis results to CSV file"""
    try:
        # Flatten results for CSV export
        flattened_data = []
        
        for model_name, result in results.items():
            row = {'model': model_name}
            for key, value in result.items():
                if key not in ['model', 'confusion_matrix', 'predictions']:
                    row[key] = value
            flattened_data.append(row)
        
        df_results = pd.DataFrame(flattened_data)
        df_results.to_csv(filepath, index=False)
        
        print(f"Results exported to {filepath}")
        return True
    
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        return False

def get_system_info():
    """Get system information for diagnostics"""
    import platform
    import psutil
    
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return info
