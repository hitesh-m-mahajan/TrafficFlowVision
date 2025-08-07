
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
warnings.filterwarnings('ignore')

class TrafficAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=200
        )
        self.scaler = StandardScaler()
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.baseline_established = False
        self.normal_patterns = {}
        
    def establish_baseline(self, historical_data):
        """Establish baseline traffic patterns"""
        if isinstance(historical_data, pd.DataFrame):
            # Extract features for anomaly detection
            features = self._extract_anomaly_features(historical_data)
            
            # Fit isolation forest on normal data
            scaled_features = self.scaler.fit_transform(features)
            self.isolation_forest.fit(scaled_features)
            
            # Establish normal patterns by hour and day
            historical_data['datetime'] = pd.to_datetime(historical_data['timestamp'], unit='s')
            historical_data['hour'] = historical_data['datetime'].dt.hour
            historical_data['day_of_week'] = historical_data['datetime'].dt.dayofweek
            
            self.normal_patterns = {
                'hourly_speed': historical_data.groupby('hour')['average_speed_kmph'].agg(['mean', 'std']).to_dict(),
                'hourly_volume': historical_data.groupby('hour')['vehicle_count'].agg(['mean', 'std']).to_dict(),
                'daily_patterns': historical_data.groupby(['day_of_week', 'hour']).agg({
                    'average_speed_kmph': ['mean', 'std'],
                    'vehicle_count': ['mean', 'std']
                }).to_dict()
            }
            
            self.baseline_established = True
        
    def detect_anomalies(self, current_data):
        """Detect anomalies in current traffic data"""
        if not self.baseline_established:
            return {'anomaly_detected': False, 'reason': 'Baseline not established'}
        
        anomalies = {
            'anomaly_detected': False,
            'anomaly_type': [],
            'severity': 'low',
            'confidence': 0.0,
            'recommendations': []
        }
        
        # Statistical anomaly detection
        current_hour = pd.to_datetime(current_data.get('timestamp', 0), unit='s').hour
        current_speed = current_data.get('average_speed_kmph', 0)
        current_volume = current_data.get('vehicle_count', 0)
        
        # Check speed anomalies
        normal_speed = self.normal_patterns['hourly_speed']['mean'][current_hour]
        speed_std = self.normal_patterns['hourly_speed']['std'][current_hour]
        
        speed_z_score = abs(current_speed - normal_speed) / (speed_std + 1e-8)
        if speed_z_score > 2.5:  # Beyond 2.5 standard deviations
            anomalies['anomaly_detected'] = True
            anomalies['anomaly_type'].append('speed_anomaly')
            if current_speed < normal_speed:
                anomalies['recommendations'].append('Investigate potential traffic incident or congestion')
            else:
                anomalies['recommendations'].append('Verify speed limit compliance and safety measures')
        
        # Check volume anomalies
        normal_volume = self.normal_patterns['hourly_volume']['mean'][current_hour]
        volume_std = self.normal_patterns['hourly_volume']['std'][current_hour]
        
        volume_z_score = abs(current_volume - normal_volume) / (volume_std + 1e-8)
        if volume_z_score > 2.5:
            anomalies['anomaly_detected'] = True
            anomalies['anomaly_type'].append('volume_anomaly')
            if current_volume > normal_volume:
                anomalies['recommendations'].append('Prepare for high traffic volume management')
            else:
                anomalies['recommendations'].append('Investigate unusually low traffic volume')
        
        # Machine learning based anomaly detection
        features = np.array([[
            current_speed,
            current_volume,
            current_hour,
            current_data.get('weather_encoded', 0),
            current_data.get('is_weekend', 0)
        ]])
        
        scaled_features = self.scaler.transform(features)
        anomaly_score = self.isolation_forest.decision_function(scaled_features)[0]
        is_ml_anomaly = self.isolation_forest.predict(scaled_features)[0] == -1
        
        if is_ml_anomaly:
            anomalies['anomaly_detected'] = True
            anomalies['anomaly_type'].append('ml_pattern_anomaly')
            anomalies['recommendations'].append('Unusual traffic pattern detected by ML model')
        
        # Calculate overall confidence and severity
        max_z_score = max(speed_z_score, volume_z_score)
        anomalies['confidence'] = min(0.95, max_z_score / 5.0)  # Normalize to 0-1
        
        if max_z_score > 4:
            anomalies['severity'] = 'high'
        elif max_z_score > 3:
            anomalies['severity'] = 'medium'
        else:
            anomalies['severity'] = 'low'
        
        return anomalies
    
    def _extract_anomaly_features(self, data):
        """Extract features for anomaly detection"""
        features = []
        
        for _, row in data.iterrows():
            feature_vector = [
                row['average_speed_kmph'],
                row['vehicle_count'],
                pd.to_datetime(row['timestamp'], unit='s').hour,
                row.get('weather_encoded', 0),
                row.get('is_weekend', 0)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def detect_traffic_incidents(self, recent_data):
        """Detect potential traffic incidents"""
        incidents = []
        
        if len(recent_data) < 5:
            return incidents
        
        # Sudden speed drops
        speeds = [d.get('average_speed_kmph', 0) for d in recent_data]
        speed_changes = np.diff(speeds)
        
        # Look for sudden drops > 20 km/h
        for i, change in enumerate(speed_changes):
            if change < -20:
                incidents.append({
                    'type': 'sudden_slowdown',
                    'severity': 'high' if change < -40 else 'medium',
                    'location': recent_data[i+1].get('location', 'unknown'),
                    'timestamp': recent_data[i+1].get('timestamp', 0),
                    'description': f'Sudden speed drop of {abs(change):.1f} km/h detected'
                })
        
        # Persistent low speeds
        recent_speeds = speeds[-3:]  # Last 3 readings
        if all(speed < 15 for speed in recent_speeds):
            incidents.append({
                'type': 'persistent_congestion',
                'severity': 'medium',
                'location': recent_data[-1].get('location', 'unknown'),
                'timestamp': recent_data[-1].get('timestamp', 0),
                'description': 'Persistent severe congestion detected'
            })
        
        return incidents
    
    def predict_future_anomalies(self, current_trends):
        """Predict potential future anomalies based on current trends"""
        predictions = []
        
        # Analyze trends
        if len(current_trends) >= 3:
            speeds = [d.get('average_speed_kmph', 0) for d in current_trends]
            volumes = [d.get('vehicle_count', 0) for d in current_trends]
            
            # Calculate trends
            speed_trend = np.polyfit(range(len(speeds)), speeds, 1)[0]
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            
            # Predict future values
            future_speed = speeds[-1] + speed_trend * 3  # 3 time steps ahead
            future_volume = volumes[-1] + volume_trend * 3
            
            # Check if future values would be anomalous
            current_hour = pd.to_datetime(current_trends[-1].get('timestamp', 0), unit='s').hour
            
            if self.baseline_established:
                normal_speed = self.normal_patterns['hourly_speed']['mean'][current_hour]
                normal_volume = self.normal_patterns['hourly_volume']['mean'][current_hour]
                
                if abs(future_speed - normal_speed) > 30:
                    predictions.append({
                        'type': 'speed_anomaly_predicted',
                        'time_horizon': '15_minutes',
                        'predicted_value': future_speed,
                        'normal_value': normal_speed,
                        'confidence': 0.7
                    })
                
                if abs(future_volume - normal_volume) > 50:
                    predictions.append({
                        'type': 'volume_anomaly_predicted',
                        'time_horizon': '15_minutes',
                        'predicted_value': future_volume,
                        'normal_value': normal_volume,
                        'confidence': 0.7
                    })
        
        return predictions

