
import json
import time
import random
import numpy as np
from datetime import datetime, timedelta

class IoTSensorNetwork:
    def __init__(self):
        self.sensors = {
            'speed_sensors': {
                'sensor_001': {'location': 'Main_St_North', 'type': 'radar', 'status': 'active'},
                'sensor_002': {'location': 'Highway_Exit', 'type': 'lidar', 'status': 'active'},
                'sensor_003': {'location': 'Downtown_Central', 'type': 'camera_ai', 'status': 'active'},
                'sensor_004': {'location': 'Industrial_Zone', 'type': 'magnetic_loop', 'status': 'maintenance'}
            },
            'weather_sensors': {
                'weather_001': {'location': 'City_Center', 'type': 'meteorological', 'status': 'active'},
                'weather_002': {'location': 'Highway_Junction', 'type': 'road_surface', 'status': 'active'}
            },
            'air_quality_sensors': {
                'air_001': {'location': 'Main_Intersection', 'type': 'particulate', 'status': 'active'},
                'air_002': {'location': 'School_Zone', 'type': 'co2_nox', 'status': 'active'}
            },
            'noise_sensors': {
                'noise_001': {'location': 'Residential_Area', 'type': 'acoustic', 'status': 'active'},
                'noise_002': {'location': 'Hospital_Zone', 'type': 'acoustic', 'status': 'active'}
            }
        }
        
        self.data_cache = []
        self.fusion_weights = {
            'speed_sensors': 0.4,
            'weather_sensors': 0.25,
            'air_quality_sensors': 0.2,
            'noise_sensors': 0.15
        }
        
    def collect_sensor_data(self):
        """Collect data from all active sensors"""
        current_time = time.time()
        sensor_data = {
            'timestamp': current_time,
            'collected_data': {}
        }
        
        # Collect speed sensor data
        for sensor_id, sensor_info in self.sensors['speed_sensors'].items():
            if sensor_info['status'] == 'active':
                sensor_data['collected_data'][sensor_id] = {
                    'vehicle_count': random.randint(5, 150),
                    'average_speed': random.uniform(15, 85),
                    'speed_variance': random.uniform(5, 25),
                    'lane_occupancy': [random.uniform(0.1, 0.9) for _ in range(3)],
                    'vehicle_classification': {
                        'light_vehicles': random.randint(60, 85),
                        'heavy_vehicles': random.randint(10, 30),
                        'motorcycles': random.randint(2, 15)
                    }
                }
        
        # Collect weather sensor data
        for sensor_id, sensor_info in self.sensors['weather_sensors'].items():
            if sensor_info['status'] == 'active':
                sensor_data['collected_data'][sensor_id] = {
                    'temperature': random.uniform(-10, 40),
                    'humidity': random.uniform(20, 90),
                    'precipitation': random.uniform(0, 25),
                    'wind_speed': random.uniform(0, 30),
                    'visibility': random.uniform(50, 10000),
                    'road_surface_temp': random.uniform(-5, 60),
                    'road_condition': random.choice(['dry', 'wet', 'icy', 'snow'])
                }
        
        # Collect air quality data
        for sensor_id, sensor_info in self.sensors['air_quality_sensors'].items():
            if sensor_info['status'] == 'active':
                sensor_data['collected_data'][sensor_id] = {
                    'pm25': random.uniform(5, 150),
                    'pm10': random.uniform(10, 200),
                    'co2': random.uniform(400, 1500),
                    'nox': random.uniform(10, 200),
                    'air_quality_index': random.randint(20, 180)
                }
        
        # Collect noise data
        for sensor_id, sensor_info in self.sensors['noise_sensors'].items():
            if sensor_info['status'] == 'active':
                sensor_data['collected_data'][sensor_id] = {
                    'decibel_level': random.uniform(45, 85),
                    'frequency_analysis': {
                        'low_freq': random.uniform(0.2, 0.8),
                        'mid_freq': random.uniform(0.3, 0.9),
                        'high_freq': random.uniform(0.1, 0.6)
                    },
                    'noise_source': random.choice(['traffic', 'construction', 'ambient'])
                }
        
        self.data_cache.append(sensor_data)
        return sensor_data
    
    def fuse_sensor_data(self, sensor_readings):
        """Perform sensor data fusion for comprehensive traffic state"""
        fused_data = {
            'timestamp': sensor_readings['timestamp'],
            'location': 'city_wide',
            'fused_metrics': {}
        }
        
        # Extract speed and volume data
        speed_data = []
        volume_data = []
        
        for sensor_id, data in sensor_readings['collected_data'].items():
            if sensor_id.startswith('sensor_'):
                speed_data.append(data['average_speed'])
                volume_data.append(data['vehicle_count'])
        
        # Fuse speed data with confidence weighting
        if speed_data:
            weights = np.array([1.0] * len(speed_data))  # Equal weights for simplicity
            weights /= weights.sum()
            fused_data['fused_metrics']['average_speed'] = np.average(speed_data, weights=weights)
            fused_data['fused_metrics']['speed_confidence'] = 1 - np.std(speed_data) / np.mean(speed_data)
        
        # Fuse volume data
        if volume_data:
            fused_data['fused_metrics']['total_volume'] = sum(volume_data)
            fused_data['fused_metrics']['average_volume'] = np.mean(volume_data)
        
        # Environmental factor fusion
        weather_impact = self._calculate_weather_impact(sensor_readings)
        air_quality_impact = self._calculate_air_quality_impact(sensor_readings)
        noise_impact = self._calculate_noise_impact(sensor_readings)
        
        fused_data['fused_metrics']['environmental_factors'] = {
            'weather_impact_score': weather_impact,
            'air_quality_impact_score': air_quality_impact,
            'noise_impact_score': noise_impact,
            'overall_environmental_score': (weather_impact + air_quality_impact + noise_impact) / 3
        }
        
        # Calculate comprehensive traffic health score
        traffic_health = self._calculate_traffic_health_score(fused_data['fused_metrics'])
        fused_data['fused_metrics']['traffic_health_score'] = traffic_health
        
        return fused_data
    
    def _calculate_weather_impact(self, sensor_readings):
        """Calculate weather impact on traffic"""
        impact_score = 1.0  # 1.0 = no impact, 0.0 = severe impact
        
        for sensor_id, data in sensor_readings['collected_data'].items():
            if sensor_id.startswith('weather_'):
                # Precipitation impact
                if data['precipitation'] > 10:
                    impact_score *= 0.7
                elif data['precipitation'] > 5:
                    impact_score *= 0.85
                
                # Visibility impact
                if data['visibility'] < 100:
                    impact_score *= 0.5
                elif data['visibility'] < 500:
                    impact_score *= 0.75
                
                # Wind impact
                if data['wind_speed'] > 20:
                    impact_score *= 0.8
                
                # Road condition impact
                if data['road_condition'] in ['icy', 'snow']:
                    impact_score *= 0.6
                elif data['road_condition'] == 'wet':
                    impact_score *= 0.8
        
        return max(0.0, min(1.0, impact_score))
    
    def _calculate_air_quality_impact(self, sensor_readings):
        """Calculate air quality impact score"""
        impact_score = 1.0
        
        for sensor_id, data in sensor_readings['collected_data'].items():
            if sensor_id.startswith('air_'):
                aqi = data['air_quality_index']
                
                if aqi > 150:  # Unhealthy
                    impact_score *= 0.6
                elif aqi > 100:  # Unhealthy for sensitive groups
                    impact_score *= 0.75
                elif aqi > 50:  # Moderate
                    impact_score *= 0.9
        
        return max(0.0, min(1.0, impact_score))
    
    def _calculate_noise_impact(self, sensor_readings):
        """Calculate noise pollution impact score"""
        impact_score = 1.0
        
        for sensor_id, data in sensor_readings['collected_data'].items():
            if sensor_id.startswith('noise_'):
                decibel = data['decibel_level']
                
                if decibel > 80:  # Very loud
                    impact_score *= 0.5
                elif decibel > 70:  # Loud
                    impact_score *= 0.7
                elif decibel > 60:  # Moderate
                    impact_score *= 0.85
        
        return max(0.0, min(1.0, impact_score))
    
    def _calculate_traffic_health_score(self, metrics):
        """Calculate overall traffic health score"""
        health_factors = []
        
        # Speed factor
        if 'average_speed' in metrics:
            speed = metrics['average_speed']
            if 40 <= speed <= 70:  # Optimal speed range
                speed_factor = 1.0
            elif speed < 20:  # Very slow
                speed_factor = 0.3
            elif speed < 40:  # Slow
                speed_factor = 0.6
            else:  # Too fast
                speed_factor = 0.8
            health_factors.append(speed_factor)
        
        # Environmental factor
        if 'environmental_factors' in metrics:
            env_score = metrics['environmental_factors']['overall_environmental_score']
            health_factors.append(env_score)
        
        # Volume factor (relative to capacity)
        if 'average_volume' in metrics:
            volume = metrics['average_volume']
            if volume < 30:  # Low traffic
                volume_factor = 0.8
            elif volume <= 80:  # Optimal
                volume_factor = 1.0
            elif volume <= 120:  # High
                volume_factor = 0.7
            else:  # Very high
                volume_factor = 0.4
            health_factors.append(volume_factor)
        
        return np.mean(health_factors) if health_factors else 0.5
    
    def get_predictive_maintenance_alerts(self):
        """Generate predictive maintenance alerts for sensors"""
        alerts = []
        current_time = time.time()
        
        for sensor_category, sensors in self.sensors.items():
            for sensor_id, sensor_info in sensors.items():
                # Simulate sensor health monitoring
                uptime_hours = random.randint(100, 8760)  # 100 hours to 1 year
                last_calibration = random.randint(1, 365)  # days since calibration
                error_rate = random.uniform(0, 0.1)  # 0-10% error rate
                
                # Generate alerts based on conditions
                if sensor_info['status'] == 'maintenance':
                    alerts.append({
                        'sensor_id': sensor_id,
                        'alert_type': 'maintenance_required',
                        'priority': 'high',
                        'description': 'Sensor is currently in maintenance mode',
                        'estimated_fix_time': '2-4 hours'
                    })
                elif uptime_hours > 8000:  # Over 8000 hours
                    alerts.append({
                        'sensor_id': sensor_id,
                        'alert_type': 'replacement_recommended',
                        'priority': 'medium',
                        'description': f'Sensor has been operational for {uptime_hours} hours',
                        'estimated_fix_time': '4-8 hours'
                    })
                elif last_calibration > 90:  # Over 90 days since calibration
                    alerts.append({
                        'sensor_id': sensor_id,
                        'alert_type': 'calibration_required',
                        'priority': 'medium',
                        'description': f'Last calibration was {last_calibration} days ago',
                        'estimated_fix_time': '1-2 hours'
                    })
                elif error_rate > 0.05:  # Over 5% error rate
                    alerts.append({
                        'sensor_id': sensor_id,
                        'alert_type': 'accuracy_degraded',
                        'priority': 'high',
                        'description': f'Error rate is {error_rate:.1%}',
                        'estimated_fix_time': '2-6 hours'
                    })
        
        return alerts
    
    def optimize_sensor_placement(self, traffic_patterns):
        """Suggest optimal sensor placement based on traffic patterns"""
        recommendations = []
        
        # Analyze current coverage
        covered_locations = set()
        for sensor_category, sensors in self.sensors.items():
            for sensor_info in sensors.values():
                covered_locations.add(sensor_info['location'])
        
        # Suggest new sensor locations based on traffic density
        high_traffic_areas = [
            'School_Zone_Main', 'Hospital_Access_Road', 'Shopping_Center_Entry',
            'Stadium_Vicinity', 'Airport_Connection', 'Port_Access'
        ]
        
        for location in high_traffic_areas:
            if location not in covered_locations:
                recommendations.append({
                    'location': location,
                    'sensor_type': 'multi_modal',  # Camera + Radar + Environmental
                    'priority': 'high',
                    'expected_improvement': '15-25% better traffic management',
                    'installation_cost': '$15,000 - $25,000',
                    'roi_estimate': '6-12 months'
                })
        
        return recommendations

