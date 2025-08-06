import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TrafficDecisionEngine:
    def __init__(self):
        self.congestion_thresholds = {
            'low': 40,    # Below 40 km/h average speed
            'medium': 70, # 40-70 km/h average speed
            'high': 100   # Above 70 km/h average speed
        }
        
        self.base_timing = {
            'green': 30,
            'yellow': 5,
            'red': 25
        }
    
    def determine_congestion_level(self, predicted_speed):
        """Determine congestion level based on predicted speed"""
        if predicted_speed < 40:
            return "High"
        elif predicted_speed < 70:
            return "Medium"
        else:
            return "Low"
    
    def get_traffic_light_timing(self, predicted_speed):
        """Calculate optimal traffic light timing based on traffic conditions"""
        congestion_level = self.determine_congestion_level(predicted_speed)
        
        if congestion_level == "High":
            # Longer green light for congested traffic
            return {
                'green': 45,
                'yellow': 5,
                'red': 20
            }
        elif congestion_level == "Medium":
            # Standard timing with slight adjustment
            return {
                'green': 35,
                'yellow': 5,
                'red': 25
            }
        else:
            # Normal timing for free-flowing traffic
            return {
                'green': 30,
                'yellow': 4,
                'red': 26
            }
    
    def get_route_recommendation(self, predicted_speed):
        """Provide route recommendations based on traffic conditions"""
        congestion_level = self.determine_congestion_level(predicted_speed)
        
        if congestion_level == "High":
            return """
            🔴 **High Congestion Detected**
            - Consider alternative routes
            - Use public transportation if available
            - Avoid non-essential trips
            - Expected delay: 15-30 minutes
            """
        elif congestion_level == "Medium":
            return """
            🟡 **Moderate Traffic**
            - Minor delays expected
            - Consider departure time adjustment
            - Alternative routes may be beneficial
            - Expected delay: 5-15 minutes
            """
        else:
            return """
            🟢 **Light Traffic**
            - Optimal driving conditions
            - All routes are clear
            - Good time for travel
            - Expected delay: < 5 minutes
            """
    
    def calculate_signal_optimization(self, vehicle_counts, weather_condition, time_of_day):
        """Calculate optimized signal timing based on multiple factors"""
        base_cycle_time = 60  # seconds
        
        # Adjust for vehicle count
        if vehicle_counts > 100:
            cycle_multiplier = 1.3
        elif vehicle_counts > 50:
            cycle_multiplier = 1.1
        else:
            cycle_multiplier = 0.9
        
        # Adjust for weather
        weather_multiplier = {
            'Sunny': 1.0,
            'Cloudy': 1.05,
            'Rainy': 1.2,
            'Snowy': 1.3,
            'Foggy': 1.4
        }.get(weather_condition, 1.0)
        
        # Adjust for time of day (rush hours)
        if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19:
            time_multiplier = 1.2
        else:
            time_multiplier = 1.0
        
        optimized_cycle = base_cycle_time * cycle_multiplier * weather_multiplier * time_multiplier
        
        # Calculate phase timing
        green_ratio = 0.55
        yellow_ratio = 0.08
        red_ratio = 0.37
        
        return {
            'total_cycle_time': int(optimized_cycle),
            'green_time': int(optimized_cycle * green_ratio),
            'yellow_time': int(optimized_cycle * yellow_ratio),
            'red_time': int(optimized_cycle * red_ratio),
            'efficiency_score': min(100, (100 / weather_multiplier / time_multiplier))
        }
    
    def generate_adaptive_timing(self, traffic_data):
        """Generate adaptive timing based on real-time traffic data"""
        if not traffic_data:
            return self.base_timing
        
        # Analyze traffic patterns
        avg_vehicle_count = np.mean([data.get('vehicle_count', 0) for data in traffic_data])
        avg_speed = np.mean([data.get('speed', 50) for data in traffic_data])
        
        # Calculate adaptive multipliers
        count_multiplier = min(2.0, max(0.5, avg_vehicle_count / 50))
        speed_multiplier = min(1.5, max(0.7, 50 / avg_speed)) if avg_speed > 0 else 1.0
        
        # Apply multipliers
        adaptive_timing = {}
        for phase, base_time in self.base_timing.items():
            if phase == 'green':
                adaptive_timing[phase] = int(base_time * count_multiplier * speed_multiplier)
            elif phase == 'yellow':
                adaptive_timing[phase] = base_time  # Keep yellow constant
            else:  # red
                adaptive_timing[phase] = max(15, int(base_time * (2 - count_multiplier)))
        
        return adaptive_timing
    
    def get_infrastructure_insights(self, predicted_speed, vehicle_count, weather_condition):
        """Provide infrastructure planning insights"""
        insights = []
        
        # Traffic volume analysis
        if vehicle_count > 150:
            insights.append({
                'Priority': 'High',
                'Category': 'Capacity',
                'Recommendation': 'Consider additional lanes or alternate routes',
                'Impact': 'Reduce congestion by 20-30%',
                'Cost_Estimate': 'High ($1-5M)',
                'Timeline': '2-3 years'
            })
        
        # Speed-based recommendations
        if predicted_speed < 30:
            insights.append({
                'Priority': 'High',
                'Category': 'Traffic Flow',
                'Recommendation': 'Implement intelligent traffic signals',
                'Impact': 'Improve average speed by 15-25%',
                'Cost_Estimate': 'Medium ($100K-500K)',
                'Timeline': '6-12 months'
            })
        
        # Weather-based infrastructure
        if weather_condition in ['Rainy', 'Snowy', 'Foggy']:
            insights.append({
                'Priority': 'Medium',
                'Category': 'Safety',
                'Recommendation': 'Enhanced weather monitoring and variable speed limits',
                'Impact': 'Reduce weather-related accidents by 40%',
                'Cost_Estimate': 'Low ($50K-100K)',
                'Timeline': '3-6 months'
            })
        
        # General improvements
        insights.append({
            'Priority': 'Low',
            'Category': 'Technology',
            'Recommendation': 'Deploy IoT sensors for real-time monitoring',
            'Impact': 'Improve traffic management efficiency by 35%',
            'Cost_Estimate': 'Medium ($200K-800K)',
            'Timeline': '1-2 years'
        })
        
        return insights
    
    def calculate_environmental_impact(self, vehicle_count, average_speed, weather_condition):
        """Calculate environmental impact metrics"""
        # Base emissions per vehicle (grams CO2 per km)
        base_emissions = {
            'car': 120,
            'truck': 250,
            'bus': 180,
            'motorcycle': 80,
            'bicycle': 0
        }
        
        # Speed efficiency curve (optimal around 50-60 km/h)
        if average_speed < 20:
            speed_factor = 1.5  # Heavy congestion increases emissions
        elif 20 <= average_speed < 40:
            speed_factor = 1.3
        elif 40 <= average_speed < 80:
            speed_factor = 1.0  # Optimal range
        else:
            speed_factor = 1.2  # High speed increases emissions
        
        # Weather impact on emissions
        weather_factor = {
            'Sunny': 1.0,
            'Cloudy': 1.02,
            'Rainy': 1.15,
            'Snowy': 1.25,
            'Foggy': 1.1
        }.get(weather_condition, 1.0)
        
        # Estimate vehicle type distribution (simplified)
        car_count = int(vehicle_count * 0.75)
        truck_count = int(vehicle_count * 0.15)
        bus_count = int(vehicle_count * 0.05)
        motorcycle_count = int(vehicle_count * 0.05)
        
        # Calculate total emissions
        total_emissions = (
            (car_count * base_emissions['car']) +
            (truck_count * base_emissions['truck']) +
            (bus_count * base_emissions['bus']) +
            (motorcycle_count * base_emissions['motorcycle'])
        ) * speed_factor * weather_factor
        
        return {
            'total_co2_per_hour': total_emissions,
            'co2_per_vehicle': total_emissions / vehicle_count if vehicle_count > 0 else 0,
            'efficiency_rating': max(0, min(100, 100 - (speed_factor - 1) * 50)),
            'recommendations': self._get_environmental_recommendations(speed_factor, weather_factor)
        }
    
    def _get_environmental_recommendations(self, speed_factor, weather_factor):
        """Get environmental improvement recommendations"""
        recommendations = []
        
        if speed_factor > 1.2:
            recommendations.append("Improve traffic flow to reduce emissions")
        
        if weather_factor > 1.1:
            recommendations.append("Implement weather-adaptive traffic management")
        
        recommendations.append("Encourage public transportation and carpooling")
        recommendations.append("Consider electric vehicle charging infrastructure")
        
        return recommendations
    
    def predict_peak_hours(self, historical_data):
        """Predict peak traffic hours based on historical data"""
        if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
            # Return default peak hours if no data
            return {
                'morning_peak': {'start': 7, 'end': 9, 'intensity': 'High'},
                'evening_peak': {'start': 17, 'end': 19, 'intensity': 'High'},
                'lunch_peak': {'start': 12, 'end': 13, 'intensity': 'Medium'}
            }
        
        # Convert timestamp to hour if needed
        if 'timestamp' in historical_data.columns:
            historical_data['hour'] = pd.to_datetime(historical_data['timestamp'], unit='s').dt.hour
        
        # Analyze traffic patterns by hour
        hourly_traffic = historical_data.groupby('hour').agg({
            'vehicle_count': 'mean',
            'average_speed_kmph': 'mean'
        }).round(2)
        
        # Find peak hours (high vehicle count, low speed)
        traffic_intensity = (hourly_traffic['vehicle_count'] / hourly_traffic['vehicle_count'].max()) * 0.6 + \
                          ((100 - hourly_traffic['average_speed_kmph']) / 100) * 0.4
        
        # Identify peak periods
        peak_threshold = traffic_intensity.quantile(0.75)
        peak_hours = hourly_traffic[traffic_intensity >= peak_threshold].index.tolist()
        
        # Group consecutive hours
        peak_periods = []
        if peak_hours:
            current_period = [peak_hours[0]]
            
            for hour in peak_hours[1:]:
                if hour == current_period[-1] + 1:
                    current_period.append(hour)
                else:
                    peak_periods.append(current_period)
                    current_period = [hour]
            peak_periods.append(current_period)
        
        # Format results
        results = {}
        for i, period in enumerate(peak_periods):
            intensity = 'High' if traffic_intensity[period].mean() > 0.8 else \
                       'Medium' if traffic_intensity[period].mean() > 0.6 else 'Low'
            
            results[f'peak_{i+1}'] = {
                'start': min(period),
                'end': max(period),
                'intensity': intensity,
                'avg_vehicles': hourly_traffic.loc[period, 'vehicle_count'].mean(),
                'avg_speed': hourly_traffic.loc[period, 'average_speed_kmph'].mean()
            }
        
        return results
