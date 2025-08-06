import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class TrafficVisualizer:
    def __init__(self):
        self.color_palette = {
            'primary': '#FF6B6B',
            'secondary': '#4ECDC4', 
            'success': '#45B7D1',
            'warning': '#FFA07A',
            'danger': '#FF4757',
            'info': '#5352ED'
        }
        
        self.weather_colors = {
            'Sunny': '#FFD700',
            'Cloudy': '#B0C4DE',
            'Rainy': '#4169E1',
            'Snowy': '#F0F8FF',
            'Foggy': '#696969'
        }
    
    def create_density_heatmap(self, image, density_map):
        """Create traffic density heatmap overlay on image"""
        if density_map is None or image is None:
            return image
        
        if CV2_AVAILABLE:
            # Normalize density map
            normalized_density = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
            
            # Create heatmap
            heatmap = cv2.applyColorMap(normalized_density.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Resize heatmap to match image size if needed
            if heatmap.shape[:2] != image.shape[:2]:
                heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            
            # Overlay heatmap on original image
            overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        else:
            # Simple heatmap without OpenCV
            normalized_density = ((density_map - density_map.min()) * 255 / 
                                (density_map.max() - density_map.min() + 1e-8)).astype(np.uint8)
            
            # Create a simple red-scale heatmap
            heatmap = np.zeros((*normalized_density.shape, 3), dtype=np.uint8)
            heatmap[:, :, 0] = normalized_density  # Red channel
            
            # Resize if needed using simple numpy operations
            if heatmap.shape[:2] != image.shape[:2]:
                # Simple nearest neighbor resize
                h_ratio = image.shape[0] / heatmap.shape[0]
                w_ratio = image.shape[1] / heatmap.shape[1]
                
                new_heatmap = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
                for i in range(image.shape[0]):
                    for j in range(image.shape[1]):
                        orig_i = int(i / h_ratio)
                        orig_j = int(j / w_ratio)
                        if orig_i < heatmap.shape[0] and orig_j < heatmap.shape[1]:
                            new_heatmap[i, j] = heatmap[orig_i, orig_j]
                heatmap = new_heatmap
            
            # Simple alpha blending
            overlay = (image * 0.6 + heatmap * 0.4).astype(np.uint8)
        
        return overlay
    
    def analyze_weather_impact(self, df):
        """Create weather impact analysis visualization"""
        # Group by weather and calculate metrics
        weather_stats = df.groupby('weather').agg({
            'average_speed_kmph': ['mean', 'std'],
            'vehicle_count': ['mean', 'std']
        }).round(2)
        
        weather_stats.columns = ['avg_speed', 'speed_std', 'avg_vehicles', 'vehicles_std']
        weather_stats = weather_stats.reset_index()
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Speed by Weather', 'Speed Variability', 
                          'Vehicle Count by Weather', 'Weather Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "domain"}]]
        )
        
        # Average speed by weather
        fig.add_trace(
            go.Bar(
                x=weather_stats['weather'],
                y=weather_stats['avg_speed'],
                name='Average Speed',
                marker_color=[self.weather_colors.get(w, '#666666') for w in weather_stats['weather']],
                error_y=dict(
                    type='data',
                    array=weather_stats['speed_std'],
                    visible=True
                )
            ),
            row=1, col=1
        )
        
        # Speed variability (box plot data simulation)
        weather_groups = df.groupby('weather')['average_speed_kmph']
        for weather, speeds in weather_groups:
            fig.add_trace(
                go.Box(
                    y=speeds,
                    name=weather,
                    marker_color=self.weather_colors.get(weather, '#666666'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Vehicle count by weather
        fig.add_trace(
            go.Bar(
                x=weather_stats['weather'],
                y=weather_stats['avg_vehicles'],
                name='Average Vehicle Count',
                marker_color=[self.weather_colors.get(w, '#666666') for w in weather_stats['weather']],
                showlegend=False,
                error_y=dict(
                    type='data',
                    array=weather_stats['vehicles_std'],
                    visible=True
                )
            ),
            row=2, col=1
        )
        
        # Weather distribution pie chart
        weather_counts = df['weather'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=weather_counts.index,
                values=weather_counts.values,
                marker_colors=[self.weather_colors.get(w, '#666666') for w in weather_counts.index],
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Weather Impact on Traffic Analysis",
            showlegend=True
        )
        
        return fig
    
    def create_speed_time_analysis(self, df):
        """Create speed vs time analysis"""
        # Convert timestamp to datetime
        df_time = df.copy()
        df_time['datetime'] = pd.to_datetime(df_time['timestamp'], unit='s')
        df_time['hour'] = df_time['datetime'].dt.hour
        df_time['date'] = df_time['datetime'].dt.date
        
        # Daily speed trends
        daily_speed = df_time.groupby(['date', 'hour'])['average_speed_kmph'].mean().reset_index()
        
        # Hourly average across all days
        hourly_avg = df_time.groupby('hour')['average_speed_kmph'].agg(['mean', 'std']).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Hourly Speed Patterns', 'Speed Distribution by Location'),
            row_heights=[0.6, 0.4]
        )
        
        # Hourly speed pattern with confidence interval
        fig.add_trace(
            go.Scatter(
                x=hourly_avg['hour'],
                y=hourly_avg['mean'],
                mode='lines+markers',
                name='Average Speed',
                line=dict(color=self.color_palette['primary'], width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=list(hourly_avg['hour']) + list(hourly_avg['hour'][::-1]),
                y=list(hourly_avg['mean'] + hourly_avg['std']) + list((hourly_avg['mean'] - hourly_avg['std'])[::-1]),
                fill='toself',
                fillcolor='rgba(255, 107, 107, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='±1 Std Dev',
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Speed by location
        location_speed = df.groupby('location')['average_speed_kmph'].mean().sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=location_speed.values,
                y=location_speed.index,
                orientation='h',
                name='Location Average',
                marker_color=self.color_palette['secondary'],
                showlegend=False
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=700,
            title_text="Traffic Speed Analysis Over Time",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Hour of Day", row=1, col=1)
        fig.update_yaxes(title_text="Average Speed (km/h)", row=1, col=1)
        fig.update_xaxes(title_text="Average Speed (km/h)", row=2, col=1)
        fig.update_yaxes(title_text="Location", row=2, col=1)
        
        return fig
    
    def create_congestion_heatmap(self, df):
        """Create congestion heatmap by time and location"""
        # Prepare data
        df_pivot = df.copy()
        df_pivot['datetime'] = pd.to_datetime(df_pivot['timestamp'], unit='s')
        df_pivot['hour'] = df_pivot['datetime'].dt.hour
        
        # Create congestion score (inverse of speed)
        df_pivot['congestion_score'] = 100 - df_pivot['average_speed_kmph']
        
        # Pivot for heatmap
        heatmap_data = df_pivot.groupby(['location', 'hour'])['congestion_score'].mean().unstack(fill_value=0)
        
        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Reds',
                colorbar=dict(title="Congestion Level"),
                hoverongaps=False
            )
        )
        
        fig.update_layout(
            title='Traffic Congestion Heatmap by Location and Time',
            xaxis_title='Hour of Day',
            yaxis_title='Location',
            height=500
        )
        
        return fig
    
    def create_traffic_flow_visualization(self, detections_history):
        """Create traffic flow visualization"""
        if not detections_history:
            return go.Figure().add_annotation(
                text="No detection history available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Simulate traffic flow data
        time_points = list(range(len(detections_history)))
        vehicle_counts = [len(detections) for detections in detections_history]
        
        # Calculate flow rate (vehicles per minute)
        flow_rates = []
        for i in range(1, len(vehicle_counts)):
            flow_rate = max(0, vehicle_counts[i] - vehicle_counts[i-1])
            flow_rates.append(flow_rate)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Vehicle Count Over Time', 'Traffic Flow Rate')
        )
        
        # Vehicle count
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=vehicle_counts,
                mode='lines+markers',
                name='Vehicle Count',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # Flow rate
        if flow_rates:
            fig.add_trace(
                go.Bar(
                    x=time_points[1:],
                    y=flow_rates,
                    name='Flow Rate',
                    marker_color=self.color_palette['secondary']
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=600, title_text="Traffic Flow Analysis")
        return fig
    
    def create_performance_dashboard(self, model_results):
        """Create ML model performance dashboard"""
        if not model_results:
            return go.Figure().add_annotation(
                text="No model results available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        models = list(model_results.keys())
        accuracies = [result['accuracy'] for result in model_results.values()]
        precisions = [result['precision'] for result in model_results.values()]
        recalls = [result['recall'] for result in model_results.values()]
        f1_scores = [result['f1_score'] for result in model_results.values()]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Accuracy', 'Precision vs Recall', 'F1 Scores', 'Performance Comparison'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "radar"}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(
                x=models,
                y=accuracies,
                name='Accuracy',
                marker_color=self.color_palette['primary']
            ),
            row=1, col=1
        )
        
        # Precision vs Recall scatter
        fig.add_trace(
            go.Scatter(
                x=precisions,
                y=recalls,
                mode='markers+text',
                text=models,
                textposition="top center",
                name='Models',
                marker=dict(size=15, color=self.color_palette['secondary'])
            ),
            row=1, col=2
        )
        
        # F1 scores
        fig.add_trace(
            go.Bar(
                x=models,
                y=f1_scores,
                name='F1 Score',
                marker_color=self.color_palette['success']
            ),
            row=2, col=1
        )
        
        # Radar chart for overall performance
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for i, model in enumerate(models):
            result = model_results[model]
            values = [result['accuracy'], result['precision'], result['recall'], result['f1_score']]
            
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=model,
                    line_color=list(self.color_palette.values())[i % len(self.color_palette)]
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Machine Learning Model Performance Dashboard",
            showlegend=True
        )
        
        return fig
    
    def create_real_time_metrics(self, current_data):
        """Create real-time metrics visualization"""
        # Create gauge charts for key metrics
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Traffic Density', 'Average Speed', 'Congestion Level',
                          'Weather Impact', 'System Efficiency', 'Alert Level'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Traffic density gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=current_data.get('density', 0) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Density (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ),
            row=1, col=1
        )
        
        # Average speed gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=current_data.get('speed', 50),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Speed (km/h)"},
                gauge={
                    'axis': {'range': [None, 120]},
                    'bar': {'color': self.color_palette['success']},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 80], 'color': "yellow"},
                        {'range': [80, 120], 'color': "green"}
                    ]
                }
            ),
            row=1, col=2
        )
        
        # Add other gauges...
        congestion_score = max(0, min(100, 100 - current_data.get('speed', 50)))
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=congestion_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Congestion (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_palette['warning']},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ),
            row=1, col=3
        )
        
        fig.update_layout(height=600, title_text="Real-Time Traffic Metrics")
        return fig
    
    def create_prediction_comparison(self, actual_values, predicted_values, model_names):
        """Create prediction vs actual comparison chart"""
        fig = go.Figure()
        
        # Perfect prediction line
        min_val = min(min(actual_values), min(predicted_values))
        max_val = max(max(actual_values), max(predicted_values))
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash', color='gray')
            )
        )
        
        # Predicted vs Actual scatter
        colors = list(self.color_palette.values())
        for i, (actual, predicted, model) in enumerate(zip(actual_values, predicted_values, model_names)):
            fig.add_trace(
                go.Scatter(
                    x=actual,
                    y=predicted,
                    mode='markers',
                    name=f'{model} Predictions',
                    marker=dict(color=colors[i % len(colors)], size=8, opacity=0.7)
                )
            )
        
        fig.update_layout(
            title='Prediction Accuracy Comparison',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=500
        )
        
        return fig
