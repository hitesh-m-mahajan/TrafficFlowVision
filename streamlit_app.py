import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import pickle

# Import custom modules
from ml_model import TrafficMLModel
from image_utils import TrafficImageProcessor
from decision_engine import TrafficDecisionEngine
from fog_computing import FogComputingSimulator
from visualization import TrafficVisualizer
from utils import load_data, preprocess_data

# Set page config
st.set_page_config(
    page_title="Intelligent Traffic Management System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'dataset' not in st.session_state:
    st.session_state.dataset = None

# Main title
st.title("🚦 Intelligent Traffic Management System")
st.markdown("---")

# Sidebar
st.sidebar.title("Navigation")
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Train Model", "Upload Image & Predict", "Dashboard Overview"]
)

# Load dataset
@st.cache_data
def load_traffic_data():
    """Load and cache the traffic dataset"""
    try:
        # Try to load from the uploaded file
        df = pd.read_csv("attached_assets/traffic_weather_speed_dataset_1754508149737.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure the CSV file is in the correct location.")
        return None

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize and cache the system components"""
    ml_model = TrafficMLModel()
    image_processor = TrafficImageProcessor()
    decision_engine = TrafficDecisionEngine()
    fog_simulator = FogComputingSimulator()
    visualizer = TrafficVisualizer()
    return ml_model, image_processor, decision_engine, fog_simulator, visualizer

ml_model, image_processor, decision_engine, fog_simulator, visualizer = initialize_components()

# Load data
if st.session_state.dataset is None:
    with st.spinner("Loading dataset..."):
        st.session_state.dataset = load_traffic_data()

if st.session_state.dataset is not None:
    df = st.session_state.dataset
    
    # Train Model Mode
    if mode == "Train Model":
        st.header("🤖 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.write(f"**Total Records:** {len(df):,}")
            st.write(f"**Features:** {list(df.columns)}")
            st.write(f"**Date Range:** {pd.to_datetime(df['timestamp'], unit='s').min()} to {pd.to_datetime(df['timestamp'], unit='s').max()}")
            
            # Dataset preview
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10))
            
            # Data statistics
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())
        
        with col2:
            st.subheader("Training Configuration")
            
            # Model selection
            selected_models = st.multiselect(
                "Select Models to Train",
                ["Random Forest", "LSTM", "XGBoost"],
                default=["Random Forest", "XGBoost"]
            )
            
            # Training parameters
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("Random State", value=42, min_value=0)
            
            if st.button("🚀 Start Training", type="primary"):
                if selected_models:
                    with st.spinner("Training models... This may take a few minutes."):
                        # Prepare data
                        X, y = preprocess_data(df)
                        
                        # Train selected models
                        results = ml_model.train_models(X, y, selected_models, test_size, random_state)
                        
                        st.session_state.trained_models = results
                        st.session_state.model_trained = True
                        
                        st.success("✅ Models trained successfully!")
                        st.rerun()
                else:
                    st.error("Please select at least one model to train.")
        
        # Display training results
        if st.session_state.model_trained and st.session_state.trained_models:
            st.markdown("---")
            st.subheader("📊 Training Results")
            
            for model_name, result in st.session_state.trained_models.items():
                with st.expander(f"{model_name} Results"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{result['accuracy']:.3f}")
                        st.metric("Precision", f"{result['precision']:.3f}")
                    
                    with col2:
                        st.metric("Recall", f"{result['recall']:.3f}")
                        st.metric("F1-Score", f"{result['f1_score']:.3f}")
                    
                    with col3:
                        # Confusion Matrix
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', ax=ax, cmap='Blues')
                        ax.set_title(f'{model_name} - Confusion Matrix')
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
    
    # Upload Image & Predict Mode
    elif mode == "Upload Image & Predict":
        st.header("📸 Traffic Image Analysis & Prediction")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train models first before making predictions.")
            if st.button("Go to Train Model"):
                st.rerun()
        else:
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload Traffic Image",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a traffic image for analysis"
            )
            
            if uploaded_file is not None:
                # Load and display image
                image = Image.open(uploaded_file)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, caption="Uploaded Traffic Image", use_column_width=True)
                
                with col2:
                    st.subheader("Image Processing")
                    with st.spinner("Processing image..."):
                        # Process image
                        processed_results = image_processor.process_image(np.array(image))
                        
                        # Display processed image
                        st.image(
                            processed_results['processed_image'], 
                            caption="Processed Image (CLAHE + Gaussian + Canny)", 
                            use_column_width=True
                        )
                
                # Vehicle detection
                st.markdown("---")
                st.subheader("🚗 Vehicle Detection Results")
                
                with st.spinner("Detecting vehicles with YOLOv8..."):
                    detection_results = image_processor.detect_vehicles(np.array(image))
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(
                        detection_results['annotated_image'],
                        caption="Vehicle Detection Results",
                        use_column_width=True
                    )
                
                with col2:
                    # Vehicle statistics
                    vehicle_counts = detection_results['vehicle_counts']
                    total_vehicles = sum(vehicle_counts.values())
                    
                    st.metric("Total Vehicles Detected", total_vehicles)
                    
                    # Vehicle type breakdown
                    if vehicle_counts:
                        fig = px.bar(
                            x=list(vehicle_counts.keys()),
                            y=list(vehicle_counts.values()),
                            title="Vehicle Count by Type"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Traffic prediction
                st.markdown("---")
                st.subheader("🔮 Traffic Prediction & Analysis")
                
                # Create feature vector for prediction
                current_time = datetime.now()
                weather_condition = st.selectbox(
                    "Current Weather Condition",
                    ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"]
                )
                
                # Prepare features
                features = {
                    'vehicle_count': total_vehicles,
                    'weather': weather_condition,
                    'hour': current_time.hour,
                    'day_of_week': current_time.weekday(),
                    'density': detection_results.get('density', 0)
                }
                
                # Make predictions
                predictions = {}
                for model_name, model_data in st.session_state.trained_models.items():
                    if 'model' in model_data:
                        try:
                            pred = ml_model.predict(model_data['model'], features)
                            predictions[model_name] = pred
                        except Exception as e:
                            st.error(f"Prediction error for {model_name}: {str(e)}")
                
                # Display predictions
                if predictions:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("Congestion Prediction")
                        for model_name, pred in predictions.items():
                            congestion_level = decision_engine.determine_congestion_level(pred)
                            color = {"Low": "green", "Medium": "orange", "High": "red"}[congestion_level]
                            st.markdown(f"**{model_name}:** :{color}[{congestion_level}]")
                    
                    with col2:
                        st.subheader("Traffic Light Timing")
                        if predictions:
                            avg_prediction = np.mean(list(predictions.values()))
                            timing = decision_engine.get_traffic_light_timing(avg_prediction)
                            st.write(f"**Green Light:** {timing['green']}s")
                            st.write(f"**Yellow Light:** {timing['yellow']}s")
                            st.write(f"**Red Light:** {timing['red']}s")
                        else:
                            st.write("No predictions available")
                    
                    with col3:
                        st.subheader("Route Recommendation")
                        if predictions:
                            avg_prediction = np.mean(list(predictions.values()))
                            route_advice = decision_engine.get_route_recommendation(avg_prediction)
                            st.write(route_advice)
                        else:
                            st.write("No route recommendations available")
                
                # Visualizations
                st.markdown("---")
                st.subheader("📊 Detailed Analysis")
                
                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4 = st.tabs(["Density Heatmap", "Weather Impact", "Speed Analysis", "Infrastructure Planning"])
                
                with tab1:
                    # Density heatmap overlay
                    if 'density_map' in detection_results:
                        st.subheader("Traffic Density Heatmap")
                        density_overlay = visualizer.create_density_heatmap(
                            np.array(image), 
                            detection_results['density_map']
                        )
                        st.image(density_overlay, caption="Traffic Density Overlay")
                
                with tab2:
                    # Weather impact analysis
                    weather_analysis = visualizer.analyze_weather_impact(df)
                    st.plotly_chart(weather_analysis, use_container_width=True)
                
                with tab3:
                    # Speed vs time analysis
                    speed_analysis = visualizer.create_speed_time_analysis(df)
                    st.plotly_chart(speed_analysis, use_container_width=True)
                
                with tab4:
                    # Infrastructure planning insights
                    avg_pred_value = np.mean(list(predictions.values())) if predictions else 50
                    planning_data = decision_engine.get_infrastructure_insights(
                        avg_pred_value,
                        total_vehicles,
                        weather_condition
                    )
                    st.table(pd.DataFrame(planning_data))
                
                # Fog Computing Simulation
                st.markdown("---")
                st.subheader("☁️ Fog Computing Simulation")
                
                with st.expander("View Fog Computing Process"):
                    simulation_result = fog_simulator.simulate_fog_computing(features)
                    
                    for step, details in simulation_result.items():
                        st.write(f"**{step}:**")
                        for key, value in details.items():
                            if key == "encrypted_data":
                                st.code(value[:100] + "..." if len(value) > 100 else value)
                            else:
                                st.write(f"  - {key}: {value}")
                        st.write("")

    # Dashboard Overview Mode
    elif mode == "Dashboard Overview":
        st.header("📈 Traffic Management Dashboard")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            avg_speed = df['average_speed_kmph'].mean()
            st.metric("Average Speed", f"{avg_speed:.1f} km/h")
        with col3:
            avg_vehicles = df['vehicle_count'].mean()
            st.metric("Average Vehicle Count", f"{avg_vehicles:.0f}")
        with col4:
            locations = df['location'].nunique()
            st.metric("Monitored Locations", locations)
        
        # Create dashboard visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Speed distribution by weather
            fig = px.box(
                df, 
                x='weather', 
                y='average_speed_kmph',
                title="Speed Distribution by Weather Condition"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Vehicle count by location
            location_stats = df.groupby('location')['vehicle_count'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=location_stats.index,
                y=location_stats.values,
                title="Average Vehicle Count by Location"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weather distribution
            weather_counts = df['weather'].value_counts()
            fig = px.pie(
                values=weather_counts.values,
                names=weather_counts.index,
                title="Weather Condition Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Speed vs Vehicle Count correlation
            fig = px.scatter(
                df,
                x='vehicle_count',
                y='average_speed_kmph',
                color='weather',
                title="Speed vs Vehicle Count Correlation"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        st.markdown("---")
        st.subheader("Time Series Analysis")
        
        # Convert timestamp and create time-based analysis
        df_time = df.copy()
        df_time['datetime'] = pd.to_datetime(df_time['timestamp'], unit='s')
        df_time['hour'] = df_time['datetime'].dt.hour
        df_time['day_of_week'] = df_time['datetime'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly traffic pattern
            hourly_stats = df_time.groupby('hour').agg({
                'vehicle_count': 'mean',
                'average_speed_kmph': 'mean'
            }).round(2)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Scatter(x=hourly_stats.index, y=hourly_stats['vehicle_count'], 
                          name="Vehicle Count", line=dict(color='blue')),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=hourly_stats.index, y=hourly_stats['average_speed_kmph'], 
                          name="Average Speed", line=dict(color='red')),
                secondary_y=True
            )
            fig.update_layout(title="Traffic Patterns by Hour")
            fig.update_xaxes(title_text="Hour of Day")
            fig.update_yaxes(title_text="Vehicle Count", secondary_y=False)
            fig.update_yaxes(title_text="Speed (km/h)", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weekly traffic pattern
            weekly_stats = df_time.groupby('day_of_week').agg({
                'vehicle_count': 'mean',
                'average_speed_kmph': 'mean'
            }).round(2)
            
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekly_stats = weekly_stats.reindex([day for day in days_order if day in weekly_stats.index])
            
            fig = px.bar(
                x=weekly_stats.index,
                y=weekly_stats['vehicle_count'],
                title="Average Traffic by Day of Week"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Unable to load the traffic dataset. Please check if the file exists.")
    st.info("Expected file: `attached_assets/traffic_weather_speed_dataset_1754508149737.csv`")

# Footer
st.markdown("---")
st.markdown("🚦 **Intelligent Traffic Management System** | Built with Streamlit")
