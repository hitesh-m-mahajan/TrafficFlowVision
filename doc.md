# Intelligent Traffic Management System

## Overview

This project is an AI-powered traffic management system that combines machine learning, computer vision, and fog computing to optimize traffic flow in real-time. The system processes traffic camera images to detect vehicles, predicts traffic conditions using multiple ML algorithms, and makes intelligent decisions for traffic light timing. It features a Streamlit web interface for monitoring and management, with emphasis on edge computing and data encryption for efficient and secure operations.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Web Application**: Interactive dashboard built with Streamlit for real-time traffic monitoring and system management
- **Multi-page Interface**: Organized into three main sections - model training, image prediction, and dashboard overview
- **Data Visualization**: Plotly and Matplotlib integration for interactive charts, heatmaps, and traffic analytics
- **Real-time Updates**: Session state management for maintaining model training status and data persistence

### Backend Architecture
- **Modular Design**: Component-based architecture with separate modules for ML models, image processing, decision engine, fog computing, and visualization
- **Machine Learning Pipeline**: Multi-algorithm approach using Random Forest, XGBoost, and LSTM neural networks for traffic prediction
- **Computer Vision Processing**: OpenCV-based image processing with CLAHE enhancement, Gaussian filtering, and Canny edge detection
- **Decision Engine**: Rule-based system for dynamic traffic light timing optimization based on congestion levels

### Data Processing
- **Feature Engineering**: Automatic extraction of temporal features (hour, day of week, rush hour indicators) from timestamp data
- **Data Preprocessing**: StandardScaler normalization and LabelEncoder for categorical variables
- **Real-time Analytics**: Traffic density analysis, weather impact assessment, and congestion level determination

### Fog Computing Architecture
- **Edge Node Simulation**: Distributed processing across multiple fog nodes with varying capacities and latencies
- **Hierarchical Processing**: Edge-first approach with cloud fallback for complex computations
- **Security Layer**: AES-256 encryption using PBKDF2 key derivation for secure data transmission
- **Load Balancing**: Intelligent task distribution based on node capacity and proximity

### Traffic Intelligence
- **Multi-threshold Congestion Detection**: Speed-based classification system (High <40 km/h, Medium 40-70 km/h, Low >70 km/h)
- **Adaptive Traffic Light Control**: Dynamic timing adjustment based on real-time traffic conditions
- **Vehicle Detection**: YOLOv8-ready implementation with fallback simulation for object detection and tracking

## External Dependencies

### Machine Learning Libraries
- **scikit-learn**: Random Forest regression and data preprocessing utilities
- **TensorFlow/Keras**: LSTM neural network implementation for time series prediction  
  *Install with:*  
  ```
  pip install tensorflow
  ```
- **XGBoost**: Gradient boosting for traffic pattern analysis  
  *Install with:*  
  ```
  pip install xgboost
  ```
- **NumPy/Pandas**: Data manipulation and numerical computations  
  *Install with:*  
  ```
  pip install numpy pandas
  ```

### Computer Vision Stack
- **OpenCV**: Image processing, vehicle detection, and video stream handling
- **PIL (Pillow)**: Image manipulation and annotation capabilities
- **Ultralytics YOLO**: Vehicle detection and classification (optional integration)

### Web Framework and Visualization
- **Streamlit**: Web application framework for the dashboard interface
- **Plotly**: Interactive plotting and real-time data visualization
- **Matplotlib/Seaborn**: Statistical plotting and heatmap generation

### Security and Encryption
- **Cryptography**: AES-256 encryption for fog computing data security
- **PBKDF2**: Key derivation function for secure password-based encryption

### Data Storage
- **CSV File Processing**: Traffic and weather dataset handling
- **Pickle**: Model serialization and session state persistence
- **JSON**: Configuration and metadata storage

### Development Tools
- **joblib**: Model persistence and parallel processing
- **warnings**: Runtime warning management for ML operations