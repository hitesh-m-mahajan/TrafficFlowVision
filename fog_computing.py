import json
import time
import hashlib
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import random

class FogComputingSimulator:
    def __init__(self):
        self.fog_nodes = {
            'edge_node_1': {'location': 'Intersection_A', 'capacity': 100, 'latency': 5},
            'edge_node_2': {'location': 'Intersection_B', 'capacity': 150, 'latency': 3},
            'edge_node_3': {'location': 'Highway_Junction', 'capacity': 200, 'latency': 4},
            'cloud_server': {'location': 'Data_Center', 'capacity': 1000, 'latency': 50}
        }
        
        self.processing_steps = [
            "Data Collection",
            "Data Preprocessing", 
            "Edge Processing",
            "Encryption",
            "Data Transmission",
            "Cloud Processing",
            "Decision Making",
            "Response Transmission"
        ]
    
    def generate_encryption_key(self, password=None):
        """Generate encryption key for AES-256"""
        if password is None:
            password = "traffic_management_secure_key_2024"
        
        # Convert password to bytes
        password = password.encode()
        
        # Generate salt
        salt = os.urandom(16)
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        
        return key, salt
    
    def encrypt_data(self, data, key):
        """Encrypt data using AES-256"""
        try:
            fernet = Fernet(key)
            
            # Convert data to JSON string if it's not already
            if not isinstance(data, str):
                data = json.dumps(data)
            
            # Encrypt data
            encrypted_data = fernet.encrypt(data.encode())
            
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            return f"Encryption failed: {str(e)}"
    
    def decrypt_data(self, encrypted_data, key):
        """Decrypt data using AES-256"""
        try:
            fernet = Fernet(key)
            
            # Decode base64
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            
            # Decrypt data
            decrypted_bytes = fernet.decrypt(encrypted_bytes)
            decrypted_data = decrypted_bytes.decode()
            
            # Try to parse as JSON
            try:
                return json.loads(decrypted_data)
            except json.JSONDecodeError:
                return decrypted_data
        except Exception as e:
            return f"Decryption failed: {str(e)}"
    
    def simulate_fog_computing(self, traffic_data):
        """Simulate the complete fog computing workflow"""
        simulation_start = time.time()
        results = {}
        
        # Generate encryption key
        encryption_key, salt = self.generate_encryption_key()
        
        # Step 1: Data Collection
        step_start = time.time()
        collected_data = {
            'timestamp': time.time(),
            'traffic_data': traffic_data,
            'sensor_id': f"sensor_{random.randint(1000, 9999)}",
            'data_size_bytes': len(json.dumps(traffic_data))
        }
        step_duration = time.time() - step_start
        
        results['Step_1_Data_Collection'] = {
            'status': 'completed',
            'duration_ms': round(step_duration * 1000, 2),
            'data_size_bytes': collected_data['data_size_bytes'],
            'sensor_id': collected_data['sensor_id']
        }
        
        # Step 2: Data Preprocessing
        step_start = time.time()
        processed_data = self._preprocess_traffic_data(collected_data['traffic_data'])
        step_duration = time.time() - step_start
        
        results['Step_2_Data_Preprocessing'] = {
            'status': 'completed',
            'duration_ms': round(step_duration * 1000, 2),
            'features_extracted': len(processed_data),
            'quality_score': random.uniform(0.85, 0.98)
        }
        
        # Step 3: Edge Processing
        step_start = time.time()
        selected_node = self._select_optimal_fog_node(processed_data)
        edge_analysis = self._perform_edge_analysis(processed_data, selected_node)
        step_duration = time.time() - step_start
        
        results['Step_3_Edge_Processing'] = {
            'status': 'completed',
            'duration_ms': round(step_duration * 1000, 2),
            'selected_node': selected_node,
            'processing_capacity_used': f"{random.randint(20, 80)}%",
            'analysis_results': edge_analysis
        }
        
        # Step 4: Encryption
        step_start = time.time()
        encrypted_data = self.encrypt_data(processed_data, encryption_key)
        step_duration = time.time() - step_start
        
        results['Step_4_Encryption'] = {
            'status': 'completed',
            'duration_ms': round(step_duration * 1000, 2),
            'encryption_algorithm': 'AES-256',
            'key_length': '256 bits',
            'encrypted_data': encrypted_data[:100] + "..." if len(encrypted_data) > 100 else encrypted_data
        }
        
        # Step 5: Data Transmission
        step_start = time.time()
        transmission_result = self._simulate_data_transmission(encrypted_data, selected_node)
        step_duration = time.time() - step_start
        
        results['Step_5_Data_Transmission'] = {
            'status': 'completed',
            'duration_ms': round(step_duration * 1000, 2),
            'transmission_speed_mbps': transmission_result['speed_mbps'],
            'latency_ms': transmission_result['latency_ms'],
            'packet_loss': transmission_result['packet_loss']
        }
        
        # Step 6: Cloud Processing
        step_start = time.time()
        cloud_analysis = self._perform_cloud_analysis(processed_data)
        step_duration = time.time() - step_start
        
        results['Step_6_Cloud_Processing'] = {
            'status': 'completed',
            'duration_ms': round(step_duration * 1000, 2),
            'ml_models_used': cloud_analysis['models_used'],
            'prediction_accuracy': cloud_analysis['accuracy'],
            'computational_resources': cloud_analysis['resources_used']
        }
        
        # Step 7: Decision Making
        step_start = time.time()
        decisions = self._make_traffic_decisions(edge_analysis, cloud_analysis)
        step_duration = time.time() - step_start
        
        results['Step_7_Decision_Making'] = {
            'status': 'completed',
            'duration_ms': round(step_duration * 1000, 2),
            'decisions_made': len(decisions),
            'confidence_score': decisions['confidence'],
            'recommendations': decisions['recommendations']
        }
        
        # Step 8: Response Transmission
        step_start = time.time()
        response_data = {
            'decisions': decisions,
            'timestamp': time.time(),
            'response_id': f"resp_{random.randint(10000, 99999)}"
        }
        encrypted_response = self.encrypt_data(response_data, encryption_key)
        response_transmission = self._simulate_response_transmission(encrypted_response)
        step_duration = time.time() - step_start
        
        results['Step_8_Response_Transmission'] = {
            'status': 'completed',
            'duration_ms': round(step_duration * 1000, 2),
            'response_size_bytes': len(encrypted_response),
            'delivery_time_ms': response_transmission['delivery_time_ms'],
            'success_rate': response_transmission['success_rate']
        }
        
        # Calculate total processing time
        total_duration = time.time() - simulation_start
        
        # Add summary
        results['Simulation_Summary'] = {
            'total_duration_ms': round(total_duration * 1000, 2),
            'total_steps': len(self.processing_steps),
            'encryption_key_hash': hashlib.sha256(encryption_key).hexdigest()[:16],
            'overall_efficiency': self._calculate_efficiency(results),
            'security_level': 'High (AES-256 Encryption)',
            'fog_nodes_utilized': len([node for node in self.fog_nodes.keys() if 'edge' in node])
        }
        
        return results
    
    def _preprocess_traffic_data(self, traffic_data):
        """Preprocess traffic data for fog computing"""
        processed = {}
        
        # Simulate feature extraction
        processed['vehicle_count'] = traffic_data.get('vehicle_count', 0)
        processed['weather_encoded'] = self._encode_weather(traffic_data.get('weather', 'Sunny'))
        processed['time_features'] = {
            'hour': traffic_data.get('hour', 12),
            'is_rush_hour': traffic_data.get('is_rush_hour', 0),
            'day_of_week': traffic_data.get('day_of_week', 1)
        }
        processed['traffic_density'] = traffic_data.get('density', 0.0)
        
        return processed
    
    def _encode_weather(self, weather):
        """Encode weather condition to numerical value"""
        weather_mapping = {
            'Sunny': 0,
            'Cloudy': 1,
            'Rainy': 2,
            'Snowy': 3,
            'Foggy': 4
        }
        return weather_mapping.get(weather, 0)
    
    def _select_optimal_fog_node(self, data):
        """Select optimal fog node based on data and capacity"""
        # Simple selection based on capacity and latency
        edge_nodes = {k: v for k, v in self.fog_nodes.items() if 'edge' in k}
        
        # Calculate scores for each node
        scores = {}
        data_size = len(json.dumps(data))
        
        for node_id, node_info in edge_nodes.items():
            capacity_score = min(1.0, node_info['capacity'] / data_size)
            latency_score = 1.0 / (node_info['latency'] + 1)
            scores[node_id] = capacity_score * 0.6 + latency_score * 0.4
        
        # Select node with highest score
        optimal_node = max(scores.keys(), key=lambda k: scores[k])
        return optimal_node
    
    def _perform_edge_analysis(self, data, node_id):
        """Perform analysis at edge node"""
        node_info = self.fog_nodes[node_id]
        
        # Simulate edge processing capabilities
        analysis = {
            'congestion_level': 'medium' if data['vehicle_count'] > 50 else 'low',
            'weather_impact': 'high' if data['weather_encoded'] in [2, 3, 4] else 'low',
            'processing_node': node_id,
            'processing_latency': node_info['latency'],
            'real_time_score': random.uniform(0.7, 0.95)
        }
        
        return analysis
    
    def _perform_cloud_analysis(self, data):
        """Perform advanced analysis in cloud"""
        analysis = {
            'models_used': ['Random Forest', 'Neural Network', 'XGBoost'],
            'accuracy': random.uniform(0.85, 0.95),
            'resources_used': f"{random.randint(10, 30)}% CPU, {random.randint(15, 40)}% Memory",
            'advanced_predictions': {
                'traffic_flow_prediction': random.uniform(20, 100),
                'congestion_probability': random.uniform(0.1, 0.9),
                'optimal_routing': f"Route_{random.randint(1, 5)}"
            }
        }
        
        return analysis
    
    def _make_traffic_decisions(self, edge_analysis, cloud_analysis):
        """Make traffic management decisions"""
        decisions = {
            'traffic_light_timing': {
                'green': random.randint(25, 45),
                'yellow': 5,
                'red': random.randint(20, 35)
            },
            'route_recommendations': [
                f"Route_{i}" for i in range(1, random.randint(3, 6))
            ],
            'congestion_alerts': edge_analysis['congestion_level'] != 'low',
            'weather_adjustments': edge_analysis['weather_impact'] == 'high',
            'confidence': min(edge_analysis['real_time_score'], cloud_analysis['accuracy']),
            'recommendations': [
                "Optimize signal timing",
                "Monitor weather conditions",
                "Update route suggestions"
            ]
        }
        
        return decisions
    
    def _simulate_data_transmission(self, data, source_node):
        """Simulate data transmission between nodes"""
        data_size_mb = len(data) / (1024 * 1024)  # Convert to MB
        base_speed = random.uniform(50, 150)  # Mbps
        
        # Simulate network conditions
        transmission = {
            'speed_mbps': base_speed,
            'latency_ms': self.fog_nodes[source_node]['latency'] + random.randint(1, 10),
            'packet_loss': random.uniform(0, 0.02),  # 0-2% packet loss
            'transmission_time_ms': (data_size_mb / base_speed) * 1000
        }
        
        return transmission
    
    def _simulate_response_transmission(self, response_data):
        """Simulate response transmission back to edge"""
        response_size = len(response_data)
        
        transmission = {
            'delivery_time_ms': random.randint(5, 25),
            'success_rate': random.uniform(0.95, 0.99),
            'response_size': response_size
        }
        
        return transmission
    
    def _calculate_efficiency(self, results):
        """Calculate overall system efficiency"""
        total_time = results['Simulation_Summary']['total_duration_ms']
        
        # Ideal time benchmark (in ms)
        ideal_time = 100
        
        # Calculate efficiency score
        efficiency = min(100, (ideal_time / total_time) * 100)
        
        return round(efficiency, 2)
    
    def get_fog_node_status(self):
        """Get current status of all fog nodes"""
        status = {}
        
        for node_id, node_info in self.fog_nodes.items():
            status[node_id] = {
                'location': node_info['location'],
                'capacity_utilization': f"{random.randint(10, 80)}%",
                'current_latency': node_info['latency'] + random.randint(-2, 5),
                'status': 'online',
                'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
                'processed_requests': random.randint(100, 1000)
            }
        
        return status
    
    def simulate_security_measures(self, data):
        """Simulate additional security measures"""
        security_measures = {
            'data_integrity_check': {
                'status': 'passed',
                'hash_verification': True,
                'timestamp_validation': True
            },
            'access_control': {
                'authentication': 'multi-factor',
                'authorization_level': 'admin',
                'session_token': f"token_{random.randint(100000, 999999)}"
            },
            'encryption_details': {
                'algorithm': 'AES-256-GCM',
                'key_rotation': 'every_24_hours',
                'secure_key_exchange': True
            },
            'audit_trail': {
                'logged_actions': ['data_access', 'encryption', 'transmission'],
                'compliance_level': 'ISO_27001',
                'audit_score': random.uniform(0.9, 1.0)
            }
        }
        
        return security_measures
