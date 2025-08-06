import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict, deque
import math

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class TrafficImageProcessor:
    def __init__(self):
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        self.trackers = {}
        self.next_id = 0
        
        # Initialize YOLOv8n model
        self.model = None
        self.model_loaded = False
        
        if YOLO_AVAILABLE:
            try:
                print("Loading YOLOv8n model...")
                self.model = YOLO('yolov8n.pt')  # This will download the model if not present
                self.model_loaded = True
                print("YOLOv8n model loaded successfully!")
            except Exception as e:
                print(f"Failed to load YOLOv8 model: {e}")
                print("Using simulated detection instead.")
                self.model_loaded = False
        else:
            print("Ultralytics not available. Using simulated detection.")
            self.model_loaded = False
    
    def process_image(self, image):
        """Apply CLAHE, Gaussian filter, and Canny edge detection"""
        if not CV2_AVAILABLE:
            return self._simple_image_processing(image)
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        
        # Apply Gaussian blur
        gaussian_blur = cv2.GaussianBlur(clahe_image, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gaussian_blur, 50, 150)
        
        # Combine processed images for visualization
        processed_image = np.hstack([clahe_image, gaussian_blur, edges])
        
        return {
            'original': image,
            'clahe': clahe_image,
            'gaussian': gaussian_blur,
            'edges': edges,
            'processed_image': processed_image,
            'combined': processed_image
        }
    
    def _simple_image_processing(self, image):
        """Simple image processing without OpenCV"""
        # Convert to grayscale using weighted average
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = image.copy()
        
        # Simple contrast enhancement (histogram stretching)
        min_val, max_val = gray.min(), gray.max()
        if max_val > min_val:
            enhanced = ((gray - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
        else:
            enhanced = gray
        
        # Simple edge detection using gradient
        grad_x = np.abs(np.diff(enhanced, axis=1))
        grad_y = np.abs(np.diff(enhanced, axis=0))
        
        # Pad to maintain original size
        grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
        grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
        
        edges = (grad_x + grad_y).clip(0, 255).astype(np.uint8)
        
        # Combine processed images
        height, width = gray.shape
        processed_image = np.zeros((height, width * 3), dtype=np.uint8)
        processed_image[:, :width] = gray
        processed_image[:, width:2*width] = enhanced
        processed_image[:, 2*width:] = edges
        
        return {
            'original': image,
            'clahe': enhanced,
            'gaussian': enhanced,
            'edges': edges,
            'processed_image': processed_image,
            'combined': processed_image
        }
    
    def detect_vehicles(self, image):
        """Detect vehicles using YOLOv8 (simulated if model not available)"""
        if self.model_loaded:
            return self._yolo_detection(image)
        else:
            return self._simulated_detection(image)
    
    def _yolo_detection(self, image):
        """Real YOLOv8 detection implementation"""
        try:
            # Run inference
            results = self.model(image, verbose=False)
            
            # Extract detections
            detections = []
            vehicle_counts = defaultdict(int)
            annotated_image = image.copy()
            
            # COCO class names for vehicles
            vehicle_class_names = {
                2: 'car',      # car
                3: 'motorcycle', # motorcycle  
                5: 'bus',      # bus
                7: 'truck',    # truck
                1: 'bicycle'   # bicycle
            }
            
            # Colors for different vehicle types
            colors = {
                'car': (0, 255, 0),      # Green
                'truck': (255, 0, 0),    # Red
                'bus': (0, 0, 255),      # Blue
                'motorcycle': (255, 255, 0),  # Yellow
                'bicycle': (255, 0, 255)  # Magenta
            }
            
            # Process detections
            if results[0].boxes is not None:
                boxes = results[0].boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    
                    # Get class and confidence
                    cls = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())
                    
                    # Check if it's a vehicle class and confidence is above threshold
                    if cls in vehicle_class_names and conf > 0.3:
                        vehicle_type = vehicle_class_names[cls]
                        
                        detection = {
                            'bbox': [x1, y1, x2, y2],
                            'class': vehicle_type,
                            'confidence': conf,
                            'centroid': [(x1 + x2) // 2, (y1 + y2) // 2]
                        }
                        detections.append(detection)
                        vehicle_counts[vehicle_type] += 1
                        
                        # Draw bounding box and label
                        if CV2_AVAILABLE:
                            color = colors.get(vehicle_type, (0, 255, 0))
                            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw label
                            label = f"{vehicle_type}: {conf:.2f}"
                            cv2.putText(annotated_image, label, (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            # Use PIL for drawing when OpenCV is not available
                            pil_image = Image.fromarray(annotated_image)
                            draw = ImageDraw.Draw(pil_image)
                            
                            color = colors.get(vehicle_type, (0, 255, 0))
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            
                            # Draw label
                            label = f"{vehicle_type}: {conf:.2f}"
                            draw.text((x1, y1-15), label, fill=color)
                            
                            annotated_image = np.array(pil_image)
            
            # Calculate traffic density
            height, width = image.shape[:2]
            total_area = width * height
            vehicle_area = sum([(det['bbox'][2] - det['bbox'][0]) * 
                               (det['bbox'][3] - det['bbox'][1]) for det in detections])
            density = vehicle_area / total_area if total_area > 0 else 0
            
            # Create density map
            density_map = self._create_density_map(detections, (height, width))
            
            return {
                'detections': detections,
                'vehicle_counts': dict(vehicle_counts),
                'annotated_image': annotated_image,
                'density': density,
                'density_map': density_map,
                'total_vehicles': len(detections)
            }
            
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            print("Falling back to simulated detection")
            return self._simulated_detection(image)
    
    def _simulated_detection(self, image):
        """Simulate vehicle detection for demonstration"""
        height, width = image.shape[:2]
        
        # Generate random detections
        num_vehicles = np.random.randint(5, 25)
        detections = []
        vehicle_counts = defaultdict(int)
        
        # Create annotated image
        annotated_image = image.copy()
        
        # Colors for different vehicle types
        colors = {
            'car': (0, 255, 0),      # Green
            'truck': (255, 0, 0),    # Red
            'bus': (0, 0, 255),      # Blue
            'motorcycle': (255, 255, 0),  # Cyan
            'bicycle': (255, 0, 255)  # Magenta
        }
        
        for i in range(num_vehicles):
            # Random vehicle type
            vehicle_type = np.random.choice(self.vehicle_classes, p=[0.7, 0.1, 0.05, 0.1, 0.05])
            
            # Random bounding box
            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 60)
            w = np.random.randint(60, 150)
            h = np.random.randint(40, 100)
            x2 = min(x1 + w, width)
            y2 = min(y1 + h, height)
            
            # Confidence score
            confidence = np.random.uniform(0.5, 0.95)
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'class': vehicle_type,
                'confidence': confidence,
                'centroid': [(x1 + x2) // 2, (y1 + y2) // 2]
            }
            detections.append(detection)
            vehicle_counts[vehicle_type] += 1
            
            # Draw bounding box and label
            if CV2_AVAILABLE:
                color = colors[vehicle_type]
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{vehicle_type}: {confidence:.2f}"
                cv2.putText(annotated_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Use PIL for drawing when OpenCV is not available
                pil_image = Image.fromarray(annotated_image)
                draw = ImageDraw.Draw(pil_image)
                
                color = colors[vehicle_type]
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # Draw label
                label = f"{vehicle_type}: {confidence:.2f}"
                draw.text((x1, y1-15), label, fill=color)
                
                annotated_image = np.array(pil_image)
        
        # Calculate traffic density
        total_area = width * height
        vehicle_area = sum([(det['bbox'][2] - det['bbox'][0]) * 
                           (det['bbox'][3] - det['bbox'][1]) for det in detections])
        density = vehicle_area / total_area
        
        # Create density map
        density_map = self._create_density_map(detections, (height, width))
        
        return {
            'detections': detections,
            'vehicle_counts': dict(vehicle_counts),
            'annotated_image': annotated_image,
            'density': density,
            'density_map': density_map,
            'total_vehicles': len(detections)
        }
    
    def _create_density_map(self, detections, image_shape):
        """Create a density heatmap based on detected vehicles"""
        height, width = image_shape
        density_map = np.zeros((height, width), dtype=np.float32)
        
        for detection in detections:
            center_x, center_y = detection['centroid']
            
            # Create Gaussian blob around each vehicle
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= 50**2
            density_map[mask] += 1
        
        # Normalize
        if density_map.max() > 0:
            density_map = density_map / density_map.max()
        
        return density_map
    
    def track_vehicles(self, detections, frame_id):
        """Track vehicles across frames using centroid tracking"""
        if not detections:
            return []
        
        centroids = [det['centroid'] for det in detections]
        
        # If no existing trackers, create new ones
        if not self.trackers:
            for i, centroid in enumerate(centroids):
                self.trackers[self.next_id] = {
                    'centroid': centroid,
                    'frames': deque([frame_id], maxlen=10),
                    'path': deque([centroid], maxlen=20)
                }
                detections[i]['track_id'] = self.next_id
                self.next_id += 1
        else:
            # Match centroids to existing trackers
            tracker_centroids = [tracker['centroid'] for tracker in self.trackers.values()]
            tracker_ids = list(self.trackers.keys())
            
            # Calculate distances
            distances = []
            for detection_centroid in centroids:
                row_distances = []
                for tracker_centroid in tracker_centroids:
                    distance = math.sqrt((detection_centroid[0] - tracker_centroid[0])**2 + 
                                       (detection_centroid[1] - tracker_centroid[1])**2)
                    row_distances.append(distance)
                distances.append(row_distances)
            
            # Assign detections to trackers
            distance_threshold = 50
            assigned_detections = set()
            assigned_trackers = set()
            
            # Find minimum distance assignments
            for _ in range(min(len(centroids), len(tracker_ids))):
                min_distance = float('inf')
                min_detection_idx = -1
                min_tracker_idx = -1
                
                for i, row in enumerate(distances):
                    if i in assigned_detections:
                        continue
                    for j, distance in enumerate(row):
                        if j in assigned_trackers:
                            continue
                        if distance < min_distance and distance < distance_threshold:
                            min_distance = distance
                            min_detection_idx = i
                            min_tracker_idx = j
                
                if min_detection_idx >= 0 and min_tracker_idx >= 0:
                    # Update existing tracker
                    tracker_id = tracker_ids[min_tracker_idx]
                    self.trackers[tracker_id]['centroid'] = centroids[min_detection_idx]
                    self.trackers[tracker_id]['frames'].append(frame_id)
                    self.trackers[tracker_id]['path'].append(centroids[min_detection_idx])
                    detections[min_detection_idx]['track_id'] = tracker_id
                    
                    assigned_detections.add(min_detection_idx)
                    assigned_trackers.add(min_tracker_idx)
            
            # Create new trackers for unassigned detections
            for i, detection in enumerate(detections):
                if i not in assigned_detections:
                    self.trackers[self.next_id] = {
                        'centroid': centroids[i],
                        'frames': deque([frame_id], maxlen=10),
                        'path': deque([centroids[i]], maxlen=20)
                    }
                    detection['track_id'] = self.next_id
                    self.next_id += 1
            
            # Remove old trackers
            inactive_trackers = []
            for tracker_id, tracker in self.trackers.items():
                if frame_id - tracker['frames'][-1] > 5:  # 5 frames without detection
                    inactive_trackers.append(tracker_id)
            
            for tracker_id in inactive_trackers:
                del self.trackers[tracker_id]
        
        return detections
    
    def apply_clahe(self, image, clip_limit=3.0, tile_grid_size=(8, 8)):
        """Apply CLAHE to enhance contrast"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_clahe = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l_clahe, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def gaussian_filter(self, image, kernel_size=5, sigma=0):
        """Apply Gaussian blur filter"""
        if sigma == 0:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges
    
    def create_traffic_flow_visualization(self, detections_history):
        """Create traffic flow visualization from detection history"""
        if not detections_history:
            return None
        
        # This would analyze movement patterns over multiple frames
        # For now, return a placeholder
        flow_data = {
            'incoming_vehicles': len(detections_history[-1]) if detections_history else 0,
            'outgoing_vehicles': 0,
            'average_speed': 25.0,  # km/h
            'flow_direction': 'mixed'
        }
        
        return flow_data
    
    def set_confidence_threshold(self, threshold=0.5):
        """Set confidence threshold for YOLO detections"""
        self.confidence_threshold = threshold
        if self.model_loaded and hasattr(self.model, 'conf'):
            self.model.conf = threshold
