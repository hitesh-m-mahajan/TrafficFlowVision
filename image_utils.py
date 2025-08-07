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
        # Define vehicle classes with their COCO IDs
        self.vehicle_classes = {
            'car': 2,
            'motorcycle': 3,
            'bus': 5,
            'truck': 7,
            'bicycle': 1
        }
        self.trackers = {}
        self.next_id = 0

        # Initialize YOLOv8n model
        self.model = None
        self.model_loaded = False
        self.confidence_threshold = 0.5 # Default confidence threshold

        if YOLO_AVAILABLE:
            try:
                print("Loading YOLOv8n model...")
                # YOLOv8n.pt will be downloaded automatically if not present
                self.model = YOLO('yolov8n.pt')  
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
        """Detect vehicles with advanced tracking and analysis"""
        if self.model_loaded:
            try:
                # Use YOLOv8 for real detection with tracking
                results = self.model.track(image, persist=True)

                # Process results with tracking
                vehicle_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
                annotated_image = image.copy()
                detections = []

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get class ID and confidence
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])

                            if confidence > self.confidence_threshold and class_id in self.vehicle_classes.values():
                                # Get vehicle type and tracking ID
                                vehicle_type = [k for k, v in self.vehicle_classes.items() if v == class_id][0]
                                vehicle_counts[vehicle_type] += 1

                                # Get bounding box coordinates
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2

                                # Get tracking ID if available
                                track_id = int(box.id[0]) if box.id is not None else -1

                                # Store detection info
                                detections.append({
                                    'type': vehicle_type,
                                    'confidence': confidence,
                                    'bbox': [x1, y1, x2, y2],
                                    'center': [center_x, center_y],
                                    'track_id': track_id,
                                    'area': (x2 - x1) * (y2 - y1)
                                })

                                # Draw enhanced bounding box with tracking
                                color = self._get_vehicle_color(vehicle_type)
                                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)

                                # Draw center point
                                cv2.circle(annotated_image, (center_x, center_y), 5, color, -1)

                                # Add enhanced label with tracking ID
                                label = f"{vehicle_type}({track_id}): {confidence:.2f}"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                cv2.rectangle(annotated_image, (x1, y1-label_size[1]-10), 
                                            (x1+label_size[0], y1), color, -1)
                                cv2.putText(annotated_image, label, (x1, y1-5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Calculate advanced metrics
                density_map = self._calculate_traffic_density(image, detections)
                flow_analysis = self._analyze_traffic_flow(detections)

                return {
                    'vehicle_counts': vehicle_counts,
                    'annotated_image': annotated_image,
                    'total_vehicles': sum(vehicle_counts.values()),
                    'detections': detections,
                    'density': len(detections) / (image.shape[0] * image.shape[1] / 10000),  # vehicles per 100x100 area
                    'density_map': density_map,
                    'flow_analysis': flow_analysis,
                    'detection_method': 'YOLO_with_tracking',
                    'confidence_threshold': self.confidence_threshold
                }

            except Exception as e:
                print(f"YOLO detection failed: {str(e)}")
                return self._simulate_advanced_detection(image)
        else:
            return self._simulate_advanced_detection(image)

    def _get_vehicle_color(self, vehicle_type):
        """Get color for vehicle type visualization"""
        colors = {
            'car': (0, 255, 0),      # Green
            'truck': (255, 0, 0),    # Red
            'bus': (0, 0, 255),      # Blue
            'motorcycle': (255, 255, 0)  # Yellow
        }
        return colors.get(vehicle_type, (128, 128, 128)) # Default to grey

    def _calculate_traffic_density(self, image, detections):
        """Calculate traffic density heatmap"""
        h, w = image.shape[:2]
        # Create a smaller grid for density map for performance
        density_map_grid_size = 10
        density_map = np.zeros((h // density_map_grid_size, w // density_map_grid_size), dtype=np.float32)

        for detection in detections:
            center_x, center_y = detection['center']
            # Map center to grid coordinates
            grid_x = min(center_x // density_map_grid_size, density_map.shape[1] - 1)
            grid_y = min(center_y // density_map_grid_size, density_map.shape[0] - 1)

            # Weight by vehicle size (larger vehicles contribute more)
            # Normalize area to be between 0 and 1 (approx)
            normalized_area = (detection['area'] / (w * h)) * 100 if w * h > 0 else 0
            density_map[grid_y, grid_x] += normalized_area

        # Apply Gaussian smoothing for a smoother heatmap
        if CV2_AVAILABLE:
            density_map = cv2.GaussianBlur(density_map, (5, 5), 0)

        # Normalize the density map to [0, 1]
        if density_map.max() > 0:
            density_map = density_map / density_map.max()
            
        return density_map

    def _analyze_traffic_flow(self, detections):
        """Analyze traffic flow patterns"""
        if not detections:
            return {'average_speed': 0, 'flow_direction': 'unknown', 'congestion_points': [], 'vehicle_trajectories': 0, 'lane_occupancy': {}}

        # Simulate flow analysis (in real implementation, this would use tracking data)
        # For now, we use detection centers and track_ids as proxies
        
        # Estimate average speed and direction (very basic simulation)
        speeds = []
        directions = []
        for detection in detections:
            if detection['track_id'] > 0: # Only consider tracked vehicles for speed/direction
                # In a real scenario, we would compare positions across frames
                # For simulation, we assign random plausible values
                speeds.append(np.random.uniform(15, 70)) # km/h
                directions.append(np.random.choice(['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']))
        
        avg_speed = np.mean(speeds) if speeds else 0
        # Determine dominant direction (simplified)
        dominant_direction = "mixed"
        if directions:
            from collections import Counter
            direction_counts = Counter(directions)
            most_common_dir, _ = direction_counts.most_common(1)[0] if direction_counts else (None, 0)
            if most_common_dir:
                dominant_direction = most_common_dir

        flow_analysis = {
            'average_speed': round(avg_speed, 2),  # km/h
            'flow_direction': dominant_direction,
            'congestion_points': [],
            'vehicle_trajectories': len([d for d in detections if d['track_id'] > 0]),
            'lane_occupancy': {
                'lane_1': np.random.uniform(0.3, 0.9),
                'lane_2': np.random.uniform(0.2, 0.8),
                'lane_3': np.random.uniform(0.1, 0.7)
            }
        }

        # Identify potential congestion points using clustering
        vehicle_positions = [d['center'] for d in detections]
        if len(vehicle_positions) > 5:
            from sklearn.cluster import KMeans
            
            # Use KMeans to find clusters of vehicles
            # Number of clusters can be dynamic or a parameter
            n_clusters = min(max(2, len(vehicle_positions) // 5), 10) # Heuristic for number of clusters
            
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
                kmeans.fit(vehicle_positions)
                
                # Analyze cluster density
                for i in range(n_clusters):
                    cluster_points = np.array(vehicle_positions)[kmeans.labels_ == i]
                    if len(cluster_points) > 2: # Consider clusters with more than 2 vehicles as potentially congested
                        center = kmeans.cluster_centers_[i]
                        # Estimate density within the cluster (e.g., vehicles per unit area)
                        # A simple proxy: count of points in cluster / area covered by cluster (approximated by bounding box)
                        min_x, min_y = np.min(cluster_points, axis=0)
                        max_x, max_y = np.max(cluster_points, axis=0)
                        cluster_area = (max_x - min_x) * (max_y - min_y) if (max_x > min_x and max_y > min_y) else 1
                        density_score = len(cluster_points) / cluster_area if cluster_area > 0 else len(cluster_points)

                        flow_analysis['congestion_points'].append({
                            'center': [int(center[0]), int(center[1])],
                            'vehicle_count': len(cluster_points),
                            'density_score': round(density_score, 4)
                        })
            except Exception as e:
                print(f"KMeans clustering failed: {e}")


        return flow_analysis

    def _simulate_advanced_detection(self, image):
        """Simulate advanced vehicle detection when YOLO is not available"""
        h, w = image.shape[:2]

        # Simulate random detections with tracking
        num_vehicles = np.random.randint(15, 60)
        vehicle_counts = {
            'car': np.random.randint(8, 35),
            'truck': np.random.randint(1, 12),
            'bus': np.random.randint(0, 5),
            'motorcycle': np.random.randint(0, 8)
        }

        # Ensure total matches
        total = sum(vehicle_counts.values())
        if total != num_vehicles:
            # Adjust 'car' count to match the total desired vehicles
            vehicle_counts['car'] += (num_vehicles - total)
            # Ensure car count doesn't become negative
            vehicle_counts['car'] = max(0, vehicle_counts['car'])


        # Create annotated image with simulated boxes
        annotated_image = image.copy()
        detections = []

        track_id_counter = 1
        for vehicle_type, count in vehicle_counts.items():
            for i in range(count):
                # Random bounding box
                x1 = np.random.randint(0, w-100)
                y1 = np.random.randint(0, h-80)
                x2 = x1 + np.random.randint(40, 120)
                y2 = y1 + np.random.randint(30, 80)

                # Ensure box is within image boundaries
                x2 = min(x2, w)
                y2 = min(y2, h)

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Store detection
                detections.append({
                    'type': vehicle_type,
                    'confidence': np.random.uniform(0.7, 0.95),
                    'bbox': [x1, y1, x2, y2],
                    'center': [center_x, center_y],
                    'track_id': track_id_counter,
                    'area': (x2 - x1) * (y2 - y1)
                })

                # Draw enhanced box
                color = self._get_vehicle_color(vehicle_type)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                cv2.circle(annotated_image, (center_x, center_y), 5, color, -1)

                # Add label with tracking ID
                label = f"{vehicle_type}({track_id_counter}): {detections[-1]['confidence']:.2f}"
                # Calculate text size for positioning the label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                              (x1 + label_size[0], y1), color, cv2.FILLED)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                track_id_counter += 1

        # Calculate advanced metrics
        density_map = self._calculate_traffic_density(image, detections)
        flow_analysis = self._analyze_traffic_flow(detections)

        return {
            'vehicle_counts': vehicle_counts,
            'annotated_image': annotated_image,
            'total_vehicles': sum(vehicle_counts.values()),
            'detections': detections,
            'density': len(detections) / (h * w / 10000) if h*w > 0 else 0, # vehicles per 100x100 area
            'density_map': density_map,
            'flow_analysis': flow_analysis,
            'detection_method': 'Advanced_Simulation'
        }

    def track_vehicles(self, detections, frame_id):
        """Track vehicles across frames using centroid tracking"""
        if not detections:
            return []

        # Use detections directly if they already contain track_ids from YOLO tracking
        # If not, fall back to centroid tracking (though YOLO's track method is preferred)
        if all('track_id' in det and det['track_id'] > 0 for det in detections):
            # YOLO's tracking is already integrated, no need for separate centroid tracker
            # We just return the detections with their track_ids
            return detections
        else:
            # Fallback to manual centroid tracking if YOLO tracking is not used or failed
            # This part of the code might be redundant if YOLO's persist=True is always used
            # and provides track IDs. Keeping for robustness.
            
            current_centroids = {det['track_id']: det['center'] for det in detections if 'track_id' in det and det['track_id'] > 0}
            
            # If no existing trackers or detections have track IDs, initialize new trackers
            if not self.trackers or not current_centroids:
                for i, det in enumerate(detections):
                    if 'track_id' not in det or det['track_id'] <= 0: # Assign new track ID if missing
                        det['track_id'] = self.next_id
                        self.next_id += 1
                    self.trackers[det['track_id']] = {
                        'centroid': det['center'],
                        'frames': deque([frame_id], maxlen=10),
                        'path': deque([det['center']], maxlen=20)
                    }
                return detections

            # Match current detections to existing trackers
            matched_detections = []
            used_tracker_ids = set()

            for det in detections:
                best_match_id = -1
                min_dist = float('inf')

                if 'track_id' in det and det['track_id'] > 0 and det['track_id'] in self.trackers:
                    # If detection already has a valid track ID, try to update its tracker
                    tracker_id = det['track_id']
                    if tracker_id in self.trackers:
                        tracker_centroid = self.trackers[tracker_id]['centroid']
                        dist = math.sqrt((det['center'][0] - tracker_centroid[0])**2 + (det['center'][1] - tracker_centroid[1])**2)
                        if dist < 50: # Threshold for matching
                            self.trackers[tracker_id]['centroid'] = det['center']
                            self.trackers[tracker_id]['frames'].append(frame_id)
                            self.trackers[tracker_id]['path'].append(det['center'])
                            matched_detections.append(det)
                            used_tracker_ids.add(tracker_id)
                            continue # Move to next detection, this one is matched

                # If no direct track ID match or it failed, try to find nearest tracker
                # This part is crucial if YOLO's track IDs are not persistent or reliable
                # However, given the use of persist=True, this fallback might be less critical
                # but good for robustness.
                
                # For now, prioritize existing track IDs from YOLO. If a detection has a track_id
                # and it's not found in self.trackers (e.g., tracker was removed),
                # we should treat it as a new detection to potentially create a new tracker.

            # Create new trackers for detections that were not matched or had no track ID
            new_detection_ids = set()
            for det in detections:
                if 'track_id' not in det or det['track_id'] <= 0 or det['track_id'] not in self.trackers:
                    # Check if this detection center is already close to an existing tracker
                    # This is a simplified check to avoid creating too many duplicate trackers
                    is_close_to_existing = False
                    for tid, tracker in self.trackers.items():
                        if tid not in used_tracker_ids:
                             dist = math.sqrt((det['center'][0] - tracker['centroid'][0])**2 + (det['center'][1] - tracker['centroid'][1])**2)
                             if dist < 50: # Threshold for considering it a match to an existing but not explicitly ID'd tracker
                                 is_close_to_existing = True
                                 break
                    
                    if not is_close_to_existing:
                        det['track_id'] = self.next_id
                        self.trackers[det['track_id']] = {
                            'centroid': det['center'],
                            'frames': deque([frame_id], maxlen=10),
                            'path': deque([det['center']], maxlen=20)
                        }
                        self.next_id += 1
                        matched_detections.append(det)
                    else:
                        # If it's close to an existing tracker but didn't get matched,
                        # we might need to update that tracker. This logic can get complex.
                        # For now, we assume YOLO's tracking is good enough to provide initial IDs.
                        # If YOLO's track_id management is problematic, a full Hungarian algorithm
                        # based matching would be needed here.
                        pass # Handled by the matching logic if it were more advanced

            # Remove trackers that have not been updated for a while
            inactive_tracker_ids = []
            for tracker_id, tracker in self.trackers.items():
                if frame_id - tracker['frames'][-1] > 5: # If last update was more than 5 frames ago
                    inactive_tracker_ids.append(tracker_id)
            
            for tid in inactive_tracker_ids:
                del self.trackers[tid]

            # Ensure all detections have a track_id assigned (even if it's a new one)
            final_detections = []
            assigned_track_ids = set()
            for det in detections:
                if 'track_id' in det and det['track_id'] > 0:
                    final_detections.append(det)
                    assigned_track_ids.add(det['track_id'])
                else:
                    # This case should be handled by the new tracker creation above
                    # but as a safeguard:
                    if 'track_id' not in det or det['track_id'] <= 0:
                        det['track_id'] = self.next_id
                        self.trackers[det['track_id']] = {
                            'centroid': det['center'],
                            'frames': deque([frame_id], maxlen=10),
                            'path': deque([det['center']], maxlen=20)
                        }
                        self.next_id += 1
                        final_detections.append(det)
                        assigned_track_ids.add(det['track_id'])


            return final_detections


    def apply_clahe(self, image, clip_limit=3.0, tile_grid_size=(8, 8)):
        """Apply CLAHE to enhance contrast"""
        if not CV2_AVAILABLE:
            print("OpenCV not available. Cannot apply CLAHE.")
            return image

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
        if not CV2_AVAILABLE:
            print("OpenCV not available. Cannot apply Gaussian filter.")
            return image
        
        if sigma == 0:
            # Default sigma calculation if not provided
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Apply Canny edge detection"""
        if not CV2_AVAILABLE:
            print("OpenCV not available. Cannot apply Canny edge detection.")
            return image.copy() # Return copy to maintain consistent return type

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges

    def create_traffic_flow_visualization(self, detections_history):
        """Create traffic flow visualization from detection history"""
        # This method is a placeholder. Real implementation would require
        # analyzing the `detections_history` (a list of detection dictionaries per frame)
        # to compute metrics like average speed, flow rate, queue lengths, etc.
        
        if not detections_history:
            return {
                'incoming_vehicles': 0,
                'outgoing_vehicles': 0,
                'average_speed': 0.0,  # km/h
                'flow_direction': 'unknown',
                'congestion_level': 'low'
            }

        # Example: Using the last frame's detections for a basic visualization
        last_frame_detections = detections_history[-1]
        
        # Placeholder calculations
        flow_data = {
            'incoming_vehicles': len([d for d in last_frame_detections if d.get('center', [0,0])[1] < 100]), # Simplified: vehicles in top part of image
            'outgoing_vehicles': len([d for d in last_frame_detections if d.get('center', [0,0])[1] > last_frame_detections[0].get('bbox', [0,0,0,0])[3] - 100]), # Simplified: vehicles in bottom part
            'average_speed': np.random.uniform(20, 80),  # km/h (simulated)
            'flow_direction': np.random.choice(['north', 'south', 'east', 'west', 'mixed']),
            'congestion_level': np.random.choice(['low', 'medium', 'high'])
        }

        return flow_data

    def set_confidence_threshold(self, threshold=0.5):
        """Set confidence threshold for YOLO detections"""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
            print(f"Confidence threshold set to {self.confidence_threshold}")
            # Update model's confidence if model is loaded
            if self.model_loaded and hasattr(self.model, 'conf'):
                self.model.conf = threshold
        else:
            print("Confidence threshold must be between 0.0 and 1.0")