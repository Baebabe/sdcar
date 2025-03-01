# carla_env.py





def __init__(self, sim_port=2000, tm_port=8000):
        self.sim_port = sim_port
        self.tm_port = tm_port
        self.client = None
        self.world = None
        self.camera = None
        self.vehicle = None
        self.collision_hist = []
        self.collision_sensor = None
        self.yolo_model = None
        self.max_objects = 5
        self.last_location = None
        self.stuck_time = 0
        self.episode_start = 0

        # Add MPC controller
        self.controller = None  # This will hold your MPC controller

        # Add missing attributes
        self.front_camera = None
        self.npc_vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []

        pygame.init()
        self.display = None
        self.clock = None
        self.init_pygame_display()

        # Camera settings
        self.camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7),
            carla.Rotation(pitch=0)
        )

        # Initialize CARLA world first
        self.setup_world()

        # Initialize YOLO model
        self._init_yolo()

def setup_world(self):
    try:
        print(f"Connecting to CARLA server on port {self.sim_port}...")
        self.client = carla.Client('localhost', self.sim_port)
        self.client.set_timeout(20.0)
        
        print("Getting world...")
        self.world = self.client.get_world()
        
        # Set up traffic manager with the specified port
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.traffic_manager.global_percentage_speed_difference(10.0)
        
        # Configure synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        
        # Stabilize the world
        for _ in range(10):
            self.world.tick()
        
        print(f"CARLA world setup completed for port {self.sim_port}")
        
    except Exception as e:
        print(f"Error setting up CARLA world on port {self.sim_port}: {e}")
        raise





















import os
import sys
import glob
import time
import random
import numpy as np
import cv2
import weakref
import math
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import requests 
import pygame
from pygame import surfarray
import traceback
from mpc import MPCController

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Add YOLOv5 to path
yolov11_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ultralytics')
if yolov11_path not in sys.path:
    sys.path.append(yolov11_path)

# Now import YOLOv11 modules
try:
    try:
        from models.experimental import attempt_load
        print("YOLOv11 modules imported successfully!")
    except ImportError as e:
        print(f"Import error details: {e}")
        print("Current sys.path:", sys.path)
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device
except ImportError:
    print("Error importing YOLOv5 modules. Make sure YOLOv5 is properly installed.")
    sys.exit(1)

# Constants
IM_WIDTH = 640
IM_HEIGHT = 480
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_spatial_info(bbox, class_id, image_width=IM_WIDTH, image_height=IM_HEIGHT, fov=90):
    """
    Calculate depth and relative position using bounding box and class-specific dimensions
    
    Returns:
        - depth: estimated distance to object in meters
        - confidence: confidence in the depth estimate
        - relative_angle: angle to object in degrees (0 = straight ahead, negative = left, positive = right)
        - normalized_x_pos: horizontal position in normalized coordinates (-1 to 1, where 0 is center)
        - lane_position: estimated lane position relative to ego vehicle (-1: left lane, 0: same lane, 1: right lane)
    """
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    
    # Center of the bounding box
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Define typical widths for different object classes (in meters)
    REAL_WIDTHS = {
        0: 0.45,   # person - average shoulder width
        
        # Vehicles
        1: 0.8,    # bicycle - typical handlebar width
        2: 0.8,    # motorcycle - typical handlebar width
        3: 1.8,    # car - average car width
        4: 2.5,    # truck - average truck width
        5: 2.9,    # bus - average bus width
        6: 3.0,    # train - typical train car width
        
        # Outdoor objects
        7: 0.6,    # fire hydrant - typical width
        8: 0.3,    # stop sign - standard width
        9: 0.3,    # parking meter - typical width
        10: 0.4,   # bench - typical seat width
    }
    
    # Get real width based on class, default to car width if class not found
    real_width = REAL_WIDTHS.get(class_id, 1.8)
    
    # Calculate focal length using camera parameters
    focal_length = (image_width / 2) / np.tan(np.radians(fov / 2))
    
    # Calculate depth using similar triangles principle
    if bbox_width > 0:  # Avoid division by zero
        depth = (real_width * focal_length) / bbox_width
    else:
        depth = float('inf')
    
    # Horizontal position normalized to [-1, 1] where 0 is center
    normalized_x_pos = (center_x - (image_width / 2)) / (image_width / 2)
    
    # Calculate relative angle in degrees
    relative_angle = np.degrees(np.arctan2(normalized_x_pos * np.tan(np.radians(fov / 2)), 1))
    
    # Estimate lane position
    # This is a simple heuristic based on horizontal position and object width
    # More sophisticated lane detection would use road markings
    if abs(normalized_x_pos) < 0.2:
        # Object is roughly centered - likely in same lane
        lane_position = 0
    elif normalized_x_pos < 0:
        # Object is to the left
        lane_position = -1
    else:
        # Object is to the right
        lane_position = 1
    
    # For vehicles (class 1-6), refine lane estimation based on size and position
    if 1 <= class_id <= 6:
        # Calculate expected width at this depth if in same lane
        expected_width_in_px = (real_width * focal_length) / depth
        
        # Ratio of actual width to expected width if centered
        width_ratio = bbox_width / expected_width_in_px
        
        # If object seems too small for its position, might be in adjacent lane
        if width_ratio < 0.7 and abs(normalized_x_pos) < 0.4:
            # Object appears smaller than expected for this position
            if normalized_x_pos < 0:
                lane_position = -1
            else:
                lane_position = 1
    
    # Calculate confidence based on multiple factors
    size_confidence = min(1.0, bbox_width / (image_width * 0.5))  # Higher confidence for larger objects
    center_confidence = 1.0 - abs(normalized_x_pos)  # Higher confidence for centered objects
    aspect_confidence = min(1.0, bbox_height / (bbox_width + 1e-6) / 0.75)  # Expected aspect ratio
    
    # Combined confidence score
    confidence = (size_confidence * 0.5 + center_confidence * 0.3 + aspect_confidence * 0.2)
    
    return {
        'depth': depth,
        'confidence': confidence,
        'relative_angle': relative_angle,
        'normalized_x_pos': normalized_x_pos,
        'lane_position': lane_position
    }

class CarEnv:
    def __init__(self):
        # Existing attributes
        self.client = None
        self.world = None
        self.camera = None
        self.vehicle = None
        self.collision_hist = []
        self.collision_sensor = None
        self.yolo_model = None
        self.max_objects = 5
        self.last_location = None
        self.stuck_time = 0
        self.episode_start = 0

        # Add MPC controller
        self.controller = None  # This will hold your MPC controller

        # Add missing attributes
        self.front_camera = None
        self.npc_vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []

        pygame.init()
        self.display = None
        self.clock = None
        self.init_pygame_display()

        # Camera settings
        self.camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7),
            carla.Rotation(pitch=0)
        )

        # Initialize CARLA world first
        self.setup_world()

        # Initialize YOLO model
        self._init_yolo()

    def init_pygame_display(self):
        """Initialize pygame display with error handling"""
        try:
            if self.display is None:
                pygame.init()
                self.display = pygame.display.set_mode((IM_WIDTH, IM_HEIGHT))
                pygame.display.set_caption("CARLA + YOLO View")
                self.clock = pygame.time.Clock()
        except Exception as e:
            print(f"Error initializing pygame display: {e}")
            traceback.print_exc()

    def _process_image(self, weak_self, image):
        self = weak_self()
        if self is not None:
            try:
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]  # Remove alpha channel
                self.front_camera = array   

                # Process YOLO detection with proper scaling
                detections = self.process_yolo_detection(array) 

                # Create pygame surface
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))   

                # Draw detections with proper scaling
                for obj in detections:
                    # Get original image coordinates
                    x1 = int(obj['position'][0] - obj['bbox_width']/2)
                    y1 = int(obj['position'][1] - obj['bbox_height']/2)
                    x2 = int(obj['position'][0] + obj['bbox_width']/2)
                    y2 = int(obj['position'][1] + obj['bbox_height']/2) 

                    # Scale coordinates to match display size
                    x1 = int(x1 * IM_WIDTH / 640)
                    y1 = int(y1 * IM_HEIGHT / 640)
                    x2 = int(x2 * IM_WIDTH / 640)
                    y2 = int(y2 * IM_HEIGHT / 640)  

                    # Determine rectangle color based on object type or traffic light color
                    rect_color = (0, 255, 0)  # Default green
                    
                    # For traffic lights, use color based on detected state
                    if "traffic light" in obj['class_name'].lower() and obj['traffic_light_color']:
                        if obj['traffic_light_color'] == "red":
                            rect_color = (255, 0, 0)  # Red
                        elif obj['traffic_light_color'] == "yellow":
                            rect_color = (255, 255, 0)  # Yellow
                        elif obj['traffic_light_color'] == "green":
                            rect_color = (0, 255, 0)  # Green   

                    # Draw rectangle with appropriate color
                    pygame.draw.rect(surface, rect_color, (x1, y1, x2-x1, y2-y1), 2)    

                    # Draw label
                    if pygame.font.get_init():
                        font = pygame.font.Font(None, 24)
                        # Include traffic light color in label if available
                        if "traffic light" in obj['class_name'].lower() and obj['traffic_light_color']:
                            label = f"{obj['class_name']} ({obj['traffic_light_color']}) {obj['depth']:.1f}m"
                        else:
                            label = f"{obj['class_name']} {obj['depth']:.1f}m"
                        text = font.render(label, True, (255, 255, 255))
                        surface.blit(text, (x1, y1-20)) 

                # Update display
                if self.display is not None:
                    self.display.blit(surface, (0, 0))
                    pygame.display.flip()   

            except Exception as e:
                print(f"Error in image processing: {e}")
                traceback.print_exc()

    def cleanup_actors(self):
        """Clean up all spawned actors"""
        try:
            print("Starting cleanup of actors...")
            # Clean up sensors and vehicle
            if hasattr(self, 'collision_sensor') and self.collision_sensor and self.collision_sensor.is_alive:
                self.collision_sensor.destroy()
                print("Collision sensor destroyed")
            if hasattr(self, 'camera') and self.camera and self.camera.is_alive:
                self.camera.destroy()
                print("Camera destroyed") 
            if hasattr(self, 'vehicle') and self.vehicle and self.vehicle.is_alive:
                self.vehicle.destroy()
                print("Vehicle destroyed")

            self.collision_sensor = None
            self.camera = None
            self.vehicle = None
            self.front_camera = None

            # Clear MPC controller
            self.controller = None
            print("MPC controller cleared")

            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error cleaning up actors: {e}")
            import traceback
            traceback.print_exc()

    def setup_world(self):
        """Initialize CARLA world and settings"""
        try:
            print("Connecting to CARLA server...")
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(20.0)
            
            print("Getting world...")
            self.world = self.client.get_world()
            
            # Set up traffic manager
            self.traffic_manager = self.client.get_trafficmanager(8000)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            self.traffic_manager.global_percentage_speed_difference(10.0)
            
            # Set synchronous mode settings
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            # Wait for the world to stabilize
            for _ in range(10):
                self.world.tick()
            
            print("CARLA world setup completed successfully")
            
        except Exception as e:
            print(f"Error setting up CARLA world: {e}")
            raise
        
    def _init_yolo(self):
        """Initialize YOLOv11 model from Ultralytics"""
        try:
            print("Loading YOLOv11 model...")
            
            # Define path to ultralytics YOLOv11
            ultralytics_path = r"C:\Users\msi\miniconda3\envs\sdcarAB\Carla-0.10.0-Win64-Shipping\PythonAPI\sdcar\ultralytics"
            if ultralytics_path not in sys.path:
                sys.path.append(ultralytics_path)
            
            # Define model path
            model_path = os.path.join(ultralytics_path, 'yolov11n.pt')
            
            # Check if the model exists, if not, download it
            if not os.path.exists(model_path):
                print(f"Model not found at {model_path}, downloading YOLOv11 model")
                self._download_yolov11_weights(model_path)
            
            # Import the YOLO model from Ultralytics package
            from ultralytics import YOLO
            
            print(f"Loading model from: {model_path}")
            self.yolo_model = YOLO(model_path)
            
            # Configure model settings
            self.yolo_model.conf = 0.25  # Confidence threshold
            self.yolo_model.iou = 0.45   # NMS IoU threshold
            
            print("YOLOv11 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading YOLOv11 model: {e}")
            import traceback
            traceback.print_exc()
            raise   

    def _download_yolov11_weights(self, save_path):
        """Download YOLOv11 weights if they don't exist"""
        import requests
        
        # URL for YOLOv11 weights (you'll need to update this with the correct URL)
        # This is a placeholder URL - you'll need to replace it with the actual URL
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11n.pt"
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            print(f"Downloading YOLOv11 weights to {save_path}")
            response = requests.get(url, stream=True)
            
            # Check if the request was successful
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                block_size = 8192
                downloaded = 0
                
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=block_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Print progress
                            if total_size > 0:
                                percent = downloaded * 100 / total_size
                                print(f"Download progress: {percent:.1f}%", end="\r")
                
                print(f"\nDownload complete! YOLOv11 weights saved to {save_path}")
                return True
            else:
                print(f"Failed to download YOLOv11 weights. Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error downloading YOLOv11 weights: {e}")
            return False

    def process_yolo_detection(self, image):
        """Process image with YOLOv11 and return detections with enhanced spatial information"""
        if image is None:
            return []   

        try:
            # YOLOv11 (Ultralytics) expects BGR format for inference
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Run inference with YOLOv11
            results = self.yolo_model(img_bgr, verbose=False)

            objects = []

            # Process each detection
            for result in results:
                # Get the boxes, confidences, and class IDs
                boxes = result.boxes.xyxy.cpu().numpy()  
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                # Process each detection
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confidences[i]
                    cls = class_ids[i]

                    # Get class name
                    class_name = result.names[cls]

                    # Get enhanced spatial information
                    spatial_info = calculate_spatial_info([x1, y1, x2, y2], cls)

                    # Calculate time-to-collision (TTC) if vehicle is moving
                    ttc = float('inf')
                    if hasattr(self, 'vehicle') and self.vehicle:
                        velocity = self.vehicle.get_velocity()
                        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s

                        # Only calculate TTC for objects roughly in our path
                        if abs(spatial_info['relative_angle']) < 30 and speed > 0.5:
                            ttc = spatial_info['depth'] / speed  # seconds

                    # Initialize traffic light color
                    traffic_light_color = None

                    # Direct traffic light color detection
                    if "traffic light" in class_name.lower():
                        # Extract the portion of the image containing the traffic light
                        tl_img = img_bgr[int(y1):int(y2), int(x1):int(x2)]

                        # Check if the cropped image is not empty
                        if tl_img.size > 0:
                            # Convert to HSV for better color detection
                            tl_hsv = cv2.cvtColor(tl_img, cv2.COLOR_BGR2HSV)

                            # Define color ranges for traffic lights
                            # Adjust these ranges based on testing with your specific environment
                            red_lower1 = np.array([0, 120, 70])
                            red_upper1 = np.array([10, 255, 255])
                            red_lower2 = np.array([170, 120, 70])
                            red_upper2 = np.array([180, 255, 255])
                            yellow_lower = np.array([20, 100, 100])
                            yellow_upper = np.array([30, 255, 255])
                            green_lower = np.array([40, 50, 50])
                            green_upper = np.array([90, 255, 255])

                            # Create masks for each color
                            red_mask1 = cv2.inRange(tl_hsv, red_lower1, red_upper1)
                            red_mask2 = cv2.inRange(tl_hsv, red_lower2, red_upper2)
                            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
                            yellow_mask = cv2.inRange(tl_hsv, yellow_lower, yellow_upper)
                            green_mask = cv2.inRange(tl_hsv, green_lower, green_upper)

                            # Count pixels for each color
                            red_pixels = cv2.countNonZero(red_mask)
                            yellow_pixels = cv2.countNonZero(yellow_mask)
                            green_pixels = cv2.countNonZero(green_mask)

                            # Determine color based on pixel counts
                            color_counts = {
                                "red": red_pixels,
                                "yellow": yellow_pixels,
                                "green": green_pixels
                            }

                            # Get the color with the most pixels if it exceeds a threshold
                            max_color = max(color_counts, key=color_counts.get)
                            if color_counts[max_color] > 20:  # Adjust threshold as needed
                                traffic_light_color = max_color

                    # Create enriched object dictionary
                    objects.append({
                        'position': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'depth': spatial_info['depth'],
                        'depth_confidence': spatial_info['confidence'],
                        'relative_angle': spatial_info['relative_angle'],
                        'normalized_x_pos': spatial_info['normalized_x_pos'],
                        'lane_position': spatial_info['lane_position'],
                        'time_to_collision': ttc,
                        'class': int(cls),
                        'class_name': class_name,
                        'traffic_light_color': traffic_light_color,
                        'detection_confidence': float(conf),
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1,
                        # Calculate risk score
                        'risk_score': self._calculate_risk_score(
                            spatial_info['depth'],
                            spatial_info['lane_position'],
                            ttc,
                            int(cls),
                            traffic_light_color
                        )
                    })

            # Sort by risk score (higher risk first)
            objects.sort(key=lambda x: x['risk_score'], reverse=True)
            return objects[:self.max_objects]

        except Exception as e:
            print(f"Error in YOLOv11 detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _calculate_risk_score(self, depth, lane_position, ttc, class_id, traffic_light_color=None):
        """Calculate risk score for object based on multiple factors including traffic light status"""
        # Base risk inversely proportional to distance
        distance_factor = 10.0 / max(1.0, depth)    

        # Lane position factor (higher if in same lane)
        lane_factor = 1.0 if lane_position == 0 else 0.3    

        # Time to collision factor (higher for imminent collisions)
        ttc_factor = 1.0 if ttc < 3.0 else (0.5 if ttc < 6.0 else 0.2)  

        # Object type factor - adapt to YOLOv11 classes
        # Get class names from model if available
        if hasattr(self, 'yolo_model') and hasattr(self.yolo_model, 'names'):
            class_name = self.yolo_model.names[class_id].lower()
        else:
            class_name = ""

        # Set type factors based on class name instead of fixed indices
        type_factor = 0.5  # Default value

        if 'person' in class_name:
            type_factor = 1.0  # Highest risk
        elif any(vehicle in class_name for vehicle in ['car', 'truck', 'bus']):
            type_factor = 0.9  # High risk for large vehicles
        elif any(vehicle in class_name for vehicle in ['motorcycle', 'bicycle']):
            type_factor = 0.8  # Medium-high risk
        elif 'stop sign' in class_name:
            type_factor = 0.8  # Important for traffic rules

        # Check if this is a traffic light
        is_traffic_light = 'traffic light' in class_name    

        # Traffic light color modifies risk significantly
        if is_traffic_light and traffic_light_color:
            if traffic_light_color == "red":
                # Red light is high risk if we're moving
                velocity = self.vehicle.get_velocity() if hasattr(self, 'vehicle') else None
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) if velocity else 0 

                # Higher risk for red light when moving fast
                type_factor = 1.5 if speed > 5.0 else 1.0
            elif traffic_light_color == "yellow":
                type_factor = 0.9  # Yellow is medium risk
            elif traffic_light_color == "green":
                type_factor = 0.2  # Green is low risk  

        # Combine factors
        risk_score = distance_factor * lane_factor * ttc_factor * type_factor
        return risk_score
     
    def get_waypoint_info(self):
        """Get waypoint information"""
        location = self.vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(location)
        
        # Calculate distance and angle to waypoint
        distance = location.distance(waypoint.transform.location)
        
        vehicle_forward = self.vehicle.get_transform().get_forward_vector()
        waypoint_forward = waypoint.transform.get_forward_vector()
        
        dot = vehicle_forward.x * waypoint_forward.x + vehicle_forward.y * waypoint_forward.y
        cross = vehicle_forward.x * waypoint_forward.y - vehicle_forward.y * waypoint_forward.x
        angle = math.atan2(cross, dot)
        
        return {
            'distance': distance,
            'angle': angle
        }
    
    def get_vehicle_state(self):
        """Get vehicle state information"""
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)  # km/h
        steering = self.vehicle.get_control().steer
        
        return {
            'speed': speed,
            'steering': steering
        }
    
    def get_state(self):
        """Get enhanced state representation with improved spatial awareness including traffic lights"""
        if self.front_camera is None:
            return None

        # Initialize state array
        state_array = []

        # 1. Process YOLO detections with enhanced spatial information
        detections = self.process_yolo_detection(self.front_camera)

        # Track if we have a traffic light in our detections
        traffic_light_detected = False
        traffic_light_features = [0.0, 0.0, 0.0, 100.0]  # [red, yellow, green, distance]

        # Create a feature vector for each detected object
        for obj in detections:
            # Special handling for traffic lights
            if "traffic_light" in obj['class_name']:
                traffic_light_detected = True
                color = obj.get('traffic_light_color')
                if color == "red":
                    traffic_light_features = [1.0, 0.0, 0.0, obj['depth'] / 100.0]
                elif color == "yellow":
                    traffic_light_features = [0.0, 1.0, 0.0, obj['depth'] / 100.0]
                elif color == "green":
                    traffic_light_features = [0.0, 0.0, 1.0, obj['depth'] / 100.0]

                # Continue processing the traffic light as a regular object too

            state_array.extend([
                obj['normalized_x_pos'],            # Horizontal position (-1 to 1)
                obj['depth'] / 100.0,               # Normalized depth 
                obj['relative_angle'] / 90.0,       # Normalized angle (-1 to 1)
                float(obj['lane_position']),        # Lane position (-1, 0, 1)
                min(1.0, 10.0 / max(0.1, obj['time_to_collision'])),  # Time-to-collision normalized
                obj['risk_score'] / 10.0            # Normalized risk score
            ])

        # Pad if fewer than max objects
        remaining_objects = self.max_objects - len(detections)
        state_array.extend([0.0] * (remaining_objects * 6))  # 6 features per object

        # Add traffic light information
        state_array.extend(traffic_light_features)  # [red, yellow, green, distance]

        # 2. Get waypoint information for path following
        waypoint_info = self.get_waypoint_info()
        state_array.extend([
            waypoint_info['distance'] / 50.0,      # distance normalized
            waypoint_info['angle'] / math.pi       # angle normalized
        ])

        # 3. Get vehicle state
        vehicle_state = self.get_vehicle_state()
        state_array.extend([
            vehicle_state['speed'] / 50.0,         # speed normalized
            vehicle_state['steering']              # steering already normalized
        ])

        # 4. Add lane-specific information
        try:
            # Get location and waypoint
            location = self.vehicle.get_location()
            waypoint = self.world.get_map().get_waypoint(location)

            # Get left and right lane markings
            has_left_lane = 1.0 if waypoint.get_left_lane() is not None else 0.0
            has_right_lane = 1.0 if waypoint.get_right_lane() is not None else 0.0

            # Add lane info to state
            state_array.extend([
                has_left_lane,
                has_right_lane,
                float(waypoint.lane_id),
                float(waypoint.lane_width) / 4.0  # Normalize typical lane width
            ])
        except:
            # Fallback if lane info can't be accessed
            state_array.extend([0.0, 0.0, 0.0, 0.75])

        return np.array(state_array, dtype=np.float16)
    

    def calculate_reward(self):
        """Improved reward function focusing on speed control for RL agent"""
        try:
            reward = 0.0
            done = False
            info = {}

            # Get current state
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            location = self.vehicle.get_location()

            # Process YOLO detections to get traffic information
            detections = self.process_yolo_detection(self.front_camera)
            safety_info = self._analyze_safety(detections)

            # Traffic light checks
            approaching_red_light = safety_info['approaching_red_light']
            approaching_yellow_light = safety_info['approaching_yellow_light'] 
            approaching_green_light = safety_info['approaching_green_light']
            traffic_light_distance = safety_info['traffic_light_distance']

            # Obstacle checks
            nearest_same_lane_dist = safety_info['nearest_same_lane_dist']
            nearest_cross_lane_dist = safety_info['nearest_cross_lane_dist']

            # Base reward for staying alive
            reward += 0.1

            # Determine target speed based on environment conditions
            target_speed = 30.0  # Default target speed in km/h

            # Adjust target speed based on traffic lights
            if approaching_red_light and traffic_light_distance < 30.0:
                target_speed = 0.0  # Should stop for red
            elif approaching_yellow_light and traffic_light_distance < 20.0:
                target_speed = 10.0  # Should slow for yellow
            elif nearest_same_lane_dist < 15.0:
                # Obstacle ahead - reduce speed based on distance
                target_speed = max(0, min(30.0, nearest_same_lane_dist * 2))
            elif approaching_green_light:
                # Can proceed at normal speed through green
                target_speed = 30.0

            # Enhanced speed reward (more weight since this is RL's main responsibility)
            speed_diff = abs(speed - target_speed)
            if speed_diff < 5.0:  # Very close to target speed
                speed_reward = 2.0
            elif speed_diff < 10.0:  # Reasonably close
                speed_reward = 1.0 - (speed_diff / 10.0)
            else:  # Too far from target
                speed_reward = -1.0 * (speed_diff / 10.0)

            # Amplify speed reward since it's the primary RL task
            reward += speed_reward * 2.0

            # Traffic light behavior rewards
            if approaching_red_light:
                if traffic_light_distance < 20.0:
                    if speed < 5.0:  # Good behavior: stopping at red light
                        reward += 5.0
                        info['traffic_behavior'] = 'stopped_at_red'
                    else:  # Bad behavior: speeding through red light
                        reward -= 20.0
                        info['traffic_violation'] = 'ran_red_light'
                        # Don't end episode, but heavily penalize
                elif traffic_light_distance < 40.0 and speed > 30.0:
                    # Penalize for approaching red light too fast
                    reward -= 2.0
                    info['traffic_warning'] = 'approaching_red_too_fast'

            elif approaching_yellow_light:
                if traffic_light_distance < 15.0:
                    if speed < 5.0:  # Safely stopped at yellow
                        reward += 2.0
                        info['traffic_behavior'] = 'stopped_at_yellow'
                    elif speed > 30.0:  # Speeding through late yellow
                        reward -= 1.0
                        info['traffic_warning'] = 'speeding_through_yellow'
                    elif traffic_light_distance < 40.0 and speed > 50.0:
                        # Penalize for approaching yellow too fast
                        reward -= 0.5

            # Handle stopped vehicle cases
            if speed < 1.0:  # Almost stopped
                # Check if there are obstacles or red lights that justify stopping
                near_obstacle = nearest_same_lane_dist < 10.0
                should_stop = near_obstacle or (approaching_red_light and traffic_light_distance < 20.0)

                if should_stop:
                    # It's good to stop for obstacles or red lights
                    reward += 1.0
                    if approaching_red_light:
                        info['stop_reason'] = 'red_light'
                    else:
                        info['stop_reason'] = 'obstacle'
                else:
                    # No reason to stop
                    reward -= 0.5
                    self.stuck_time += 0.1
                    if self.stuck_time > 3.0:
                        done = True
                        reward -= 10.0
                        info['termination_reason'] = 'stuck'
            else:
                self.stuck_time = 0

            # Distance traveled reward - reduced when approaching red/yellow
            if self.last_location is not None:
                distance_traveled = location.distance(self.last_location)
                # Reduce distance reward when approaching red light
                if approaching_red_light and traffic_light_distance < 30.0:
                    reward += distance_traveled * 0.1  # Reduced reward
                else:
                    reward += distance_traveled * 0.5  # Normal reward

            # Collision penalty
            if len(self.collision_hist) > 0:
                reward -= 50.0
                done = True
                info['termination_reason'] = 'collision'

            # Lane keeping is no longer RL's responsibility, but we'll track it for info
            waypoint = self.world.get_map().get_waypoint(location)
            distance_from_center = location.distance(waypoint.transform.location)

            # Store current location for next step
            self.last_location = location

            # Add detailed information to info dictionary
            info['speed'] = speed
            info['target_speed'] = target_speed
            info['speed_diff'] = speed_diff
            info['reward'] = reward
            info['distance_from_center'] = distance_from_center

            # Add traffic light info to debug info
            info.update({
                'red_light': approaching_red_light,
                'yellow_light': approaching_yellow_light,
                'green_light': approaching_green_light,
                'traffic_light_distance': traffic_light_distance,
                'nearest_obstacle_distance': nearest_same_lane_dist
            })

            return reward, done, info

        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            traceback.print_exc()
            return 0.0, True, {'error': str(e)}
    



    def step(self, rl_action=None):
        """Hybrid control with MPC for normal driving and RL for complex scenarios"""
        try:
            # 1. Get enhanced object detections first
            detections = self.process_yolo_detection(self.front_camera)
    
            # 2. Extract key safety information from detections
            safety_info = self._analyze_safety(detections)
    
            # 3. Determine if environment is complex (obstacles, traffic lights)
            complex_environment = (
                safety_info['approaching_red_light'] or 
                safety_info['approaching_yellow_light'] or
                safety_info['approaching_green_light'] or  # Include green for consistency
                safety_info['nearest_same_lane_dist'] < 50.0 or  # Vehicle ahead within 50m
                safety_info['nearest_cross_lane_dist'] < 20.0 or  # Cross traffic within 20m
                safety_info['emergency_braking'] or 
                safety_info['collision_avoidance']
            )
    
            # 4. Get MPC control (base behavior)
            mpc_control = self.controller.get_control(self.vehicle, self.world)
            
            # 5. Default control setup
            control = mpc_control  # Start with MPC as default
            control_source = "MPC_FULL"
            
            # Emergency override takes highest priority regardless of mode
            emergency_braking = safety_info['emergency_braking']
            if emergency_braking:
                control = carla.VehicleControl(
                    throttle=0.0,
                    steer=mpc_control.steer,  # Keep MPC steering
                    brake=1.0,  # Full braking
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False
                )
                control_source = "EMERGENCY_BRAKE"
            
            # If in complex environment, use RL for throttle/brake
            elif complex_environment and rl_action is not None:
                # Convert RL action to throttle/brake
                rl_value = float(rl_action[0])
                
                throttle = 0.0
                brake = 0.0
                
                if rl_value >= 0:  # Positive values control throttle
                    throttle = float(np.clip(rl_value, 0.0, 1.0))
                else:  # Negative values control brake
                    brake = float(np.clip(-rl_value, 0.0, 1.0))
                
                # Create hybrid control with RL throttle/brake and MPC steering
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=mpc_control.steer,  # Always use MPC steering
                    brake=brake,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False
                )
                control_source = "RL_THROTTLE_MPC_STEER"
            
            # If environment is clear, use full MPC control
            else:
                control = mpc_control
                control_source = "MPC_FULL"
            
            # Apply the final control
            self.vehicle.apply_control(control)
    
            # Store control values and environment state for info
            control_info = {
                'throttle': control.throttle,
                'steer': control.steer,
                'brake': control.brake,
                'control_source': control_source,
                'complex_environment': complex_environment
            }
            control_info.update(safety_info)  # Add safety analysis data
            
            # Continue with physics, reward calculation, etc.
            for _ in range(4):
                self.world.tick()   
    
            new_state = self.get_state()
            reward, done, info = self.calculate_reward()    
    
            info.update(control_info)   
    
            return new_state, reward, done, info    
    
        except Exception as e:
            print(f"Error in step: {e}")
            traceback.print_exc()
            return None, 0, True, {'error': str(e)}

    def _analyze_safety(self, detections):
        """Analyze detections to determine safety-critical information for control"""
        # Get vehicle velocity
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

        # Initialize safety flags
        emergency_braking = False
        collision_avoidance = False
        avoidance_steering = False
        steer_amount = 0.0

        # Traffic light specific flags
        approaching_red_light = False
        approaching_yellow_light = False
        approaching_green_light = False
        traffic_light_distance = float('inf')

        # Initialize distances
        nearest_same_lane_dist = float('inf')
        nearest_left_lane_dist = float('inf')
        nearest_right_lane_dist = float('inf')
        min_ttc = float('inf')

        # Important object classes for driving
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        critical_classes = ['person'] + vehicle_classes
        traffic_signals = ['stop sign']  # Removed traffic light as we handle it separately now

        # Analyze each detection
        for obj in detections:
            obj_depth = obj['depth']
            obj_lane = obj['lane_position']
            obj_angle = obj['relative_angle']
            obj_ttc = obj['time_to_collision']
            obj_class = obj['class_name']

            # Process traffic lights specifically
            if "traffic_light" in obj_class and abs(obj_angle) < 30 and obj_depth < 50.0:
                traffic_light_color = obj.get('traffic_light_color')

                if traffic_light_color and obj_depth < traffic_light_distance:
                    traffic_light_distance = obj_depth

                    if traffic_light_color == "red":
                        approaching_red_light = True
                        # Emergency braking for close red lights when moving at speed
                        if obj_depth < 15.0 and speed > 10.0:
                            emergency_braking = True
                        # Regular braking for red lights
                        elif obj_depth < 30.0:
                            collision_avoidance = True

                    elif traffic_light_color == "yellow":
                        approaching_yellow_light = True
                        # Decision based on distance: brake if close, proceed if far
                        if obj_depth < 20.0 and speed > 30.0:
                            collision_avoidance = True  # Slow down for yellow

                    elif traffic_light_color == "green":
                        approaching_green_light = True
                        # No special action needed for green

                continue  # Skip the standard processing for traffic lights

            # Update nearest distances by lane
            if obj_class in critical_classes:
                if obj_lane == 0:  # Same lane
                    nearest_same_lane_dist = min(nearest_same_lane_dist, obj_depth)
                elif obj_lane == -1:  # Left lane
                    nearest_left_lane_dist = min(nearest_left_lane_dist, obj_depth)
                elif obj_lane == 1:  # Right lane
                    nearest_right_lane_dist = min(nearest_right_lane_dist, obj_depth)

                # Update minimum time-to-collision for objects in our path
                if abs(obj_angle) < 30 and obj_ttc < min_ttc:
                    min_ttc = obj_ttc

            # Pedestrian checks (highest priority)
            if obj_class == 'person' and obj_depth < 15 and abs(obj_angle) < 45:
                emergency_braking = True

            # Vehicle collision checks
            elif obj_class in vehicle_classes:
                # Same lane checks
                if obj_lane == 0 and abs(obj_angle) < 30:
                    # Emergency braking condition - close vehicle in same lane
                    if obj_ttc < 2.0 or obj_depth < 5.0:
                        emergency_braking = True
                    # Collision avoidance - vehicle in our lane but not imminent emergency
                    elif obj_ttc < 6.0 or obj_depth < 15.0:
                        collision_avoidance = True

                        # Determine if we should steer to avoid
                        if speed > 5.0 and obj_depth > 8.0:
                            avoidance_steering = True

                            # Decide which way to steer based on adjacent lane distances
                            if nearest_left_lane_dist > nearest_right_lane_dist and nearest_left_lane_dist > obj_depth:
                                steer_amount = -0.5  # Steer left
                            elif nearest_right_lane_dist > obj_depth:
                                steer_amount = 0.5   # Steer right

                # Adjacent lane - vehicle cutting in
                elif obj_depth < 10.0 and abs(obj_angle) < 40:
                    collision_avoidance = True

                    # Light steering away from adjacent vehicle
                    if speed > 10.0:
                        avoidance_steering = True
                        steer_amount = 0.2 if obj_angle < 0 else -0.2  # Steer away slightly

            # Other traffic signal checks
            elif obj_class in traffic_signals and obj_depth < 25.0 and abs(obj_angle) < 20:
                collision_avoidance = True

        # Calculate nearest_cross_lane_dist as the minimum of adjacent lane distances
        nearest_cross_lane_dist = min(nearest_left_lane_dist, nearest_right_lane_dist)
        if nearest_cross_lane_dist == float('inf'):
            nearest_cross_lane_dist = 100.0
    
        # Handle other infinite distances
        if nearest_same_lane_dist == float('inf'):
            nearest_same_lane_dist = 100.0
        if min_ttc == float('inf'):
            min_ttc = 100.0
    
        # Return comprehensive safety information
        return {
            'emergency_braking': emergency_braking,
            'collision_avoidance': collision_avoidance,
            'avoidance_steering': avoidance_steering,
            'steer_amount': steer_amount,
            'nearest_same_lane_dist': nearest_same_lane_dist,
            'nearest_left_lane_dist': nearest_left_lane_dist,
            'nearest_right_lane_dist': nearest_right_lane_dist,
            'nearest_cross_lane_dist': nearest_cross_lane_dist,  # Added
            'min_ttc': min_ttc,
            'approaching_red_light': approaching_red_light,
            'approaching_yellow_light': approaching_yellow_light,
            'approaching_green_light': approaching_green_light,
            'traffic_light_distance': traffic_light_distance
        }

    def run(self):
        """Main game loop using MPC controller"""
        try:
            running = True
            while running:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_r:
                            # Reset simulation
                            self.reset()
                        elif event.key == pygame.K_p:
                            # Reset path
                            self.reset_path()

                # Execute step with MPC controller
                _, reward, done, info = self.step()

                # Handle termination
                if done:
                    print(f"Episode terminated: {info.get('termination_reason', 'unknown')}")
                    self.reset()

                # Maintain fps
                if self.clock:
                    self.clock.tick(20)

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Cleanup
            self.cleanup_actors()
            pygame.quit()

            # Disable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def reset_path(self):
        """Reset the MPC controller path with a new destination"""
        if self.vehicle and self.controller:
            # Get current location
            start_location = self.vehicle.get_location()

            # Get random destination point
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                return False

            end_point = random.choice(spawn_points)
            while end_point.location.distance(start_location) < 100:
                end_point = random.choice(spawn_points)

            # Set path using A*
            success = self.controller.set_path(
                self.world,
                start_location,
                end_point.location
            )

            return success

        return False
    
    def setup_vehicle(self):
        """Spawn and setup the ego vehicle with MPC controller"""
        try:
            print("Starting vehicle setup...")

            # Get the blueprint library
            blueprint_library = self.world.get_blueprint_library()
            print("Got blueprint library")

            # Try to get vehicle blueprints in order of priority
            vehicle_names = ['vehicle.dodge.charger_2020',
                'vehicle.lincoln.mkz_2017',
                'vehicle.nissan.patrol']
            vehicle_bp = None

            for vehicle_name in vehicle_names:
                vehicle_filters = blueprint_library.filter(vehicle_name)
                if vehicle_filters:
                    vehicle_bp = vehicle_filters[0]
                    print(f"Selected vehicle: {vehicle_name}")
                    break

            if vehicle_bp is None:
                # Fallback to any available vehicle if none of the preferred ones exist
                available_vehicles = blueprint_library.filter('vehicle.*')
                if available_vehicles:
                    vehicle_bp = random.choice(available_vehicles)
                    print(f"Fallback to available vehicle: {vehicle_bp.id}")
                else:
                    raise Exception("No vehicle blueprints found")

            # Get random spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points found")
            spawn_point = random.choice(spawn_points)
            print(f"Selected spawn point: {spawn_point}")

            # Spawn the vehicle
            self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if self.vehicle is None:
                # Try alternative spawn points if the first one fails
                for i in range(10):  # Try up to 10 different spawn points
                    spawn_point = random.choice(spawn_points)
                    self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if self.vehicle is not None:
                        break

                if self.vehicle is None:
                    raise Exception("Failed to spawn vehicle after multiple attempts")

            print("Vehicle spawned successfully")

            # Set up the camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
            camera_bp.set_attribute('fov', '90')
            print("Camera blueprint configured")

            # Spawn and set up camera
            self.camera = self.world.spawn_actor(
                camera_bp,
                self.camera_transform,
                attach_to=self.vehicle
            )
            print("Camera spawned successfully")

            # Set up camera callback
            weak_self = weakref.ref(self)
            self.camera.listen(lambda image: self._process_image(weak_self, image))
            print("Camera callback set up")

            # Set up collision sensor
            collision_bp = blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                collision_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
            print("Collision sensor spawned")

            # Set up collision callback
            self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
            print("Collision callback set up")

            # Initialize MPC controller
            self.controller = MPCController()
            print("MPC controller initialized")

            # Get random destination point for MPC path
            end_point = random.choice(spawn_points)
            while end_point.location.distance(spawn_point.location) < 100:
                end_point = random.choice(spawn_points)

            # Set path using A*
            success = self.controller.set_path(
                self.world,
                spawn_point.location,
                end_point.location
            )

            if not success:
                print("Failed to find path! Trying another destination...")
                # Try another destination if path finding fails
                for _ in range(5):  # Try up to 5 times
                    end_point = random.choice(spawn_points)
                    success = self.controller.set_path(
                        self.world,
                        spawn_point.location,
                        end_point.location
                    )
                    if success:
                        break
                    
                if not success:
                    raise Exception("Failed to find a valid path after multiple attempts")

            # Wait for sensors to initialize
            for _ in range(10):  # Wait up to 10 ticks
                self.world.tick()
                if self.front_camera is not None:
                    print("Sensors initialized successfully")
                    return True
                time.sleep(0.1)

            if self.front_camera is None:
                raise Exception("Camera failed to initialize")

            return True

        except Exception as e:
            print(f"Error setting up vehicle: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup_actors()
            return False


    def reset(self):
        """Reset environment with improved movement verification and visualization cleanup"""
        print("Starting environment reset...")

        # Clear all visualizations before cleanup
        if hasattr(self, 'world') and self.world is not None:
            try:
                # Clear all debug visualizations
                self.world.debug.draw_line(
                    carla.Location(0, 0, 0),
                    carla.Location(0, 0, 0.1),
                    thickness=0.0,
                    color=carla.Color(0, 0, 0),
                    life_time=0.0,
                    persistent_lines=False
                )

                # A workaround to clear persistent visualization is to call 
                # clear_all_debug_shapes which is available in CARLA 0.10.0
                if hasattr(self.world.debug, 'clear_all_debug_shapes'):
                    self.world.debug.clear_all_debug_shapes()
                    print("Cleared all debug visualizations")
            except Exception as e:
                print(f"Warning: Failed to clear visualizations: {e}")

        # Reset path visualization flag in MPC controller if it exists
        if hasattr(self, 'controller') and self.controller is not None:
            if hasattr(self.controller, 'path_visualization_done'):
                self.controller.path_visualization_done = False
                print("Reset MPC visualization flags")

        # Cleanup actors and NPCs
        self.cleanup_actors()
        self.cleanup_npcs()

        # Clear state variables
        self.collision_hist = []  # Clear collision history
        self.stuck_time = 0
        self.episode_start = time.time()
        self.last_location = None

        # Set up new episode
        if not self.setup_vehicle():
            raise Exception("Failed to setup vehicle")

        # Wait for initial camera frame
        print("Waiting for camera initialization...")
        timeout = time.time() + 10.0
        while self.front_camera is None:
            self.world.tick()
            if time.time() > timeout:
                raise Exception("Camera initialization timeout")
            time.sleep(0.1)

        # Spawn NPCs after camera is ready
        self.spawn_npcs()

        # Let the physics settle and NPCs initialize
        for _ in range(20):
            self.world.tick()
            time.sleep(0.05)

        # Get initial state after everything is set up
        state = self.get_state()
        if state is None:
            raise Exception("Failed to get initial state")

        print("Environment reset complete")
        return state




    def spawn_npcs(self):
        """Spawn NPC vehicles and pedestrians near the training vehicle"""
        try:
            number_of_vehicles = 15
            spawn_radius = 40.0

            if self.vehicle is None:
                print("Training vehicle not found! Cannot spawn NPCs.")
                return

            # Get training vehicle's location
            vehicle_location = self.vehicle.get_location()

            # Configure traffic manager
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(1.5)
            traffic_manager.global_percentage_speed_difference(-50.0)

            # Spawn strategic vehicles
            self._spawn_strategic_npcs(close_npcs=3, far_npcs=number_of_vehicles)

            # Spawn strategic pedestrians
            self._spawn_strategic_pedestrians(close_peds=3, far_peds=5)

            # Visualize spawn radius using points and lines
            try:
                debug = self.world.debug
                if debug:
                    # Draw points around the radius (approximating a circle)
                    num_points = 36  # Number of points to approximate circle
                    for i in range(num_points):
                        angle = (2 * math.pi * i) / num_points
                        point = carla.Location(
                            x=vehicle_location.x + spawn_radius * math.cos(angle),
                            y=vehicle_location.y + spawn_radius * math.sin(angle),
                            z=vehicle_location.z
                        )
                        debug.draw_point(
                            point,
                            size=0.1,
                            color=carla.Color(r=0, g=255, b=0),
                            life_time=1.0
                        )
                        # Draw lines between points
                        if i > 0:
                            prev_point = carla.Location(
                                x=vehicle_location.x + spawn_radius * math.cos((2 * math.pi * (i-1)) / num_points),
                                y=vehicle_location.y + spawn_radius * math.sin((2 * math.pi * (i-1)) / num_points),
                                z=vehicle_location.z
                            )
                            debug.draw_line(
                                prev_point,
                                point,
                                thickness=0.1,
                                color=carla.Color(r=0, g=255, b=0),
                                life_time=1.0
                            )
            except Exception as debug_error:
                print(f"Warning: Could not draw debug visualization: {debug_error}")

        except Exception as e:
            print(f"Error spawning NPCs: {e}")
            traceback.print_exc()
            self.cleanup_npcs()

    def _spawn_strategic_npcs(self, close_npcs=5, far_npcs=15):
        """
        Spawn NPC vehicles with some specifically placed in front of the player vehicle
        """
        try:
            blueprints = self.world.get_blueprint_library().filter('vehicle.*')

            # Filter for cars (no bikes/motorcycles)
            car_blueprints = [
                bp for bp in blueprints 
                if any(car_type in bp.id.lower() 
                    for car_type in ['car', 'tesla', 'audi', 'bmw', 'mercedes', 'toyota', 'ford'])
            ]

            if not car_blueprints:
                print("Warning: Using all vehicle blueprints as fallback")
                car_blueprints = blueprints

            spawn_points = self.world.get_map().get_spawn_points()

            # Get player's transform
            player_transform = self.vehicle.get_transform()
            player_location = player_transform.location
            player_forward = player_transform.get_forward_vector()

            # Setup traffic manager
            traffic_manager = self.client.get_trafficmanager()

            # First, spawn NPCs close to the player
            close_spawn_points = []
            for spawn_point in spawn_points:
                # Calculate vector from player to spawn point
                to_spawn_x = spawn_point.location.x - player_location.x
                to_spawn_y = spawn_point.location.y - player_location.y
                to_spawn_z = spawn_point.location.z - player_location.z

                # Calculate distance using pythagorean theorem
                distance = math.sqrt(to_spawn_x**2 + to_spawn_y**2 + to_spawn_z**2)

                # Calculate dot product manually
                dot_product = (to_spawn_x * player_forward.x + 
                            to_spawn_y * player_forward.y + 
                            to_spawn_z * player_forward.z)

                # Check if point is in front of player and within distance range
                if (distance < 50.0 and distance > 30.0 and  # Between 30m and 50m
                    dot_product > 0):                        # In front of player
                    close_spawn_points.append(spawn_point)

            # Spawn close NPCs
            print(f"\nSpawning {close_npcs} NPCs near player...")
            random.shuffle(close_spawn_points)
            for i in range(min(close_npcs, len(close_spawn_points))):
                try:
                    blueprint = random.choice(car_blueprints)
                    if blueprint.has_attribute('color'):
                        blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').    recommended_values))

                    vehicle = self.world.spawn_actor(blueprint, close_spawn_points[i])
                    vehicle.set_autopilot(True)
                    self.npc_vehicles.append(vehicle)

                    # Set traffic manager parameters
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(0, 10))
                    traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 2.0))

                    print(f"Spawned close NPC {i+1}/{close_npcs}")

                    # Give time for physics to settle
                    self.world.tick()

                    # Draw debug line to show spawn location
                    debug = self.world.debug
                    if debug:
                        # Draw a line from ego vehicle to spawned vehicle
                        debug.draw_line(
                            player_location,
                            vehicle.get_location(),
                            thickness=0.1,
                            color=carla.Color(r=0, g=255, b=0),
                            life_time=5.0
                        )

                except Exception as e:
                    print(f"Failed to spawn close NPC: {e}")
                    continue
                
            # Then spawn far NPCs randomly in the map
            print(f"\nSpawning {far_npcs} NPCs around the map...")
            random.shuffle(spawn_points)
            for i in range(far_npcs):
                try:
                    blueprint = random.choice(car_blueprints)
                    if blueprint.has_attribute('color'):
                        blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').    recommended_values))

                    # Try to find a spawn point not too close to player
                    spawn_point = None
                    for point in spawn_points:
                        # Calculate distance manually
                        dx = point.location.x - player_location.x
                        dy = point.location.y - player_location.y
                        dz = point.location.z - player_location.z
                        distance = math.sqrt(dx**2 + dy**2 + dz**2)

                        if distance > 50.0:  # More than 50m away
                            spawn_point = point
                            spawn_points.remove(point)
                            break
                        
                    if spawn_point is None:
                        print("Couldn't find suitable spawn point for far NPC")
                        continue

                    vehicle = self.world.spawn_actor(blueprint, spawn_point)
                    vehicle.set_autopilot(True)
                    self.npc_vehicles.append(vehicle)

                    # Set traffic manager parameters
                    traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(0, 10))
                    traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 2.0))

                    print(f"Spawned far NPC {i+1}/{far_npcs}")

                    # Give time for physics to settle
                    self.world.tick()

                except Exception as e:
                    print(f"Failed to spawn far NPC: {e}")
                    continue
                
            print(f"\nSuccessfully spawned {len(self.npc_vehicles)} NPC vehicles total")

        except Exception as e:
            print(f"Error in spawn_strategic_npcs: {e}")

    def _spawn_strategic_pedestrians(self, close_peds=3, far_peds=5):
        """
        Spawn pedestrians with robust error handling and careful placement
        to avoid simulation crashes
        """
        try:
            # Filter walker blueprints
            walker_bps = self.world.get_blueprint_library().filter('walker.pedestrian.*')

            if len(walker_bps) == 0:
                print("Warning: No pedestrian blueprints found!")
                return

            # Get player's transform
            player_transform = self.vehicle.get_transform()
            player_location = player_transform.location

            # Create spawn points manually instead of using road spawn points
            print(f"\nSpawning {close_peds} pedestrians near player...")

            # For close pedestrians, use sidewalks near player
            spawn_attempts = 0
            close_peds_spawned = 0

            while close_peds_spawned < close_peds and spawn_attempts < 20:
                spawn_attempts += 1

                try:
                    # Get a waypoint near the player
                    player_waypoint = self.world.get_map().get_waypoint(player_location)

                    # Find nearby random location within 30-60m, biased to sidewalks
                    distance = random.uniform(20.0, 80.0)
                    angle = random.uniform(-45, 45)  # Degrees, roughly in front of player
                    angle_rad = math.radians(angle)

                    # Calculate offset position based on player forward direction
                    # Create right vector manually from forward vector
                    forward = player_transform.get_forward_vector()

                    # Calculate right vector using cross product with up vector (0,0,1)
                    right_x = -forward.y  # Cross product with up vector
                    right_y = forward.x

                    # Calculate the target position
                    target_x = player_location.x + forward.x * distance * math.cos(angle_rad) + right_x * distance *    math.sin(angle_rad)
                    target_y = player_location.y + forward.y * distance * math.cos(angle_rad) + right_y * distance *    math.sin(angle_rad)
                    target_location = carla.Location(x=target_x, y=target_y, z=player_location.z)

                    # Get waypoint near this location
                    waypoint = self.world.get_map().get_waypoint(target_location)
                    if not waypoint:
                        continue
                    
                    # Try to find nearby sidewalk
                    sidewalk_wp = None
                    for _ in range(5):  # Try both sides and offsets
                        try:
                            # Try right side first (usually where sidewalks are)
                            temp_wp = waypoint.get_right_lane()
                            if temp_wp and temp_wp.lane_type == carla.LaneType.Sidewalk:
                                sidewalk_wp = temp_wp
                                break
                            
                            # Try left side
                            temp_wp = waypoint.get_left_lane()
                            if temp_wp and temp_wp.lane_type == carla.LaneType.Sidewalk:
                                sidewalk_wp = temp_wp
                                break
                            
                            # Move the waypoint forward and try again
                            next_wps = waypoint.next(5.0)
                            if next_wps:
                                waypoint = next_wps[0]
                        except:
                            pass
                        
                    # If no sidewalk found, use original waypoint but offset from road
                    if sidewalk_wp:
                        spawn_transform = sidewalk_wp.transform
                        # Raise slightly to avoid ground collision
                        spawn_transform.location.z += 0.5
                    else:
                        # Offset from road - Use manual right vector calculation
                        spawn_transform = waypoint.transform
                        # Get right vector from waypoint transform
                        wp_forward = waypoint.transform.get_forward_vector()
                        # Calculate right vector from forward vector
                        wp_right_x = -wp_forward.y
                        wp_right_y = wp_forward.x

                        # Apply offset
                        spawn_transform.location.x += wp_right_x * 3.0
                        spawn_transform.location.y += wp_right_y * 3.0
                        spawn_transform.location.z += 0.5

                    # Choose random walker blueprint
                    walker_bp = random.choice(walker_bps)

                    # Make sure pedestrian is not invincible
                    if walker_bp.has_attribute('is_invincible'):
                        walker_bp.set_attribute('is_invincible', 'false')

                    # Simplified attribute handling
                    try:
                        # Just try common attributes without checking if they exist
                        for attr_name in ['color', 'texture']:
                            attr = walker_bp.get_attribute(attr_name)
                            if attr and attr.recommended_values:
                                walker_bp.set_attribute(attr_name, random.choice(attr.recommended_values))
                    except:
                        # If this fails, just continue with default appearance
                        pass
                    
                    # Spawn the walker directly instead of using batch command
                    try:
                        walker = self.world.spawn_actor(walker_bp, spawn_transform)
                        if not walker:
                            print("Failed to spawn walker actor")
                            continue
                        
                        # Add to the class's walker list
                        if not hasattr(self, 'pedestrians'):
                            self.pedestrians = []
                        self.pedestrians.append(walker)
                    except Exception as e:
                        print(f"Failed to spawn pedestrian: {e}")
                        continue
                    
                    # Wait for physics to settle
                    for _ in range(5):
                        self.world.tick()

                    # Create walker controller directly instead of using batch
                    controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

                    try:
                        controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                        if not controller:
                            print("Failed to spawn controller")
                            walker.destroy()
                            self.pedestrians.pop()
                            continue
                        
                        # Add to the class's controller list
                        self.pedestrian_controllers.append(controller)
                    except Exception as e:
                        print(f"Failed to spawn controller: {e}")
                        walker.destroy()
                        self.pedestrians.pop()
                        continue
                    
                    # Give time for actors to initialize properly
                    self.world.tick()

                    # Initialize controller to make pedestrian walk randomly
                    try:
                        # Start walking to random destination
                        controller.start()
                        controller.go_to_location(self.world.get_random_location_from_navigation())

                        # Set walking speed (lower speed for safety)
                        controller.set_max_speed(1.2)

                        close_peds_spawned += 1
                        print(f"Spawned close pedestrian {close_peds_spawned}/{close_peds}")
                    except Exception as e:
                        print(f"Error initializing pedestrian controller: {e}")
                        controller.destroy()
                        walker.destroy()
                        self.pedestrians.pop()
                        self.pedestrian_controllers.pop()
                        continue
                    
                    # Additional delay to ensure stability
                    for _ in range(3):
                        self.world.tick()

                except Exception as e:
                    print(f"Error during close pedestrian spawn attempt {spawn_attempts}: {e}")
                    continue
                
            # For far pedestrians, use navigation system
            print(f"\nSpawning {far_peds} pedestrians around the map...")
            spawn_attempts = 0
            far_peds_spawned = 0

            while far_peds_spawned < far_peds and spawn_attempts < 20:
                spawn_attempts += 1

                try:
                    # Get random location from navigation
                    nav_location = None
                    for _ in range(10):
                        try:
                            test_location = self.world.get_random_location_from_navigation()
                            # Check distance from player
                            dx = test_location.x - player_location.x
                            dy = test_location.y - player_location.y
                            dz = test_location.z - player_location.z
                            distance = math.sqrt(dx**2 + dy**2 + dz**2)

                            if distance > 70.0:  # Farther away for safety
                                nav_location = test_location
                                break
                        except:
                            continue
                        
                    if nav_location is None:
                        continue
                    
                    # Create spawn transform
                    spawn_transform = carla.Transform(nav_location)

                    # Choose random walker blueprint
                    walker_bp = random.choice(walker_bps)

                    # Configure blueprint
                    if walker_bp.has_attribute('is_invincible'):
                        walker_bp.set_attribute('is_invincible', 'false')

                    # Simplified attribute handling
                    try:
                        # Just try common attributes without checking if they exist
                        for attr_name in ['color', 'texture']:
                            attr = walker_bp.get_attribute(attr_name)
                            if attr and attr.recommended_values:
                                walker_bp.set_attribute(attr_name, random.choice(attr.recommended_values))
                    except:
                        # If this fails, just continue with default appearance
                        pass
                    
                    # Spawn walker directly
                    try:
                        walker = self.world.spawn_actor(walker_bp, spawn_transform)
                        if not walker:
                            print("Failed to spawn far walker actor")
                            continue
                        
                        # Add to the class's walker list
                        if not hasattr(self, 'pedestrians'):
                            self.pedestrians = []
                        self.pedestrians.append(walker)
                    except Exception as e:
                        print(f"Failed to spawn far pedestrian: {e}")
                        continue
                    
                    # Wait for physics to settle
                    for _ in range(5):
                        self.world.tick()

                    # Create walker controller directly
                    controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')

                    try:
                        controller = self.world.spawn_actor(controller_bp, carla.Transform(), walker)
                        if not controller:
                            print("Failed to retrieve far controller actor")
                            walker.destroy()
                            self.pedestrians.pop()
                            continue
                        
                        self.pedestrian_controllers.append(controller)
                    except Exception as e:
                        print(f"Failed to spawn far controller: {e}")
                        walker.destroy()
                        self.pedestrians.pop()
                        continue
                    
                    # Give time for actors to initialize properly
                    self.world.tick()

                    # Initialize controller
                    try:
                        controller.start()
                        controller.go_to_location(self.world.get_random_location_from_navigation())
                        controller.set_max_speed(1.2)

                        far_peds_spawned += 1
                        print(f"Spawned far pedestrian {far_peds_spawned}/{far_peds}")
                    except Exception as e:
                        print(f"Error initializing far pedestrian controller: {e}")
                        controller.destroy()
                        walker.destroy()
                        self.pedestrians.pop()
                        self.pedestrian_controllers.pop()
                        continue
                    
                    # Additional delay for stability
                    for _ in range(3):
                        self.world.tick()

                except Exception as e:
                    print(f"Error during far pedestrian spawn attempt {spawn_attempts}: {e}")
                    continue
                
            total_peds = len(self.pedestrians) if hasattr(self, 'pedestrians') else 0
            print(f"\nSuccessfully spawned {total_peds} pedestrians total")

        except Exception as e:
            print(f"Error in spawn_strategic_pedestrians: {e}")

    def cleanup_npcs(self):
        """Clean up all spawned NPCs, pedestrians, and controllers"""
        try:
            # Stop pedestrian controllers
            for controller in self.pedestrian_controllers:
                if controller and controller.is_alive:
                    controller.stop()
                    controller.destroy()
            self.pedestrian_controllers.clear()

            # Destroy pedestrians
            if hasattr(self, 'pedestrians'):
                for pedestrian in self.pedestrians:
                    if pedestrian and pedestrian.is_alive:
                        pedestrian.destroy()
                self.pedestrians.clear()

            # Destroy vehicles
            for vehicle in self.npc_vehicles:
                if vehicle and vehicle.is_alive:
                    vehicle.destroy()
            self.npc_vehicles.clear()

            print("Successfully cleaned up all NPCs")

        except Exception as e:
            print(f"Error cleaning up NPCs: {e}")
            traceback.print_exc()

    def close(self):
        """Close environment and cleanup"""
        try:
            self.cleanup_actors()
            self.cleanup_npcs()

            if pygame.get_init():
                pygame.quit()

            if self.world is not None:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                self.world.apply_settings(settings)

            print("Environment closed successfully")

        except Exception as e:
            print(f"Error closing environment: {e}")
            traceback.print_exc()
    





















#main.py

import os
import sys
import time
import numpy as np
from tqdm import tqdm
import pygame
import carla
import random
import math
from sklearn.cluster import DBSCAN
from safety_controller import SafetyController
from vehicle_detector import VehicleDetector 


# Pygame window settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 30

class CameraManager:
    def __init__(self, parent_actor, world):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self._world = world
        
        # Set up camera blueprint
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        blueprint.set_attribute('fov', '110')
        
        # Find camera spawn point (behind and above the vehicle)
        spawn_point = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        
        # Spawn camera
        self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=self._parent)
        
        # Setup callback for camera data
        self.sensor.listen(self._parse_image)
    
    def _parse_image(self, image):
        """Convert CARLA raw image to Pygame surface"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def render(self, display):
        """Render camera image to Pygame display"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    
    def destroy(self):
        """Clean up camera sensor"""
        if self.sensor is not None:
            self.sensor.destroy()


def connect_to_carla(retries=10, timeout=5.0):
    """Attempt to connect to CARLA with retries"""
    for attempt in range(retries):
        try:
            print(f"Attempting to connect to CARLA (Attempt {attempt + 1}/{retries})")
            client = carla.Client('localhost', 2000)
            client.set_timeout(timeout)
            world = client.get_world()
            print("Successfully connected to CARLA")
            return client, world
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
    raise ConnectionError("Failed to connect to CARLA after multiple attempts")

def find_spawn_points(world, min_distance=30.0):
    """Find suitable spawn points with minimum distance between them"""
    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) < 2:
        raise ValueError("Not enough spawn points found!")
    
    for _ in range(50):
        start_point = random.choice(spawn_points)
        end_point = random.choice(spawn_points)
        distance = start_point.location.distance(end_point.location)
        if distance >= min_distance:
            return start_point, end_point
    return spawn_points[0], spawn_points[1]

def spawn_strategic_npcs(world, player_vehicle, close_npcs=5, far_npcs=15):
    """
    Spawn NPCs with some specifically placed in front of the player vehicle
    """
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    
    # Filter for cars (no bikes/motorcycles)
    car_blueprints = [
        bp for bp in blueprints 
        if any(car_type in bp.id.lower() 
               for car_type in ['car', 'tesla', 'audi', 'bmw', 'mercedes', 'toyota', 'ford'])
    ]
    
    if not car_blueprints:
        print("Warning: Using all vehicle blueprints as fallback")
        car_blueprints = blueprints
    
    spawn_points = world.get_map().get_spawn_points()
    
    # Get player's transform
    player_transform = player_vehicle.get_transform()
    player_location = player_transform.location
    player_forward = player_transform.get_forward_vector()
    
    vehicles = []
    
    # First, spawn NPCs close to the player
    close_spawn_points = []
    for spawn_point in spawn_points:
        # Calculate vector from player to spawn point
        to_spawn_x = spawn_point.location.x - player_location.x
        to_spawn_y = spawn_point.location.y - player_location.y
        to_spawn_z = spawn_point.location.z - player_location.z
        
        # Calculate distance using pythagorean theorem
        distance = math.sqrt(to_spawn_x**2 + to_spawn_y**2 + to_spawn_z**2)
        
        # Calculate dot product manually
        dot_product = (to_spawn_x * player_forward.x + 
                      to_spawn_y * player_forward.y + 
                      to_spawn_z * player_forward.z)
        
        # Check if point is in front of player and within distance range
        if (distance < 50.0 and distance > 30.0 and  # Between 30m and 50m
            dot_product > 0):                        # In front of player
            close_spawn_points.append(spawn_point)
    
    # Spawn close NPCs
    print(f"\nSpawning {close_npcs} NPCs near player...")
    random.shuffle(close_spawn_points)
    for i in range(min(close_npcs, len(close_spawn_points))):
        try:
            blueprint = random.choice(car_blueprints)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
            
            vehicle = world.spawn_actor(blueprint, close_spawn_points[i])
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            print(f"Spawned close NPC {i+1}/{close_npcs}")
            
            # Give time for physics to settle
            world.tick(2.0)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed to spawn close NPC: {e}")
            continue
    
    # Then spawn far NPCs randomly in the map
    print(f"\nSpawning {far_npcs} NPCs around the map...")
    random.shuffle(spawn_points)
    for i in range(far_npcs):
        try:
            blueprint = random.choice(car_blueprints)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
            
            # Try to find a spawn point not too close to player
            spawn_point = None
            for point in spawn_points:
                # Calculate distance manually
                dx = point.location.x - player_location.x
                dy = point.location.y - player_location.y
                dz = point.location.z - player_location.z
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if distance > 50.0:  # More than 50m away
                    spawn_point = point
                    spawn_points.remove(point)
                    break
            
            if spawn_point is None:
                print("Couldn't find suitable spawn point for far NPC")
                continue
                
            vehicle = world.spawn_actor(blueprint, spawn_point)
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            print(f"Spawned far NPC {i+1}/{far_npcs}")
            
            # Give time for physics to settle
            world.tick(2.0)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed to spawn far NPC: {e}")
            continue
    
    print(f"\nSuccessfully spawned {len(vehicles)} NPCs total")
    return vehicles

def spawn_strategic_pedestrians(world, player_vehicle, close_peds=5, far_peds=2):
    """
    Spawn pedestrians with robust error handling and careful placement
    to avoid simulation crashes - Fixed version for older CARLA API
    """
    import random
    import math
    import time
    import carla
    
    # Lists to store spawned walkers and controllers
    walkers = []
    walker_controllers = []
    
    try:
        # Filter walker blueprints
        walker_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
        
        if len(walker_bps) == 0:
            print("Warning: No pedestrian blueprints found!")
            return [], []
        
        # Get player's transform
        player_transform = player_vehicle.get_transform()
        player_location = player_transform.location
        
        # Create spawn points manually instead of using road spawn points
        print(f"\nSpawning {close_peds} pedestrians near player...")
        
        # For close pedestrians, use sidewalks near player
        spawn_attempts = 0
        close_peds_spawned = 0
        
        while close_peds_spawned < close_peds and spawn_attempts < 20:
            spawn_attempts += 1
            
            try:
                # Get a waypoint near the player
                player_waypoint = world.get_map().get_waypoint(player_location)
                
                # Find nearby random location within 30-60m, biased to sidewalks
                distance = random.uniform(20.0, 80.0)
                angle = random.uniform(-45, 45)  # Degrees, roughly in front of player
                angle_rad = math.radians(angle)
                
                # Calculate offset position based on player forward direction
                # Create right vector manually from forward vector
                forward = player_transform.get_forward_vector()
                
                # Calculate right vector using cross product with up vector (0,0,1)
                right_x = -forward.y  # Cross product with up vector
                right_y = forward.x
                
                # Calculate the target position
                target_x = player_location.x + forward.x * distance * math.cos(angle_rad) + right_x * distance * math.sin(angle_rad)
                target_y = player_location.y + forward.y * distance * math.cos(angle_rad) + right_y * distance * math.sin(angle_rad)
                target_location = carla.Location(x=target_x, y=target_y, z=player_location.z)
                
                # Get waypoint near this location
                waypoint = world.get_map().get_waypoint(target_location)
                if not waypoint:
                    continue
                
                # Try to find nearby sidewalk
                sidewalk_wp = None
                for _ in range(5):  # Try both sides and offsets
                    try:
                        # Try right side first (usually where sidewalks are)
                        temp_wp = waypoint.get_right_lane()
                        if temp_wp and temp_wp.lane_type == carla.LaneType.Sidewalk:
                            sidewalk_wp = temp_wp
                            break
                        
                        # Try left side
                        temp_wp = waypoint.get_left_lane()
                        if temp_wp and temp_wp.lane_type == carla.LaneType.Sidewalk:
                            sidewalk_wp = temp_wp
                            break
                        
                        # Move the waypoint forward and try again
                        next_wps = waypoint.next(5.0)
                        if next_wps:
                            waypoint = next_wps[0]
                    except:
                        pass
                
                # If no sidewalk found, use original waypoint but offset from road
                if sidewalk_wp:
                    spawn_transform = sidewalk_wp.transform
                    # Raise slightly to avoid ground collision
                    spawn_transform.location.z += 0.5
                else:
                    # Offset from road - Use manual right vector calculation
                    spawn_transform = waypoint.transform
                    # Get right vector from waypoint transform
                    wp_forward = waypoint.transform.get_forward_vector()
                    # Calculate right vector from forward vector
                    wp_right_x = -wp_forward.y
                    wp_right_y = wp_forward.x
                    
                    # Apply offset
                    spawn_transform.location.x += wp_right_x * 3.0
                    spawn_transform.location.y += wp_right_y * 3.0
                    spawn_transform.location.z += 0.5
                
                # Choose random walker blueprint
                walker_bp = random.choice(walker_bps)
                
                # Make sure pedestrian is not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                
                # Simplified attribute handling
                try:
                    # Just try common attributes without checking if they exist
                    for attr_name in ['color', 'texture']:
                        attr = walker_bp.get_attribute(attr_name)
                        if attr and attr.recommended_values:
                            walker_bp.set_attribute(attr_name, random.choice(attr.recommended_values))
                except:
                    # If this fails, just continue with default appearance
                    pass
                
                # FIX: Replace batch spawning with direct spawn_actor
                # Spawn the walker directly instead of using batch command
                try:
                    walker = world.spawn_actor(walker_bp, spawn_transform)
                    if not walker:
                        print("Failed to spawn walker actor")
                        continue
                    
                    walkers.append(walker)
                except Exception as e:
                    print(f"Failed to spawn pedestrian: {e}")
                    continue
                
                # Wait for physics to settle
                for _ in range(5):
                    world.tick()
                
                # Create walker controller directly instead of using batch
                controller_bp = world.get_blueprint_library().find('controller.ai.walker')
                
                try:
                    controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
                    if not controller:
                        print("Failed to spawn controller")
                        walker.destroy()
                        walkers.pop()
                        continue
                        
                    walker_controllers.append(controller)
                except Exception as e:
                    print(f"Failed to spawn controller: {e}")
                    walker.destroy()
                    walkers.pop()
                    continue
                
                # Give time for actors to initialize properly
                world.tick()
                
                # Initialize controller to make pedestrian walk randomly
                try:
                    # Start walking to random destination
                    controller.start()
                    controller.go_to_location(world.get_random_location_from_navigation())
                    
                    # Set walking speed (lower speed for safety)
                    controller.set_max_speed(1.2)
                    
                    close_peds_spawned += 1
                    print(f"Spawned close pedestrian {close_peds_spawned}/{close_peds}")
                except Exception as e:
                    print(f"Error initializing pedestrian controller: {e}")
                    controller.destroy()
                    walker.destroy()
                    walkers.pop()
                    walker_controllers.pop()
                    continue
                
                # Additional delay to ensure stability
                for _ in range(3):
                    world.tick()
                
            except Exception as e:
                print(f"Error during close pedestrian spawn attempt {spawn_attempts}: {e}")
                continue
        
        # For far pedestrians, use navigation system
        print(f"\nSpawning {far_peds} pedestrians around the map...")
        spawn_attempts = 0
        far_peds_spawned = 0
        
        while far_peds_spawned < far_peds and spawn_attempts < 20:
            spawn_attempts += 1
            
            try:
                # Get random location from navigation
                nav_location = None
                for _ in range(10):
                    try:
                        test_location = world.get_random_location_from_navigation()
                        # Check distance from player
                        dx = test_location.x - player_location.x
                        dy = test_location.y - player_location.y
                        dz = test_location.z - player_location.z
                        distance = math.sqrt(dx**2 + dy**2 + dz**2)
                        
                        if distance > 70.0:  # Farther away for safety
                            nav_location = test_location
                            break
                    except:
                        continue
                
                if nav_location is None:
                    continue
                
                # Create spawn transform
                spawn_transform = carla.Transform(nav_location)
                
                # Choose random walker blueprint
                walker_bp = random.choice(walker_bps)
                
                # Configure blueprint
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                
                # Simplified attribute handling
                try:
                    # Just try common attributes without checking if they exist
                    for attr_name in ['color', 'texture']:
                        attr = walker_bp.get_attribute(attr_name)
                        if attr and attr.recommended_values:
                            walker_bp.set_attribute(attr_name, random.choice(attr.recommended_values))
                except:
                    # If this fails, just continue with default appearance
                    pass
                
                # FIX: Replace batch spawning with direct spawn_actor 
                # Spawn walker directly
                try:
                    walker = world.spawn_actor(walker_bp, spawn_transform)
                    if not walker:
                        print("Failed to spawn far walker actor")
                        continue
                        
                    walkers.append(walker)
                except Exception as e:
                    print(f"Failed to spawn far pedestrian: {e}")
                    continue
                
                # Wait for physics to settle
                for _ in range(5):
                    world.tick()
                
                # Create walker controller directly
                controller_bp = world.get_blueprint_library().find('controller.ai.walker')
                
                try:
                    controller = world.spawn_actor(controller_bp, carla.Transform(), walker)
                    if not controller:
                        print("Failed to retrieve far controller actor")
                        walker.destroy()
                        walkers.pop()
                        continue
                        
                    walker_controllers.append(controller)
                except Exception as e:
                    print(f"Failed to spawn far controller: {e}")
                    walker.destroy()
                    walkers.pop()
                    continue
                
                # Give time for actors to initialize properly
                world.tick()
                
                # Initialize controller
                try:
                    controller.start()
                    controller.go_to_location(world.get_random_location_from_navigation())
                    controller.set_max_speed(1.2)
                    
                    far_peds_spawned += 1
                    print(f"Spawned far pedestrian {far_peds_spawned}/{far_peds}")
                except Exception as e:
                    print(f"Error initializing far pedestrian controller: {e}")
                    controller.destroy()
                    walker.destroy()
                    walkers.pop()
                    walker_controllers.pop()
                    continue
                
                # Additional delay for stability
                for _ in range(3):
                    world.tick()
                
            except Exception as e:
                print(f"Error during far pedestrian spawn attempt {spawn_attempts}: {e}")
                continue
        
        total_peds = len(walkers)
        print(f"\nSuccessfully spawned {total_peds} pedestrians total")
        
    except Exception as e:
        print(f"Error in spawn_strategic_pedestrians: {e}")
        # Clean up any partially created pedestrians
        safe_cleanup_pedestrians(world, walkers, walker_controllers)
        return [], []
    
    return walkers, walker_controllers

def safe_cleanup_pedestrians(world, walkers, walker_controllers):
    """
    Safely clean up pedestrians and their controllers
    """
    print("Performing safe pedestrian cleanup...")
    
    # First stop all controllers
    for controller in walker_controllers:
        try:
            if controller is not None and controller.is_alive:
                controller.stop()
        except:
            pass
    
    # Give time for controllers to stop
    for _ in range(5):
        try:
            world.tick()
        except:
            pass
    
    # Remove controllers
    for controller in walker_controllers:
        try:
            if controller is not None and controller.is_alive:
                controller.destroy()
        except:
            pass
    
    # Remove walkers
    for walker in walkers:
        try:
            if walker is not None and walker.is_alive:
                walker.destroy()
        except:
            pass
# Modify main() to include this function and update the cleanup logic
def main():
    try:
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("CARLA Navigation")
        display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()
        
        # Connect to CARLA
        print("Connecting to CARLA...")
        client, world = connect_to_carla()
        
        # Set synchronous mode with enhanced physics settings
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        settings.substepping = True  # Enable physics substepping
        settings.max_substep_delta_time = 0.01  # Maximum physics substep size
        settings.max_substeps = 10  # Maximum number of substeps
        world.apply_settings(settings)
        
        # Wait for the world to stabilize
        print("Waiting for world to stabilize...")
        for _ in range(20):
            world.tick(2.0)
            time.sleep(0.1)
        
        # Find suitable spawn points
        print("Finding suitable spawn points...")
        start_point, end_point = find_spawn_points(world)
        print(f"Start point: {start_point.location}")
        print(f"End point: {end_point.location}")
        
        # Spawn vehicle
                # Get vehicle blueprint with priority order (compatible with CARLA 0.10.0)
        print("Selecting vehicle blueprint...")
        blueprint_library = world.get_blueprint_library()
        
        # Try to get vehicle blueprints in order of priority
        vehicle_names = ['vehicle.dodge.charger_2020',
            'vehicle.lincoln.mkz_2017',
            'vehicle.nissan.patrol']
        vehicle_bp = None
        
        for vehicle_name in vehicle_names:
            vehicle_filters = blueprint_library.filter(vehicle_name)
            if vehicle_filters:
                vehicle_bp = vehicle_filters[0]
                print(f"Selected vehicle: {vehicle_name}")
                break
                
        if vehicle_bp is None:
            # Fallback to any available vehicle if none of the preferred ones exist
            available_vehicles = blueprint_library.filter('vehicle.*')
            if available_vehicles:
                vehicle_bp = random.choice(available_vehicles)
                print(f"Fallback to available vehicle: {vehicle_bp.id}")
            else:
                raise Exception("No vehicle blueprints found")
        vehicle = None
        camera = None
        detector = None
        safety_controller = None
        
        try:
            vehicle = world.spawn_actor(blueprint, start_point)
            print("Vehicle spawned successfully")
            
            # Set up camera
            camera = CameraManager(vehicle, world)
            
            # Set up Vehicle Detector and Controllers
            from navigation_controller import NavigationController
            controller = NavigationController()
            detector = VehicleDetector(vehicle, world, controller)
            safety_controller = SafetyController(vehicle, world, controller)
            
            # Allow everything to settle
            world.tick()
            time.sleep(0.5)
            
            # Plan path
            print("Planning path...")
            success = controller.set_path(world, start_point.location, end_point.location)
            
            if not success:
                print("Failed to plan path!")
                return
            
            print(f"Path planned with {len(controller.waypoints)} waypoints")
            
            # Spawn NPC vehicles
            print("Spawning NPC vehicles...")
            npcs = spawn_strategic_npcs(world, vehicle, close_npcs=5, far_npcs=15)
            print(f"Spawned {len(npcs)} total NPCs")
            
            # Wait for physics to settle after vehicle spawning
            for _ in range(10):
                world.tick()
                time.sleep(0.05)
            
            # Spawn pedestrians
            print("Spawning pedestrians...")
            walkers, walker_controllers = spawn_strategic_pedestrians(world, vehicle, close_peds=5, far_peds=2)
            
            # Main simulation loop
            with tqdm(total=5000, desc="Navigation") as pbar:
                # Inside the main simulation loop
                try:
                    while True:
                        try:
                            # Tick the world with correct parameter
                            start_time = time.time()
                            while True:
                                try:
                                    world.tick(2.0)
                                    break
                                except RuntimeError as e:
                                    if time.time() - start_time > 10.0:  # Overall timeout
                                        raise
                                    time.sleep(0.1)
                                    continue
                                
                            # Update Pygame display
                            display.fill((0, 0, 0))
                            camera.render(display)
                            if detector is not None:
                                detector.detect_vehicles()
                                detector.render(display)
                            pygame.display.flip()

                            # Safety check before applying control
                            safety_controller.check_safety()

                            # Apply the updated safety control logic
                            if safety_controller.last_detected:
                                # Apply emergency brake
                                vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0, steer=0))
                                print("Safety stop active")
                            else:
                                # Resume normal path following
                                control = controller.get_control(vehicle, world)
                                vehicle.apply_control(control)
                                # Debug output for normal operation
                                print(f"Control commands - Throttle: {control.throttle:.2f}, "
                                      f"Brake: {control.brake:.2f}, "
                                      f"Steer: {control.steer:.2f}")

                            # Debug vehicle state
                            velocity = vehicle.get_velocity()
                            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
                            print(f"Vehicle speed: {speed:.2f} km/h")

                            # Update progress
                            if controller.waypoints:
                                progress = (len(controller.visited_waypoints) / 
                                          len(controller.waypoints)) * 100
                            else:
                                progress = 0

                            pbar.update(1)
                            pbar.set_postfix({
                                'speed': f"{speed:.1f}km/h",
                                'progress': f"{progress:.1f}%",
                                'safety': 'ACTIVE' if safety_controller.last_detected else 'OK'
                            })

                            # Check if destination reached
                            if (controller.current_waypoint_index >= 
                                len(controller.waypoints) - 1):
                                print("\nDestination reached!")
                                time.sleep(2)
                                break
                            
                            # Handle pygame events
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    return
                                elif event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_ESCAPE:
                                        return

                            clock.tick(FPS)

                        except Exception as e:
                            print(f"\nError during simulation step: {str(e)}")
                            if "time-out" in str(e).lower():
                                print("Attempting to recover from timeout...")
                                time.sleep(1.0)  # Give the simulator time to recover
                                continue
                            else:
                                raise
                
                except KeyboardInterrupt:
                    print("\nNavigation interrupted by user")
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                        
        finally:
            print("Cleaning up...")
            try:
                # First safely clean up pedestrians
                if 'walkers' in locals() and 'walker_controllers' in locals():
                    safe_cleanup_pedestrians(world, walkers, walker_controllers)
                
                # Then clean up the rest
                if detector is not None:
                    detector.destroy()
                if camera is not None:
                    camera.destroy()
                if safety_controller is not None:
                    safety_controller.destroy()
                if vehicle is not None:
                    vehicle.destroy()
                
                # Destroy NPC vehicles
                if 'npcs' in locals() and npcs:
                    for npc in npcs:
                        if npc is not None and npc.is_alive:
                            try:
                                npc.set_autopilot(False)  # Disable autopilot before destroying
                                npc.destroy()
                            except:
                                pass
                
                # Restore original settings
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.substepping = False  # Disable physics substepping
                world.apply_settings(settings)
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
                
            print("Cleanup complete")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        pygame.quit()
        print("Pygame quit")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user.')
    except Exception as e:
        print(f'Error occurred: {e}')
        import traceback
        traceback.print_exc()




































#vehicle_detector.py
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import pygame
import carla
import random
import math
from sklearn.cluster import DBSCAN


class VehicleDetector:
    def __init__(self, parent_actor, world, controller, detection_distance=40.0, detection_angle=160.0):
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.debug = True
        self.detection_distance = detection_distance 
        self.detection_angle = detection_angle  
        self.min_distance = float('inf')

        # Multiple sensor configuration
        self.sensors = []
        self._setup_sensors()

        # Visualization parameters
        self.width = 375
        self.height = 375
        self.scale = (self.height * 0.7) / self.detection_distance
        self.surface = None

        # Detection history for stability
        self.detection_history = []
        self.history_length = 5  # Keep track of last 5 frames

        # Add pedestrian tracking
        self.pedestrian_history = []
        self.pedestrian_history_length = 5

        print(f"Improved Vehicle & Pedestrian Detector initialized with {detection_distance}m range and {detection_angle}° angle")
    
    def _setup_sensors(self):
        """Setup multiple sensors for redundant detection"""
        try:
            # Main forward-facing radar
            radar_bp = self._world.get_blueprint_library().find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov', str(self.detection_angle))
            radar_bp.set_attribute('vertical_fov', '20')
            radar_bp.set_attribute('range', str(self.detection_distance))
            radar_location = carla.Transform(carla.Location(x=2.0, z=1.0))
            radar = self._world.spawn_actor(radar_bp, radar_location, attach_to=self._parent)
            radar.listen(lambda data: self._radar_callback(data))
            self.sensors.append(radar)
            
            # Semantic LIDAR for additional detection
            # lidar_bp = self._world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp = self._world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', str(self.detection_distance))
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('channels', '32')
            lidar_bp.set_attribute('points_per_second', '90000')
            lidar_location = carla.Transform(carla.Location(x=0.0, z=2.0))
            lidar = self._world.spawn_actor(lidar_bp, lidar_location, attach_to=self._parent)
            lidar.listen(lambda data: self._lidar_callback(data))
            self.sensors.append(lidar)
            
            print("Successfully set up multiple detection sensors")
            
        except Exception as e:
            print(f"Error setting up sensors: {e}")
    
    def _radar_callback(self, radar_data):
        """Process radar data for vehicle detection"""
        try:
            points = []
            for detection in radar_data:
                # Get the detection coordinates in world space
                distance = detection.depth
                if distance < self.detection_distance:
                    azimuth = math.degrees(detection.azimuth)
                    if abs(azimuth) < self.detection_angle / 2:
                        points.append({
                            'distance': distance,
                            'azimuth': azimuth,
                            'altitude': math.degrees(detection.altitude),
                            'velocity': detection.velocity
                        })
            
            # Store radar detections
            self.radar_points = points
            
        except Exception as e:
            print(f"Error in radar callback: {e}")
    
    def _lidar_callback(self, lidar_data):
        """Process semantic LIDAR data for vehicle and pedestrian detection"""
        try:
            # Filter points that belong to vehicles (semantic tag 10 in CARLA)
            vehicle_points = [point for point in lidar_data 
                            if point.object_tag == 10 
                            and point.distance < self.detection_distance]

            # Filter points that belong to pedestrians (semantic tag 4 in CARLA)
            pedestrian_points = [point for point in lidar_data 
                               if point.object_tag == 4 
                               and point.distance < self.detection_distance]

            # Store LIDAR detections
            self.lidar_points = vehicle_points
            self.lidar_pedestrian_points = pedestrian_points

        except Exception as e:
            print(f"Error in LIDAR callback: {e}")
    
    def detect_pedestrians(self):
        """Detect pedestrians in the scene"""
        try:
            detected_pedestrians = set()

            # 1. Direct object detection from world
            all_pedestrians = self._world.get_actors().filter('walker.pedestrian.*')
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            ego_right = self._calculate_right_vector(ego_transform.rotation)

            for pedestrian in all_pedestrians:
                pedestrian_location = pedestrian.get_location()
                distance = ego_location.distance(pedestrian_location)

                if distance <= self.detection_distance:
                    # Calculate relative position and angle
                    relative_loc = pedestrian_location - ego_location
                    forward_dot = (relative_loc.x * ego_forward.x + 
                                 relative_loc.y * ego_forward.y)
                    right_dot = (relative_loc.x * ego_right.x + 
                               relative_loc.y * ego_right.y)

                    angle = math.degrees(math.atan2(right_dot, forward_dot))

                    # Check if pedestrian is within detection angle
                    if abs(angle) < self.detection_angle / 2:
                        detected_pedestrians.add((pedestrian, distance))

                        if self.debug:
                            self._draw_debug_pedestrian(pedestrian, distance)

            # 2. Sensor fusion - combine with LIDAR detection
            if hasattr(self, 'lidar_pedestrian_points'):
                for point in self.lidar_pedestrian_points:
                    self._add_potential_pedestrian_location(point, detected_pedestrians)

            # 3. Update detection history
            self.pedestrian_history.append(detected_pedestrians)
            if len(self.pedestrian_history) > self.pedestrian_history_length:
                self.pedestrian_history.pop(0)

            # 4. Get stable detections
            stable_detections = self._get_stable_pedestrian_detections()

            return stable_detections

        except Exception as e:
            print(f"Error in pedestrian detection: {e}")
            import traceback
            traceback.print_exc()
            return set()

    def _add_potential_pedestrian_location(self, point, detected_pedestrians):
        """Add potential pedestrian location from sensor data"""
        try:
            # Convert LIDAR point to world location
            x, y, z = point.point
            distance = point.distance

            location = carla.Location(x=x, y=y, z=z)

            # Check if there's already a pedestrian detected at this location
            for pedestrian, _ in detected_pedestrians:
                if pedestrian and pedestrian.get_location().distance(location) < 1.0:  # 1m threshold (smaller than for vehicles)
                    return

            # Add as potential detection
            detected_pedestrians.add((None, distance))

        except Exception as e:
            print(f"Error adding potential pedestrian: {e}")

    def _get_stable_pedestrian_detections(self):
        """Get pedestrian detections that have been stable across multiple frames"""
        if not self.pedestrian_history:
            return set()

        # Count how many times each pedestrian appears in history
        pedestrian_counts = {}
        for detections in self.pedestrian_history:
            for pedestrian, distance in detections:
                if pedestrian:  # Only count actual pedestrians, not sensor-only detections
                    pedestrian_id = pedestrian.id
                    pedestrian_counts[pedestrian_id] = pedestrian_counts.get(pedestrian_id, 0) + 1

        # Return pedestrians that appear in majority of frames
        threshold = self.pedestrian_history_length * 0.6  # 60% of frames
        stable_pedestrians = {p_id for p_id, count in pedestrian_counts.items() 
                            if count >= threshold}

        # Get the most recent detection for stable pedestrians
        latest_detections = self.pedestrian_history[-1]
        return {(p, d) for p, d in latest_detections 
                if p and p.id in stable_pedestrians}

    def _draw_debug_pedestrian(self, pedestrian, distance):
        """Draw debug visualization for detected pedestrian"""
        try:
            if self.debug:
                # Draw bounding box (slightly smaller than vehicles)
                pedestrian_bbox = pedestrian.bounding_box
                pedestrian_transform = pedestrian.get_transform()

                # Color based on distance
                if distance < 5:
                    color = carla.Color(255, 0, 255)  # Magenta for very close
                elif distance < 15:
                    color = carla.Color(255, 165, 0)  # Orange for intermediate
                else:
                    color = carla.Color(0, 255, 0)  # Green for far

                self._world.debug.draw_box(
                    box=pedestrian_bbox,
                    rotation=pedestrian_transform.rotation,
                    thickness=0.5,
                    color=color,
                    life_time=0.1
                )

                # Draw distance and velocity info
                velocity = pedestrian.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h

                info_text = f"PED {distance:.1f}m, {speed:.1f}km/h"
                self._world.debug.draw_string(
                    pedestrian.get_location() + carla.Location(z=2.0),
                    info_text,
                    color=color,
                    life_time=0.1
                )

        except Exception as e:
            print(f"Error in pedestrian debug drawing: {e}")

    def detect_vehicles(self):
        
        try:
            detected_vehicles = set()
            
            # 1. Direct object detection from world
            all_vehicles = self._world.get_actors().filter('vehicle.*')
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            ego_right = self._calculate_right_vector(ego_transform.rotation)
            
            for vehicle in all_vehicles:
                # Skip self
                if vehicle.id == self._parent.id:
                    continue
                    
                vehicle_location = vehicle.get_location()
                distance = ego_location.distance(vehicle_location)
                
                if distance <= self.detection_distance:
                    # Calculate relative position and angle
                    relative_loc = vehicle_location - ego_location
                    forward_dot = (relative_loc.x * ego_forward.x + 
                                 relative_loc.y * ego_forward.y)
                    right_dot = (relative_loc.x * ego_right.x + 
                               relative_loc.y * ego_right.y)
                    
                    angle = math.degrees(math.atan2(right_dot, forward_dot))
                    
                    # Check if vehicle is within detection angle
                    if abs(angle) < self.detection_angle / 2:
                        detected_vehicles.add((vehicle, distance))
                        
                        if self.debug:
                            self._draw_debug_vehicle(vehicle, distance)
            
            # 2. Sensor fusion - combine with radar detection
            if hasattr(self, 'radar_points'):
                for point in self.radar_points:
                    # Filter out static objects
                    if point['velocity'] > 0.5:
                        self._add_potential_vehicle_location(point, detected_vehicles)
            
            # 3. Sensor fusion - combine with lidar detection
            if hasattr(self, 'lidar_points'):
                for point in self.lidar_points:
                    self._add_potential_vehicle_location(point, detected_vehicles)
            
            # 4. Update detection history
            self.detection_history.append(detected_vehicles)
            if len(self.detection_history) > self.history_length:
                self.detection_history.pop(0)
            
            # 5. Update minimum distance tracking
            self.min_distance = (min((d for _, d in detected_vehicles))
                               if detected_vehicles else float('inf'))
            
            # 6. Get stable detections and update controller
            stable_detections = self._get_stable_detections()
            self._controller.update_obstacles(
                [v.get_location() for v, _ in stable_detections]
            )
            
            detected_pedestrians = self.detect_pedestrians()
    
            # 7. Update visualization to include pedestrians
            self._create_visualization(stable_detections, detected_pedestrians)
            
            return stable_detections
            
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            import traceback
            traceback.print_exc()
            return set()
    
    def _add_potential_vehicle_location(self, point, detected_vehicles):
        """Add potential vehicle location from sensor data"""
        try:
            # Convert sensor point to world location
            if hasattr(point, 'distance'):  # Radar point
                distance = point['distance']
                azimuth = math.radians(point['azimuth'])
                altitude = math.radians(point['altitude'])
                
                x = distance * math.cos(azimuth) * math.cos(altitude)
                y = distance * math.sin(azimuth) * math.cos(altitude)
                z = distance * math.sin(altitude)
                
            else:  # LIDAR point
                x, y, z = point.point
                distance = point.distance
            
            location = carla.Location(x=x, y=y, z=z)
            
            # Check if there's already a vehicle detected at this location
            for vehicle, _ in detected_vehicles:
                if vehicle.get_location().distance(location) < 2.0:  # 2m threshold
                    return
            
            # Add as potential detection
            detected_vehicles.add((None, distance))
            
        except Exception as e:
            print(f"Error adding potential vehicle: {e}")
    
    def _get_stable_detections(self):
        """Get detections that have been stable across multiple frames"""
        if not self.detection_history:
            return set()
        
        # Count how many times each vehicle appears in history
        vehicle_counts = {}
        for detections in self.detection_history:
            for vehicle, distance in detections:
                if vehicle:  # Only count actual vehicles, not sensor-only detections
                    vehicle_id = vehicle.id
                    vehicle_counts[vehicle_id] = vehicle_counts.get(vehicle_id, 0) + 1
        
        # Return vehicles that appear in majority of frames
        threshold = self.history_length * 0.6  # 60% of frames
        stable_vehicles = {v_id for v_id, count in vehicle_counts.items() 
                         if count >= threshold}
        
        # Get the most recent detection for stable vehicles
        latest_detections = self.detection_history[-1]
        return {(v, d) for v, d in latest_detections 
                if v and v.id in stable_vehicles}
    
    def _calculate_right_vector(self, rotation):
        """Calculate right vector from rotation"""
        yaw = math.radians(rotation.yaw)
        return carla.Vector3D(x=math.cos(yaw + math.pi/2),
                            y=math.sin(yaw + math.pi/2),
                            z=0.0)
    
    def _draw_debug_vehicle(self, vehicle, distance):
        """Draw enhanced debug visualization for detected vehicle"""
        try:
            if self.debug:
                # Draw bounding box
                vehicle_bbox = vehicle.bounding_box
                vehicle_transform = vehicle.get_transform()
                
                # Color based on distance
                if distance < 10:
                    color = carla.Color(255, 0, 0)  # Red for very close
                elif distance < 20:
                    color = carla.Color(255, 165, 0)  # Orange for intermediate
                else:
                    color = carla.Color(0, 255, 0)  # Green for far
                
                self._world.debug.draw_box(
                    box=vehicle_bbox,
                    rotation=vehicle_transform.rotation,
                    thickness=0.5,
                    color=color,
                    life_time=0.1
                )
                
                # Draw distance and velocity info
                velocity = vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
                
                info_text = f"{distance:.1f}m, {speed:.1f}km/h"
                self._world.debug.draw_string(
                    vehicle.get_location() + carla.Location(z=2.0),
                    info_text,
                    color=color,
                    life_time=0.1
                )
                
        except Exception as e:
            print(f"Error in debug drawing: {e}")
    
    def _get_relative_velocity(self, other_vehicle):
        """Calculate relative velocity between ego vehicle and other vehicle"""
        try:
            ego_vel = self._parent.get_velocity()
            other_vel = other_vehicle.get_velocity()

            # Convert to numpy arrays for easier calculation
            ego_vel_array = np.array([ego_vel.x, ego_vel.y])
            other_vel_array = np.array([other_vel.x, other_vel.y])

            # Calculate relative velocity
            rel_vel = np.linalg.norm(ego_vel_array - other_vel_array)

            # Determine if approaching
            ego_fwd = self._parent.get_transform().get_forward_vector()
            ego_fwd_array = np.array([ego_fwd.x, ego_fwd.y])

            # If dot product is positive, we're approaching
            if np.dot(ego_vel_array - other_vel_array, ego_fwd_array) > 0:
                return rel_vel
            return -rel_vel

        except Exception as e:
            print(f"Error calculating relative velocity: {e}")
            return 0.0

    def _create_visualization(self, detected_vehicles, detected_pedestrians=None):
        try:
            if self.surface is None:
                self.surface = pygame.Surface((self.width, self.height))
                self.font = pygame.font.Font(None, 24)
                self.small_font = pygame.font.Font(None, 20)
    
            # Fill background with a premium dark gradient
            gradient = pygame.Surface((self.width, self.height))
            for y in range(self.height):
                alpha = y / self.height
                color = (20, 22, 30, int(255 * (0.95 - alpha * 0.3)))
                pygame.draw.line(gradient, color, (0, y), (self.width, y))
            self.surface.blit(gradient, (0, 0))
    
            # Constants for visualization
            center_x = self.width // 2
            center_y = int(self.height * 0.8)
    
            # Draw sensor range indicators with premium styling
            ranges = [10, 20, 30, 40]
            for range_dist in ranges:
                radius = int(range_dist * self.scale)
                points = []
                for angle in range(0, 360, 2):
                    rad = math.radians(angle)
                    x = center_x + radius * math.sin(rad)
                    y = center_y - radius * math.cos(rad)
                    points.append((x, y))
    
                # Draw circles with fade effect
                alpha = 100 - (range_dist / 40) * 60
                circle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.lines(circle_surface, (0, 255, 255, int(alpha)), False, points, 1)
                self.surface.blit(circle_surface, (0, 0))
    
                # Premium distance labels
                label = self.small_font.render(f"{range_dist}m", True, (0, 255, 255, int(alpha)))
                self.surface.blit(label, (center_x + radius + 5, center_y - 10))
    
            def draw_realistic_car(surface, x, y, width, height, color, is_ego=False):
                """Draw a more realistic car shape with corrected dimensions"""
                # Scale down size for more realistic proportions
                width = int(width * 0.8)
                height = int(height * 0.8)
    
                # Create a surface for the car with alpha channel
                car_surface = pygame.Surface((width * 2, height * 2), pygame.SRCALPHA)
                center_x = width
                center_y = height
    
                # Main body shape points (more curved)
                body_points = [
                    (center_x - width//2, center_y),                    # Left bottom
                    (center_x - width//2, center_y - height*0.7),      # Left middle
                    (center_x - width//2.2, center_y - height*0.8),    # Left front curve
                    (center_x - width//3, center_y - height*0.85),     # Left hood
                    (center_x, center_y - height*0.9),                 # Front center
                    (center_x + width//3, center_y - height*0.85),     # Right hood
                    (center_x + width//2.2, center_y - height*0.8),    # Right front curve
                    (center_x + width//2, center_y - height*0.7),      # Right middle
                    (center_x + width//2, center_y),                   # Right bottom
                ]
    
                # Draw car body
                pygame.draw.polygon(car_surface, color, body_points)
    
                # Windshield (more curved)
                windshield_points = [
                    (center_x - width//3, center_y - height*0.85),     # Left bottom
                    (center_x - width//3.5, center_y - height*0.95),   # Left top
                    (center_x + width//3.5, center_y - height*0.95),   # Right top
                    (center_x + width//3, center_y - height*0.85),     # Right bottom
                ]
                pygame.draw.polygon(car_surface, (*color[:3], 150), windshield_points)
    
                # Wheels (more elliptical)
                wheel_width = width // 6
                wheel_height = height // 8
    
                # Front wheels (black ellipses)
                pygame.draw.ellipse(car_surface, (30, 30, 30),
                                  (center_x - width//2 - wheel_width//4,
                                   center_y - wheel_height//2,
                                   wheel_width, wheel_height))
                pygame.draw.ellipse(car_surface, (30, 30, 30),
                                  (center_x + width//2 - wheel_width*3//4,
                                   center_y - wheel_height//2,
                                   wheel_width, wheel_height))
    
                # Wheel well highlights (curved lines instead of arcs)
                # Left wheel well
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x - width//2 - wheel_width//4, center_y - wheel_height),
                               (center_x - width//2 + wheel_width, center_y - wheel_height), 2)
    
                # Right wheel well
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x + width//2 - wheel_width*3//4, center_y - wheel_height),
                               (center_x + width//2 + wheel_width//4, center_y - wheel_height), 2)
    
                # Add detail lines for body contour
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x - width//3, center_y - height*0.75),
                               (center_x + width//3, center_y - height*0.75), 1)
    
                # Add headlights or taillights
                light_width = width // 8
                light_height = height // 12
    
                if is_ego:
                    # Headlights (bright yellow)
                    light_color = (255, 255, 200, 200)
                else:
                    # Taillights (red)
                    light_color = (255, 50, 50, 200)
    
                # Left light
                pygame.draw.ellipse(car_surface, light_color,
                                  (center_x - width//2.5,
                                   center_y - height*0.85,
                                   light_width, light_height))
    
                # Right light
                pygame.draw.ellipse(car_surface, light_color,
                                  (center_x + width//2.5 - light_width,
                                   center_y - height*0.85,
                                   light_width, light_height))
    
                # Blit the car onto the main surface
                surface.blit(car_surface, (x - width, y - height))
    
            # Add the new function to draw realistic pedestrians
            def draw_realistic_pedestrian(surface, x, y, width, height, color):
                """Draw a more realistic pedestrian shape"""
                # Scale down size for realistic pedestrian proportions
                width = int(width * 0.5)  # Pedestrians are narrower than cars
                height = int(height * 0.6)  # And shorter
                
                # Create a surface for the pedestrian with alpha channel
                ped_surface = pygame.Surface((width * 2, height * 2), pygame.SRCALPHA)
                center_x = width
                center_y = height
                
                # Head
                head_radius = width // 2
                head_y = center_y - height * 0.75
                pygame.draw.circle(ped_surface, color, (center_x, head_y), head_radius)
                
                # Body (roughly human-shaped)
                body_points = [
                    (center_x - width//2, center_y),              # Left bottom
                    (center_x - width//3, head_y + head_radius),  # Left shoulder
                    (center_x + width//3, head_y + head_radius),  # Right shoulder
                    (center_x + width//2, center_y),              # Right bottom
                ]
                pygame.draw.polygon(ped_surface, color, body_points)
                
                # Arms (lines)
                arm_length = height * 0.3
                left_arm_x = center_x - width//3
                right_arm_x = center_x + width//3
                arm_y = head_y + head_radius + height * 0.1
                
                # Left arm
                pygame.draw.line(ped_surface, (*color[:3], 220),
                               (left_arm_x, arm_y),
                               (left_arm_x - width//3, arm_y + arm_length), 2)
                               
                # Right arm
                pygame.draw.line(ped_surface, (*color[:3], 220),
                               (right_arm_x, arm_y),
                               (right_arm_x + width//3, arm_y + arm_length), 2)
                
                # Legs (lines)
                leg_length = height * 0.35
                leg_spacing = width//3
                leg_y = center_y
                
                # Left leg
                pygame.draw.line(ped_surface, (*color[:3], 220),
                               (center_x - leg_spacing, leg_y),
                               (center_x - leg_spacing, leg_y + leg_length), 2)
                               
                # Right leg
                pygame.draw.line(ped_surface, (*color[:3], 220),
                               (center_x + leg_spacing, leg_y),
                               (center_x + leg_spacing, leg_y + leg_length), 2)
                
                # Blit the pedestrian onto the main surface
                surface.blit(ped_surface, (x - width, y - height))
    
            # Draw ego vehicle (smaller size)
            draw_realistic_car(self.surface, center_x, center_y, 24, 40, (0, 255, 255), True)
    
            # Draw detection cone with gradient
            fov_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            fov_radius = self.detection_distance * self.scale
    
            # Create detection cone
            fov_points = [(center_x, center_y)]
            for angle in range(-int(self.detection_angle/2), int(self.detection_angle/2)):
                rad = math.radians(angle)
                x = center_x + fov_radius * math.sin(rad)
                y = center_y - fov_radius * math.cos(rad)
                fov_points.append((x, y))
            fov_points.append((center_x, center_y))
    
            # Draw cone with gradient
            pygame.draw.polygon(fov_surface, (0, 255, 255, 20), fov_points)
            self.surface.blit(fov_surface, (0, 0))
    
            # Colors for detection
            vehicle_colors = {
                'alert': (255, 50, 50),
                'warning': (255, 165, 0),
                'safe': (0, 255, 255)
            }
            
            # New colors for pedestrians - slightly different hues
            pedestrian_colors = {
                'alert': (255, 50, 255),  # Magenta for close pedestrians
                'warning': (255, 120, 180),  # Pink for medium
                'safe': (180, 180, 255)   # Light blue for far
            }
    
            detected_info = []
            pedestrian_info = []
    
            # First draw the connection lines for vehicles
            line_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
    
            for vehicle, distance in detected_vehicles:
                if vehicle is None or distance > 40:  # Only process vehicles within 40m radius
                    continue
                
                try:
                    ego_transform = self._parent.get_transform()
                    ego_forward = ego_transform.get_forward_vector()
                    ego_right = self._calculate_right_vector(ego_transform.rotation)
    
                    vehicle_location = vehicle.get_location()
                    ego_location = self._parent.get_location()
    
                    rel_loc = vehicle_location - ego_location
                    forward_dot = (rel_loc.x * ego_forward.x + rel_loc.y * ego_forward.y)
                    right_dot = (rel_loc.x * ego_right.x + rel_loc.y * ego_right.y)
    
                    screen_x = center_x + right_dot * self.scale
                    screen_y = center_y - forward_dot * self.scale
    
                    # Determine color based on distance
                    if distance < 10:
                        color = vehicle_colors['alert']
                        status = 'ALERT'
                    elif distance < 20:
                        color = vehicle_colors['warning']
                        status = 'WARNING'
                    else:
                        color = vehicle_colors['safe']
                        status = 'TRACKED'
    
                    # Draw connection line with gradient effect
                    for i in range(3):
                        alpha = 150 - (i * 40)
                        pygame.draw.line(line_surface, (*color[:3], alpha),
                                       (center_x, center_y),
                                       (screen_x, screen_y), 1 + i)
    
                    # Calculate relative velocity
                    rel_velocity = self._get_relative_velocity(vehicle)
    
                    detected_info.append({
                        'distance': distance,
                        'velocity': rel_velocity,
                        'status': status,
                        'color': color,
                        'position': (screen_x, screen_y),
                        'type': 'vehicle'
                    })
    
                except Exception as e:
                    print(f"Error processing vehicle: {e}")
            
            # Draw connection lines for pedestrians if they exist
            if detected_pedestrians:
                for pedestrian, distance in detected_pedestrians:
                    if pedestrian is None or distance > 40:  # Only process within 40m radius
                        continue
                    
                    try:
                        ego_transform = self._parent.get_transform()
                        ego_forward = ego_transform.get_forward_vector()
                        ego_right = self._calculate_right_vector(ego_transform.rotation)
    
                        pedestrian_location = pedestrian.get_location()
                        ego_location = self._parent.get_location()
    
                        rel_loc = pedestrian_location - ego_location
                        forward_dot = (rel_loc.x * ego_forward.x + rel_loc.y * ego_forward.y)
                        right_dot = (rel_loc.x * ego_right.x + rel_loc.y * ego_right.y)
    
                        screen_x = center_x + right_dot * self.scale
                        screen_y = center_y - forward_dot * self.scale
    
                        # Pedestrians are more dangerous at closer ranges
                        if distance < 5:
                            color = pedestrian_colors['alert']
                            status = 'PED ALERT'
                        elif distance < 15:
                            color = pedestrian_colors['warning']
                            status = 'PED WARNING'
                        else:
                            color = pedestrian_colors['safe']
                            status = 'PED TRACKED'
    
                        # Draw connection line with dashed style for pedestrians
                        dash_length = 5
                        gap_length = 3
                        total_length = dash_length + gap_length
                        
                        # Calculate direction vector
                        dir_x = screen_x - center_x
                        dir_y = screen_y - center_y
                        line_length = math.sqrt(dir_x**2 + dir_y**2)
                        
                        if line_length > 0:
                            unit_x = dir_x / line_length
                            unit_y = dir_y / line_length
                            
                            # Draw dashed line
                            pos = 0
                            while pos < line_length:
                                dash_end = min(pos + dash_length, line_length)
                                
                                start_x = center_x + unit_x * pos
                                start_y = center_y + unit_y * pos
                                end_x = center_x + unit_x * dash_end
                                end_y = center_y + unit_y * dash_end
                                
                                pygame.draw.line(line_surface, (*color[:3], 180),
                                               (start_x, start_y),
                                               (end_x, end_y), 2)
                                
                                pos += total_length
    
                        # Calculate velocity
                        velocity = pedestrian.get_velocity()
                        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
    
                        pedestrian_info.append({
                            'distance': distance,
                            'velocity': speed,  # Just use absolute speed for pedestrians
                            'status': status,
                            'color': color,
                            'position': (screen_x, screen_y),
                            'type': 'pedestrian'
                        })
    
                    except Exception as e:
                        print(f"Error processing pedestrian: {e}")
    
            # Blit all connection lines
            self.surface.blit(line_surface, (0, 0))
    
            # Then draw all vehicles
            for info in detected_info:
                draw_realistic_car(self.surface, 
                                 int(info['position'][0]), 
                                 int(info['position'][1]), 
                                 20, 32,  # Smaller size for other vehicles
                                 info['color'])
                                 
            # Draw all pedestrians
            for info in pedestrian_info:
                draw_realistic_pedestrian(self.surface,
                                        int(info['position'][0]),
                                        int(info['position'][1]),
                                        14, 26,  # Smaller size for pedestrians
                                        info['color'])
    
            # Draw premium status panel - updated to include pedestrians
            panel_height = 130
            panel_surface = pygame.Surface((self.width, panel_height), pygame.SRCALPHA)
    
            # Panel background with gradient
            for y in range(panel_height):
                alpha = 200 - (y / panel_height) * 100
                pygame.draw.line(panel_surface, (20, 22, 30, int(alpha)),
                               (0, y), (self.width, y))
    
            # Status header
            header = self.font.render("Detection Status", True, (0, 255, 255))
            panel_surface.blit(header, (15, 10))
    
            # Combine vehicle and pedestrian info for display, sorted by distance
            all_detections = detected_info + pedestrian_info
            sorted_detections = sorted(all_detections, key=lambda x: x['distance'])
    
            # Detection information
            y_offset = 40
            for i, info in enumerate(sorted_detections):
                if i < 4:  # Limit to 4 entries to avoid overcrowding
                    if info['type'] == 'pedestrian':
                        status_text = f"{info['status']}: {info['distance']:.1f}m"
                        if abs(info['velocity']) > 0.1:
                            status_text += f" | {info['velocity']:.1f} km/h"
                    else:
                        status_text = f"{info['status']}: {info['distance']:.1f}m"
                        if abs(info['velocity']) > 0.1:
                            status_text += f" | {info['velocity']:.1f} m/s"
    
                    pygame.draw.circle(panel_surface, info['color'], (20, y_offset + 7), 4)
    
                    shadow = self.small_font.render(status_text, True, (0, 0, 0))
                    text = self.small_font.render(status_text, True, info['color'])
                    panel_surface.blit(shadow, (32, y_offset + 1))
                    panel_surface.blit(text, (30, y_offset))
                    y_offset += 22
    
            self.surface.blit(panel_surface, (0, 0))
    
            # Footer with system status - updated to include pedestrian count
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            vehicle_count = len(detected_info)
            pedestrian_count = len(pedestrian_info)
            stats_text = f"System Active | Time: {timestamp} | Vehicles: {vehicle_count} | Pedestrians: {pedestrian_count}"
            
            footer_surface = pygame.Surface((self.width, 30), pygame.SRCALPHA)
            pygame.draw.rect(footer_surface, (20, 22, 30, 200), (0, 0, self.width, 30))
            stats = self.small_font.render(stats_text, True, (0, 255, 255))
            footer_surface.blit(stats, (15, 8))
            self.surface.blit(footer_surface, (0, self.height - 30))
    
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()


    def render(self, display):
        """Render the detection visualization"""
        if self.surface is not None:
            display.blit(self.surface, (display.get_width() - self.width - 20, 20))
    
    def destroy(self):
        """Clean up sensors and resources"""
        try:
            for sensor in self.sensors:
                if sensor is not None:
                    sensor.destroy()
            self.sensors.clear()
        except Exception as e:
            print(f"Error destroying sensors: {e}")






































#vehicle_detetion_yolo.py
import os
import sys
import time
import numpy as np
import pygame
import carla
import random
import math
from sklearn.cluster import DBSCAN
import torch
import cv2
import weakref

class VehicleDetector:
    def __init__(self, parent_actor, world, controller, detection_distance=40.0, detection_angle=160.0):
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.debug = True
        self.detection_distance = detection_distance 
        self.detection_angle = detection_angle
        self.min_distance = float('inf')
        
        # YOLO model initialization
        self.yolo_model = None
        self._init_yolo()
        
        # Camera settings
        self.camera = None
        self.camera_image = None
        self.camera_width = 640
        self.camera_height = 480
        
        # Multiple sensor configuration
        self.sensors = []
        self._setup_sensors()

        # Visualization parameters
        self.width = 375
        self.height = 375
        self.scale = (self.height * 0.7) / self.detection_distance
        self.surface = None

        # Detection history for stability
        self.vehicle_history = []
        self.history_length = 5  # Keep track of last 5 frames
        
        # Add pedestrian tracking
        self.pedestrian_history = []
        self.pedestrian_history_length = 5
        
        # Store detections
        self.vehicle_detections = set()
        self.pedestrian_detections = set()
        
        print(f"YOLOv11 Vehicle & Pedestrian Detector initialized with {detection_distance}m range and {detection_angle}° angle")
    
    def _init_yolo(self):
        """Initialize YOLOv11 model"""
        try:
            print("Loading YOLOv11 model...")

            # For YOLOv11, we use the appropriate import and model path
            from ultralytics import YOLO

            # Define path to your YOLOv11 weights
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolo11m.pt')

            # Check if weights file exists, download if not available
            if not os.path.exists(weights_path):
                print(f"YOLOv11 weights not found at {weights_path}")
                print("Downloading YOLOv11 weights...")
                try:
                    # Using ultralytics built-in download mechanism
                    self.yolo_model = YOLO('yolov11m')  # This will download the official weights

                    # Save the model to the specified path
                    self.yolo_model.save(weights_path)
                    print(f"YOLOv11 weights downloaded and saved to {weights_path}")
                except Exception as download_error:
                    print(f"Error downloading YOLOv11 weights: {download_error}")
                    raise
                
            # Load the YOLOv11 model
            self.yolo_model = YOLO(weights_path)

            # Set model parameters
            self.yolo_model.conf = 0.25  # Confidence threshold
            self.yolo_model.iou = 0.45   # NMS IoU threshold

            # Specify the classes we're most interested in (vehicles and pedestrians)
            # The class IDs depend on the model but typically include:
            # Person: 0, Bicycle: 1, Car: 2, Motorcycle: 3, Bus: 5, Truck: 7
            self.vehicle_classes = [2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck
            self.pedestrian_classes = [0]        # Person

            print("YOLOv11 model loaded successfully!")

        except Exception as e:
            print(f"Error loading YOLOv11 model: {e}")
            import traceback
            traceback.print_exc()
            self.yolo_model = None
    
    def _setup_sensors(self):
        """Setup multiple sensors for redundant detection"""
        try:
            # Main camera for YOLO detection
            camera_bp = self._world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.camera_width))
            camera_bp.set_attribute('image_size_y', str(self.camera_height))
            camera_bp.set_attribute('fov', '90')
            camera_location = carla.Transform(carla.Location(x=2.0, z=1.5))
            self.camera = self._world.spawn_actor(camera_bp, camera_location, attach_to=self._parent)
            
            # Use weak reference to avoid circular reference
            weak_self = weakref.ref(self)
            self.camera.listen(lambda image: self._process_camera(weak_self, image))
            self.sensors.append(self.camera)
            
            # Main forward-facing radar
            radar_bp = self._world.get_blueprint_library().find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov', str(self.detection_angle))
            radar_bp.set_attribute('vertical_fov', '20')
            radar_bp.set_attribute('range', str(self.detection_distance))
            radar_location = carla.Transform(carla.Location(x=2.0, z=1.0))
            radar = self._world.spawn_actor(radar_bp, radar_location, attach_to=self._parent)
            radar.listen(lambda data: self._radar_callback(data))
            self.sensors.append(radar)
            
            # Semantic LIDAR for additional detection
            lidar_bp = self._world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', str(self.detection_distance))
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('channels', '32')
            lidar_bp.set_attribute('points_per_second', '90000')
            lidar_location = carla.Transform(carla.Location(x=0.0, z=2.0))
            lidar = self._world.spawn_actor(lidar_bp, lidar_location, attach_to=self._parent)
            lidar.listen(lambda data: self._lidar_callback(data))
            self.sensors.append(lidar)
            
            print("Successfully set up multiple detection sensors")
            
        except Exception as e:
            print(f"Error setting up sensors: {e}")
    
    @staticmethod
    def _process_camera(weak_self, image):
        """Process camera images for YOLO detection"""
        self = weak_self()
        if self is None:
            return
            
        try:
            # Convert image data to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            
            # Store the camera image for YOLO processing
            self.camera_image = array
            
            # Process with YOLO
            self._process_yolo_detection()
            
        except Exception as e:
            print(f"Error in camera callback: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_yolo_detection(self):
        """Process current camera image with YOLOv11 model"""
        if self.camera_image is None or self.yolo_model is None:
            return

        try:
            # Run YOLO detection on the image
            results = self.yolo_model(self.camera_image)

            # Get current ego vehicle transform
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            ego_right = self._calculate_right_vector(ego_transform.rotation)

            # Process detected vehicles
            vehicle_detections = set()
            pedestrian_detections = set()

            # Calculate focal length using camera parameters
            fov = 90  # Camera FOV is set to 90 in the sensor setup
            focal_length = (self.camera_width / 2) / np.tan(np.radians(fov / 2))

            # Extract detections from results
            for result in results:
                # Get bounding boxes, confidence scores, and class IDs
                boxes = result.boxes

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()

                    # Skip if confidence is too low
                    if conf < 0.25:
                        continue
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Calculate object center in image
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    # Normalize position to center of image
                    norm_x = (center_x - self.camera_width / 2) / (self.camera_width / 2)

                    # Calculate angle from forward direction
                    angle = norm_x * 45  # Assuming 90-degree FOV, so each half is 45 degrees

                    # Filter by angle (detection angle)
                    if abs(angle) > self.detection_angle / 2:
                        continue
                    
                    # Define mapping for YOLO classes to our internal class IDs
                    # YOLO class mapping: person=0, bicycle=1, car=2, motorcycle=3, bus=5, truck=7
                    # Our internal class mapping from calculate_spatial_info:
                    # 0: person, 1: bicycle, 2: motorcycle, 3: car, 4: truck, 5: bus
                    yolo_to_internal_class = {
                        0: 0,  # person
                        1: 1,  # bicycle
                        2: 3,  # car
                        3: 2,  # motorcycle
                        5: 5,  # bus
                        7: 4,  # truck
                    }

                    internal_class_id = yolo_to_internal_class.get(cls_id, 3)  # Default to car if not found

                    # Get bounding box dimensions
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1

                    # Use the calculate_spatial_info logic for more accurate depth calculation
                    # Define typical widths for different object classes (in meters)
                    REAL_WIDTHS = {
                        0: 0.45,   # person - average shoulder width
                        1: 0.8,    # bicycle - typical handlebar width
                        2: 0.8,    # motorcycle - typical handlebar width
                        3: 1.8,    # car - average car width
                        4: 2.5,    # truck - average truck width
                        5: 2.9,    # bus - average bus width
                        6: 3.0,    # train - typical train car width
                    }

                    # Get real width based on class, default to car width if class not found
                    real_width = REAL_WIDTHS.get(internal_class_id, 1.8)

                    # Calculate depth using similar triangles principle
                    if bbox_width > 0:  # Avoid division by zero
                        depth = (real_width * focal_length) / bbox_width
                    else:
                        depth = float('inf')

                    # Apply distance limitation
                    if depth > self.detection_distance:
                        continue
                    
                    # Add to appropriate detection set
                    if cls_id in self.vehicle_classes:
                        vehicle_detections.add((None, depth, angle))

                        if self.debug:
                            self._draw_debug_yolo_detection(cls_id, x1, y1, x2, y2, depth, angle)

                    elif cls_id in self.pedestrian_classes:
                        pedestrian_detections.add((None, depth, angle))

                        if self.debug:
                            self._draw_debug_yolo_detection(cls_id, x1, y1, x2, y2, depth, angle)

            # Update detection sets
            self.vehicle_detections = vehicle_detections
            self.pedestrian_detections = pedestrian_detections

            # Update detection history
            self.vehicle_history.append(vehicle_detections)
            if len(self.vehicle_history) > self.history_length:
                self.vehicle_history.pop(0)

            self.pedestrian_history.append(pedestrian_detections)
            if len(self.pedestrian_history) > self.pedestrian_history_length:
                self.pedestrian_history.pop(0)

        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            import traceback
            traceback.print_exc()
    
    def _draw_debug_yolo_detection(self, cls_id, x1, y1, x2, y2, distance, angle):
        """Draw debug visualization for YOLO detection"""
        if not self.debug:
            return
            
        try:
            # Get class name
            class_name = "Vehicle" if cls_id in self.vehicle_classes else "Pedestrian" if cls_id in self.pedestrian_classes else "Other"
            
            # Convert normalized camera coordinates to world space (approximate)
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            ego_right = self._calculate_right_vector(ego_transform.rotation)
            
            # Calculate object position in world space (approximate based on distance and angle)
            angle_rad = math.radians(angle)
            dx = distance * math.cos(angle_rad)
            dy = distance * math.sin(angle_rad)
            
            # Position in world coordinates
            world_pos = ego_location + ego_forward * dx + ego_right * dy
            
            # Color based on distance
            if distance < 5:
                color = carla.Color(255, 0, 0)  # Red for very close
            elif distance < 15:
                color = carla.Color(255, 165, 0)  # Orange for intermediate
            else:
                color = carla.Color(0, 255, 0)  # Green for far
            
            # Draw debug sphere at the estimated position
            self._world.debug.draw_point(
                world_pos,
                size=0.2,
                color=color,
                life_time=0.1
            )
            
            # Draw debug text
            self._world.debug.draw_string(
                world_pos + carla.Location(z=1.0),
                f"{class_name} {distance:.1f}m",
                color=color,
                life_time=0.1
            )
            
        except Exception as e:
            print(f"Error in YOLO debug drawing: {e}")
    
    def _radar_callback(self, radar_data):
        """Process radar data for vehicle detection"""
        try:
            points = []
            for detection in radar_data:
                # Get the detection coordinates in world space
                distance = detection.depth
                if distance < self.detection_distance:
                    azimuth = math.degrees(detection.azimuth)
                    if abs(azimuth) < self.detection_angle / 2:
                        # Radar points for sensor fusion
                        points.append({
                            'distance': distance,
                            'azimuth': azimuth,
                            'altitude': math.degrees(detection.altitude),
                            'velocity': detection.velocity
                        })
                        
                        # Add to vehicle detections (focused on vehicles)
                        self.vehicle_detections.add((None, distance, azimuth))
            
            # Store radar detections
            self.radar_points = points
            
        except Exception as e:
            print(f"Error in radar callback: {e}")
    
    def _lidar_callback(self, lidar_data):
        """Process semantic LIDAR data for vehicle and pedestrian detection"""
        try:
            # Filter points that belong to vehicles (semantic tag 10 in CARLA)
            vehicle_points = [point for point in lidar_data 
                             if point.object_tag == 10 
                             and point.distance < self.detection_distance]

            # Filter points that belong to pedestrians (semantic tag 4 in CARLA)
            pedestrian_points = [point for point in lidar_data 
                               if point.object_tag == 4 
                               and point.distance < self.detection_distance]

            # Process vehicle points
            vehicle_detections = set()
            for point in vehicle_points:
                # Convert point to local coordinates relative to ego vehicle
                ego_transform = self._parent.get_transform()
                ego_location = ego_transform.location
                ego_forward = ego_transform.get_forward_vector()
                ego_right = self._calculate_right_vector(ego_transform.rotation)
                
                # Get point location in world coordinates
                point_location = carla.Location(x=point.point.x, y=point.point.y, z=point.point.z)
                
                # Calculate distance and angle
                distance = point.distance
                
                # Calculate relative position for angle
                relative_loc = point_location - ego_location
                forward_dot = (relative_loc.x * ego_forward.x + relative_loc.y * ego_forward.y)
                right_dot = (relative_loc.x * ego_right.x + relative_loc.y * ego_right.y)
                angle = math.degrees(math.atan2(right_dot, forward_dot))
                
                # Check if point is within detection angle
                if abs(angle) < self.detection_angle / 2:
                    vehicle_detections.add((None, distance, angle))
            
            # Process pedestrian points
            pedestrian_detections = set()
            for point in pedestrian_points:
                # Same process for pedestrians
                ego_transform = self._parent.get_transform()
                ego_location = ego_transform.location
                ego_forward = ego_transform.get_forward_vector()
                ego_right = self._calculate_right_vector(ego_transform.rotation)
                
                point_location = carla.Location(x=point.point.x, y=point.point.y, z=point.point.z)
                distance = point.distance
                
                relative_loc = point_location - ego_location
                forward_dot = (relative_loc.x * ego_forward.x + relative_loc.y * ego_forward.y)
                right_dot = (relative_loc.x * ego_right.x + relative_loc.y * ego_right.y)
                angle = math.degrees(math.atan2(right_dot, forward_dot))
                
                if abs(angle) < self.detection_angle / 2:
                    pedestrian_detections.add((None, distance, angle))
            
            # Add to current detections (combine with existing from YOLO)
            self.vehicle_detections.update(vehicle_detections)
            self.pedestrian_detections.update(pedestrian_detections)
            
            # Store LIDAR detections
            self.lidar_points = vehicle_points
            self.lidar_pedestrian_points = pedestrian_points

        except Exception as e:
            print(f"Error in LIDAR callback: {e}")
    
    def detect_vehicles(self):
        """Get stable vehicle detections from all sensors"""
        try:
            # Ensure YOLO detection is processed (if camera image is available)
            if self.camera_image is not None and self.yolo_model is not None:
                self._process_yolo_detection()

            # Get stable detections from history
            if len(self.vehicle_history) > 0:
                # Fuse detections using DBSCAN clustering
                all_points = []
                for frame in self.vehicle_history:
                    for _, distance, angle in frame:
                        all_points.append([distance, angle])

                if len(all_points) == 0:
                    return set()

                # Cluster detections
                X = np.array(all_points)
                clustering = DBSCAN(eps=5.0, min_samples=2).fit(X)

                # Get cluster centers
                labels = clustering.labels_
                unique_labels = set(labels)

                stable_detections = set()
                for label in unique_labels:
                    if label == -1:  # Noise
                        continue

                    # Get points in this cluster
                    cluster_points = X[labels == label]

                    # Calculate mean distance and angle
                    mean_distance = np.mean(cluster_points[:, 0])
                    mean_angle = np.mean(cluster_points[:, 1])

                    # Add as stable detection
                    stable_detections.add((None, mean_distance, mean_angle))

                # Get pedestrian detections
                detected_pedestrians = self.detect_pedestrians()

                # Update visualization with both vehicle and pedestrian detections
                self._create_visualization(stable_detections, detected_pedestrians)

                return stable_detections

            return set()

        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            import traceback
            traceback.print_exc()
            return set()    

    def detect_pedestrians(self):
        """Get stable pedestrian detections from all sensors"""
        try:
            # Ensure YOLO detection is processed (if camera image is available)
            if self.camera_image is not None and self.yolo_model is not None:
                self._process_yolo_detection()

            # Get stable detections from history
            if len(self.pedestrian_history) > 0:
                # Fuse detections using DBSCAN clustering
                all_points = []
                for frame in self.pedestrian_history:
                    for _, distance, angle in frame:
                        all_points.append([distance, angle])

                if len(all_points) == 0:
                    return set()

                # Cluster detections
                X = np.array(all_points)
                clustering = DBSCAN(eps=3.0, min_samples=2).fit(X)  # Tighter clustering for pedestrians

                # Get cluster centers
                labels = clustering.labels_
                unique_labels = set(labels)

                stable_detections = set()
                for label in unique_labels:
                    if label == -1:  # Noise
                        continue

                    # Get points in this cluster
                    cluster_points = X[labels == label]

                    # Calculate mean distance and angle
                    mean_distance = np.mean(cluster_points[:, 0])
                    mean_angle = np.mean(cluster_points[:, 1])

                    # Add as stable detection
                    stable_detections.add((None, mean_distance, mean_angle))

                    # Debug visualization
                    if self.debug:
                        self._draw_debug_pedestrian(mean_distance, mean_angle)

                return stable_detections

            return set()

        except Exception as e:
            print(f"Error in pedestrian detection: {e}")
            import traceback
            traceback.print_exc()
            return set()
    
    def _draw_debug_vehicle(self, distance, angle):
        """Draw debug visualization for detected vehicle"""
        try:
            if self.debug:
                # Calculate position in world coordinates
                ego_transform = self._parent.get_transform()
                ego_location = ego_transform.location
                ego_forward = ego_transform.get_forward_vector()
                ego_right = self._calculate_right_vector(ego_transform.rotation)
                
                # Convert angle to radians
                angle_rad = math.radians(angle)
                
                # Calculate position
                dx = distance * math.cos(angle_rad)
                dy = distance * math.sin(angle_rad)
                
                # Position in world coordinates
                world_pos = ego_location + ego_forward * dx + ego_right * dy
                
                # Adjust height to approximate car height
                world_pos.z += 0.5
                
                # Color based on distance
                if distance < 5:
                    color = carla.Color(255, 0, 0)  # Red for very close
                elif distance < 15:
                    color = carla.Color(255, 165, 0)  # Orange for intermediate
                else:
                    color = carla.Color(0, 255, 0)  # Green for far
                
                # Draw a box to represent vehicle
                box_size = carla.Vector3D(2.0, 1.0, 1.5)  # Approximate car size
                self._world.debug.draw_box(
                    box=carla.BoundingBox(world_pos, box_size),
                    rotation=ego_transform.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=0.1
                )
                
                # Draw distance info
                self._world.debug.draw_string(
                    world_pos + carla.Location(z=2.0),
                    f"VEH {distance:.1f}m",
                    color=color,
                    life_time=0.1
                )
                
        except Exception as e:
            print(f"Error in vehicle debug drawing: {e}")
    
    def _draw_debug_pedestrian(self, distance, angle):
        """Draw debug visualization for detected pedestrian"""
        try:
            if self.debug:
                # Calculate position in world coordinates
                ego_transform = self._parent.get_transform()
                ego_location = ego_transform.location
                ego_forward = ego_transform.get_forward_vector()
                ego_right = self._calculate_right_vector(ego_transform.rotation)
                
                # Convert angle to radians
                angle_rad = math.radians(angle)
                
                # Calculate position
                dx = distance * math.cos(angle_rad)
                dy = distance * math.sin(angle_rad)
                
                # Position in world coordinates
                world_pos = ego_location + ego_forward * dx + ego_right * dy
                
                # Adjust height to approximate pedestrian height
                world_pos.z += 1.0
                
                # Color based on distance
                if distance < 5:
                    color = carla.Color(255, 0, 255)  # Magenta for very close pedestrians
                elif distance < 15:
                    color = carla.Color(255, 165, 0)  # Orange for intermediate
                else:
                    color = carla.Color(0, 255, 0)  # Green for far
                
                # Draw a cylinder to represent pedestrian
                self._world.debug.draw_point(
                    world_pos,
                    size=0.2,
                    color=color,
                    life_time=0.1
                )
                
                # Draw pedestrian representation (smaller than vehicle)
                box_size = carla.Vector3D(0.5, 0.5, 1.8)  # Approximate pedestrian size
                self._world.debug.draw_box(
                    box=carla.BoundingBox(world_pos, box_size),
                    rotation=ego_transform.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=0.1
                )
                
                # Draw distance info
                self._world.debug.draw_string(
                    world_pos + carla.Location(z=2.0),
                    f"PED {distance:.1f}m",
                    color=color,
                    life_time=0.1
                )
                
        except Exception as e:
            print(f"Error in pedestrian debug drawing: {e}")
    
    def _calculate_right_vector(self, rotation):
        """Calculate the right vector from a rotation"""
        forward = rotation.get_forward_vector()
        right = carla.Vector3D(x=-forward.y, y=forward.x, z=0.0)
        return right.make_unit_vector()
    
    def _get_relative_velocity(self, other_vehicle):
        """Calculate relative velocity between ego vehicle and other vehicle"""
        try:
            ego_vel = self._parent.get_velocity()
            other_vel = other_vehicle.get_velocity()

            # Convert to numpy arrays for easier calculation
            ego_vel_array = np.array([ego_vel.x, ego_vel.y])
            other_vel_array = np.array([other_vel.x, other_vel.y])

            # Calculate relative velocity
            rel_vel = np.linalg.norm(ego_vel_array - other_vel_array)

            # Determine if approaching
            ego_fwd = self._parent.get_transform().get_forward_vector()
            ego_fwd_array = np.array([ego_fwd.x, ego_fwd.y])

            # If dot product is positive, we're approaching
            if np.dot(ego_vel_array - other_vel_array, ego_fwd_array) > 0:
                return rel_vel
            return -rel_vel

        except Exception as e:
            print(f"Error calculating relative velocity: {e}")
            return 0.0

    def _create_visualization(self, detected_vehicles, detected_pedestrians=None):
        try:
            if self.surface is None:
                self.surface = pygame.Surface((self.width, self.height))
                self.font = pygame.font.Font(None, 24)
                self.small_font = pygame.font.Font(None, 20)
    
            # Fill background with a premium dark gradient
            gradient = pygame.Surface((self.width, self.height))
            for y in range(self.height):
                alpha = y / self.height
                color = (20, 22, 30, int(255 * (0.95 - alpha * 0.3)))
                pygame.draw.line(gradient, color, (0, y), (self.width, y))
            self.surface.blit(gradient, (0, 0))
    
            # Constants for visualization
            center_x = self.width // 2
            center_y = int(self.height * 0.8)
    
            # Draw sensor range indicators with premium styling
            ranges = [10, 20, 30, 40]
            for range_dist in ranges:
                radius = int(range_dist * self.scale)
                points = []
                for angle in range(0, 360, 2):
                    rad = math.radians(angle)
                    x = center_x + radius * math.sin(rad)
                    y = center_y - radius * math.cos(rad)
                    points.append((x, y))
    
                # Draw circles with fade effect
                alpha = 100 - (range_dist / 40) * 60
                circle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.lines(circle_surface, (0, 255, 255, int(alpha)), False, points, 1)
                self.surface.blit(circle_surface, (0, 0))
    
                # Premium distance labels
                label = self.small_font.render(f"{range_dist}m", True, (0, 255, 255, int(alpha)))
                self.surface.blit(label, (center_x + radius + 5, center_y - 10))
    
            def draw_realistic_car(surface, x, y, width, height, color, is_ego=False):
                """Draw a more realistic car shape with corrected dimensions"""
                # Scale down size for more realistic proportions
                width = int(width * 0.8)
                height = int(height * 0.8)
    
                # Create a surface for the car with alpha channel
                car_surface = pygame.Surface((width * 2, height * 2), pygame.SRCALPHA)
                center_x = width
                center_y = height
    
                # Main body shape points (more curved)
                body_points = [
                    (center_x - width//2, center_y),                    # Left bottom
                    (center_x - width//2, center_y - height*0.7),      # Left middle
                    (center_x - width//2.2, center_y - height*0.8),    # Left front curve
                    (center_x - width//3, center_y - height*0.85),     # Left hood
                    (center_x, center_y - height*0.9),                 # Front center
                    (center_x + width//3, center_y - height*0.85),     # Right hood
                    (center_x + width//2.2, center_y - height*0.8),    # Right front curve
                    (center_x + width//2, center_y - height*0.7),      # Right middle
                    (center_x + width//2, center_y),                   # Right bottom
                ]
    
                # Draw car body
                pygame.draw.polygon(car_surface, color, body_points)
    
                # Windshield (more curved)
                windshield_points = [
                    (center_x - width//3, center_y - height*0.85),     # Left bottom
                    (center_x - width//3.5, center_y - height*0.95),   # Left top
                    (center_x + width//3.5, center_y - height*0.95),   # Right top
                    (center_x + width//3, center_y - height*0.85),     # Right bottom
                ]
                pygame.draw.polygon(car_surface, (*color[:3], 150), windshield_points)
    
                # Wheels (more elliptical)
                wheel_width = width // 6
                wheel_height = height // 8
    
                # Front wheels (black ellipses)
                pygame.draw.ellipse(car_surface, (30, 30, 30),
                                  (center_x - width//2 - wheel_width//4,
                                   center_y - wheel_height//2,
                                   wheel_width, wheel_height))
                pygame.draw.ellipse(car_surface, (30, 30, 30),
                                  (center_x + width//2 - wheel_width*3//4,
                                   center_y - wheel_height//2,
                                   wheel_width, wheel_height))
    
                # Wheel well highlights (curved lines instead of arcs)
                # Left wheel well
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x - width//2 - wheel_width//4, center_y - wheel_height),
                               (center_x - width//2 + wheel_width, center_y - wheel_height), 2)
    
                # Right wheel well
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x + width//2 - wheel_width*3//4, center_y - wheel_height),
                               (center_x + width//2 + wheel_width//4, center_y - wheel_height), 2)
    
                # Add detail lines for body contour
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x - width//3, center_y - height*0.75),
                               (center_x + width//3, center_y - height*0.75), 1)
    
                # Add headlights or taillights
                light_width = width // 8
                light_height = height // 12
    
                if is_ego:
                    # Headlights (bright yellow)
                    light_color = (255, 255, 200, 200)
                else:
                    # Taillights (red)
                    light_color = (255, 50, 50, 200)
    
                # Left light
                pygame.draw.ellipse(car_surface, light_color,
                                  (center_x - width//2.5,
                                   center_y - height*0.85,
                                   light_width, light_height))
    
                # Right light
                pygame.draw.ellipse(car_surface, light_color,
                                  (center_x + width//2.5 - light_width,
                                   center_y - height*0.85,
                                   light_width, light_height))
    
                # Blit the car onto the main surface
                surface.blit(car_surface, (x - width, y - height))
    
            # Add the new function to draw realistic pedestrians
            def draw_realistic_pedestrian(surface, x, y, width, height, color):
                """Draw a more realistic pedestrian shape"""
                # Scale down size for realistic pedestrian proportions
                width = int(width * 0.5)  # Pedestrians are narrower than cars
                height = int(height * 0.6)  # And shorter
                
                # Create a surface for the pedestrian with alpha channel
                ped_surface = pygame.Surface((width * 2, height * 2), pygame.SRCALPHA)
                center_x = width
                center_y = height
                
                # Head
                head_radius = width // 2
                head_y = center_y - height * 0.75
                pygame.draw.circle(ped_surface, color, (center_x, head_y), head_radius)
                
                # Body (roughly human-shaped)
                body_points = [
                    (center_x - width//2, center_y),              # Left bottom
                    (center_x - width//3, head_y + head_radius),  # Left shoulder
                    (center_x + width//3, head_y + head_radius),  # Right shoulder
                    (center_x + width//2, center_y),              # Right bottom
                ]
                pygame.draw.polygon(ped_surface, color, body_points)
                
                # Arms (lines)
                arm_length = height * 0.3
                left_arm_x = center_x - width//3
                right_arm_x = center_x + width//3
                arm_y = head_y + head_radius + height * 0.1
                
                # Left arm
                pygame.draw.line(ped_surface, (*color[:3], 220),
                               (left_arm_x, arm_y),
                               (left_arm_x - width//3, arm_y + arm_length), 2)
                               
                # Right arm
                pygame.draw.line(ped_surface, (*color[:3], 220),
                               (right_arm_x, arm_y),
                               (right_arm_x + width//3, arm_y + arm_length), 2)
                
                # Legs (lines)
                leg_length = height * 0.35
                leg_spacing = width//3
                leg_y = center_y
                
                # Left leg
                pygame.draw.line(ped_surface, (*color[:3], 220),
                               (center_x - leg_spacing, leg_y),
                               (center_x - leg_spacing, leg_y + leg_length), 2)
                               
                # Right leg
                pygame.draw.line(ped_surface, (*color[:3], 220),
                               (center_x + leg_spacing, leg_y),
                               (center_x + leg_spacing, leg_y + leg_length), 2)
                
                # Blit the pedestrian onto the main surface
                surface.blit(ped_surface, (x - width, y - height))
    
            # Draw ego vehicle (smaller size)
            draw_realistic_car(self.surface, center_x, center_y, 24, 40, (0, 255, 255), True)
    
            # Draw detection cone with gradient
            fov_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            fov_radius = self.detection_distance * self.scale
    
            # Create detection cone
            fov_points = [(center_x, center_y)]
            for angle in range(-int(self.detection_angle/2), int(self.detection_angle/2)):
                rad = math.radians(angle)
                x = center_x + fov_radius * math.sin(rad)
                y = center_y - fov_radius * math.cos(rad)
                fov_points.append((x, y))
            fov_points.append((center_x, center_y))
    
            # Draw cone with gradient
            pygame.draw.polygon(fov_surface, (0, 255, 255, 20), fov_points)
            self.surface.blit(fov_surface, (0, 0))
    
            # Colors for detection
            vehicle_colors = {
                'alert': (255, 50, 50),
                'warning': (255, 165, 0),
                'safe': (0, 255, 255)
            }
            
            # New colors for pedestrians - slightly different hues
            pedestrian_colors = {
                'alert': (255, 50, 255),  # Magenta for close pedestrians
                'warning': (255, 120, 180),  # Pink for medium
                'safe': (180, 180, 255)   # Light blue for far
            }
    
            detected_info = []
            pedestrian_info = []
    
            # First draw the connection lines for vehicles
            line_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
    
            for vehicle, distance in detected_vehicles:
                if vehicle is None or distance > 40:  # Only process vehicles within 40m radius
                    continue
                
                try:
                    ego_transform = self._parent.get_transform()
                    ego_forward = ego_transform.get_forward_vector()
                    ego_right = self._calculate_right_vector(ego_transform.rotation)
    
                    vehicle_location = vehicle.get_location()
                    ego_location = self._parent.get_location()
    
                    rel_loc = vehicle_location - ego_location
                    forward_dot = (rel_loc.x * ego_forward.x + rel_loc.y * ego_forward.y)
                    right_dot = (rel_loc.x * ego_right.x + rel_loc.y * ego_right.y)
    
                    screen_x = center_x + right_dot * self.scale
                    screen_y = center_y - forward_dot * self.scale
    
                    # Determine color based on distance
                    if distance < 10:
                        color = vehicle_colors['alert']
                        status = 'ALERT'
                    elif distance < 20:
                        color = vehicle_colors['warning']
                        status = 'WARNING'
                    else:
                        color = vehicle_colors['safe']
                        status = 'TRACKED'
    
                    # Draw connection line with gradient effect
                    for i in range(3):
                        alpha = 150 - (i * 40)
                        pygame.draw.line(line_surface, (*color[:3], alpha),
                                       (center_x, center_y),
                                       (screen_x, screen_y), 1 + i)
    
                    # Calculate relative velocity
                    rel_velocity = self._get_relative_velocity(vehicle)
    
                    detected_info.append({
                        'distance': distance,
                        'velocity': rel_velocity,
                        'status': status,
                        'color': color,
                        'position': (screen_x, screen_y),
                        'type': 'vehicle'
                    })
    
                except Exception as e:
                    print(f"Error processing vehicle: {e}")
            
            # Draw connection lines for pedestrians if they exist
            if detected_pedestrians:
                for pedestrian, distance in detected_pedestrians:
                    if pedestrian is None or distance > 40:  # Only process within 40m radius
                        continue
                    
                    try:
                        ego_transform = self._parent.get_transform()
                        ego_forward = ego_transform.get_forward_vector()
                        ego_right = self._calculate_right_vector(ego_transform.rotation)
    
                        pedestrian_location = pedestrian.get_location()
                        ego_location = self._parent.get_location()
    
                        rel_loc = pedestrian_location - ego_location
                        forward_dot = (rel_loc.x * ego_forward.x + rel_loc.y * ego_forward.y)
                        right_dot = (rel_loc.x * ego_right.x + rel_loc.y * ego_right.y)
    
                        screen_x = center_x + right_dot * self.scale
                        screen_y = center_y - forward_dot * self.scale
    
                        # Pedestrians are more dangerous at closer ranges
                        if distance < 5:
                            color = pedestrian_colors['alert']
                            status = 'PED ALERT'
                        elif distance < 15:
                            color = pedestrian_colors['warning']
                            status = 'PED WARNING'
                        else:
                            color = pedestrian_colors['safe']
                            status = 'PED TRACKED'
    
                        # Draw connection line with dashed style for pedestrians
                        dash_length = 5
                        gap_length = 3
                        total_length = dash_length + gap_length
                        
                        # Calculate direction vector
                        dir_x = screen_x - center_x
                        dir_y = screen_y - center_y
                        line_length = math.sqrt(dir_x**2 + dir_y**2)
                        
                        if line_length > 0:
                            unit_x = dir_x / line_length
                            unit_y = dir_y / line_length
                            
                            # Draw dashed line
                            pos = 0
                            while pos < line_length:
                                dash_end = min(pos + dash_length, line_length)
                                
                                start_x = center_x + unit_x * pos
                                start_y = center_y + unit_y * pos
                                end_x = center_x + unit_x * dash_end
                                end_y = center_y + unit_y * dash_end
                                
                                pygame.draw.line(line_surface, (*color[:3], 180),
                                               (start_x, start_y),
                                               (end_x, end_y), 2)
                                
                                pos += total_length
    
                        # Calculate velocity
                        velocity = pedestrian.get_velocity()
                        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
    
                        pedestrian_info.append({
                            'distance': distance,
                            'velocity': speed,  # Just use absolute speed for pedestrians
                            'status': status,
                            'color': color,
                            'position': (screen_x, screen_y),
                            'type': 'pedestrian'
                        })
    
                    except Exception as e:
                        print(f"Error processing pedestrian: {e}")
    
            # Blit all connection lines
            self.surface.blit(line_surface, (0, 0))
    
            # Then draw all vehicles
            for info in detected_info:
                draw_realistic_car(self.surface, 
                                 int(info['position'][0]), 
                                 int(info['position'][1]), 
                                 20, 32,  # Smaller size for other vehicles
                                 info['color'])
                                 
            # Draw all pedestrians
            for info in pedestrian_info:
                draw_realistic_pedestrian(self.surface,
                                        int(info['position'][0]),
                                        int(info['position'][1]),
                                        14, 26,  # Smaller size for pedestrians
                                        info['color'])
    
            # Draw premium status panel - updated to include pedestrians
            panel_height = 130
            panel_surface = pygame.Surface((self.width, panel_height), pygame.SRCALPHA)
    
            # Panel background with gradient
            for y in range(panel_height):
                alpha = 200 - (y / panel_height) * 100
                pygame.draw.line(panel_surface, (20, 22, 30, int(alpha)),
                               (0, y), (self.width, y))
    
            # Status header
            header = self.font.render("Detection Status", True, (0, 255, 255))
            panel_surface.blit(header, (15, 10))
    
            # Combine vehicle and pedestrian info for display, sorted by distance
            all_detections = detected_info + pedestrian_info
            sorted_detections = sorted(all_detections, key=lambda x: x['distance'])
    
            # Detection information
            y_offset = 40
            for i, info in enumerate(sorted_detections):
                if i < 4:  # Limit to 4 entries to avoid overcrowding
                    if info['type'] == 'pedestrian':
                        status_text = f"{info['status']}: {info['distance']:.1f}m"
                        if abs(info['velocity']) > 0.1:
                            status_text += f" | {info['velocity']:.1f} km/h"
                    else:
                        status_text = f"{info['status']}: {info['distance']:.1f}m"
                        if abs(info['velocity']) > 0.1:
                            status_text += f" | {info['velocity']:.1f} m/s"
    
                    pygame.draw.circle(panel_surface, info['color'], (20, y_offset + 7), 4)
    
                    shadow = self.small_font.render(status_text, True, (0, 0, 0))
                    text = self.small_font.render(status_text, True, info['color'])
                    panel_surface.blit(shadow, (32, y_offset + 1))
                    panel_surface.blit(text, (30, y_offset))
                    y_offset += 22
    
            self.surface.blit(panel_surface, (0, 0))
    
            # Footer with system status - updated to include pedestrian count
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            vehicle_count = len(detected_info)
            pedestrian_count = len(pedestrian_info)
            stats_text = f"System Active | Time: {timestamp} | Vehicles: {vehicle_count} | Pedestrians: {pedestrian_count}"
            
            footer_surface = pygame.Surface((self.width, 30), pygame.SRCALPHA)
            pygame.draw.rect(footer_surface, (20, 22, 30, 200), (0, 0, self.width, 30))
            stats = self.small_font.render(stats_text, True, (0, 255, 255))
            footer_surface.blit(stats, (15, 8))
            self.surface.blit(footer_surface, (0, self.height - 30))
    
        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()


    def render(self, display):
        """Render the detection visualization"""
        if self.surface is not None:
            display.blit(self.surface, (display.get_width() - self.width - 20, 20))
    
    def destroy(self):
        """Clean up sensors and resources"""
        try:
            for sensor in self.sensors:
                if sensor is not None:
                    sensor.destroy()
            self.sensors.clear()
        except Exception as e:
            print(f"Error destroying sensors: {e}")






























#sb3_ppo_train.py


class CarlaGymEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, carla_env=None, sim_port=2000, tm_port=8000):

        super(CarlaGymEnv, self).__init__()
        
        # Create or use provided CARLA environment
        if carla_env is None:
            self.env = CarEnv(sim_port=sim_port, tm_port=tm_port)
        else:
            self.env = carla_env
            
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Action space: [steering, throttle/brake]
        # Both ranging from -1 to 1
        self.action_space = spaces.Box(
            low=np.array([-1.0]),  # -1 for maximum braking
            high=np.array([1.0]),  # +1 for maximum throttle
            dtype=np.float32
        )
        
        # Observation space:
        # - object detections: max_objects * 6 features
        # - lane info: 4 features
        # - vehicle state: 2 features (speed, steering)
        # - additional features: 2 + 4
        obs_dim = self.env.max_objects * 6 + 4 + 2 + 2 + 4

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.episode_reward = 0
        self.episode_length = 0
        self.metrics = {
            'speeds': [],
            'collisions': 0,
            'lane_deviations': []
        }





def make_env(env_id, rank, seed=0, base_sim_port=2000, base_tm_port=8000):
    def _init():
        # Assign unique ports for each environment
        sim_port = base_sim_port + rank * 2
        tm_port = base_tm_port + rank * 2
        env = CarlaGymEnv(sim_port=sim_port, tm_port=tm_port)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init










env = SubprocVecEnv([make_env("CarlaGymEnv", i, base_sim_port=2000, base_tm_port=8000) for i in range(num_envs)])


















from Vehicle_detector_rl import VehicleDetector 
def main():
    """Main function to run CARLA environment with MPC controller integration"""
    # Set random seeds for reproducibility
    set_random_seed(42)
    
    # Configuration parameters
    TRAIN_RL = True  # Set to False to run only with MPC controller
    EVALUATE_ONLY = False  # Set to True to only evaluate without training
    TUNE_HYPERPARAMETERS = False
    CONTINUE_TRAINING = False
    CHECKPOINT_PATH = "models/ppo_carla_model_990000_steps.zip"  # Your checkpoint path
    
    try:
        print("Starting CARLA with hybrid MPC-RL control system")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        # Initialize the CarlaEnv with MPC controller
        env = CarEnv()
        if not env.setup_vehicle():
            print("Failed to set up CARLA environment")
            return
            
        # Set up Vehicle Detector
        detector = None
        
        try:
            # Create the vehicle detector
            # Assuming env has controller and vehicle attributes
            detector = VehicleDetector(env.vehicle, env.world, env.controller)
            
            # If we're just running with the MPC controller
            if not TRAIN_RL:
                print("Running with MPC controller only...")
                
                # Modify env.run() to include vehicle detection
                def custom_run():
                    # This would replace or extend env.run() to include vehicle detection
                    while env.running:
                        # Update vehicle detector
                        detector.detect_vehicles()
                        
                        # Continue with normal env step
                        env.step()
                
                custom_run()
                return
                
            # If we're only evaluating a trained model
            if EVALUATE_ONLY:
                print("Evaluating model with MPC controller...")
                if os.path.exists(CHECKPOINT_PATH):
                    # Modify evaluate_model to include vehicle detection
                    def evaluate_with_detector(model_path, env):
                        # Load the model
                        model = PPO.load(model_path)
                        
                        # Run evaluation with vehicle detection
                        obs = env.reset()
                        for _ in range(1000):  # Evaluate for 1000 steps
                            # Update vehicle detector
                            detector.detect_vehicles()
                            
                            # Get action from model
                            action, _ = model.predict(obs, deterministic=True)
                            obs, _, done, _ = env.step(action)
                            
                            if done:
                                obs = env.reset()
                    
                    evaluate_with_detector(CHECKPOINT_PATH, env)
                else:
                    print(f"Checkpoint not found at {CHECKPOINT_PATH}")
                return
            
            # Extend training to include vehicle detection
            # Wrap the env.step method to include vehicle detection
            original_step = env.step
            
            def step_with_detector(action):
                # Update vehicle detector
                detector.detect_vehicles()
                
                # Normal step
                return original_step(action)
            
            # Replace the step method with our wrapped version
            env.step = step_with_detector
            
            # Train the PPO model with MPC controller for lane keeping and path following
            model = train_ppo(
                env=env,  # Pass the CARLA environment with MPC
                num_envs=1,
                total_timesteps=1_000_000,
                hyperparameter_tuning=TUNE_HYPERPARAMETERS,
                continue_training=CONTINUE_TRAINING,
                checkpoint_path=CHECKPOINT_PATH if CONTINUE_TRAINING else None
            )
            
            # Evaluate the best model
            best_model_path = "models/best_model/best_model.zip"
            if os.path.exists(best_model_path):
                print("\nEvaluating best model...")
                evaluate_model(best_model_path, env=env)
                
        finally:
            # Clean up detector resources
            if detector is not None:
                detector.destroy()
        
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program terminated")
        # Make sure to clean up CARLA actors
        if 'env' in locals() and hasattr(env, 'cleanup_actors'):
            env.cleanup_actors()
        if 'env' in locals() and hasattr(env, 'cleanup_npcs'):
            env.cleanup_npcs()








# ./CarlaUE4.sh -carla-port=2000 -carla-tm-port=8000

#safety_controller.py


import os
import sys
import time
import numpy as np
import pygame
import carla
import random
import math

class SafetyController:
    def __init__(self, parent_actor, world, controller):
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.debug = True

        # MODIFIED: Adjusted detection parameters for more realistic stopping
        self.detection_distance = 70.0  # Keep detection distance high (70m)
        self.lane_width = 3.5
        self.last_detected = False

        self.recovery_mode = False
        self.recovery_start_time = None
        self.max_recovery_time = 5.0  # INCREASED from 3s to 5s to give more time to recover
        self.min_speed_threshold = 0.5  # m/s
        self.last_brake_time = None
        self.brake_cooldown = 1.0  # seconds

        # MODIFIED: Braking parameters for more realistic stopping
        self.time_to_collision_threshold = 5.0
        self.min_safe_distance = 8.0  # REDUCED from 15m to 8m
        self.emergency_brake_distance = 5.0  # REDUCED from 7m to 5m

        # MODIFIED: Gradual braking parameters with closer distances
        self.target_stop_distance = 5.0  # Stop 5m from obstacles, not 30m
        self.deceleration_start_distance = 30.0  # REDUCED from 40m to 30m
        self.deceleration_profile = [ # (distance, brake_intensity)
            (30.0, 0.03),  # REDUCED brake intensity at long distances
            (20.0, 0.10),  # REDUCED brake intensity at medium distances
            (15.0, 0.20),  # REDUCED brake intensity
            (10.0, 0.35),  # REDUCED brake intensity
            (7.0, 0.6),    # MODIFIED distance and intensity
            (5.0, 1.0)     # Full brake at 5m
        ]

        # MODIFIED: Traffic light parameters
        self.traffic_light_detection_distance = 100.0  # INCREASED from 70m to 100m
        self.traffic_light_detection_angle = 60.0  # INCREASED from 45° to 60° for better detection
        self.yellow_light_threshold = 2.5  # REDUCED from 3s to 2.5s
        self.traffic_light_slowdown_distance = 25.0  # Keep at 25m
        self.traffic_light_stop_distance = 3.0  # Keep at 3m
        self.override_timeout = 15.0  # INCREASED from 10s to 15s for more reliable stopping
        self.last_tl_brake_time = None
        self.tl_override_start_time = None
        self.is_tl_override_active = False
        self.green_light_resume_attempts = 0
        self.max_green_light_resume_attempts = 5  # INCREASED from 3 to 5 attempts
        self.last_green_light_time = None
        self.green_light_grace_period = 0.3  # REDUCED from 0.5s to 0.3s for faster response
        self.last_tl_state = None
        self.stuck_detection_timeout = 10.0  # NEW: Detect if we're stuck at a green light
        self.stuck_at_light = False  # NEW: Flag for stuck state

        # Create collision sensor
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

        print("Enhanced Safety controller initialized with improved obstacle and traffic light handling")


    def _on_collision(self, event):
        """Collision event handler"""
        print("!!! COLLISION DETECTED !!!")
        self._emergency_stop()

    def _calculate_time_to_collision(self, ego_velocity, other_velocity, distance):
        """Calculate time to collision based on relative velocity"""
        # Get velocity magnitudes
        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        other_speed = math.sqrt(other_velocity.x**2 + other_velocity.y**2)
        
        # If both vehicles are stationary or moving at same speed
        if abs(ego_speed - other_speed) < 0.1:
            return float('inf') if distance > self.emergency_brake_distance else 0
            
        # Calculate time to collision
        relative_speed = ego_speed - other_speed
        if relative_speed > 0:  # Only if we're moving faster than the other vehicle
            return distance / relative_speed
        return float('inf')

    def _is_in_same_lane(self, ego_location, ego_forward, other_location):
        """
        More accurate lane detection that handles curves by checking waypoints and road curvature
        """
        try:
            # Get the current waypoint of ego vehicle
            ego_waypoint = self._world.get_map().get_waypoint(ego_location)
            other_waypoint = self._world.get_map().get_waypoint(other_location)

            if not ego_waypoint or not other_waypoint:
                return True, 0  # Conservative approach if waypoints can't be found

            # Check if vehicles are on the same road and in the same lane
            same_road = (ego_waypoint.road_id == other_waypoint.road_id)
            same_lane = (ego_waypoint.lane_id == other_waypoint.lane_id)

            # Vector to other vehicle
            to_other = other_location - ego_location

            # Project onto forward direction for distance calculation
            forward_dist = (to_other.x * ego_forward.x + to_other.y * ego_forward.y)

            if forward_dist <= 0:  # Vehicle is behind
                return False, forward_dist

            # If they're on different roads/lanes, no need for further checks
            if not (same_road and same_lane):
                return False, forward_dist

            # For vehicles on the same lane, check if they're within reasonable distance
            # Get a series of waypoints ahead to account for road curvature
            next_waypoints = ego_waypoint.next(forward_dist)
            if not next_waypoints:
                return True, forward_dist  # Conservative approach if can't get waypoints

            # Find the closest waypoint to the other vehicle
            min_dist = float('inf')
            closest_wp = None

            for wp in next_waypoints:
                dist = wp.transform.location.distance(other_location)
                if dist < min_dist:
                    min_dist = dist
                    closest_wp = wp

            # If no waypoint found within reasonable distance, vehicles aren't in same lane
            if not closest_wp:
                return False, forward_dist

            # Check lateral distance from the predicted path
            # Use a more generous threshold in curves
            road_curvature = abs(ego_waypoint.transform.rotation.yaw - closest_wp.transform.rotation.yaw)
            lateral_threshold = self.lane_width * (0.5 + (road_curvature / 90.0) * 0.3)  # Increase threshold in curves

            is_in_lane = min_dist < lateral_threshold

            if self.debug and forward_dist > 0:
                color = carla.Color(0, 255, 0) if is_in_lane else carla.Color(128, 128, 128)
                self._world.debug.draw_line(
                    ego_location,
                    other_location,
                    thickness=0.1,
                    color=color,
                    life_time=0.1
                )
                # Draw predicted path for debugging
                for wp in next_waypoints:
                    self._world.debug.draw_point(
                        wp.transform.location,
                        size=0.1,
                        color=carla.Color(0, 0, 255),
                        life_time=0.1
                    )

            return is_in_lane, forward_dist

        except Exception as e:
            print(f"Lane check error: {str(e)}")
            return True, 0  # Conservative approach - assume in lane if error
    
    def _get_traffic_light_state(self):
        """
        Improved traffic light detection with better error handling and debugging
        """
        try:
            ego_location = self._parent.get_location()
            ego_transform = self._parent.get_transform()
            ego_forward = ego_transform.get_forward_vector()
            ego_waypoint = self._world.get_map().get_waypoint(ego_location)
            
            # Debug information for diagnosing detection issues
            if self.debug:
                self._world.debug.draw_arrow(
                    ego_location,
                    ego_location + ego_forward * 5.0,
                    thickness=0.2,
                    arrow_size=0.2,
                    color=carla.Color(0, 255, 255),
                    life_time=0.1
                )
                
            # Method 1: Direct API call (most reliable when available)
            traffic_light = self._parent.get_traffic_light()
            if traffic_light:
                light_location = traffic_light.get_location()
                distance = ego_location.distance(light_location)
                
                if self.debug:
                    print(f"Method 1: Direct API detected traffic light at {distance:.1f}m")
                    
                return traffic_light.get_state(), distance, traffic_light, light_location
            
            # Method 2: Current waypoint's traffic light
            if ego_waypoint and hasattr(ego_waypoint, 'get_traffic_light'):
                traffic_light = ego_waypoint.get_traffic_light()
                if traffic_light:
                    light_location = traffic_light.get_location()
                    distance = ego_location.distance(light_location)
                    
                    if self.debug:
                        print(f"Method 2: Waypoint detected traffic light at {distance:.1f}m")
                        
                    return traffic_light.get_state(), distance, traffic_light, light_location
            
            # Method 3: Scan upcoming waypoints with progress logging
            if ego_waypoint:
                next_wp = ego_waypoint
                distance_accumulated = 0
                scan_points = int(self.traffic_light_detection_distance / 2.0)
                
                if self.debug:
                    print(f"Method 3: Scanning {scan_points} waypoints ahead...")
                    
                for i in range(scan_points):
                    next_wps = next_wp.next(2.0)
                    if not next_wps:
                        if self.debug and i < 5:
                            print(f"Method 3: Path ends after {i} waypoints")
                        break
                        
                    next_wp = next_wps[0]
                    distance_accumulated += 2.0
                    
                    # Log waypoint positions for debugging
                    if self.debug and i % 5 == 0:
                        self._world.debug.draw_point(
                            next_wp.transform.location,
                            size=0.1,
                            color=carla.Color(0, 255, 0),
                            life_time=0.1
                        )
                    
                    # Check traffic light at this waypoint
                    if hasattr(next_wp, 'get_traffic_light'):
                        traffic_light = next_wp.get_traffic_light()
                        if traffic_light:
                            light_location = traffic_light.get_location()
                            distance = ego_location.distance(light_location)
                            
                            if self.debug:
                                print(f"Method 3: Found traffic light at waypoint {i}, distance {distance:.1f}m")
                                
                            return traffic_light.get_state(), distance, traffic_light, light_location
                    
                    # Check if at junction
                    if next_wp.is_junction:
                        if self.debug:
                            print(f"Method 3: Junction found at waypoint {i}")
            
            # Method 4: Direct search with improved angle calculation
            traffic_lights = self._world.get_actors().filter('traffic.traffic_light*')
            if self.debug:
                light_count = len(traffic_lights)
                print(f"Method 4: Scanning {light_count} traffic lights in the world")
                
            min_distance = float('inf')
            closest_light = None
            closest_light_loc = None
            
            for light in traffic_lights:
                light_loc = light.get_location()
                distance = ego_location.distance(light_loc)
                
                # Only check lights within detection distance
                if distance < self.traffic_light_detection_distance:
                    # Improved vector calculation
                    to_light = light_loc - ego_location
                    
                    # Normalize to avoid math errors
                    norm = math.sqrt(to_light.x**2 + to_light.y**2 + to_light.z**2)
                    if norm < 0.001:  # Avoid division by zero
                        continue
                        
                    to_light_normalized = carla.Vector3D(
                        to_light.x / norm,
                        to_light.y / norm,
                        to_light.z / norm
                    )
                    
                    # Calculate dot product with forward vector (cosine of angle)
                    forward_dot = ego_forward.x * to_light_normalized.x + ego_forward.y * to_light_normalized.y
                    
                    # Prevent math domain errors
                    clamped_dot = max(-1.0, min(1.0, forward_dot))
                    angle = math.acos(clamped_dot) * 180 / math.pi
                    
                    # Log all detected lights
                    if self.debug and angle < 90:  # Only log lights somewhat in front
                        self._world.debug.draw_line(
                            ego_location,
                            light_loc,
                            thickness=0.1,
                            color=carla.Color(255, 255, 0),
                            life_time=0.1
                        )
                        self._world.debug.draw_string(
                            light_loc + carla.Location(z=1.0),
                            f'Light: {angle:.1f}°, {distance:.1f}m',
                            color=carla.Color(255, 255, 0),
                            life_time=0.1
                        )
                    
                    # Check if light is within our detection angle and closer than any previously found
                    if angle < self.traffic_light_detection_angle and distance < min_distance:
                        # Additional verification for lane association
                        if ego_waypoint:
                            light_waypoint = self._world.get_map().get_waypoint(light_loc)
                            if light_waypoint and light_waypoint.road_id == ego_waypoint.road_id:
                                min_distance = distance
                                closest_light = light
                                closest_light_loc = light_loc
                                
                                if self.debug:
                                    print(f"Method 4: Verified light on same road {light_waypoint.road_id}, angle {angle:.1f}°")
                        else:
                            # If we don't have ego waypoint, be more permissive
                            min_distance = distance
                            closest_light = light
                            closest_light_loc = light_loc
            
            if closest_light:
                if self.debug:
                    print(f"Method 4: Found closest traffic light at {min_distance:.1f}m")
                    
                    # Draw debug visualization
                    color = carla.Color(255, 255, 0)  # Yellow by default
                    if closest_light.get_state() == carla.TrafficLightState.Red:
                        color = carla.Color(255, 0, 0)  # Red
                    elif closest_light.get_state() == carla.TrafficLightState.Green:
                        color = carla.Color(0, 255, 0)  # Green
                    
                    self._world.debug.draw_line(
                        ego_location,
                        closest_light_loc,
                        thickness=0.3,
                        color=color,
                        life_time=0.1
                    )
                    self._world.debug.draw_point(
                        closest_light_loc,
                        size=0.3,
                        color=color,
                        life_time=0.1
                    )
                    self._world.debug.draw_string(
                        closest_light_loc + carla.Location(z=2.0),
                        f'Traffic Light: {closest_light.get_state()}, {min_distance:.1f}m',
                        color=color,
                        life_time=0.1
                    )
                
                return closest_light.get_state(), min_distance, closest_light, closest_light_loc
            
            # No traffic light found
            if self.debug:
                print("No traffic light detected by any method")
                
            return None, float('inf'), None, None
            
        except Exception as e:
            print(f"ERROR in traffic light detection: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Create a visual warning in the world
            if self.debug:
                self._world.debug.draw_string(
                    ego_location + carla.Location(z=3.0),
                    f'!!! TRAFFIC LIGHT DETECTION ERROR !!!',
                    color=carla.Color(255, 0, 0),
                    life_time=0.5
                )
                
            return None, float('inf'), None, None
    
    def _handle_traffic_light(self):
        """Enhanced traffic light handling with more robust stopping and resuming"""
        light_state, distance, light_actor, light_location = self._get_traffic_light_state()
        current_time = time.time()
        
        # Get current speed
        ego_velocity = self._parent.get_velocity()
        speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        speed_kmh = 3.6 * speed  # Convert to km/h for display
        
        # NEW: Store the last traffic light state for comparison
        previous_state = self.last_tl_state
        self.last_tl_state = light_state
        
        # ENHANCED: If no traffic light is detected, check if we need to release override
        if light_state is None or distance == float('inf'):
            # Reset traffic light override if we're no longer detecting a light
            if self.is_tl_override_active:
                # NEW: Check if we're stuck (stopped but no light)
                if speed < 0.5 and current_time - self.tl_override_start_time > self.stuck_detection_timeout:
                    print("WARNING: Stuck in traffic with no light detected. Forcing resume...")
                    self._force_resume_path()
                    self.is_tl_override_active = False
                    self.tl_override_start_time = None
                    self.stuck_at_light = False
                    return False
                    
                if (current_time - self.tl_override_start_time > self.override_timeout):
                    print("Releasing traffic light override (timeout)")
                    self.is_tl_override_active = False
                    self.tl_override_start_time = None
                    self._force_resume_path()  # Force resume
            return False
        
        # Convert to carla.TrafficLightState object if needed
        if isinstance(light_state, str):
            if light_state == "Red":
                light_state = carla.TrafficLightState.Red
            elif light_state == "Yellow":
                light_state = carla.TrafficLightState.Yellow
            elif light_state == "Green":
                light_state = carla.TrafficLightState.Green
        
        # Handle different light states
        if light_state == carla.TrafficLightState.Red:
            # Red light - we should come to a stop
            if distance < self.traffic_light_slowdown_distance:
                # Reset green light counter
                self.green_light_resume_attempts = 0
                self.last_green_light_time = None
                self.stuck_at_light = False
                
                # Calculate stopping distance
                stopping_distance = distance - self.traffic_light_stop_distance
                
                # Activate traffic light override
                if not self.is_tl_override_active:
                    self.is_tl_override_active = True
                    self.tl_override_start_time = time.time()
                    print(f"\n!!! RED LIGHT DETECTED - Distance: {distance:.1f}m, Speed: {speed_kmh:.1f} km/h !!!")
                
                if stopping_distance <= 0:
                    # We've reached or passed the stopping point
                    self._emergency_stop()
                    print(f"RED LIGHT STOP - Distance: {distance:.1f}m")
                else:
                    # Apply gradual braking based on distance
                    brake_intensity = self._calculate_brake_intensity(stopping_distance)
                    
                    # Increase braking intensity for red lights
                    brake_intensity = min(1.0, brake_intensity * 1.5)
                    
                    print(f"RED LIGHT BRAKING - Distance: {distance:.1f}m, Brake: {brake_intensity:.2f}")
                    control = carla.VehicleControl(
                        throttle=0.0,
                        brake=brake_intensity,
                        steer=self._maintain_path_steer(),  # Keep steering while braking
                        hand_brake=False
                    )
                    self._parent.apply_control(control)
                    self.last_tl_brake_time = time.time()
                return True
        
        elif light_state == carla.TrafficLightState.Yellow:
            # ENHANCED Yellow light handling with more reliable stopping
            time_to_light = distance / max(speed, 0.1)  # Avoid division by zero
            
            # MODIFIED: More aggressive yellow light handling
            if time_to_light > self.yellow_light_threshold or distance < 15.0:
                # We can't clear the intersection in time, stop
                if distance < self.traffic_light_slowdown_distance:
                    # Reset green light counter
                    self.green_light_resume_attempts = 0
                    self.last_green_light_time = None
                    self.stuck_at_light = False
                    
                    # Activate traffic light override
                    if not self.is_tl_override_active:
                        self.is_tl_override_active = True
                        self.tl_override_start_time = time.time()
                        print(f"\n!!! YELLOW LIGHT DETECTED (stopping) - Distance: {distance:.1f}m, Time: {time_to_light:.1f}s !!!")
                    
                    # Calculate stopping distance
                    stopping_distance = distance - self.traffic_light_stop_distance
                    
                    if stopping_distance <= 0:
                        # We've reached or passed the stopping point
                        self._emergency_stop()
                        print(f"YELLOW LIGHT STOP - Distance: {distance:.1f}m")
                    else:
                        # Apply gradual braking based on distance
                        brake_intensity = self._calculate_brake_intensity(stopping_distance)
                        # INCREASED braking for yellow lights
                        brake_intensity = min(1.0, brake_intensity * 1.3)
                        
                        print(f"YELLOW LIGHT BRAKING - Distance: {distance:.1f}m, Brake: {brake_intensity:.2f}")
                        control = carla.VehicleControl(
                            throttle=0.0,
                            brake=brake_intensity,
                            steer=self._maintain_path_steer(),  # Keep steering while braking
                            hand_brake=False
                        )
                        self._parent.apply_control(control)
                        self.last_tl_brake_time = time.time()
                    return True
            else:
                # We can clear the intersection, proceed with caution
                if self.is_tl_override_active:
                    self.is_tl_override_active = False
                    self.tl_override_start_time = None
                print(f"YELLOW LIGHT (proceeding) - Distance: {distance:.1f}m, Time: {time_to_light:.1f}s")
                return False
        
        elif light_state == carla.TrafficLightState.Green:
            # ENHANCED: Improved green light handling with stuck detection
            
            # Check if we're coming from a different state to green
            state_change_to_green = (previous_state != carla.TrafficLightState.Green and 
                                    light_state == carla.TrafficLightState.Green)
            
            # Track when we first see the green light
            if state_change_to_green:
                self.last_green_light_time = current_time
                print(f"\n!!! GREEN LIGHT DETECTED - Distance: {distance:.1f}m !!!")
            
            # If we were stopped for a red/yellow light and now it's green
            if self.is_tl_override_active:
                # Wait for a short grace period before trying to resume
                if self.last_green_light_time and (current_time - self.last_green_light_time) >= self.green_light_grace_period:
                    # Increment resume attempt counter
                    self.green_light_resume_attempts += 1
                    
                    # ENHANCED: Check if we're stuck at a green light
                    if speed < 0.5 and self.green_light_resume_attempts > 2:
                        self.stuck_at_light = True
                        
                    print(f"GREEN LIGHT - Resuming operation (attempt {self.green_light_resume_attempts})")
                    
                    # NEW: Apply increasing throttle based on number of attempts
                    throttle_value = min(0.7 + (self.green_light_resume_attempts * 0.05), 1.0)
                    
                    # Force resume with stronger throttle if we're stuck
                    if self.stuck_at_light:
                        control = carla.VehicleControl(
                            throttle=1.0,  # Maximum throttle to get unstuck
                            brake=0.0,
                            steer=self._maintain_path_steer(),
                            hand_brake=False
                        )
                        self._parent.apply_control(control)
                        print("STUCK AT GREEN LIGHT - Applying maximum throttle")
                    else:
                        # Normal resume
                        control = carla.VehicleControl(
                            throttle=throttle_value,
                            brake=0.0,
                            steer=self._maintain_path_steer(),
                            hand_brake=False
                        )
                        self._parent.apply_control(control)
                    
                    # Only release override after we're moving or max attempts reached
                    if speed > 1.0 or self.green_light_resume_attempts >= self.max_green_light_resume_attempts:
                        self.is_tl_override_active = False
                        self.tl_override_start_time = None
                        self.green_light_resume_attempts = 0
                        self.stuck_at_light = False
                        print(f"GREEN LIGHT - Successfully resumed normal operation")
                    
                    # Always return False on green to allow controller to work
                    return False
            else:
                # Already moving - no need to do anything special
                self.green_light_resume_attempts = 0
                self.stuck_at_light = False
            
            return False
        
        # Reset override for unknown light state
        if self.is_tl_override_active:
            self.is_tl_override_active = False
            self.tl_override_start_time = None
        
        return False
    
    def _force_resume_path(self):
        """Force vehicle to resume movement after stopping for a traffic light"""
        try:
            # Reset brake and recovery state
            self.last_brake_time = None
            self.recovery_mode = False
            self.recovery_start_time = None

            # ENHANCED: Get current velocity to determine needed thrust
            ego_velocity = self._parent.get_velocity()
            current_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            # Apply stronger initial acceleration when completely stopped
            throttle_value = 0.9 if current_speed < 0.5 else 0.7

            control = carla.VehicleControl(
                throttle=throttle_value,
                brake=0.0,
                steer=self._maintain_path_steer(),
                hand_brake=False
            )
            self._parent.apply_control(control)

            # NEW: Apply control multiple times to overcome inertia
            for _ in range(3):
                self._parent.apply_control(control)
                time.sleep(0.05)  # Short delay between control applications

            # Reset controller state if method exists
            if hasattr(self._controller, '_reset_control_state'):
                self._controller._reset_control_state()

        except Exception as e:
            print(f"Error forcing resume: {str(e)}")
    
    def _maintain_path_steer(self):
        """Get steering value to maintain path while braking"""
        try:
            if not self._controller:
                return 0.0
                
            if hasattr(self._controller, 'waypoints') and hasattr(self._controller, 'current_waypoint_index'):
                if self._controller.current_waypoint_index < len(self._controller.waypoints):
                    target_wp = self._controller.waypoints[self._controller.current_waypoint_index]
                    vehicle_transform = self._parent.get_transform()
                    
                    if hasattr(self._controller, '_calculate_steering'):
                        return self._controller._calculate_steering(vehicle_transform, target_wp.transform)
            
            return 0.0
        
        except Exception as e:
            print(f"Error calculating steering: {str(e)}")
            return 0.0
    
    def _calculate_brake_intensity(self, distance):
        """Calculate brake intensity based on distance to obstacle/traffic light"""
        # Find the appropriate braking level from the deceleration profile
        for dist, brake in self.deceleration_profile:
            if distance <= dist:
                return brake
        
        return 0.0  # No braking needed
    
    def _apply_gradual_braking(self, distance, ttc):
        """Apply gradual braking based on distance and time to collision"""
        # ENHANCED: Emergency stop for very close obstacles
        if distance < 3.0 or ttc < 0.5:
            print(f"!!! EMERGENCY STOP !!! Distance: {distance:.1f}m, TTC: {ttc:.1f}s")
            self._emergency_stop()
            return True

        # MODIFIED: More realistic and appropriate braking intensity
        brake_intensity = self._calculate_brake_intensity(distance)

        # If TTC is very small, increase braking
        if ttc < 2.0:
            brake_intensity = max(brake_intensity, 0.8)
        elif ttc < 3.0:
            brake_intensity = max(brake_intensity, 0.5)

        if brake_intensity > 0:
            ego_velocity = self._parent.get_velocity()
            speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)  # km/h

            # NEW: Don't brake too hard at moderate distances
            if distance > 15.0 and brake_intensity > 0.2:
                brake_intensity = 0.2

            print(f"Gradual braking - Distance: {distance:.1f}m, TTC: {ttc:.1f}s, Speed: {speed:.1f} km/h, Brake: {brake_intensity:.2f}")

            # ENHANCED: Apply some throttle at long distances to prevent premature stopping
            throttle = 0.0
            if distance > 20.0 and brake_intensity < 0.3:
                throttle = 0.1

            control = carla.VehicleControl(
                throttle=throttle,
                brake=brake_intensity,
                steer=self._maintain_path_steer(),  # Keep steering while braking
                hand_brake=False
            )
            self._parent.apply_control(control)
            return True

        return False
    
    def check_safety(self):
        """Enhanced safety checking with prioritized traffic light handling"""
        try:
            # First check traffic lights - THIS MUST TAKE PRIORITY
            if self._handle_traffic_light():
                # If we're handling a traffic light, skip other checks
                return

            # Only check for other vehicles if we're not dealing with a traffic light
            all_vehicles = self._world.get_actors().filter('vehicle.*')
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            ego_velocity = self._parent.get_velocity()

            # NEW: Also check for pedestrians
            all_pedestrians = self._world.get_actors().filter('walker.*')

            detected_obstacles = []
            min_distance = float('inf')
            min_ttc = float('inf')  # Time to collision

            # Current speed in km/h
            speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            # Check for vehicles
            for vehicle in all_vehicles:
                if vehicle.id == self._parent.id:
                    continue

                vehicle_location = vehicle.get_location()
                distance = ego_location.distance(vehicle_location)

                # Check vehicles within detection range
                if distance < self.detection_distance:
                    is_in_lane, forward_dist = self._is_in_same_lane(ego_location, ego_forward, vehicle_location)

                    if is_in_lane and forward_dist > 0:  # Only consider vehicles ahead
                        other_velocity = vehicle.get_velocity()
                        ttc = self._calculate_time_to_collision(ego_velocity, other_velocity, forward_dist)

                        detected_obstacles.append((vehicle, forward_dist, ttc))
                        min_distance = min(min_distance, forward_dist)
                        min_ttc = min(min_ttc, ttc)

                        if self.debug:
                            self._world.debug.draw_box(
                                vehicle.bounding_box,
                                vehicle.get_transform().rotation,
                                thickness=0.5,
                                color=carla.Color(255, 0, 0, 255),
                                life_time=0.1
                            )
                            self._world.debug.draw_string(
                                vehicle_location + carla.Location(z=2.0),
                                f'!!! {forward_dist:.1f}m, TTC: {ttc:.1f}s !!!',
                                color=carla.Color(255, 0, 0, 255),
                                life_time=0.1
                            )

            # NEW: Check for pedestrians
            for pedestrian in all_pedestrians:
                pedestrian_location = pedestrian.get_location()
                distance = ego_location.distance(pedestrian_location)

                # Use a wider detection range for pedestrians
                if distance < self.detection_distance:
                    # For pedestrians, be more cautious - use a wider detection area
                    to_ped = pedestrian_location - ego_location
                    forward_dot = ego_forward.x * to_ped.x + ego_forward.y * to_ped.y

                    # Only consider pedestrians somewhat ahead of us (allow for wider angle)
                    if forward_dot > 0:
                        # Calculate lateral distance to be more cautious with pedestrians
                        lateral_vector = to_ped - ego_forward * forward_dot
                        lateral_distance = math.sqrt(lateral_vector.x**2 + lateral_vector.y**2)

                        # Use a wider lane width for pedestrians
                        if lateral_distance < self.lane_width * 1.5:
                            other_velocity = pedestrian.get_velocity()
                            ttc = self._calculate_time_to_collision(ego_velocity, other_velocity, forward_dot)

                            # Be extra cautious with pedestrians - reduce TTC
                            ttc = ttc * 0.7

                            detected_obstacles.append((pedestrian, forward_dot, ttc))
                            min_distance = min(min_distance, forward_dot)
                            min_ttc = min(min_ttc, ttc)

                            if self.debug:
                                self._world.debug.draw_point(
                                    pedestrian_location,
                                    size=0.2,
                                    color=carla.Color(255, 0, 0, 255),
                                    life_time=0.1
                                )
                                self._world.debug.draw_string(
                                    pedestrian_location + carla.Location(z=2.0),
                                    f'!!! Pedestrian {forward_dot:.1f}m, TTC: {ttc:.1f}s !!!',
                                    color=carla.Color(255, 0, 0, 255),
                                    life_time=0.1
                                )

            if detected_obstacles:
                print(f"\nObstacle detected - Distance: {min_distance:.1f}m, TTC: {min_ttc:.1f}s, Speed: {speed:.1f} km/h")

                # Apply gradual braking based on distance and TTC
                if self._apply_gradual_braking(min_distance, min_ttc):
                    self.last_detected = True
                else:
                    # Not braking, proceed normally
                    if self.last_detected:
                        print("Path clear - resuming normal operation")
                        self._resume_path()
                    self.last_detected = False
            else:
                if self.last_detected:
                    print("Path clear - resuming normal operation")
                    self._resume_path()
                self.last_detected = False

            # Update controller with obstacles
            if self._controller and hasattr(self._controller, 'update_obstacles'):
                self._controller.update_obstacles([v[0].get_location() for v in detected_obstacles])

        except Exception as e:
            print(f"Error in safety check: {str(e)}")
            import traceback
            traceback.print_exc()
            self._emergency_stop()

    def _emergency_stop(self):
        """Maximum braking force"""
        control = carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True,
            steer=self._maintain_path_steer()  # Keep steering while emergency braking
        )
        self._parent.apply_control(control)
        self.last_brake_time = time.time()

    def _maintain_path(self):
        """Enhanced path maintenance during braking"""
        try:
            if not self._controller:
                return
                
            current_time = time.time()
            
            # Check if we need to enter recovery mode
            if not self.recovery_mode:
                velocity = self._parent.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                
                if speed < self.min_speed_threshold:
                    self.recovery_mode = True
                    self.recovery_start_time = current_time
                    print("Entering path recovery mode")
            
            # Handle recovery mode
            if self.recovery_mode:
                if current_time - self.recovery_start_time > self.max_recovery_time:
                    self.recovery_mode = False
                    print("Exiting recovery mode - timeout")
                else:
                    # Force path recovery
                    if hasattr(self._controller, 'force_path_recovery'):
                        control = self._controller.force_path_recovery(self._parent)
                        self._parent.apply_control(control)
                    return
            
            # Normal path maintenance
            if hasattr(self._controller, 'waypoints') and hasattr(self._controller, 'current_waypoint_index'):
                if self._controller.current_waypoint_index < len(self._controller.waypoints):
                    target_wp = self._controller.waypoints[self._controller.current_waypoint_index]
                    vehicle_transform = self._parent.get_transform()
                    
                    if hasattr(self._controller, '_calculate_steering'):
                        steer = self._controller._calculate_steering(vehicle_transform, target_wp.transform)
                        
                        # Get current control and maintain brake while updating steering
                        control = self._parent.get_control()
                        control.steer = steer
                        self._parent.apply_control(control)
            
        except Exception as e:
            print(f"Error in path maintenance: {str(e)}")

    def _resume_path(self):
        """Improved path resumption after braking"""
        try:
            current_time = time.time()

            # Check if we're still in brake cooldown
            if self.last_brake_time and current_time - self.last_brake_time < self.brake_cooldown:
                # Maintain current path but release brake gradually
                control = self._parent.get_control()
                control.brake *= 0.3  # MORE AGGRESSIVE brake release (0.5 → 0.3)
                control.throttle = 0.2  # Add some throttle to ensure movement
                self._parent.apply_control(control)
                return

            # Reset recovery mode
            self.recovery_mode = False
            self.recovery_start_time = None

            # Reset controller state if method exists
            if hasattr(self._controller, '_reset_control_state'):
                self._controller._reset_control_state()

            # NEW: If car seems stuck (not moving), apply more thrust
            ego_velocity = self._parent.get_velocity()
            speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)

            if speed < 0.5:  # Car is basically stopped
                control = carla.VehicleControl(
                    throttle=0.7,  # Strong throttle to get moving
                    brake=0.0,
                    steer=self._maintain_path_steer(),
                    hand_brake=False
                )
                self._parent.apply_control(control)
                print("Vehicle stopped - applying extra throttle to resume")
                time.sleep(0.1)  # Short delay

            # Resume normal control
            if hasattr(self._controller, 'get_control'):
                control = self._controller.get_control(self._parent, self._world)
                self._parent.apply_control(control)

        except Exception as e:
            print(f"Error resuming path: {str(e)}")

    def destroy(self):
        """Clean up sensors"""
        if hasattr(self, 'collision_sensor'):
            self.collision_sensor.destroy()