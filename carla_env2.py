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
    

