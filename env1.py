
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
def download_weights(url, save_path):
    """Download weights if they don't exist"""
    if not os.path.exists(save_path):
        print(f"Downloading YOLOv5 weights to {save_path}")
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Add YOLOv5 to path
yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# Now import YOLOv5 modules
try:
    try:
        from models.experimental import attempt_load
        print("YOLOv5 modules imported successfully!")
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

def calculate_depth_and_angle(bbox, class_id, image_width=IM_WIDTH, image_height=IM_HEIGHT, fov=90):
    """
    Calculate depth and angle of objects using bounding box and class-specific dimensions
    
    Returns:
        depth: estimated distance to object in meters
        angle: angle from center of view in degrees (-45 to +45 for 90 degree FOV)
        confidence: confidence level of the estimate (0-1)
    """
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    
    # Center point of the bbox in image coordinates
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Calculate angle from center of image (in degrees)
    # Center of image is 0 degrees, left is negative, right is positive
    angle_x = ((center_x / image_width) - 0.5) * fov
    
    # Define typical dimensions for different object classes (in meters)
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
        
        # Adjust depth based on vertical position (objects higher in frame tend to be farther)
        # This is a heuristic adjustment that can be fine-tuned
        vertical_position_factor = 1.0 + max(0, (image_height/2 - center_y) / image_height)
        depth *= vertical_position_factor
    else:
        depth = float('inf')
    
    # Calculate confidence level
    # Larger objects (both in absolute size and relative to frame) give higher confidence
    size_confidence = min(1.0, (bbox_width * bbox_height) / (image_width * image_height * 0.25))
    # Objects in the center of frame have higher confidence
    position_confidence = 1.0 - min(1.0, 2.0 * abs(angle_x) / fov)
    # Combined confidence
    confidence = 0.7 * size_confidence + 0.3 * position_confidence
    
    return depth, angle_x, confidence


def calculate_relative_position(depth, angle_x):
    """
    Calculate relative 3D position from depth and angle
    
    Args:
        depth: distance to object in meters
        angle_x: horizontal angle from center in degrees
    
    Returns:
        x, y positions in vehicle coordinate system (meters)
        x is forward, y is right
    """
    # Convert angle to radians
    angle_rad = math.radians(angle_x)
    
    # Calculate x (forward) and y (right) components
    x = depth * math.cos(angle_rad)  # forward distance
    y = depth * math.sin(angle_rad)  # lateral distance (positive is right)
    
    return x, y

def classify_obstacle(obj_class, depth, angle_x, vehicle_speed):
    """
    Classify obstacle based on type, distance, angle and vehicle speed
    
    Returns:
        risk_level: 0-10 scale (0 = no risk, 10 = immediate collision risk)
        action: recommended action ('continue', 'slow', 'stop', 'avoid_left', 'avoid_right')
    """
    # High-risk classes (pedestrians, cyclists) have lower safety thresholds
    high_risk_classes = [0, 1, 2]  # person, bicycle, motorcycle
    medium_risk_classes = [3, 4, 5, 6]  # car, truck, bus, train
    
    # Base risk is affected by object class
    base_risk = 7 if obj_class in high_risk_classes else 5 if obj_class in medium_risk_classes else 3
    
    # Distance factor (closer = higher risk)
    # Adjust these thresholds based on your simulation needs
    if depth < 5:
        distance_factor = 3.0
    elif depth < 10:
        distance_factor = 2.0
    elif depth < 20:
        distance_factor = 1.0
    elif depth < 30:
        distance_factor = 0.5
    else:
        distance_factor = 0.2
    
    # Angle factor (objects directly ahead are higher risk)
    if abs(angle_x) < 10:
        angle_factor = 1.0  # directly ahead
    elif abs(angle_x) < 20:
        angle_factor = 0.8  # slightly off-center
    elif abs(angle_x) < 30:
        angle_factor = 0.6  # moderately off-center
    else:
        angle_factor = 0.4  # significantly off-center
    
    # Speed factor (faster = higher risk)
    speed_factor = min(1.5, max(0.5, vehicle_speed / 30.0))
    
    # Calculate final risk level (0-10 scale)
    risk_level = min(10, base_risk * distance_factor * angle_factor * speed_factor)
    
    # Determine recommended action
    if risk_level > 8:
        action = 'stop'
    elif risk_level > 6:
        action = 'slow'
    elif risk_level > 4:
        # Choose avoidance direction based on obstacle position
        if angle_x < 0:
            action = 'avoid_right'  # obstacle on left, avoid to right
        else:
            action = 'avoid_left'   # obstacle on right, avoid to left
    else:
        action = 'continue'
    
    return risk_level, action

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

                # Process YOLO detection with enhanced positioning
                detections = self.process_yolo_detection(array)

                # Create pygame surface
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

                # Draw center reference line
                pygame.draw.line(surface, (255, 255, 255), 
                                (IM_WIDTH//2, IM_HEIGHT-50), 
                                (IM_WIDTH//2, IM_HEIGHT-10), 2)

                # Draw detections with enhanced information
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

                    # Color based on risk level (green->yellow->red)
                    risk = obj['risk_level']
                    if risk < 3:
                        color = (0, 255, 0)  # green - low risk
                    elif risk < 7:
                        color = (255, 255, 0)  # yellow - medium risk
                    else:
                        color = (255, 0, 0)  # red - high risk

                    # Draw rectangle with thickness based on risk
                    thickness = max(1, min(4, int(risk / 2.5)))
                    pygame.draw.rect(surface, color, (x1, y1, x2-x1, y2-y1), thickness)

                    # Draw line from bottom center to show angle
                    center_x = (x1 + x2) // 2
                    bottom_y = y2
                    # Calculate projection point 
                    line_length = 40
                    proj_x = center_x + int(math.sin(math.radians(obj['angle'])) * line_length)
                    pygame.draw.line(surface, color, (center_x, bottom_y), (proj_x, bottom_y + line_length), 2)

                    # Draw label with enhanced information
                    if pygame.font.get_init():
                        font = pygame.font.Font(None, 24)
                        # Format angle display with sign
                        angle_str = f"{obj['angle']:.1f}°"
                        if obj['angle'] > 0:
                            angle_str = "+" + angle_str  # Explicitly show + for right

                        # Create label with class, depth and angle
                        label = f"{obj['class_name']} {obj['depth']:.1f}m {angle_str}"
                        text = font.render(label, True, (255, 255, 255))
                        surface.blit(text, (x1, y1-20))

                        # Add second line with action recommendation for high-risk objects
                        if risk > 4:
                            action_text = font.render(f"Action: {obj['recommended_action'].upper()}", True, color)
                            surface.blit(action_text, (x1, y1-40))

                # Update display
                if self.display is not None:
                    self.display.blit(surface, (0, 0))
                    pygame.display.flip()

            except Exception as e:
                print(f"Error in image processing: {e}")
                traceback.print_exc()

    def reset(self):
        """Reset environment with improved movement verification"""
        print("Starting environment reset...")
        self.cleanup_actors()
        self.cleanup_npcs()
        self.collision_hist = []  # Clear collision history
        self.stuck_time = 0
        self.episode_start = time.time()
        self.last_location = None

        # Setup new episode
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

        return state

    def setup_vehicle(self):
        """Spawn and setup the ego vehicle with MPC controller"""
        try:
            print("Starting vehicle setup...")

            # Get the blueprint library
            blueprint_library = self.world.get_blueprint_library()
            print("Got blueprint library")

            # Get the vehicle blueprint
            vehicle_bp = blueprint_library.filter('model3')[0]
            print("Got vehicle blueprint")

            # Get random spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points found")
            spawn_point = random.choice(spawn_points)
            print(f"Selected spawn point: {spawn_point}")

            # Spawn the vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
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
        """Initialize YOLO model"""
        try:
            print("Loading YOLOv5 model...")
            
            # Define paths
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5s.pt')
            
            # Download weights if needed
            weights_url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt'
            download_weights(weights_url, weights_path)
            
            print(f"Loading model from: {weights_path}")
            
            # Import YOLOv5
            import torch
            from models.experimental import attempt_load
            
            # Load the model
            self.yolo_model = attempt_load(weights_path, device=DEVICE)
            
            # Configure model settings
            self.yolo_model.conf = 0.25  # Confidence threshold
            self.yolo_model.iou = 0.45   # NMS IoU threshold
            self.yolo_model.classes = None  # Detect all classes
            self.yolo_model.eval()
            
            print("YOLOv5 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def process_yolo_detection(self, image):
        """Process image with YOLO and return enhanced detections with positioning"""
        if image is None:
            return []

        try:
            # Prepare image for YOLO
            img = cv2.resize(image, (640, 640))
            img = img.transpose((2, 0, 1))  # HWC to CHW
            img = np.ascontiguousarray(img)

            # Convert to torch tensor
            img = torch.from_numpy(img).to(DEVICE)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0

            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            with torch.no_grad():
                pred = self.yolo_model(img)[0]

            # Apply NMS
            from utils.general import non_max_suppression
            pred = non_max_suppression(pred, 
                                     conf_thres=0.25,
                                     iou_thres=0.45,
                                     classes=None,
                                     agnostic=False,
                                     max_det=300)

            objects = []
            if len(pred[0]):
                # Get vehicle speed for risk assessment
                v = self.vehicle.get_velocity()
                speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)  # km/h

                # Process detections with enhanced positioning
                for *xyxy, conf, cls in pred[0]:
                    x1, y1, x2, y2 = map(float, xyxy)

                    # Get enhanced depth, angle and confidence
                    depth, angle_x, depth_confidence = calculate_depth_and_angle([x1, y1, x2, y2], int(cls))

                    # Calculate relative position in vehicle coordinate system
                    forward_dist, lateral_dist = calculate_relative_position(depth, angle_x)

                    # Classify obstacle risk and recommended action
                    risk_level, action = classify_obstacle(int(cls), depth, angle_x, speed)

                    objects.append({
                        'position': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'depth': depth,
                        'angle': angle_x,
                        'forward_distance': forward_dist,  # distance ahead of vehicle
                        'lateral_distance': lateral_dist,  # distance to side (+ is right, - is left)
                        'depth_confidence': depth_confidence,
                        'class': int(cls),
                        'class_name': self.yolo_model.names[int(cls)],
                        'confidence': float(conf),
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1,
                        'risk_level': risk_level,
                        'recommended_action': action
                    })

            # Sort by risk level (highest risk first)
            objects.sort(key=lambda x: x['risk_level'], reverse=True)
            return objects[:self.max_objects]

        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    
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
    
    # Update the get_state method in CarEnv class
    def get_state(self):
        """Get complete state information with enhanced object features"""
        if self.front_camera is None:
            return None

        # Initialize state array
        state_array = []

        # 1. Process YOLO detections with enhanced features (max_objects * 6 features)
        detections = self.process_yolo_detection(self.front_camera)
        for obj in detections:
            # Convert action to numeric code
            action_code = 0  # continue
            if obj['recommended_action'] == 'slow':
                action_code = 1
            elif obj['recommended_action'] == 'stop':
                action_code = 2
            elif obj['recommended_action'] == 'avoid_left':
                action_code = 3
            elif obj['recommended_action'] == 'avoid_right':
                action_code = 4

            state_array.extend([
                obj['position'][0] / IM_WIDTH,         # x position normalized
                obj['position'][1] / IM_HEIGHT,        # y position normalized
                obj['depth'] / 100.0,                  # depth normalized
                obj['angle'] / 45.0,                   # angle normalized (-1 to 1)
                obj['risk_level'] / 10.0,              # risk level normalized (0 to 1)
                action_code / 4.0                      # action code normalized
            ])

        # Pad if fewer than max objects
        remaining_objects = self.max_objects - len(detections)
        state_array.extend([0.0] * (remaining_objects * 6))

        # 2. Get waypoint information
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

        return np.array(state_array, dtype=np.float32)
    

    def calculate_reward(self):
        """Reward function that incorporates risk levels from detected objects"""
        try:
            reward = 0.0
            done = False
            info = {}

            # Get current state
            velocity = self.vehicle.get_velocity()
            speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            location = self.vehicle.get_location()

            # Base reward for staying alive
            reward += 0.1

            # Get object detections and analyze risks
            detections = self.process_yolo_detection(self.front_camera)

            # Calculate risk-based reward component
            highest_risk = 0
            highest_risk_object = None
            risk_penalty = 0

            for obj in detections:
                # Track highest risk object
                if obj['risk_level'] > highest_risk:
                    highest_risk = obj['risk_level']
                    highest_risk_object = obj

                # Accumulate risk penalty based on risk level and depth
                # Higher risk and closer objects create larger penalties
                risk_factor = obj['risk_level'] / 10.0  # Normalize to 0-1
                depth_factor = max(0.1, min(1.0, 10.0 / max(1.0, obj['depth'])))
                object_penalty = risk_factor * depth_factor

                # Larger penalty for high-risk classes like pedestrians
                if obj['class'] in [0, 1, 2]:  # person, bicycle, motorcycle
                    object_penalty *= 1.5

                risk_penalty += object_penalty

            # Apply risk penalty with a scaling factor (adjustable)
            risk_scaling = 2.0
            reward -= risk_penalty * risk_scaling

            # Reward for appropriate behavior near obstacles
            if highest_risk > 7:  # High risk situation
                if speed < 5.0:  # Vehicle is slowing/stopping
                    reward += 3.0  # Big reward for appropriate caution
                else:
                    reward -= 5.0  # Big penalty for unsafe speed

            elif highest_risk > 4:  # Medium risk situation
                if speed < 15.0:  # Vehicle is moderately slowing
                    reward += 1.5  # Moderate reward for caution
                else:
                    reward -= 2.0  # Moderate penalty

            # If no high risks, reward normal driving speed
            elif highest_risk < 3:
                target_speed = 30.0  # Target speed in km/h
                if speed < 1.0:  # Almost stopped without reason
                    reward -= 0.5
                    self.stuck_time += 0.1
                    if self.stuck_time > 3.0:
                        done = True
                        reward -= 10.0
                        info['termination_reason'] = 'stuck'
                else:
                    self.stuck_time = 0
                    # Reward for moving at appropriate speed
                    speed_reward = -abs(speed - target_speed) / target_speed
                    reward += speed_reward

            # Distance traveled reward
            if self.last_location is not None:
                distance_traveled = location.distance(self.last_location)
                reward += distance_traveled * 0.5  # Reward for covering distance

            # Collision penalty
            if len(self.collision_hist) > 0:
                reward -= 50.0
                done = True
                info['termination_reason'] = 'collision'

            # Lane keeping reward
            waypoint = self.world.get_map().get_waypoint(location)
            distance_from_center = location.distance(waypoint.transform.location)
            if distance_from_center < 1.0:
                reward += 0.2
            else:
                # Penalty increases with distance from center
                lane_penalty = min(5.0, distance_from_center * 0.5)
                reward -= lane_penalty

            # Store current location for next step
            self.last_location = location

            # Store detailed information for debugging
            info['speed'] = speed
            info['reward'] = reward
            info['distance_from_center'] = distance_from_center
            info['highest_risk'] = highest_risk
            if highest_risk_object:
                info['highest_risk_object'] = {
                    'class': highest_risk_object['class_name'],
                    'depth': highest_risk_object['depth'],
                    'angle': highest_risk_object['angle'],
                    'recommended_action': highest_risk_object['recommended_action']
                }
            info['risk_penalty'] = risk_penalty * risk_scaling

            return reward, done, info

        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            traceback.print_exc()
            return 0.0, True, {'error': str(e)}

    def step(self, rl_action=None):
        """Execute step with hybrid MPC and RL control using enhanced obstacle information"""
        try:
            # Get control input from MPC (default behavior)
            if self.vehicle and self.controller:
                mpc_control = self.controller.get_control(self.vehicle, self.world)

                # Default to MPC control
                control = mpc_control
                control_source = "MPC"

                # Process YOLO detections with enhanced positioning
                detections = self.process_yolo_detection(self.front_camera)

                # Determine highest risk object and action
                highest_risk = 0
                recommended_action = "continue"
                nearest_obstacle_info = None

                for obj in detections:
                    if obj['risk_level'] > highest_risk:
                        highest_risk = obj['risk_level']
                        recommended_action = obj['recommended_action']
                        nearest_obstacle_info = {
                            'class': obj['class_name'],
                            'depth': obj['depth'],
                            'angle': obj['angle'],
                            'forward_distance': obj['forward_distance'],
                            'lateral_distance': obj['lateral_distance']
                        }

                # If we need to override MPC with reactive behavior
                if highest_risk > 3 and rl_action is not None:
                    # Convert RL action to control with awareness of obstacles
                    throttle = float(np.clip((rl_action[1] + 1) / 2, 0.0, 1.0))
                    steer = float(np.clip(rl_action[0], -1.0, 1.0))

                    # Apply action-specific modifications
                    if recommended_action == 'stop':
                        # Emergency stop
                        throttle = 0.0
                        brake = 1.0
                    elif recommended_action == 'slow':
                        # Reduce speed
                        throttle = max(0.0, throttle * 0.5)
                        brake = 0.3
                    elif recommended_action == 'avoid_left':
                        # Steer left (negative)
                        steer = max(-0.7, min(steer - 0.2, 0.0))
                        brake = 0.2
                    elif recommended_action == 'avoid_right':
                        # Steer right (positive)
                        steer = min(0.7, max(steer + 0.2, 0.0))
                        brake = 0.2
                    else:
                        # Default behavior - use RL as is
                        brake = 0.0

                    # Current control for smooth transition
                    current_control = self.vehicle.get_control()

                    # Smooth control changes
                    smooth_throttle = 0.6 * current_control.throttle + 0.4 * throttle
                    smooth_steer = 0.6 * current_control.steer + 0.4 * steer

                    # Create modified RL control
                    control = carla.VehicleControl(
                        throttle=smooth_throttle,
                        steer=smooth_steer,
                        brake=brake,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False
                    )
                    control_source = f"RL-{recommended_action}"

                # Apply the final control
                self.vehicle.apply_control(control)

                # Store current control values for info
                control_info = {
                    'throttle': control.throttle,
                    'steer': control.steer,
                    'brake': control.brake,
                    'control_source': control_source,
                    'highest_risk': highest_risk,
                    'recommended_action': recommended_action,
                    'nearest_obstacle': nearest_obstacle_info
                }
            else:
                return None, 0, True, {'error': 'Vehicle or controller not available'}

            # Tick the world multiple times for better physics
            for _ in range(4):
                self.world.tick()

            # Get new state and calculate reward
            new_state = self.get_state()
            reward, done, info = self.calculate_reward()

            # Add control info for debugging
            info.update(control_info)

            return new_state, reward, done, info

        except Exception as e:
            print(f"Error in step: {e}")
            traceback.print_exc()
            return None, 0, True, {'error': str(e)}

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
    
    def spawn_npcs(self):
        """Spawn NPC vehicles near the training vehicle"""
        try:
            number_of_vehicles = 5
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

            # Get all spawn points
            all_spawn_points = self.world.get_map().get_spawn_points()

            # Filter spawn points to only include those within radius of training vehicle
            nearby_spawn_points = []
            for spawn_point in all_spawn_points:
                if spawn_point.location.distance(vehicle_location) <= spawn_radius:
                    nearby_spawn_points.append(spawn_point)

            if not nearby_spawn_points:
                print(f"No spawn points found within {spawn_radius}m of training vehicle!")
                # Fallback to closest spawn points if none found in radius
                all_spawn_points.sort(key=lambda p: p.location.distance(vehicle_location))
                nearby_spawn_points = all_spawn_points[:number_of_vehicles]

            print(f"Found {len(nearby_spawn_points)} potential spawn points near training vehicle")

            # Spawn vehicles
            vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
            vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]

            # Try to spawn vehicles at nearby points
            spawned_count = 0
            for spawn_point in nearby_spawn_points:
                if spawned_count >= number_of_vehicles:
                    break

                blueprint = random.choice(vehicle_bps)

                # Set color for better visibility
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)

                # Try to spawn vehicle
                vehicle = self.world.try_spawn_actor(blueprint, spawn_point)

                if vehicle is not None:
                    self.npc_vehicles.append(vehicle)
                    vehicle.set_autopilot(True)

                    try:
                        # Set more conservative speed for nearby vehicles
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(0, 10))
                        traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 2.0))


                        spawned_count += 1

                        # Print spawn information
                        distance_to_ego = vehicle.get_location().distance(vehicle_location)
                        print(f"Spawned {vehicle.type_id} at {distance_to_ego:.1f}m from training vehicle")

                        # Draw debug line to show spawn location
                        debug = self.world.debug
                        if debug:
                            # Draw a line from ego vehicle to spawned vehicle
                            debug.draw_line(
                                vehicle_location,
                                vehicle.get_location(),
                                thickness=0.1,
                                color=carla.Color(r=0, g=255, b=0),
                                life_time=5.0
                            )
                            # Draw point at spawn location
                            debug.draw_point(
                                vehicle.get_location(),
                                size=0.1,
                                color=carla.Color(r=255, g=0, b=0),
                                life_time=5.0
                            )

                    except Exception as tm_error:
                        print(f"Warning: Could not set some traffic manager parameters: {tm_error}")
                        continue

            print(f"Successfully spawned {spawned_count} vehicles near training vehicle")

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
                            life_time=5.0
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
                                life_time=5.0
                            )
            except Exception as debug_error:
                print(f"Warning: Could not draw debug visualization: {debug_error}")

        except Exception as e:
            print(f"Error spawning NPCs: {e}")
            traceback.print_exc()
            self.cleanup_npcs()


    def cleanup_npcs(self):
        """Clean up all spawned NPCs"""
        try:
            # Stop pedestrian controllers
            for controller in self.pedestrian_controllers:
                if controller.is_alive:
                    controller.stop()
                    controller.destroy()
            self.pedestrian_controllers.clear()

        
            # Destroy vehicles
            for vehicle in self.npc_vehicles:
                if vehicle.is_alive:
                    vehicle.destroy()
            self.npc_vehicles.clear()
            print("Successfully cleaned up all NPCs")

        except Exception as e:
            print(f"Error cleaning up NPCs: {e}")

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
    


