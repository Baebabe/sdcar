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

                    # Draw rectangle
                    pygame.draw.rect(surface, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 2)

                    # Draw label
                    if pygame.font.get_init():
                        font = pygame.font.Font(None, 24)
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
        """Process image with YOLO and return detections with enhanced spatial information"""
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
                # Process detections
                for *xyxy, conf, cls in pred[0]:
                    x1, y1, x2, y2 = map(float, xyxy)

                    # Get enhanced spatial information
                    spatial_info = calculate_spatial_info([x1, y1, x2, y2], int(cls))

                    # Calculate time-to-collision (TTC) if vehicle is moving
                    ttc = float('inf')
                    if hasattr(self, 'vehicle') and self.vehicle:
                        velocity = self.vehicle.get_velocity()
                        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # m/s

                        # Only calculate TTC for objects roughly in our path
                        if abs(spatial_info['relative_angle']) < 30 and speed > 0.5:
                            ttc = spatial_info['depth'] / speed  # seconds

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
                        'class_name': self.yolo_model.names[int(cls)],
                        'detection_confidence': float(conf),
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1,
                        # Add a risk score based on TTC, position and object type
                        'risk_score': self._calculate_risk_score(
                            spatial_info['depth'],
                            spatial_info['lane_position'],
                            ttc,
                            int(cls)
                        )
                    })

            # Sort by risk score (higher risk first)
            objects.sort(key=lambda x: x['risk_score'], reverse=True)
            return objects[:self.max_objects]

        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _calculate_risk_score(self, depth, lane_position, ttc, class_id):
        """Calculate risk score for object based on multiple factors"""
        # Base risk inversely proportional to distance
        distance_factor = 10.0 / max(1.0, depth)

        # Lane position factor (higher if in same lane)
        lane_factor = 1.0 if lane_position == 0 else 0.3

        # Time to collision factor (higher for imminent collisions)
        ttc_factor = 1.0 if ttc < 3.0 else (0.5 if ttc < 6.0 else 0.2)

        # Object type factor (higher for vehicles and pedestrians)
        type_factors = {
            0: 1.0,  # person - highest risk
            1: 0.7,  # bicycle
            2: 0.8,  # motorcycle
            3: 0.9,  # car
            4: 1.0,  # truck - highest risk due to size
            5: 1.0,  # bus - highest risk due to size
            6: 0.8,  # train
            7: 0.3,  # fire hydrant
            8: 0.8,  # stop sign - important for traffic rules
            9: 0.3,  # parking meter
            10: 0.3, # bench
        }
        type_factor = type_factors.get(class_id, 0.5)

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
        """Get enhanced state representation with improved spatial awareness"""
        if self.front_camera is None:
            return None

        # Initialize state array
        state_array = []

        # 1. Process YOLO detections with enhanced spatial information
        detections = self.process_yolo_detection(self.front_camera)

        # Create a feature vector for each detected object
        for obj in detections:
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
        state_array.extend([0.0] * (remaining_objects * 6))  # 6 features per object now

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
        """Improved reward function to encourage movement, exploration, and obstacle avoidance"""
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

            # Speed reward
            target_speed = 30.0  # Target speed in km/h
            if speed < 1.0:  # Almost stopped
                # Check if there are obstacles nearby that justify stopping
                detections = self.process_yolo_detection(self.front_camera)
                near_obstacle = any(obj['depth'] < 10.0 and 
                                   obj['class_name'] in ['car', 'truck', 'bus', 'person', 'traffic light']
                                   for obj in detections)

                if near_obstacle:
                    # It's good to stop near obstacles
                    reward += 1.0
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

            # Store current location for next step
            self.last_location = location

            info['speed'] = speed
            info['reward'] = reward
            info['distance_from_center'] = distance_from_center

            return reward, done, info

        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            traceback.print_exc()
            return 0.0, True, {'error': str(e)}

    def step(self, rl_action=None):
        """Enhanced step function with improved spatial awareness for control decisions"""
        try:
            # 1. Get enhanced object detections first
            detections = self.process_yolo_detection(self.front_camera)

            # 2. Extract key safety information from detections
            safety_info = self._analyze_safety(detections)

            # 3. Determine control strategy
            if self.vehicle and self.controller:
                # Get standard MPC control (base behavior)
                mpc_control = self.controller.get_control(self.vehicle, self.world)

                # Default to MPC control
                control = mpc_control
                control_source = "MPC"

                # Decide if we need to override with safety controls
                emergency_braking = safety_info['emergency_braking']
                collision_avoidance = safety_info['collision_avoidance']
                avoidance_steering = safety_info['avoidance_steering']

                # If RL action is provided, use that for complex scenarios
                if rl_action is not None and (collision_avoidance or avoidance_steering):
                    # Convert RL action to control
                    throttle = float(np.clip((rl_action[1] + 1) / 2, 0.0, 1.0))  # Convert from [-1,1] to [0,1]
                    steer = float(np.clip(rl_action[0], -1.0, 1.0))

                    # Current control for smooth transition
                    current_control = self.vehicle.get_control()

                    # Apply brake only when reducing speed
                    brake = 0.0
                    if throttle < current_control.throttle:
                        brake = min((current_control.throttle - throttle) * 3.0, 1.0)

                    # Create RL control
                    control = carla.VehicleControl(
                        throttle=throttle,
                        steer=steer,
                        brake=brake,
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False
                    )
                    control_source = "RL"

                # If emergency braking needed, override with hard braking
                # This takes precedence over RL for safety
                elif emergency_braking:
                    # Emergency braking - override other controls
                    control = carla.VehicleControl(
                        throttle=0.0,
                        steer=mpc_control.steer,  # Keep MPC steering
                        brake=1.0,  # Full braking
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False
                    )
                    control_source = "EMERGENCY_BRAKE"

                # If we need to swerve and RL isn't handling it
                elif avoidance_steering and rl_action is None:
                    # Get current control and modify steering
                    current_control = self.vehicle.get_control()
                    steer_strength = safety_info['steer_amount']

                    # Blend MPC steering with avoidance steering
                    blended_steer = 0.3 * mpc_control.steer + 0.7 * steer_strength

                    # Apply gentle braking during avoidance
                    control = carla.VehicleControl(
                        throttle=mpc_control.throttle * 0.6,  # Reduce throttle
                        steer=blended_steer,
                        brake=0.2,  # Light braking
                        hand_brake=False,
                        reverse=False,
                        manual_gear_shift=False
                    )
                    control_source = "AVOIDANCE"

                # Apply the final control
                self.vehicle.apply_control(control)

                # Store control values for info
                control_info = {
                    'throttle': control.throttle,
                    'steer': control.steer,
                    'brake': control.brake,
                    'control_source': control_source,
                }
                control_info.update(safety_info)  # Add safety analysis data
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

        # Initialize distances
        nearest_same_lane_dist = float('inf')
        nearest_left_lane_dist = float('inf')
        nearest_right_lane_dist = float('inf')
        min_ttc = float('inf')

        # Important object classes for driving
        vehicle_classes = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        critical_classes = ['person'] + vehicle_classes
        traffic_signals = ['traffic light', 'stop sign']

        # Analyze each detection
        for obj in detections:
            obj_depth = obj['depth']
            obj_lane = obj['lane_position']
            obj_angle = obj['relative_angle']
            obj_ttc = obj['time_to_collision']
            obj_class = obj['class_name']

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

            # Traffic signal checks
            elif obj_class in traffic_signals and obj_depth < 25.0 and abs(obj_angle) < 20:
                # Simplification - in reality would look at light color
                collision_avoidance = True

        # Return comprehensive safety information
        return {
            'emergency_braking': emergency_braking,
            'collision_avoidance': collision_avoidance,
            'avoidance_steering': avoidance_steering,
            'steer_amount': steer_amount,
            'nearest_same_lane_dist': nearest_same_lane_dist,
            'nearest_left_lane_dist': nearest_left_lane_dist,
            'nearest_right_lane_dist': nearest_right_lane_dist,
            'min_ttc': min_ttc
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
    

