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
from torch.utils.tensorboard import SummaryWriter
from threading import Thread
from tqdm import tqdm
import requests 
import pygame
from pygame import surfarray
import traceback

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
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device
except ImportError:
    print("Error importing YOLOv5 modules. Make sure YOLOv5 is properly installed.")
    sys.exit(1)

# Constants
IM_WIDTH = 640
IM_HEIGHT = 480
EPISODES = 100
SECONDS_PER_EPISODE = 60
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
UPDATE_TIMESTEP = 1500
K_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_depth(bbox, class_id, image_width=IM_WIDTH, image_height=IM_HEIGHT, fov=90):
    """
    Calculate depth using bounding box and class-specific real-world dimensions
    """
    bbox_width = bbox[2] - bbox[0]
    
    bbox_height = bbox[3] - bbox[1]
    
    # Define typical widths for different object classes
    # These values are approximate averages in meters
    REAL_WIDTHS = {
        0: 0.45,   # person
        1: 0.8,    # bicycle
        2: 0.8,    # motorcycle
        3: 1.8,    # car
        4: 2.5,    # truck
        5: 2.9,    # bus
        6: 1.8,    # train
        # Add more classes as needed from COCO dataset
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
    
    # Add confidence measure based on bbox size
    confidence = min(1.0, bbox_width / image_width)  # Higher confidence for larger objects
    
    return depth, confidence

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Smaller initial variance for better stability
        self.action_var = torch.full((action_dim,), 0.1).to(DEVICE)
        self.cov_mat = torch.diag(self.action_var)
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Bound outputs to [-1, 1]
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            module.bias.data.zero_()
    
    def forward(self, state):
        mean = self.actor(state)
        mean = torch.clamp(mean, -1, 1)
        value = self.critic(state)
        return mean, self.cov_mat.to(state.device), value
    
    def get_action(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).to(DEVICE)
            
            mean = self.actor(state)
            mean = torch.clamp(mean, -1, 1)
            
            try:
                dist = MultivariateNormal(mean, self.cov_mat.to(state.device))
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                action = torch.clamp(action, -1, 1)
                value = self.critic(state)
                return action, value, action_log_prob
            except ValueError as e:
                print(f"Error in get_action: {e}")
                print(f"Mean: {mean}")
                print(f"Covariance matrix: {self.cov_mat}")
                raise e

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.masks = []

    def compute_returns_and_advantages(self, next_value, gamma, gae_lambda):
        values = self.values + [next_value]
        returns = []
        gae = 0

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + gamma * gae_lambda * self.masks[step] * gae
            returns.insert(0, gae + values[step])

        returns = torch.tensor(returns).float().to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.masks.clear()

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.max_objects = 5  # Track 5 closest objects
        self.state_dim = (
            self.max_objects * 3 +  # x, y, depth for each object
            2 +                     # lane distance and angle
            2                      # speed and steering
        )
        self.action_dim = action_dim
        
        self.actor_critic = ActorCritic(self.state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)
        self.memory = PPOMemory()
        
        self.writer = SummaryWriter('logs/ppo_driving')
        self.training_step = 0
        self.best_reward = float('-inf')
        self.episode_rewards = []
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def select_action(self, state):
        """Select action based on state"""
        try:
            if state is None:
                return None, None, None
                
            # State is already a numpy array, just convert to tensor
            state_tensor = torch.FloatTensor(state).to(DEVICE)
            
            action, value, log_prob = self.actor_critic.get_action(state_tensor)
            return action.cpu().numpy(), value.item(), log_prob.item()
            
        except Exception as e:
            print(f"Error in select_action: {e}")
            print(f"State shape: {state.shape if hasattr(state, 'shape') else 'No shape'}")
            print(f"State type: {type(state)}")
            print(f"State content: {state}")
            raise

    def update(self):
        if len(self.memory.states) == 0:
            return

        # Convert lists to tensors
        states = torch.FloatTensor(self.memory.states).to(DEVICE)
        actions = torch.FloatTensor(self.memory.actions).to(DEVICE)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(DEVICE)
        
        # Compute advantages
        advantages = self.memory.compute_returns_and_advantages(0, GAMMA, GAE_LAMBDA)
        
        # PPO update
        for _ in range(K_EPOCHS):
            # Get current policy outputs
            mean, cov_mat, state_values = self.actor_critic(states)
            dist = MultivariateNormal(mean, cov_mat)
            log_probs = dist.log_prob(actions)
            
            # Calculate ratio and surrogate loss
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
            
            # Calculate losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), advantages + state_values.detach().squeeze())
            entropy_loss = -dist.entropy().mean()
            
            total_loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            self.training_step += 1
            
        self.memory.clear()


    def save_checkpoint(self, episode, filepath, best_reward=None):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'episode': episode,
                'actor_critic_state_dict': self.actor_critic.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_reward': best_reward if best_reward is not None else self.best_reward,
                'training_step': self.training_step
            }

            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved: {filepath}")

        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            traceback.print_exc()
    
    def load_checkpoint(self, checkpoint_path, mode='train'):
        """Load model checkpoint"""
        try:
            if not os.path.exists(checkpoint_path):
                print(f"No checkpoint found at {checkpoint_path}")
                return False
            
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            if mode == 'train':
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.best_reward = checkpoint.get('best_reward', float('-inf'))
            self.training_step = checkpoint.get('training_step', 0)
            
            episode = checkpoint.get('episode', 0)
            print(f"Loaded checkpoint from episode {episode}")
            
            # Set to eval mode if evaluating
            if mode == 'eval':
                self.actor_critic.eval()
            else:
                self.actor_critic.train()
                
            return True
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

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
        """Reset environment with movement verification"""
        print("Starting environment reset...")
        self.cleanup_actors()
        self.cleanup_npcs()

        # Setup new episode
        if not self.setup_vehicle():
            raise Exception("Failed to setup vehicle")

        # Spawn NPCs
        self.spawn_npcs()

        # Verify vehicle movement
        print("Verifying vehicle movement...")
        initial_location = self.vehicle.get_location()

        # Apply forward movement
        control = carla.VehicleControl(throttle=0.5, steer=0.0)
        self.vehicle.apply_control(control)

        # Tick world several times
        for _ in range(20):
            self.world.tick()
            time.sleep(0.05)

        # Check if vehicle moved
        final_location = self.vehicle.get_location()
        distance_moved = initial_location.distance(final_location)

        if distance_moved < 0.1:
            print("Warning: Vehicle not moving! Attempting to fix...")
            # Try to toggle autopilot
            self.vehicle.set_autopilot(True)
            time.sleep(0.1)
            self.vehicle.set_autopilot(False)

        # Reset control
        self.vehicle.apply_control(carla.VehicleControl())

        # Wait for camera initialization
        timeout = time.time() + 10.0
        while self.front_camera is None:
            if time.time() > timeout:
                raise Exception("Timeout waiting for camera initialization")
            self.world.tick()
            time.sleep(0.1)

        return self.get_state()

    def setup_vehicle(self):
        """Spawn and setup the ego vehicle"""
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
        """Process image with YOLO and return detections"""
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
                    depth, depth_confidence = calculate_depth([x1, y1, x2, y2], int(cls))
                    
                    objects.append({
                        'position': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'depth': depth,
                        'depth_confidence': depth_confidence,
                        'class': int(cls),
                        'class_name': self.yolo_model.names[int(cls)],
                        'confidence': float(conf),
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1
                    })
            
            # Sort by depth and confidence
            objects.sort(key=lambda x: x['depth'] * (1 - x['depth_confidence']))
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
    
    def get_state(self):
        """Get complete state information"""
        if self.front_camera is None:
            return None

        # Initialize state array
        state_array = []

        # 1. Process YOLO detections (max_objects * 3 features)
        detections = self.process_yolo_detection(self.front_camera)
        for obj in detections:
            state_array.extend([
                obj['position'][0] / IM_WIDTH,     # x position normalized
                obj['position'][1] / IM_HEIGHT,    # y position normalized
                obj['depth'] / 100.0               # depth normalized
            ])

        # Pad if fewer than max objects
        remaining_objects = self.max_objects - len(detections)
        state_array.extend([0.0] * (remaining_objects * 3))

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

        return np.array(state_array, dtype=np.float16)
    
    def calculate_reward(self):
        """
        Calculate reward for self-driving car in CARLA with optimized reward values.
        """
        # Optimized constants for CARLA
        v_target = 12.0    # Target speed increased to 30 km/h for more realistic urban driving
        v_min = 10.0       # Minimum speed increased to maintain better training momentum
        v_max = 30.0       # Maximum speed increased accordingly
        d_max = 3.5        # Reduced from 2.0m to encourage tighter lane keeping
        a_max = 20.0       # Reduced from 20° to encourage smoother steering

        # Reward scaling factors
        COLLISION_PENALTY = -100.0   # Increased from -50 to strongly discourage collisions
        STUCK_PENALTY = -50.0       # New penalty for getting stuck
        SPEED_MULTIPLIER = 0.4      # Weight for speed reward
        DISTANCE_MULTIPLIER = 0.3   # Weight for distance reward
        ANGLE_MULTIPLIER = 0.3      # Weight for angle reward

        reward = 0.0
        done = False
        info = {}

        # Get current vehicle state
        location = self.vehicle.get_location()
        velocity = self.vehicle.get_velocity()
        transform = self.vehicle.get_transform()

        # Calculate current speed in km/h
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        # 1. Terminal conditions with stronger penalties
        if len(self.collision_hist) > 0:
            return COLLISION_PENALTY, True, {'termination_reason': 'collision'}

        if speed < 2.0:
            self.stuck_time += 0.1
            if self.stuck_time > 20.0:
                return STUCK_PENALTY, True, {'termination_reason': 'stuck'}
        else:
            self.stuck_time = 0

        # 2. Distance reward
        waypoint = self.world.get_map().get_waypoint(location)
        distance = location.distance(waypoint.transform.location)
        d_norm = min(1.0, distance/d_max)

        # Exponential distance penalty for more sensitive lane keeping
        distance_reward = math.exp(-2.0 * d_norm) * DISTANCE_MULTIPLIER

        # 3. Orientation reward
        vehicle_forward = transform.get_forward_vector()
        waypoint_forward = waypoint.transform.get_forward_vector()

        angle_diff = abs(math.degrees(math.atan2(
            vehicle_forward.y - waypoint_forward.y,
            vehicle_forward.x - waypoint_forward.x)))

        # Smoother angle reward with exponential decay
        a_rew = math.exp(-angle_diff/a_max) * ANGLE_MULTIPLIER if angle_diff < a_max else 0.0

        # 4. Speed reward with smoother transitions
        speed_reward = 0.0
        if speed < v_min:
            speed_reward = -0.5 * ((v_min - speed) / v_min) * SPEED_MULTIPLIER
        elif speed < v_target:
            speed_reward = ((speed - v_min) / (v_target - v_min)) * SPEED_MULTIPLIER
        else:
            speed_factor = max(0, 1 - (speed - v_target) / (v_max - v_target))
            speed_reward = speed_factor * SPEED_MULTIPLIER

        # 5. Combine rewards with continuous function
        reward = speed_reward + distance_reward + a_rew

        # Additional rewards for stable driving
        if v_min <= speed <= v_max and distance < 1.0 and angle_diff < 10:
            reward += 0.1  # Small bonus for ideal driving conditions

        # Progressive rewards for improvement
        if self.last_location:
            # Reward for making forward progress
            progress = location.distance(self.last_location)
            if progress > 0.1:  # Minimum progress threshold
                reward += 0.05 * min(progress, 1.0)  # Cap progress reward

        info = {
            'speed_kmh': speed,
            'distance_to_center': distance,
            'angle_diff': angle_diff,
            'd_norm': d_norm,
            'a_rew': a_rew,
            'speed_reward': speed_reward,
            'distance_reward': distance_reward,
            'final_reward': reward,
            'stuck_time': self.stuck_time
        }

        return reward, done, info
    
    def step(self, action):
        """Execute action and return new state, reward, done, and info"""
        try:
            # Modified control logic for better movement
            steer = float(np.clip(action[0], -1.0, 1.0))
            throttle = float(np.clip(action[1], 0.0, 1.0))

            # Apply control with more realistic values
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=0.0,  # Set to 0 initially to ensure movement
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )

            self.vehicle.apply_control(control)

            # Only tick the world - Traffic Manager updates automatically
            self.world.tick()

            # Handle pygame events and display update
            if self.display is not None:
                self.clock.tick(20)  # Limit to 20 FPS
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return None, 0, True, {'termination_reason': 'user_quit'}

            # Get new state
            new_state = self.get_state()

            # Calculate reward
            reward, done, info = self.calculate_reward()

            # Check timeout
            if time.time() - self.episode_start > SECONDS_PER_EPISODE:
                done = True
                info['termination_reason'] = 'timeout'

            return new_state, reward, done, info

        except Exception as e:
            print(f"Error in step: {e}")
            traceback.print_exc()
            return None, 0, True, {'error': str(e)}
    


    def spawn_npcs(self):
        """Spawn NPC vehicles near the training vehicle"""
        try:
            number_of_vehicles = 5
            spawn_radius = 40.0  # Radius in meters to spawn NPCs around training vehicle

            if self.vehicle is None:
                print("Training vehicle not found! Cannot spawn NPCs.")
                return

            # Get training vehicle's location
            vehicle_location = self.vehicle.get_location()

            # Configure traffic manager
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            traffic_manager.global_percentage_speed_difference(10.0)

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
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-10, 0))
                        traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(2.0, 4.0))

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
    


def evaluate(checkpoint_path, num_episodes=10):
    """Evaluate a trained agent"""
    try:
        # Initialize environment
        env = CarEnv()
        
        # Calculate state dimension
        state_dim = env.max_objects * 3 + 4
        action_dim = 2
        print(f"Initializing agent with state_dim={state_dim}, action_dim={action_dim}")
        
        agent = PPOAgent(state_dim, action_dim)
        
        # Load model
        if not agent.load_checkpoint(checkpoint_path, mode='eval'):
            print("Failed to load checkpoint. Aborting evaluation.")
            return
        
        total_rewards = []
        
        for episode in range(num_episodes):
            try:
                state = env.reset()
                if state is None:
                    print("Failed to get initial state. Skipping episode.")
                    continue
                    
                episode_reward = 0
                done = False
                
                while not done:
                    action, _, _ = agent.select_action(state)
                    state, reward, done, _ = env.step(action)
                    episode_reward += reward
                
                total_rewards.append(episode_reward)
                print(f"Episode {episode}: Reward = {episode_reward:.2f}")
                
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                continue
        
        if total_rewards:
            print(f"\nEvaluation Results:")
            print(f"Average Reward: {np.mean(total_rewards):.2f}")
            print(f"Std Dev Reward: {np.std(total_rewards):.2f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'env' in locals():
            env.cleanup_actors()

def train():
    """Main training loop with improved monitoring and visualization"""
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {DEVICE}")
    
    # Create directories for saving
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('tensorboard_logs', exist_ok=True)
    
    # Initialize metrics tracking
    training_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'best_reward': float('-inf'),
        'last_save_time': time.time(),
        'total_steps': 0
    }
    
    try:
        # Initialize environment
        print("Initializing CARLA environment...")
        env = CarEnv()
        
        # Wait for environment to stabilize
        print("Waiting for environment to stabilize...")
        time.sleep(2)
        
        # Calculate state dimension
        state_dim = env.max_objects * 3 + 4  # (max_objects * 3 features) + 4 additional features
        action_dim = 2  # steering and throttle
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        
        # Initialize agent
        print("Initializing PPO agent...")
        agent = PPOAgent(state_dim, action_dim)
        writer = SummaryWriter(f'tensorboard_logs/training_{int(time.time())}')
        
        # Try to load latest checkpoint
        latest_checkpoint = os.path.join('checkpoints', 'latest.pth')
        starting_episode = 0
        
        if os.path.exists(latest_checkpoint):
            print("Loading latest checkpoint...")
            if agent.load_checkpoint(latest_checkpoint):
                checkpoint = torch.load(latest_checkpoint, map_location=DEVICE)
                starting_episode = checkpoint.get('episode', 0) + 1
                training_metrics['best_reward'] = checkpoint.get('best_reward', float('-inf'))
                print(f"Resuming from episode {starting_episode}")
        
        print("\nStarting training loop...")
        training_start_time = time.time()
        
        for episode in range(starting_episode, EPISODES):
            episode_start_time = time.time()
            episode_metrics = {
                'reward': 0,
                'steps': 0,
                'collisions': 0,
                'avg_speed': 0,
                'speeds': []
            }
            
            try:
                print(f"\nEpisode {episode}/{EPISODES}")
                print("Resetting environment...")
                state = env.reset()
                
                if state is None:
                    print("Failed to get initial state, retrying episode")
                    continue
                
                # Episode loop
                done = False
                while not done:
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    
                    # Select action
                    action_result = agent.select_action(state)
                    if action_result is None:
                        print("Failed to select action, ending episode")
                        break
                    
                    action, value, log_prob = action_result
                    
                    # Execute action
                    next_state, reward, done, info = env.step(action)
                    
                    # Update metrics
                    episode_metrics['reward'] += reward
                    episode_metrics['steps'] += 1
                    training_metrics['total_steps'] += 1
                    
                    if 'speed' in info:
                        episode_metrics['speeds'].append(info['speed'])
                    
                    if 'collision' in info:
                        episode_metrics['collisions'] += 1
                    
                    # Store experience
                    agent.memory.states.append(state)
                    agent.memory.actions.append(action)
                    agent.memory.rewards.append(reward)
                    agent.memory.values.append(value)
                    agent.memory.log_probs.append(log_prob)
                    agent.memory.masks.append(1 - done)
                    
                    # Print progress every 100 steps
                    if episode_metrics['steps'] % 100 == 0:
                        current_speed = info.get('speed', 0)
                        print(f"Step {episode_metrics['steps']}: "
                              f"Reward = {episode_metrics['reward']:.2f}, "
                              f"Speed = {current_speed:.2f} km/h")
                        
                        # Log to tensorboard
                        writer.add_scalar('Training/Step_Reward', reward, training_metrics['total_steps'])
                        writer.add_scalar('Training/Current_Speed', current_speed, training_metrics['total_steps'])
                    
                    # Update policy if enough steps
                    if len(agent.memory.states) >= UPDATE_TIMESTEP:
                        print(f"\nPerforming policy update at step {episode_metrics['steps']}")
                        agent.update()
                    
                    if done:
                        break
                    
                    state = next_state
                
                # Episode completion statistics
                episode_duration = time.time() - episode_start_time
                episode_metrics['avg_speed'] = np.mean(episode_metrics['speeds']) if episode_metrics['speeds'] else 0
                
                # Log episode metrics
                print(f"\nEpisode {episode} Summary:")
                print(f"Total Reward: {episode_metrics['reward']:.2f}")
                print(f"Steps: {episode_metrics['steps']}")
                print(f"Average Speed: {episode_metrics['avg_speed']:.2f} km/h")
                print(f"Collisions: {episode_metrics['collisions']}")
                print(f"Duration: {episode_duration:.2f} seconds")
                print(f"Average Step Time: {episode_duration/episode_metrics['steps']:.3f} seconds")
                
                # Update training metrics
                training_metrics['episode_rewards'].append(episode_metrics['reward'])
                training_metrics['episode_lengths'].append(episode_metrics['steps'])
                
                # Log to tensorboard
                writer.add_scalar('Training/Episode_Reward', episode_metrics['reward'], episode)
                writer.add_scalar('Training/Episode_Length', episode_metrics['steps'], episode)
                writer.add_scalar('Training/Average_Speed', episode_metrics['avg_speed'], episode)
                writer.add_scalar('Training/Collisions', episode_metrics['collisions'], episode)
                
                # Save best model
                if episode_metrics['reward'] > training_metrics['best_reward']:
                    training_metrics['best_reward'] = episode_metrics['reward']
                    best_model_path = os.path.join('checkpoints', 'best_model.pth')
                    agent.save_checkpoint(episode, best_model_path, training_metrics['best_reward'])
                    print(f"New best reward: {episode_metrics['reward']:.2f}")
                
                # Regular checkpoint saving (every 10 episodes)
                if episode % 10 == 0:
                    checkpoint_path = os.path.join('checkpoints', f'checkpoint_{episode}.pth')
                    agent.save_checkpoint(episode, checkpoint_path, training_metrics['best_reward'])
                    print(f"Checkpoint saved at episode {episode}")
                
                # Update latest checkpoint
                agent.save_checkpoint(episode, latest_checkpoint, training_metrics['best_reward'])
                
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                traceback.print_exc()
                
                # Try to cleanup and continue with next episode
                try:
                    env.cleanup_actors()
                    env.cleanup_npcs()
                except:
                    pass
                
                continue
                
        # Training completion statistics
        total_training_time = time.time() - training_start_time
        print("\nTraining Complete!")
        print(f"Total training time: {total_training_time/3600:.2f} hours")
        print(f"Best reward achieved: {training_metrics['best_reward']:.2f}")
        print(f"Total steps: {training_metrics['total_steps']}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Critical error during training: {e}")
        traceback.print_exc()
    finally:
        # Final cleanup
        print("\nPerforming final cleanup...")
        try:
            # Save final checkpoint if not already saved
            if 'agent' in locals() and 'episode' in locals():
                final_checkpoint_path = os.path.join('checkpoints', 'final_model.pth')
                agent.save_checkpoint(episode, final_checkpoint_path, training_metrics['best_reward'])
                print("Final checkpoint saved")
            
            # Cleanup environment
            if 'env' in locals():
                env.close()
                print("Environment cleaned up")
            
            # Close tensorboard writer
            if 'writer' in locals():
                writer.close()
                print("TensorBoard writer closed")
            
            # Save training metrics
            if 'training_metrics' in locals():
                metrics_path = os.path.join('logs', f'training_metrics_{int(time.time())}.json')
                with open(metrics_path, 'w') as f:
                    json.dump(training_metrics, f)
                print(f"Training metrics saved to {metrics_path}")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
            traceback.print_exc()
        
        print("\nTraining session ended")


if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
        
        print("Starting CARLA PPO training with YOLO object detection")
        print(f"Running on: {DEVICE}")
        
        # Start training
        train()
        
        # Evaluate best model
        best_model_path = os.path.join('checkpoints', 'best_model.pth')
        if os.path.exists(best_model_path):
            print("\nEvaluating best model...")
            evaluate(best_model_path)
            
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Program terminated")






































#PPO USING VAE(VAE NOT IMPLEMENTED DUE TO PERFORMANCE CONSTRAINTS)

# import os
# import sys
# import glob
# import time
# import random
# import numpy as np
# import cv2
# import math
# from collections import deque
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.distributions import MultivariateNormal
# import torch.nn.functional as F
# from torch.distributions import Normal
# from torch.utils.tensorboard import SummaryWriter
# from threading import Thread
# from tqdm import tqdm

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

# import carla

# # Constants
# IM_WIDTH = 320
# IM_HEIGHT = 240
# EPISODES = 200
# SECONDS_PER_EPISODE = 30
# BATCH_SIZE = 64
# LEARNING_RATE = 3e-4
# GAMMA = 0.99
# GAE_LAMBDA = 0.95
# CLIP_EPSILON = 0.2
# VALUE_COEF = 0.5
# ENTROPY_COEF = 0.01
# LATENT_DIM = 32
# UPDATE_TIMESTEP = 1500
# K_EPOCHS = 40
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE =torch.device("cpu")


# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()
        
#         # Calculate dimensions after each convolution
#         h1, w1 = conv2d_output_size(IM_HEIGHT, IM_WIDTH, 4, 2, 1)  # After first conv
#         h2, w2 = conv2d_output_size(h1, w1, 4, 2, 1)              # After second conv
#         h3, w3 = conv2d_output_size(h2, w2, 4, 2, 1)              # After third conv
#         h4, w4 = conv2d_output_size(h3, w3, 4, 2, 1)              # After fourth conv
        
#         self.final_h = h4
#         self.final_w = w4
#         self.conv_output_size = 256 * h4 * w4
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
        
#         self.fc_mu = nn.Linear(self.conv_output_size, LATENT_DIM)
#         self.fc_var = nn.Linear(self.conv_output_size, LATENT_DIM)
        
#         self.decoder_input = nn.Linear(LATENT_DIM, self.conv_output_size)
        
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
#             nn.Sigmoid()
#         )
    
#     def encode(self, x):
#         x = self.encoder(x)
#         x = x.contiguous()
#         x = x.view(x.size(0), -1)
#         return self.fc_mu(x), self.fc_var(x)
    
#     def reparameterize(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)
#         return mu + eps * std
    
#     def decode(self, z):
#         z = self.decoder_input(z)
#         z = z.contiguous()
#         z = z.view(z.size(0), 256, self.final_h, self.final_w)
#         return self.decoder(z)
    
#     def forward(self, x):
#         # Add this method
#         mu, log_var = self.encode(x)
#         z = self.reparameterize(mu, log_var)
#         return self.decode(z), mu, log_var

# def conv2d_output_size(h_in, w_in, kernel_size, stride, padding=0):
#     h_out = (h_in + 2 * padding - (kernel_size - 1) - 1) // stride + 1
#     w_out = (w_in + 2 * padding - (kernel_size - 1) - 1) // stride + 1
#     return h_out, w_out

# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCritic, self).__init__()
        
#         self.state_dim = state_dim
#         self.action_dim = action_dim
        
#         # Smaller initial variance for better stability
#         self.action_var = torch.full((action_dim,), 0.1).to(DEVICE)
#         self.cov_mat = torch.diag(self.action_var)
        
#         # Actor network with proper initialization
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 64),
#             nn.Tanh(),
#             nn.Linear(64, action_dim),
#             nn.Tanh()  # Bound outputs to [-1, 1]
#         )
        
#         # Critic network
#         self.critic = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 64),
#             nn.Tanh(),
#             nn.Linear(64, 1)
#         )
        
#         # Initialize weights with smaller values
#         self.apply(self._init_weights)
    
#     def _init_weights(self, module):
#         if isinstance(module, nn.Linear):
#             nn.init.orthogonal_(module.weight, gain=0.01)
#             module.bias.data.zero_()
    
#     def forward(self, state):
#         mean = self.actor(state)
#         value = self.critic(state)
#         # Ensure mean is bounded
#         mean = torch.clamp(mean, -1, 1)
#         return mean, self.cov_mat.to(state.device), value
    
#     def get_action(self, state):
#         with torch.no_grad():
#             if isinstance(state, np.ndarray):
#                 state = torch.FloatTensor(state).to(DEVICE)
            
#             mean = self.actor(state)
#             # Ensure mean is bounded
#             mean = torch.clamp(mean, -1, 1)
            
#             try:
#                 dist = MultivariateNormal(mean, self.cov_mat.to(state.device))
#                 action = dist.sample()
#                 action_log_prob = dist.log_prob(action)
                
#                 # Ensure action is bounded
#                 action = torch.clamp(action, -1, 1)
                
#                 value = self.critic(state)
                
#                 return action, value, action_log_prob
#             except ValueError as e:
#                 print(f"Error in get_action: {e}")
#                 print(f"Mean: {mean}")
#                 print(f"Covariance matrix: {self.cov_mat}")
#                 raise e
    
#     def evaluate(self, states, actions):
#         mean = self.actor(states)
#         # Ensure mean is bounded
#         mean = torch.clamp(mean, -1, 1)
        
#         batch_size = states.size(0)
#         cov_mat = self.cov_mat.repeat(batch_size, 1, 1).to(states.device)
        
#         try:
#             dist = MultivariateNormal(mean, cov_mat)
#             action_log_probs = dist.log_prob(actions)
#             dist_entropy = dist.entropy()
#             state_values = self.critic(states)
            
#             return action_log_probs, state_values, dist_entropy
#         except ValueError as e:
#             print(f"Error in evaluate: {e}")
#             print(f"Mean shape: {mean.shape}")
#             print(f"Mean values: {mean}")
#             print(f"Covariance matrix shape: {cov_mat.shape}")
#             raise e
        

# class PPOMemory:
#     def __init__(self):
#         self.states = []
#         self.actions = []
#         self.rewards = []
#         self.values = []
#         self.log_probs = []
#         self.dones = []
#         self.masks = []  # Add this line

#     def compute_returns_and_advantages(self, next_value, gamma, gae_lambda):
#         values = self.values + [next_value]
#         returns = []
#         gae = 0

#         for step in reversed(range(len(self.rewards))):
#             delta = (self.rewards[step] + gamma * values[step + 1] * self.masks[step] - values[step])
#             gae = delta + gamma * gae_lambda * self.masks[step] * gae
#             returns.insert(0, gae + values[step])

#         returns = torch.tensor(returns).float().to(DEVICE)
#         returns = (returns - returns.mean()) / (returns.std() + 1e-8)
#         return returns
    
#     def clear(self):
#         self.states.clear()
#         self.actions.clear()
#         self.rewards.clear()
#         self.values.clear()
#         self.log_probs.clear()
#         self.dones.clear()
#         self.masks.clear()  # Add this line


# def calculate_reward(vehicle, world, collision_hist, last_location, stuck_time=0, last_speed=0):
#     """
#     Calculate reward for self-driving car in CARLA with optimized reward values.
#     """
#     # Optimized constants for CARLA
#     v_target = 18.0    # Target speed increased to 30 km/h for more realistic urban driving
#     v_min = 13.0       # Minimum speed increased to maintain better training momentum
#     v_max = 30.0       # Maximum speed increased accordingly
#     d_max = 3.5        # Reduced from 2.0m to encourage tighter lane keeping
#     a_max = 20.0       # Reduced from 20° to encourage smoother steering
    
#     # Reward scaling factors
#     COLLISION_PENALTY = -60.0   # Increased from -50 to strongly discourage collisions
#     STUCK_PENALTY = -30.0       # New penalty for getting stuck
#     SPEED_MULTIPLIER = 0.4       # Weight for speed reward
#     DISTANCE_MULTIPLIER = 0.3    # Weight for distance reward
#     ANGLE_MULTIPLIER = 0.3       # Weight for angle reward
    
#     reward = 0.0
#     done = False
#     info = {}
    
#     # Get current vehicle state
#     location = vehicle.get_location()
#     velocity = vehicle.get_velocity()
#     transform = vehicle.get_transform()
    
#     # Calculate current speed in km/h
#     speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    
#     # 1. Terminal conditions with stronger penalties
#     if len(collision_hist) > 0:
#         return COLLISION_PENALTY, True, {'termination_reason': 'collision'}
    
#     if speed < 2.0:
#         stuck_time += 0.1
#         if stuck_time > 3.0:
#             return STUCK_PENALTY, True, {'termination_reason': 'stuck'}
#     else:
#         stuck_time = 0
    
#     # 2. Distance reward
#     waypoint = world.get_map().get_waypoint(location)
#     distance = location.distance(waypoint.transform.location)
#     d_norm = min(1.0, distance/d_max)
    
#     # Exponential distance penalty for more sensitive lane keeping
#     distance_reward = math.exp(-2.0 * d_norm) * DISTANCE_MULTIPLIER
    
#     # 3. Orientation reward
#     vehicle_forward = transform.get_forward_vector()
#     waypoint_forward = waypoint.transform.get_forward_vector()
    
#     angle_diff = abs(math.degrees(math.atan2(
#         vehicle_forward.y - waypoint_forward.y,
#         vehicle_forward.x - waypoint_forward.x)))
    
#     # Smoother angle reward with exponential decay
#     a_rew = math.exp(-angle_diff/a_max) * ANGLE_MULTIPLIER if angle_diff < a_max else 0.0
    
#     # 4. Speed reward with smoother transitions
#     speed_reward = 0.0
#     if speed < v_min:
#         speed_reward = -0.5 * ((v_min - speed) / v_min) * SPEED_MULTIPLIER
#     elif speed < v_target:
#         speed_reward = ((speed - v_min) / (v_target - v_min)) * SPEED_MULTIPLIER
#     else:
#         speed_factor = max(0, 1 - (speed - v_target) / (v_max - v_target))
#         speed_reward = speed_factor * SPEED_MULTIPLIER
    
#     # 5. Combine rewards with continuous function
#     reward = speed_reward + distance_reward + a_rew
    
#     # Additional rewards for stable driving
#     if v_min <= speed <= v_max and distance < 1.0 and angle_diff < 10:
#         reward += 0.1  # Small bonus for ideal driving conditions
    
#     # Progressive rewards for improvement
#     if last_location:
#         # Reward for making forward progress
#         progress = location.distance(last_location)
#         if progress > 0.1:  # Minimum progress threshold
#             reward += 0.05 * min(progress, 1.0)  # Cap progress reward
    
#     info = {
#         'speed_kmh': speed,
#         'distance_to_center': distance,
#         'angle_diff': angle_diff,
#         'd_norm': d_norm,
#         'a_rew': a_rew,
#         'speed_reward': speed_reward,
#         'distance_reward': distance_reward,
#         'final_reward': reward,
#         'stuck_time': stuck_time
#     }
    
#     return reward, done, info


# class CarEnv:
#     def __init__(self):
#         self.client = carla.Client("localhost", 2000)
#         self.client.set_timeout(10.0)
#         self.world = self.client.get_world()
#         self.blueprint_library = self.world.get_blueprint_library()
#         self.model_3 = self.blueprint_library.filter("model3")[0]
#         self.actor_list = []
#         self.collision_hist = []
#         self.waypoint_list = []
#         self.front_camera = None
#         self.previous_state = None
        
#         # Fixed spawn point with initial waypoints
#         self.spawn_point = carla.Transform(
#             carla.Location(x=96.0, y=-6.5, z=0.5),
#             carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)
#         )
        
#         self._setup_waypoints()
    
#     def _setup_waypoints(self):
#         # Generate waypoints along the road
#         waypoint = self.world.get_map().get_waypoint(
#             self.spawn_point.location,
#             project_to_road=True
#         )
        
#         for _ in range(100):  # Generate 100 waypoints
#             self.waypoint_list.append(waypoint)
#             waypoint = waypoint.next(2.0)[0]  # 2 meters between waypoints
    
#     def _setup_camera(self):
#         camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
#         camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
#         camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
#         camera_bp.set_attribute('fov', '110')

#         # Ensure image dimensions are correct when processing
#         camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
#         self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
#         self.actor_list.append(self.camera)
#         self.camera.listen(lambda data: self._process_img(data))

#     def _process_img(self, image):
#         image.convert(carla.ColorConverter.CityScapesPalette)
#         i = np.array(image.raw_data)
#         i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
#         i3 = i2[:, :, :3]
#         # Ensure the image has the correct dimensions
#         if i3.shape != (IM_HEIGHT, IM_WIDTH, 3):
#             i3 = cv2.resize(i3, (IM_WIDTH, IM_HEIGHT))
#         self.front_camera = i3
    
#     def _setup_collision_sensor(self):
#         collision_bp = self.blueprint_library.find('sensor.other.collision')
#         self.collision_sensor = self.world.spawn_actor(
#             collision_bp, 
#             carla.Transform(), 
#             attach_to=self.vehicle
#         )
#         self.actor_list.append(self.collision_sensor)
#         self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
    
    
#     def get_state(self):
#         if self.front_camera is None:
#             return None
        
#         # Get vehicle state
#         v = self.vehicle.get_velocity()
#         speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)  # km/h
        
#         # Get current waypoint and calculate distance
#         vehicle_location = self.vehicle.get_location()
#         current_waypoint = self.world.get_map().get_waypoint(vehicle_location)
        
#         # Calculate distance and orientation to waypoint
#         dist = vehicle_location.distance(current_waypoint.transform.location)
#         forward_vector = self.vehicle.get_transform().get_forward_vector()
#         wp_vector = current_waypoint.transform.get_forward_vector()
#         dot = forward_vector.x * wp_vector.x + forward_vector.y * wp_vector.y
#         cross = forward_vector.x * wp_vector.y - forward_vector.y * wp_vector.x
#         orientation = math.atan2(cross, dot)
        
#         state = {
#             'image': self.front_camera,
#             'speed': speed / 20.0,  # Normalized by target speed
#             'throttle': self.vehicle.get_control().throttle,
#             'steer': self.vehicle.get_control().steer,
#             'waypoint_dist': dist,
#             'waypoint_angle': orientation
#         }
        
#         return state
    
#     def step(self, action):
#         # Unpack and apply action
#         steer, throttle = action
#         self.vehicle.apply_control(carla.VehicleControl(
#             throttle=float(throttle),
#             steer=float(steer),
#             brake=0.0
#         ))

#         # Calculate reward and check if done
#         reward, done, info = self._calculate_reward()  # Now properly unpacking all three values

#         # Get new state
#         new_state = self.get_state()

#         # Check timeout
#         if time.time() - self.episode_start > SECONDS_PER_EPISODE:
#             done = True
#             info['termination_reason'] = 'timeout'

#         return new_state, reward, done, info  # Return all values including info dict
    


#     def _calculate_reward(self):
#         """
#         Wrapper for the calculate_reward function to maintain class state
#         """
#         if not hasattr(self, 'last_location'):
#             self.last_location = None
#         if not hasattr(self, 'stuck_time'):
#             self.stuck_time = 0
#         if not hasattr(self, 'last_speed'):
#             self.last_speed = 0

#         # Get current speed
#         v = self.vehicle.get_velocity()
#         current_speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)  # km/h

#         # Calculate reward using the new function
#         reward, done, info = calculate_reward(
#             vehicle=self.vehicle,
#             world=self.world,
#             collision_hist=self.collision_hist,
#             last_location=self.last_location,
#             stuck_time=self.stuck_time,
#             last_speed=self.last_speed
#         )

#         # Update state for next iteration
#         self.last_location = self.vehicle.get_location()
#         self.last_speed = current_speed
#         if not done:
#             self.stuck_time = info.get('stuck_time', 0)
#         else:
#             self.stuck_time = 0

#         return reward, done, info

#     def reset(self):
#         """
#         Update the reset method to initialize the new reward-related variables
#         """
#         # Existing reset code...
#         for actor in self.actor_list:
#             if actor.is_alive:
#                 actor.destroy()
#         self.actor_list.clear()
#         self.collision_hist.clear()

#         while True:
#             try:
#                 self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
#                 break
#             except:
#                 print("Collision at spawn point. Retrying...")

#         self.actor_list.append(self.vehicle)

#         # Set up sensors
#         self._setup_camera()
#         self._setup_collision_sensor()

#         while self.front_camera is None:
#             time.sleep(0.1)

#         # Initialize reward-related variables
#         self.last_location = None
#         self.stuck_time = 0
#         self.last_speed = 0
#         self.episode_start = time.time()

#         # Reset previous state
#         self.previous_state = {
#             'location': self.vehicle.get_location(),
#             'velocity': 0.0,
#             'cumulative_reward': 0.0,
#             'consecutive_good_actions': 0
#         }

#         return self.get_state()

# class PPOAgent:
#     def __init__(self, state_dim, action_dim):
#         try:
#             self.best_reward = float('-inf')
#             self.episode_rewards = []
#             self.current_episode = 0

#             self.actor_critic = ActorCritic(state_dim + LATENT_DIM, action_dim).to(DEVICE)
#             self.vae = VAE().to(DEVICE)
#             self.memory = PPOMemory()

#             self.actor_critic_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)
#             self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=1e-4)

#             self.clip_param = 0.2
#             self.max_grad_norm = 0.5
#             self.ppo_epochs = 10

#             self.training_step = 0
#             try:
#                 self.writer = SummaryWriter('logs/ppo_driving')
#             except Exception as e:
#                 print(f"Warning: Could not initialize TensorBoard writer: {e}")
#                 self.writer = None
#         except Exception as e:
#             print(f"Error initializing PPOAgent: {e}")
#             raise

    
#     def select_action(self, state):
#         # Process image through VAE
#         image = torch.FloatTensor(state['image']).to(DEVICE)
#         image = image.permute(2, 0, 1).unsqueeze(0) / 255.0

#         with torch.no_grad():
#             _, mu, _ = self.vae(image)

#         # Combine VAE encoding with other state information
#         other_state = torch.FloatTensor([
#             state['speed'],
#             state['throttle'],
#             state['steer'],
#             state['waypoint_dist'],
#             state['waypoint_angle']
#         ]).to(DEVICE)

#         combined_state = torch.cat([mu.squeeze(), other_state])

#         # Get action from policy
#         action, value, log_prob = self.actor_critic.get_action(combined_state)

#         # Ensure action is numpy array with shape (2,)
#         action = action.cpu().numpy()
#         if len(action.shape) > 1:
#             action = action.squeeze()

#         return action, value.item(), log_prob.item()
    
#     def update(self):
#         if len(self.memory.states) == 0:
#             return

#         try:
#             # Convert lists to tensors
#             states = []
#             for state in self.memory.states:
#                 image = torch.FloatTensor(state['image']).to(DEVICE)
#                 image = image.permute(2, 0, 1).unsqueeze(0) / 255.0

#                 with torch.no_grad():
#                     _, mu, _ = self.vae(image)

#                 other_state = torch.FloatTensor([
#                     state['speed'],
#                     state['throttle'],
#                     state['steer'],
#                     state['waypoint_dist'],
#                     state['waypoint_angle']
#                 ]).to(DEVICE)

#                 combined_state = torch.cat([mu.squeeze(), other_state])
#                 states.append(combined_state)

#             if len(states) == 0:
#                 return

#             states = torch.stack(states)
#             actions = torch.FloatTensor(np.array(self.memory.actions)).to(DEVICE)
#             old_log_probs = torch.FloatTensor(self.memory.log_probs).to(DEVICE)

#             # Compute advantages with gradient clipping
#             advantages = self.memory.compute_returns_and_advantages(0, GAMMA, GAE_LAMBDA)
#             advantages = advantages.detach()

#             # Normalize advantages
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#             # PPO update
#             for _ in range(self.ppo_epochs):
#                 # Get current policy outputs
#                 log_probs, state_values, dist_entropy = self.actor_critic.evaluate(states, actions)

#                 # Calculate ratios
#                 ratios = torch.exp(log_probs - old_log_probs.detach())

#                 # Clip ratios to prevent numerical instability
#                 ratios = torch.clamp(ratios, 0.0, 10.0)

#                 # Calculate surrogate losses
#                 surr1 = ratios * advantages
#                 surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages

#                 # Calculate losses with gradient clipping
#                 actor_loss = -torch.min(surr1, surr2).mean()
#                 critic_loss = F.mse_loss(state_values, advantages + state_values.detach())
#                 entropy_loss = -0.01 * dist_entropy.mean()

#                 total_loss = actor_loss + 0.5 * critic_loss + entropy_loss

#                 # Optimize
#                 self.actor_critic_optimizer.zero_grad()
#                 total_loss.backward()

#                 # Clip gradients
#                 torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

#                 self.actor_critic_optimizer.step()

#                 # Logging
#                 self.writer.add_scalar('Losses/Actor', actor_loss.item(), self.training_step)
#                 self.writer.add_scalar('Losses/Critic', critic_loss.item(), self.training_step)
#                 self.writer.add_scalar('Losses/Entropy', entropy_loss.item(), self.training_step)

#                 self.training_step += 1

#         except Exception as e:
#             print(f"Error during update: {e}")
#             raise e

#         finally:
#             self.memory.clear()
           
#     def train_vae(self, images):
#         # Ensure images have correct dimensions
#         if images[0].shape != (IM_HEIGHT, IM_WIDTH, 3):
#             images = np.array([cv2.resize(img, (IM_WIDTH, IM_HEIGHT)) for img in images])

#         images = torch.FloatTensor(images).to(DEVICE)
#         images = images.permute(0, 3, 1, 2).contiguous() / 255.0

#         self.vae_optimizer.zero_grad()

#         reconstructed, mu, log_var = self.vae(images)

#         # Ensure shapes match before computing loss
#         if reconstructed.shape != images.shape:
#             print(f"Shape mismatch: reconstructed {reconstructed.shape}, images {images.shape}")
#             reconstructed = F.interpolate(reconstructed, size=(IM_HEIGHT, IM_WIDTH))

#         recon_loss = F.mse_loss(reconstructed, images)
#         kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
#         vae_loss = recon_loss + 0.1 * kl_loss

#         vae_loss.backward()
#         self.vae_optimizer.step()

#         self.writer.add_scalar('Losses/VAE', vae_loss.item(), self.training_step)

        
#     def save_checkpoint(self, episode, path='checkpoints'):
#         """
#         Save a complete training checkpoint that can be used to resume training
#         """
#         os.makedirs(path, exist_ok=True)
        
#         checkpoint = {
#             'episode': episode,
#             'training_step': self.training_step,
#             'best_reward': self.best_reward,
#             'episode_rewards': self.episode_rewards,
#             'vae_state_dict': self.vae.state_dict(),
#             'actor_critic_state_dict': self.actor_critic.state_dict(),
#             'vae_optimizer_state_dict': self.vae_optimizer.state_dict(),
#             'actor_critic_optimizer_state_dict': self.actor_critic_optimizer.state_dict(),
#         }
        
#         # Save regular checkpoint
#         checkpoint_path = os.path.join(path, f'checkpoint_episode_{episode}.pth')
#         torch.save(checkpoint, checkpoint_path)
        
#         # Save best model if current reward is best
#         if len(self.episode_rewards) > 0 and self.episode_rewards[-1] > self.best_reward:
#             self.best_reward = self.episode_rewards[-1]
#             best_model_path = os.path.join(path, 'best_model.pth')
#             torch.save(checkpoint, best_model_path)
#             print(f"Saved new best model with reward: {self.best_reward}")
        
#         # Save latest checkpoint for training continuation
#         latest_path = os.path.join(path, 'latest.pth')
#         torch.save(checkpoint, latest_path)
        
#         print(f"Saved checkpoint at episode {episode}")

#     def load_checkpoint(self, path, mode='train'):
#         if not os.path.exists(path):
#             print(f"No checkpoint found at {path}")
#             return False
            
#         checkpoint = torch.load(path)
        
#         # Load model states
#         self.vae.load_state_dict(checkpoint['vae_state_dict'])
#         self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        
#         if mode == 'train':
#             # Load training state
#             self.training_step = checkpoint['training_step']
#             self.current_episode = checkpoint['episode']
#             self.best_reward = checkpoint['best_reward']
#             self.episode_rewards = checkpoint['episode_rewards']
            
#             # Load optimizer states
#             self.vae_optimizer.load_state_dict(checkpoint['vae_optimizer_state_dict'])
#             self.actor_critic_optimizer.load_state_dict(checkpoint['actor_critic_optimizer_state_dict'])
            
#             print(f"Resumed training from episode {self.current_episode}")
#         else:
#             # For inference, just put models in eval mode
#             self.vae.eval()
#             self.actor_critic.eval()
#             print("Loaded model for inference")
        
#         return True
    
# def main():
#     env = None
#     agent = None
#     try:
#         # Setup logging
#         start_time = time.strftime('%Y-%m-%d %H:%M:%S')
#         print(f"Starting training at: {start_time}")
#         print(f"Using device: {DEVICE}")
        
#         # Create directories
#         os.makedirs('checkpoints', exist_ok=True)
#         os.makedirs('logs', exist_ok=True)
        
#         # Initialize environment with increased timeout
#         print("Initializing CARLA environment...")
#         env = CarEnv()
#         print("CARLA environment initialized successfully")
        
#         # Initialize agent
#         print("Initializing PPO agent...")
#         state_dim = 5  # speed, throttle, steer, waypoint_dist, waypoint_angle
#         action_dim = 2  # steer, throttle
#         agent = PPOAgent(state_dim, action_dim)
#         print("PPO agent initialized successfully")
        
#         # Try to load latest checkpoint
#         start_episode = 0
#         if os.path.exists('checkpoints/latest.pth'):
#             print("Found existing checkpoint. Attempting to load...")
#             try:
#                 if agent.load_checkpoint('checkpoints/latest.pth', mode='train'):
#                     start_episode = agent.current_episode
#                     print(f"Successfully loaded checkpoint. Resuming from episode {start_episode}")
#             except Exception as e:
#                 print(f"Failed to load checkpoint: {e}")
#                 print("Starting fresh training")
#         else:
#             print("No checkpoint found. Starting fresh training")
        
#         # Training loop variables
#         best_episode_reward = float('-inf')
#         total_timesteps = 0
#         update_timestep = 0
        
#         # Main training loop
#         for episode in tqdm(range(start_episode, EPISODES), desc="Training Progress"):
#             episode_start_time = time.time()
#             state = env.reset()
#             episode_reward = 0
#             episode_steps = 0
#             vae_training_images = []
            
#             print(f"\nStarting Episode {episode}/{EPISODES}")
            
#             try:
#                 # Episode loop
#                 for t in range(SECONDS_PER_EPISODE * 30):  # 30 FPS
#                     total_timesteps += 1
#                     episode_steps += 1
                    
#                     # Action selection
#                     try:
#                         action, value, log_prob = agent.select_action(state)
#                         if t % 100 == 0:
#                             print(f"Step {t}: Action: {action}, Value: {value:.3f}, Log prob: {log_prob:.3f}")
#                     except Exception as e:
#                         print(f"Error in action selection: {e}")
#                         continue
                    
#                     # Environment step
#                     try:
#                         next_state, reward, done, info = env.step(action)
#                     except Exception as e:
#                         print(f"Error in environment step: {e}")
#                         break
                    
#                     # Store experience
#                     agent.memory.states.append(state)
#                     agent.memory.actions.append(action)
#                     agent.memory.rewards.append(reward)
#                     agent.memory.values.append(value)
#                     agent.memory.log_probs.append(log_prob)
#                     agent.memory.masks.append(1 - done)
                    
#                     # Store images for VAE training
#                     if state['image'] is not None:
#                         vae_training_images.append(state['image'])
                    
#                     # Update state and reward
#                     state = next_state
#                     episode_reward += reward
                    
#                     # Policy update
#                     if len(agent.memory.states) >= UPDATE_TIMESTEP:
#                         print(f"\nPerforming update at step {t}")
#                         update_timestep += 1
                        
#                         try:
#                             # Prepare next value for advantage estimation
#                             with torch.no_grad():
#                                 image_tensor = torch.FloatTensor(state['image']).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0
#                                 mu, _ = agent.vae.encode(image_tensor)
#                                 other_state = torch.FloatTensor([
#                                     state['speed'],
#                                     state['throttle'],
#                                     state['steer'],
#                                     state['waypoint_dist'],
#                                     state['waypoint_angle']
#                                 ]).to(DEVICE)
#                                 combined_state = torch.cat([mu.squeeze(), other_state])
#                                 _, _, next_value = agent.actor_critic.forward(combined_state)
                            
#                             # Update policy
#                             agent.update()
                            
#                             # Train VAE
#                             if len(vae_training_images) > 0:
#                                 agent.train_vae(np.array(vae_training_images))
#                                 vae_training_images = []
                            
#                         except Exception as e:
#                             print(f"Error during update: {e}")
#                             continue
                    
#                     if done:
#                         print(f"\nEpisode ended after {t} steps")
#                         if 'termination_reason' in info:
#                             print(f"Termination reason: {info['termination_reason']}")
#                         break
                
#                 # Episode completion
#                 agent.episode_rewards.append(episode_reward)
#                 episode_duration = time.time() - episode_start_time
                
#                 # Log episode statistics
#                 print(f"\nEpisode {episode} Summary:")
#                 print(f"Reward: {episode_reward:.2f}")
#                 print(f"Steps: {episode_steps}")
#                 print(f"Duration: {episode_duration:.2f} seconds")
#                 print(f"Average step time: {episode_duration/episode_steps:.3f} seconds")
                
#                 # Update best reward
#                 if episode_reward > best_episode_reward:
#                     best_episode_reward = episode_reward
#                     print(f"New best reward: {best_episode_reward:.2f}")
                
#                 # TensorBoard logging
#                 if hasattr(agent, 'writer') and agent.writer is not None:
#                     agent.writer.add_scalar('Rewards/Episode', episode_reward, episode)
#                     agent.writer.add_scalar('Training/Steps_Per_Episode', episode_steps, episode)
#                     agent.writer.add_scalar('Training/Episode_Duration', episode_duration, episode)
                
#                 # Checkpoint saving
#                 if episode % 10 == 0:  # Save more frequently
#                     try:
#                         agent.save_checkpoint(episode)
#                         print(f"Saved checkpoint at episode {episode}")
#                     except Exception as e:
#                         print(f"Error saving checkpoint: {e}")
                
#             except Exception as e:
#                 print(f"Error in episode {episode}: {e}")
#                 # Emergency save
#                 try:
#                     agent.save_checkpoint(episode)
#                     print(f"Emergency checkpoint saved at episode {episode}")
#                 except Exception as save_error:
#                     print(f"Failed to save emergency checkpoint: {save_error}")
            
#             finally:
#                 # Clear any remaining images
#                 vae_training_images.clear()
        
#         # Training completion
#         try:
#             agent.save_checkpoint(EPISODES - 1)
#             print("\nTraining completed successfully!")
#             print(f"Best reward achieved: {best_episode_reward:.2f}")
#             print(f"Total timesteps: {total_timesteps}")
#             print(f"Final update timestep: {update_timestep}")
#         except Exception as e:
#             print(f"Error saving final checkpoint: {e}")
        
#     except Exception as e:
#         print(f"Fatal error in training: {e}")
#         raise e
    
#     finally:
#         # Cleanup
#         if env is not None:
#             try:
#                 env.client.apply_batch([carla.command.DestroyActor(x) for x in env.actor_list])
#                 print("Environment cleaned up")
#             except Exception as e:
#                 print(f"Error cleaning up environment: {e}")
        
#         if agent is not None and hasattr(agent, 'writer') and agent.writer is not None:
#             try:
#                 agent.writer.close()
#                 print("TensorBoard writer closed")
#             except Exception as e:
#                 print(f"Error closing TensorBoard writer: {e}")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nTraining interrupted by user")
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#     finally:
#         print("Program terminated")















































































































##DDQ LEARNING NETWORK USING TENSORFLOW(TOO BAD LEARNING)
# import glob
# import os
# import sys
# import random
# import time
# import numpy as np
# import cv2
# import math
# from collections import deque
# import shutil
# import tensorflow as tf
# from tensorflow.keras.applications import Xception
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model, load_model

# from threading import Thread
# from tqdm import tqdm

# # Add the CARLA Python API
# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

# import carla


# # Modified Constants for faster training
# SHOW_PREVIEW = False
# EPISODES = 1500
# IM_WIDTH = 320
# IM_HEIGHT = 240
# SECONDS_PER_EPISODE = 30
# REPLAY_MEMORY_SIZE = 50_000
# MIN_REPLAY_MEMORY_SIZE = 1000
# UPDATE_TARGET_EVERY = 5
# MINIBATCH_SIZE = 64  # Increased batch size
# PREDICTION_BATCH_SIZE = 1
# TRAINING_BATCH_SIZE = MINIBATCH_SIZE
# MEMORY_FRACTION = 0.7
# LEARNING_RATE = 0.001  # Reduced for stability
# DISCOUNT = 0.99
# EPSILON_DECAY = 0.999  # Slower decay
# MIN_EPSILON = 0.05
# EPSILON_GREEDY_FRAMES = 200000  # Frames over which to anneal epsilon
# AGGREGATE_STATS_EVERY = 10
# FPS = 30  # Increased for faster training
# MODEL_NAME = "Xception"
# MIN_REWARD = -200

# class ModifiedTensorBoard:
#     def __init__(self, **kwargs):
#         self.step = 1
#         self.log_dir = kwargs.get('log_dir')
#         self.writer = tf.summary.create_file_writer(self.log_dir)

#     def set_model(self, model):
#         pass

#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)

#     def on_batch_end(self, batch, logs=None):
#         pass

#     def on_train_end(self, _):
#         pass

#     def update_stats(self, **stats):
#         with self.writer.as_default():
#             for key, value in stats.items():
#                 tf.summary.scalar(key, value, step=self.step)
#             self.writer.flush()

# class CarEnv:
#     SHOW_CAM = SHOW_PREVIEW
#     STEER_AMT = 1.0
#     im_width = IM_WIDTH
#     im_height = IM_HEIGHT
#     front_camera = None
    
#     # Define fixed spawn coordinates
#     SPAWN_POINT = carla.Transform(
#         carla.Location(x=176.0, y=-10.0, z=0.5),  # Adjust these coordinates based on your map
#         carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)  # Adjust rotation as needed
#     )
    
#     def __init__(self):
#         self.client = carla.Client("localhost", 2000)
#         self.client.set_timeout(10.0)
#         self.world = self.client.get_world()
#         self.blueprint_library = self.world.get_blueprint_library()
#         self.model_3 = self.blueprint_library.filter("model3")[0]
#         self.actor_list = []
#         self.collision_hist = []
#         self.waypoint_list = []
#         self.current_waypoint = None
#         self.target_waypoint = None
#         self.last_location = None
#         self.last_steer = 0
#         self.same_steer_count = 0
#         self.stuck_time = None
#         self.last_velocity = 0
#         self.episode_start = 0
#         self.last_reward = 0
        
#     def reset(self):
#         # Clean up old actors
#         self._cleanup()
        
#         # Reset variables
#         self.collision_hist = []
#         self.last_location = None
#         self.last_steer = 0
#         self.same_steer_count = 0
#         self.stuck_time = None
#         self.last_velocity = 0
#         self.front_camera = None
        
#         # Use the fixed spawn point
#         try:
#             self.vehicle = self.world.spawn_actor(self.model_3, self.SPAWN_POINT)
#             self.actor_list.append(self.vehicle)
#         except RuntimeError as e:
#             print(f"Failed to spawn vehicle at fixed coordinates: {e}")
#             # If fixed spawn fails, try to find a clear spot nearby
#             spawn_points = self.world.get_map().get_spawn_points()
#             closest_point = min(spawn_points, 
#                               key=lambda p: p.location.distance(self.SPAWN_POINT.location))
#             try:
#                 self.vehicle = self.world.spawn_actor(self.model_3, closest_point)
#                 self.actor_list.append(self.vehicle)
#                 print("Spawned at nearest available point")
#             except RuntimeError as e2:
#                 print(f"Failed to spawn at alternate location: {e2}")
#                 return self.reset()
        
#         # Set up waypoints for navigation
#         self._setup_waypoints()
        
#         # Set up all sensors
#         self._setup_sensors()
        
#         # Wait for camera to be ready
#         while self.front_camera is None:
#             time.sleep(0.1)
        
#         # Initialize episode
#         self.episode_start = time.time()
#         self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0))
#         time.sleep(0.4)
        
#         return self.front_camera

#     def _setup_waypoints(self):
#         """Set up navigation waypoints with fixed target locations"""
#         self.current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
        
#         # Define fixed target coordinates for the route
#         target_coords = [
#             carla.Location(x=100.0, y=-10.0, z=0.5),  # First target
#             carla.Location(x=150.0, y=-10.0, z=0.5),  # Second target
#             carla.Location(x=200.0, y=-10.0, z=0.5)   # Third target
#         ]
        
#         # Get waypoints leading to the first target
#         self.waypoint_list = []
#         current_loc = self.current_waypoint.transform.location
#         target = target_coords[0]  # Use first target point
        
#         # Generate waypoints towards the target
#         next_waypoint = self.current_waypoint
#         while len(self.waypoint_list) < 100:  # Limit number of waypoints
#             next_waypoints = next_waypoint.next(2.0)  # Get waypoints every 2 meters
#             if not next_waypoints:
#                 break
#             self.waypoint_list.append(next_waypoints[0])
#             next_waypoint = next_waypoints[0]
            
#             # Break if we're close to the target
#             if next_waypoint.transform.location.distance(target) < 5.0:
#                 break
        
#         if self.waypoint_list:
#             self.target_waypoint = self.waypoint_list[-1]
#         else:
#             print("Warning: No waypoints generated!")
#             self.target_waypoint = self.current_waypoint

#     def _setup_sensors(self):
#         """Set up all vehicle sensors"""
#         # RGB Camera
#         self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
#         self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
#         self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
#         self.rgb_cam.set_attribute("fov", "110")
        
#         transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        
#         try:
#             self.camera_sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
#             self.actor_list.append(self.camera_sensor)
#             self.camera_sensor.listen(lambda data: self.process_img(data))
#         except Exception as e:
#             print(f"Failed to set up camera: {e}")
#             self._cleanup()
#             return self.reset()

#         # Collision sensor
#         col_sensor = self.blueprint_library.find("sensor.other.collision")
#         try:
#             self.collision_sensor = self.world.spawn_actor(col_sensor, transform, attach_to=self.vehicle)
#             self.actor_list.append(self.collision_sensor)
#             self.collision_sensor.listen(lambda event: self.collision_data(event))
#         except Exception as e:
#             print(f"Failed to set up collision sensor: {e}")
#             self._cleanup()
#             return self.reset()

#     def _cleanup(self):
#         """Clean up all actors"""
#         for actor in self.actor_list:
#             if actor and actor.is_alive:
#                 try:
#                     actor.destroy()
#                 except Exception as e:
#                     print(f"Error destroying actor {actor}: {e}")
#         self.actor_list.clear()

#     def collision_data(self, event):
#         """Handle collision events"""
#         self.collision_hist.append(event)

#     def process_img(self, image):
#         """Process camera images"""
#         i = np.array(image.raw_data, dtype=np.uint8)
#         i2 = i.reshape((self.im_height, self.im_width, 4))
#         i3 = i2[:, :, :3]
        
#         if self.SHOW_CAM:
#             cv2.imshow("CARLA Camera Feed", i3)
#             cv2.waitKey(1)
        
#         self.front_camera = i3



#     def calculate_reward(self, action):
#         reward = 0
#         done = False

#         # Get current state
#         v = self.vehicle.get_velocity()
#         kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
#         location = self.vehicle.get_location()
#         rotation = self.vehicle.get_transform().rotation

#         # Collision Penalty
#         if len(self.collision_hist) > 0:
#             done = True
#             reward -= 50.0  # Larger penalty for collision
#             return reward, done

#         # Speed Reward
#         if 40 <= kmh <= 80:  # Optimal driving speed range
#             reward += 2.0  # Strong reward for maintaining speed
#         elif 20 <= kmh < 40:
#             reward += 0.5 + (kmh * 0.01)  # Small reward for gradual acceleration
#         elif kmh < 20:
#             reward -= 0.05 * (20 - kmh)  # Penalize slow speeds
#         elif kmh > 80:
#             reward -= 0.02 * (kmh - 80)  # Penalize overspeeding

#         # Stuck Detection
#         if kmh < 1:
#             if self.stuck_time is None:
#                 self.stuck_time = time.time()
#             elif time.time() - self.stuck_time > 5:
#                 reward -= 10.0  # Larger penalty for being stuck
#                 done = True
#                 return reward, done
#         else:
#             self.stuck_time = None

#         # Lane-Keeping Reward
#         current_waypoint = self.world.get_map().get_waypoint(location)
#         if current_waypoint:
#             lane_dist = current_waypoint.transform.location.distance(location)
#             if lane_dist < 0.5:
#                 reward += 1.0  # Strong reward for staying close to lane center
#             elif lane_dist < 1.0:
#                 reward += 0.5  # Moderate reward
#             elif lane_dist < 2.0:
#                 reward += 0.2  # Small reward
#             else:
#                 reward -= 0.1 * lane_dist  # Penalize being far from the lane

#             # Wrong Lane Penalty
#             if current_waypoint.lane_id * self.current_waypoint.lane_id < 0:
#                 reward -= 20.0  # Strong penalty for driving on the wrong lane
#                 done = True
#                 return reward, done

#             # Heading Alignment Reward
#             heading_diff = abs(rotation.yaw - current_waypoint.transform.rotation.yaw)
#             heading_diff = min(heading_diff, 360 - heading_diff)
#             if heading_diff < 5:
#                 reward += 1.0  # Strong reward for good alignment
#             elif heading_diff < 15:
#                 reward += 0.5  # Moderate reward
#             elif heading_diff > 30:
#                 reward -= 0.1 * heading_diff  # Penalize large misalignments

#         # Progress Reward
#         if self.target_waypoint and self.last_location:
#             prev_distance = self.last_location.distance(self.target_waypoint.transform.location)
#             curr_distance = location.distance(self.target_waypoint.transform.location)
#             progress_reward = max(0, (prev_distance - curr_distance) * 0.5)  # Ensure progress reward is non-negative
#             reward += progress_reward

#             # Waypoint Reached Bonus
#             if curr_distance < 5.0:
#                 reward += 5.0  # Strong bonus for reaching waypoint
#                 self._setup_waypoints()

#         # Steering Smoothness
#         steering_penalty = abs(action - self.last_steer) * 0.1
#         reward -= steering_penalty  # Penalize abrupt changes in steering
#         if action == self.last_steer:
#             self.same_steer_count += 1
#             if self.same_steer_count > 10:
#                 reward -= 0.5  # Penalize excessive same steering
#         else:
#             self.same_steer_count = 0

#         # Exploration Bonus
#         if self.last_location:
#             distance_from_last = self.last_location.distance(location)
#             if distance_from_last > 2.0:  # Add a bonus for significant movement
#                 reward += 0.5

#         # Update Last States
#         self.last_location = location
#         self.last_steer = action
#         self.last_velocity = kmh

#         return reward, done


#     def step(self, action):
#         """Execute one environment step"""
#         # Apply vehicle control
#         if action == 0:    # Left
#             self.vehicle.apply_control(carla.VehicleControl(throttle=1, steer=-1* self.STEER_AMT))
#         elif action == 1:  # Straight
#             self.vehicle.apply_control(carla.VehicleControl(throttle=1, steer=0))
#         elif action == 2:  # Right
#             self.vehicle.apply_control(carla.VehicleControl(throttle=1, steer=1 * self.STEER_AMT))
        
#         # Calculate reward
#         reward, done = self.calculate_reward(action)
        
#         # Check for episode timeout
#         if self.episode_start + SECONDS_PER_EPISODE < time.time():
#             done = True
        
#         return self.front_camera, reward, done, None

#     def __del__(self):
#         """Destructor to clean up actors"""
#         self._cleanup()

# class DDQNAgent:
#     def __init__(self):
#         if not os.path.isdir("models"):
#             os.makedirs("models")
        
#         # Initialize model after loading state
#         self.model = None
#         self.target_model = None
#         self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
#         self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
#         self.target_update_counter = 0
#         self.frame_count = 0
#         self.terminate = False
#         self.last_logged_episode = 0
#         self.training_initialized = False
        
#         # Load or create model
#         self.initialize_models()
    
#     def create_model(self):
#         print("Creating DDQN model...")
#         base_model = Xception(weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 3))

#         x = base_model.output
#         x = GlobalAveragePooling2D(name='global_avg_pooling')(x)
#         # Simplified dense layers with adjusted sizes
#         x = Dense(256, activation='relu', name='dense_1')(x)
#         x = Dense(128, activation='relu', name='dense_2')(x)
#         # Add batch normalization and dropout for better stability
#         x = tf.keras.layers.BatchNormalization()(x)
#         x = tf.keras.layers.Dropout(0.3)(x)
#         # Change activation to softmax for proper probability distribution
#         predictions = Dense(3, activation="softmax", name='output')(x)

#         model = Model(inputs=base_model.input, outputs=predictions, name='ddqn_model')
#         # Increase learning rate and switch to MSE loss
#         model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
#         return model
    

#     def initialize_models(self):
#         """Initialize or load models with proper error handling"""
#         latest_checkpoint = 'models/checkpoint_latest.h5'
        
#         if os.path.exists(latest_checkpoint):
#             print("Attempting to load previous model...")
#             try:
#                 # Try to load the model directly
#                 self.model = load_model(latest_checkpoint, custom_objects={
#                     'huber_loss': tf.keras.losses.Huber()
#                 })
#                 print("Successfully loaded previous model")
                
#                 # Create and sync target model
#                 self.target_model = self.create_model()
#                 self.target_model.set_weights(self.model.get_weights())
                
#                 # Load training state
#                 if os.path.exists('models/training_state.npz'):
#                     training_state = np.load('models/training_state.npz', allow_pickle=True)
#                     global epsilon
#                     epsilon = float(training_state['epsilon'])
#                     self.target_update_counter = int(training_state['target_update_counter'])
#                     self.frame_count = int(training_state['frame_count'])
#                     print(f"Resumed training state - epsilon: {epsilon}, frame count: {self.frame_count}")
#             except Exception as e:
#                 print(f"Error loading model: {e}")
#                 print("Creating new models...")
#                 self.model = self.create_model()
#                 self.target_model = self.create_model()
#                 self.target_model.set_weights(self.model.get_weights())
#         else:
#             print("No previous model found. Creating new models...")
#             self.model = self.create_model()
#             self.target_model = self.create_model()
#             self.target_model.set_weights(self.model.get_weights())
    
#     def save_model(self, episode, suffix=''):
#         """Save the model and training state with proper error handling"""
#         try:
#             # Create backup of existing latest checkpoint if it exists
#             latest_path = 'models/checkpoint_latest.h5'
#             if os.path.exists(latest_path):
#                 backup_path = latest_path.replace('.h5', '_backup.h5')
#                 if os.path.exists(backup_path):
#                     os.remove(backup_path)
#                 os.rename(latest_path, backup_path)
            
#             # Save new checkpoint
#             checkpoint_path = f'models/checkpoint_{episode}{suffix}.h5'
#             self.model.save(checkpoint_path)
            
#             # Save as latest checkpoint
#             shutil.copy2(checkpoint_path, latest_path)
            
#             # Save training state
#             np.savez('models/training_state.npz', 
#                     epsilon=epsilon,
#                     target_update_counter=self.target_update_counter,
#                     frame_count=self.frame_count)
            
#             print(f"Successfully saved model and training state")
#         except Exception as e:
#             print(f"Error saving model: {e}")

#     def get_epsilon(self):
#         """Calculate epsilon based on frame count"""
#         # Linear epsilon decay
#         if self.frame_count < EPSILON_GREEDY_FRAMES:
#             return 1.0 - (self.frame_count / EPSILON_GREEDY_FRAMES)
#         return MIN_EPSILON

#     def update_replay_memory(self, transition):
#         """Add transition to replay memory"""
#         self.replay_memory.append(transition)

#     def train(self):
#         """Train the model using experience replay"""
#         if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
#             return
        
#         # Add gradient clipping to optimizer
#         optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
#         self.model.compile(loss="mse", optimizer=optimizer, metrics=["accuracy"])
#         minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

#         current_states = np.array([transition[0] for transition in minibatch])/255
#         # Get Q values for current states from model
#         current_qs_list = self.model.predict(current_states, batch_size=TRAINING_BATCH_SIZE)

#         new_current_states = np.array([transition[3] for transition in minibatch])/255
#         # Get Q values for next states from model
#         future_qs_main = self.model.predict(new_current_states, batch_size=TRAINING_BATCH_SIZE)
#         # Get Q values for next states from target model
#         future_qs_target = self.target_model.predict(new_current_states, batch_size=TRAINING_BATCH_SIZE)

#         X = []
#         y = []

#         for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
#             if not done:
#                 # DDQN Update: Use action from main model and value from target model
#                 max_future_action = np.argmax(future_qs_main[index])
#                 max_future_q = future_qs_target[index][max_future_action]
#                 new_q = reward + DISCOUNT * max_future_q
#             else:
#                 new_q = reward

#             current_qs = current_qs_list[index]
#             current_qs[action] = new_q

#             X.append(current_state)
#             y.append(current_qs)

#         # Train main network
#         self.model.fit(
#             np.array(X)/255,
#             np.array(y),
#             batch_size=TRAINING_BATCH_SIZE,
#             verbose=0,
#             shuffle=False,
#             callbacks=[self.tensorboard] if self.tensorboard.step > self.last_logged_episode else None
#         )

#         if self.tensorboard.step > self.last_logged_episode:
#             self.target_update_counter += 1
#             self.last_logged_episode = self.tensorboard.step

#         # Update target network if needed
#         if self.target_update_counter > UPDATE_TARGET_EVERY:
#             self.target_model.set_weights(self.model.get_weights())
#             self.target_update_counter = 0

#     def get_qs(self, state):
#         """Get Q values for current state"""
#         self.frame_count += 1
#         return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

#     def train_in_loop(self):
#         """Initialize training in a separate thread"""
#         X = np.random.uniform(size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
#         y = np.random.uniform(size=(1, 3)).astype(np.float32)
#         self.model.fit(X, y, verbose=0, batch_size=1)
#         self.training_initialized = True

#         while True:
#             if self.terminate:
#                 return
#             self.train()
#             time.sleep(0.01)



# if __name__ == "__main__":
#     # Set memory growth for GPU
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         except RuntimeError as e:
#             print(f"RuntimeError: {e}")
    
#     epsilon = 1.0
#     ep_rewards = [-200]

#     random.seed(1)
#     np.random.seed(1)
#     tf.random.set_seed(1)

#     agent = DDQNAgent()
#     env = CarEnv()

#     trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
#     trainer_thread.start()

#     while not agent.training_initialized:
#         time.sleep(0.01)

#     agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

#     try:
#         for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episodes"):
#             env.collision_hist = []
#             agent.tensorboard.step = episode
#             episode_reward = 0
#             step = 1
#             current_state = env.reset()
#             done = False

#             while True:
#                 if np.random.random() > agent.get_epsilon():
#                     action = np.argmax(agent.get_qs(current_state))
#                 else:
#                     action = np.random.randint(0, 3)
#                     time.sleep(1/FPS)

#                 new_state, reward, done, _ = env.step(action)
#                 episode_reward += reward
#                 agent.update_replay_memory((current_state, action, reward, new_state, done))

#                 current_state = new_state
#                 step += 1

#                 if done:
#                     break

#             # Save checkpoints and update statistics
#             if episode % 200 == 0:
#                 agent.save_model(episode)

#             ep_rewards.append(episode_reward)
#             if not episode % AGGREGATE_STATS_EVERY or episode == 1:
#                 average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
#                 min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
#                 max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
#                 agent.tensorboard.update_stats(
#                     reward_avg=average_reward,
#                     reward_min=min_reward,
#                     reward_max=max_reward,
#                     epsilon=agent.get_epsilon()
#                 )

#                 # if min_reward >= MIN_REWARD:
#                 #     agent.save_model(
#                 #         episode,
#                 #         suffix=f'_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min'
#                 #     )

#     finally:
#         # Clean up
#         for actor in env.actor_list:
#             actor.destroy()

#         agent.terminate = True
#         trainer_thread.join()
#         agent.save_model(
#             EPISODES,
#             suffix=f'_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min'
#         )























































































    # def _calculate_reward(self):
    #     """
    #     PPO-optimized reward function for self-driving car in CARLA.
    #     """
    #     # Constants optimized for PPO
    #     v_target = 40.0    # Target speed (km/h)
    #     v_margin = 10.0    # Acceptable speed margin
    #     d_threshold = 1.0  # Optimal lane center distance (meters)
        
    #     # Get current state
    #     location = self.vehicle.get_location()
    #     velocity = self.vehicle.get_velocity()
    #     transform = self.vehicle.get_transform()
    #     waypoint = self.world.get_map().get_waypoint(location)
        
    #     # Calculate speed (km/h)
    #     speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
    #     # 1. Safety Checks (Terminal Conditions)
    #     if len(self.collision_hist) > 0:
    #         return -1.0, True, {'termination_reason': 'collision'}
        
    #     # 2. Distance from Lane Center (normalized between -1 and 1)
    #     distance = location.distance(waypoint.transform.location)
    #     distance_reward = -2 * (distance / d_threshold) if distance < d_threshold else -1.0
    #     distance_reward = max(-1.0, min(0.0, distance_reward))
        
    #     # 3. Speed Reward (normalized between -1 and 1)
    #     speed_diff = abs(speed - v_target)
    #     speed_reward = 1.0 - (speed_diff / v_margin) if speed_diff < v_margin else -1.0
    #     speed_reward = max(-1.0, min(1.0, speed_reward))
        
    #     # 4. Direction Alignment (normalized between -1 and 1)
    #     forward_vec = transform.get_forward_vector()
    #     waypoint_vec = waypoint.transform.get_forward_vector()
    #     dot_product = forward_vec.x * waypoint_vec.x + forward_vec.y * waypoint_vec.y
    #     angle_reward = dot_product
        
    #     # 5. Progress Reward (normalized between 0 and 1)
    #     progress = location.distance(self.previous_state['location'])
    #     progress_reward = min(1.0, progress / 5.0)
        
    #     # 6. Smooth Driving Reward
    #     acceleration = abs(speed - self.previous_state['velocity']) / 0.1
    #     smoothness_reward = -min(1.0, acceleration / 10.0)
        
    #     # Combine rewards with weighted importance
    #     total_reward = (
    #         0.3 * speed_reward +      # Speed matching
    #         0.2 * distance_reward +   # Lane keeping
    #         0.2 * angle_reward +      # Direction alignment
    #         0.2 * progress_reward +   # Forward progress
    #         0.1 * smoothness_reward   # Smooth driving
    #     )
        
    #     # Bonus for consistent good behavior
    #     if (speed_reward > 0.5 and 
    #         distance_reward > -0.2 and 
    #         angle_reward > 0.8):
    #         self.previous_state['consecutive_good_actions'] += 1
    #         if self.previous_state['consecutive_good_actions'] >= 10:
    #             total_reward += 0.1
    #     else:
    #         self.previous_state['consecutive_good_actions'] = 0
        
    #     # Update previous state
    #     self.previous_state.update({
    #         'location': location,
    #         'velocity': speed,
    #         'cumulative_reward': self.previous_state['cumulative_reward'] + total_reward
    #     })
        
    #     # Check timeout
    #     done = False
    #     if time.time() - self.episode_start > SECONDS_PER_EPISODE:
    #         done = True
        
    #     info = {
    #         'speed_kmh': speed,
    #         'distance_to_center': distance,
    #         'speed_reward': speed_reward,
    #         'distance_reward': distance_reward,
    #         'angle_reward': angle_reward,
    #         'progress_reward': progress_reward,
    #         'smoothness_reward': smoothness_reward,
    #         'total_reward': total_reward,
    #         'cumulative_reward': self.previous_state['cumulative_reward']
    #     }
        
    #     return total_reward, done, info