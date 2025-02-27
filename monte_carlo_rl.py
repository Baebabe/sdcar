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
EPISODES = 200
SECONDS_PER_EPISODE = 150
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
    
    # Add confidence measure based on bbox size
    confidence = min(1.0, bbox_width / image_width)  # Higher confidence for larger objects
    
    return depth, confidence

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Smaller initial variance for better stability
        self.action_var = torch.full((action_dim,), 0.5).to(DEVICE)
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
        self.monte_carlo_returns = []

    def compute_monte_carlo_returns(self, GAMMA):
        returns = []
        running_return = 0
        
        # Compute returns in reverse order
        for reward in reversed(self.rewards):
            running_return = reward + GAMMA * running_return
            returns.insert(0, running_return)
            
        # Convert to tensor and normalize
        returns = torch.tensor(returns).float().to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        self.monte_carlo_returns = returns
        return returns

    def get_batch_data(self):
        states = torch.FloatTensor(self.states).to(DEVICE)
        actions = torch.FloatTensor(self.actions).to(DEVICE)
        old_log_probs = torch.FloatTensor(self.log_probs).to(DEVICE)
        returns = self.monte_carlo_returns
        
        return states, actions, old_log_probs, returns

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.masks.clear()
        self.monte_carlo_returns.clear()


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
            
        # Compute Monte Carlo returns for the entire episode
        returns = self.memory.compute_monte_carlo_returns(GAMMA)
        
        # Get batch data
        states, actions, old_log_probs, returns = self.memory.get_batch_data()
        
        # Create dataset for batch processing
        dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        # PPO update for K epochs
        for _ in range(K_EPOCHS):
            for batch_states, batch_actions, batch_old_log_probs, batch_returns in dataloader:
                # Get current policy outputs
                mean, cov_mat, state_values = self.actor_critic(batch_states)
                dist = MultivariateNormal(mean, cov_mat)
                log_probs = dist.log_prob(batch_actions)
                
                # Calculate ratio and surrogate loss
                ratios = torch.exp(log_probs - batch_old_log_probs.detach())
                
                # Calculate advantages as difference between returns and values
                advantages = batch_returns - state_values.squeeze()
                
                # PPO policy loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss using Monte Carlo returns
                critic_loss = F.mse_loss(state_values.squeeze(), batch_returns)
                
                # Entropy loss for exploration
                entropy_loss = -dist.entropy().mean()
                
                # Total loss
                total_loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()
                
                # Log metrics
                self.writer.add_scalar('Losses/Actor', actor_loss.item(), self.training_step)
                self.writer.add_scalar('Losses/Critic', critic_loss.item(), self.training_step)
                self.writer.add_scalar('Losses/Entropy', entropy_loss.item(), self.training_step)
                self.writer.add_scalar('Metrics/Value_Mean', state_values.mean().item(), self.training_step)
                self.writer.add_scalar('Metrics/Returns_Mean', batch_returns.mean().item(), self.training_step)
                
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
        """Improved reward function to encourage movement and exploration"""
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
                reward -= 0.5  # Larger penalty for not moving
                self.stuck_time += 0.1
                if self.stuck_time > 3.0:  # Reduced from 5.0 to 3.0
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
    
    def step(self, action):
        try:
            # Ensure actions are properly scaled
            throttle = float(np.clip((action[1] + 1) / 2, 0.0, 1.0))  # Convert from [-1,1] to [0,1]
            steer = float(np.clip(action[0], -1.0, 1.0))

            # Get current control
            current_control = self.vehicle.get_control()

            # Smooth control changes
            smooth_throttle = 0.8 * current_control.throttle + 0.2 * throttle
            smooth_steer = 0.8 * current_control.steer + 0.2 * steer

            # Apply brake only when reducing speed
            brake = 0.0
            if throttle < current_control.throttle:
                brake = min((current_control.throttle - throttle) * 2.0, 1.0)

            # Create and apply control
            control = carla.VehicleControl(
                throttle=smooth_throttle,
                steer=smooth_steer,
                brake=brake,
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False
            )

            self.vehicle.apply_control(control)

            # Tick the world multiple times for better physics
            for _ in range(4):
                self.world.tick()

            # Get new state and calculate reward
            new_state = self.get_state()
            reward, done, info = self.calculate_reward()

            # Add control info for debugging
            info['throttle'] = smooth_throttle
            info['steer'] = smooth_steer
            info['brake'] = brake

            return new_state, reward, done, info

        except Exception as e:
            print(f"Error in step: {e}")
            traceback.print_exc()
            return None, 0, True, {'error': str(e)}
    


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















