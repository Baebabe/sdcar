



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
import json
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
    REAL_WIDTHS = {
        0: 0.45,   # person - average shoulder width
        1: 0.8,    # bicycle - typical handlebar width
        2: 0.8,    # motorcycle - typical handlebar width
        3: 1.8,    # car - average car width
        4: 2.5,    # truck - average truck width
        5: 2.9,    # bus - average bus width
        6: 3.0,    # train - typical train car width
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

class MPCController:
    def __init__(self):
        self.horizon = 10
        self.dt = 0.05
        self.max_speed = 50.0  # km/h
        self.max_steer = 1.0  # radians

    def predict_trajectory(self, state, actions):
        """Predict trajectory given current state and actions"""
        trajectory = []
        current_state = np.array([state[0], state[1]])  # Only use speed and steering
        
        for action in actions:
            # Update state based on action
            new_state = self.update_state(current_state, action)
            trajectory.append(new_state)
            current_state = new_state
            
        return trajectory

    def update_state(self, state, action):
        """Update state based on action"""
        speed = state[0]
        steer = state[1]
        
        # Extract actions
        throttle = action[0]
        brake = action[1]
        steer_change = action[2]
        
        # Update speed based on throttle and brake
        new_speed = speed + (throttle - brake) * self.dt
        new_speed = np.clip(new_speed, 0, self.max_speed)
        
        # Update steering angle
        new_steer = steer + steer_change * self.dt
        new_steer = np.clip(new_steer, -self.max_steer, self.max_steer)
        
        return np.array([new_speed, new_steer])

    def calculate_cost(self, trajectory, waypoints):
        """Calculate cost of trajectory with proper dimension handling"""
        try:
            if not trajectory or not waypoints:
                return float('inf')
                
            cost = 0
            for i, state in enumerate(trajectory):
                if i >= len(waypoints):
                    break
                    
                wp = waypoints[i]
                
                # Calculate distance cost
                speed, steer = state
                wp_x, wp_y = wp['x'], wp['y']
                
                # Simple distance cost based on waypoint position
                distance_cost = wp_x * wp_x + wp_y * wp_y
                
                # Penalize deviation from target speed (30 km/h)
                speed_cost = abs(speed - 30.0) / 30.0
                
                # Penalize large steering angles
                steer_cost = abs(steer) / self.max_steer
                
                # Combine costs with weights
                cost += distance_cost + 0.5 * speed_cost + 0.3 * steer_cost
                
            return cost
            
        except Exception as e:
            print(f"Error in calculate_cost: {e}")
            return float('inf')

    def optimize(self, state, waypoints):
        """Optimize actions to follow waypoints"""
        try:
            if not waypoints:
                return None
                
            # Extract relevant state components (speed and steering)
            current_state = np.array([state[0], state[1]])  # Assuming speed and steering are first two components
            
            best_actions = []
            best_cost = float('inf')
            
            # Generate random actions and select the best one
            for _ in range(100):
                actions = []
                for _ in range(self.horizon):
                    throttle = np.random.uniform(0, 1)
                    brake = np.random.uniform(0, 1) if throttle < 0.5 else 0
                    steer = np.random.uniform(-self.max_steer, self.max_steer)
                    actions.append([throttle, brake, steer])
                
                trajectory = self.predict_trajectory(current_state, actions)
                cost = self.calculate_cost(trajectory, waypoints)
                
                if cost < best_cost:
                    best_cost = cost
                    best_actions = actions
            
            if best_cost == float('inf'):
                return None
                
            return best_actions[0]  # Return first action
            
        except Exception as e:
            print(f"Error in MPC optimize: {e}")
            return None


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), 0.5).to(DEVICE)
        self.cov_mat = torch.diag(self.action_var)
        
        # Ensure consistent layer sizes
        hidden1_size = 128
        hidden2_size = 64
        
        # Actor network with proper dimensions
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden1_size),
            nn.LayerNorm(hidden1_size),  # Add normalization
            nn.Tanh(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.LayerNorm(hidden2_size),  # Add normalization
            nn.Tanh(),
            nn.Linear(hidden2_size, action_dim),
            nn.Tanh()
        )
        
        # Critic network with proper dimensions
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden1_size),
            nn.LayerNorm(hidden1_size),  # Add normalization
            nn.Tanh(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.LayerNorm(hidden2_size),  # Add normalization
            nn.Tanh(),
            nn.Linear(hidden2_size, 1)
        )
        
        # Initialize weights using orthogonal initialization
        self.init_weights()
    
    def init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        mean = self.actor(state)
        mean = torch.clamp(mean, -1, 1)
        value = self.critic(state)
        return mean, self.cov_mat.to(state.device), value
    
    def get_action(self, state):
        """Get action with proper state dimension handling"""
        try:
            with torch.no_grad():
                if isinstance(state, np.ndarray):
                    # Ensure state is a batch
                    if state.ndim == 1:
                        state = state.reshape(1, -1)
                    state = torch.FloatTensor(state).to(DEVICE)
                
                # Ensure state has correct batch dimension
                if state.ndim == 1:
                    state = state.unsqueeze(0)
                
                # Debug print
                if state.shape[1] != self.state_dim:
                    raise ValueError(f"Expected state dimension {self.state_dim}, got {state.shape[1]}")
                
                mean = self.actor(state)
                mean = torch.clamp(mean, -1, 1)
                dist = MultivariateNormal(mean, self.cov_mat.to(state.device))
                action = dist.sample()
                action_log_prob = dist.log_prob(action)
                action = torch.clamp(action, -1, 1)
                value = self.critic(state)
                
                return action.squeeze(), value.squeeze(), action_log_prob.squeeze()
                
        except Exception as e:
            print(f"Error in get_action: {str(e)}")
            raise

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
        """
        Modified PPOAgent initialization with explicit state dimension check
        """
        self.max_objects = 5  # Should match CarEnv.max_objects
        # State dim should be (max_objects * 3 + 6) for vehicle state
        expected_state_dim = self.max_objects * 3 + 6
        if state_dim != expected_state_dim:
            print(f"Warning: Provided state_dim {state_dim} doesn't match expected {expected_state_dim}")
            state_dim = expected_state_dim
        
        self.state_dim = state_dim
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
        """
        Modified select_action method with better error handling
        """
        try:
            if state is None:
                return None, None, None

            # Convert state to numpy array if it isn't already
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)

            # Reshape state if necessary
            if state.ndim == 1:
                state = state.reshape(1, -1)

            # Verify state dimensions
            if state.shape[1] != self.state_dim:
                print(f"State shape mismatch. Got {state.shape}, expected (1, {self.state_dim})")
                print(f"State values: {state}")
                raise ValueError(f"Invalid state dimension. Expected {self.state_dim}, got {state.shape[1]}")

            state_tensor = torch.FloatTensor(state).to(DEVICE)
            action, value, log_prob = self.actor_critic.get_action(state_tensor)

            return action.cpu().numpy(), value.item(), log_prob.item()

        except Exception as e:
            print(f"Error in select_action: {e}")
            print(f"State shape: {state.shape if isinstance(state, np.ndarray) else 'not numpy array'}")
            print(f"State values: {state}")
            raise

    def update(self):
        if len(self.memory.states) == 0:
            return
        states = torch.FloatTensor(self.memory.states).to(DEVICE)
        actions = torch.FloatTensor(self.memory.actions).to(DEVICE)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(DEVICE)
        advantages = self.memory.compute_returns_and_advantages(0, GAMMA, GAE_LAMBDA)
        for _ in range(K_EPOCHS):
            mean, cov_mat, state_values = self.actor_critic(states)
            dist = MultivariateNormal(mean, cov_mat)
            log_probs = dist.log_prob(actions)
            ratios = torch.exp(log_probs - old_log_probs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(state_values.squeeze(), advantages + state_values.detach().squeeze())
            entropy_loss = -dist.entropy().mean()
            total_loss = actor_loss + VALUE_COEF * critic_loss + ENTROPY_COEF * entropy_loss
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
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            if mode == 'train':
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_reward = checkpoint.get('best_reward', float('-inf'))
            self.training_step = checkpoint.get('training_step', 0)
            episode = checkpoint.get('episode', 0)
            print(f"Loaded checkpoint from episode {episode}")
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
        self.front_camera = None
        self.npc_vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []
        self.mpc_controller = MPCController()
        self.use_mpc = True  # Start with MPC as primary controller
        self.mpc_failure_count = 0  # Track consecutive MPC failures
        self.mpc_failure_threshold = 3  # Switch to RL after 3 consecutive failures

        pygame.init()
        self.display = None
        self.clock = None
        self.init_pygame_display()

        self.camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7),
            carla.Rotation(pitch=0)
        )
        
        self.setup_world()
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
                array = array[:, :, :3]
                self.front_camera = array

                detections = self.process_yolo_detection(array)

                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

                for obj in detections:
                    x1 = int(obj['position'][0] - obj['bbox_width']/2)
                    y1 = int(obj['position'][1] - obj['bbox_height']/2)
                    x2 = int(obj['position'][0] + obj['bbox_width']/2)
                    y2 = int(obj['position'][1] + obj['bbox_height']/2)

                    x1 = int(x1 * IM_WIDTH / 640)
                    y1 = int(y1 * IM_HEIGHT / 640)
                    x2 = int(x2 * IM_WIDTH / 640)
                    y2 = int(y2 * IM_HEIGHT / 640)

                    pygame.draw.rect(surface, (0, 255, 0), (x1, y1, x2-x1, y2-y1), 2)

                    if pygame.font.get_init():
                        font = pygame.font.Font(None, 24)
                        label = f"{obj['class_name']} {obj['depth']:.1f}m"
                        text = font.render(label, True, (255, 255, 255))
                        surface.blit(text, (x1, y1-20))

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
        self.collision_hist = []
        self.stuck_time = 0
        self.episode_start = time.time()
        self.last_location = None

        # Reset destination
        self.reset_destination()

        if not self.setup_vehicle():
            raise Exception("Failed to setup vehicle")

        print("Waiting for camera initialization...")
        timeout = time.time() + 10.0
        while self.front_camera is None:
            self.world.tick()
            if time.time() > timeout:
                raise Exception("Camera initialization timeout")
            time.sleep(0.1)

        self.spawn_npcs()

        for _ in range(20):
            self.world.tick()
            time.sleep(0.05)

        state = self.get_state()
        if state is None:
            raise Exception("Failed to get initial state")

        return state

    def setup_vehicle(self):
        """Spawn and setup the ego vehicle"""
        try:
            print("Starting vehicle setup...")
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('model3')[0]
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points found")
            spawn_point = random.choice(spawn_points)
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print("Vehicle spawned successfully")

            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
            camera_bp.set_attribute('fov', '90')
            self.camera = self.world.spawn_actor(
                camera_bp,
                self.camera_transform,
                attach_to=self.vehicle
            )
            print("Camera spawned successfully")

            weak_self = weakref.ref(self)
            self.camera.listen(lambda image: self._process_image(weak_self, image))
            print("Camera callback set up")

            collision_bp = blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(
                collision_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )
            print("Collision sensor spawned")

            self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
            print("Collision callback set up")

            for _ in range(10):
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
            self.world = self.client.get_world()
            self.traffic_manager = self.client.get_trafficmanager(8000)
            self.traffic_manager.set_synchronous_mode(True)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            self.traffic_manager.global_percentage_speed_difference(10.0)
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)
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
            weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5s.pt')
            weights_url = 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt'
            download_weights(weights_url, weights_path)
            print(f"Loading model from: {weights_path}")
            import torch
            from models.experimental import attempt_load
            self.yolo_model = attempt_load(weights_path, device=DEVICE)
            self.yolo_model.conf = 0.25
            self.yolo_model.iou = 0.45
            self.yolo_model.classes = None
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
            img = cv2.resize(image, (640, 640))
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(DEVICE)
            img = img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            with torch.no_grad():
                pred = self.yolo_model(img)[0]
            from utils.general import non_max_suppression
            pred = non_max_suppression(pred, 
                                     conf_thres=0.25,
                                     iou_thres=0.45,
                                     classes=None,
                                     agnostic=False,
                                     max_det=300)
            objects = []
            if len(pred[0]):
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
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        steering = self.vehicle.get_control().steer
        return {
            'speed': speed,
            'steering': steering
        }
    
    def get_waypoints(self, num_waypoints=10, distance=5.0):
        """
        Get future waypoints for trajectory planning

        Args:
            num_waypoints (int): Number of waypoints to return
            distance (float): Distance between waypoints in meters

        Returns:
            list: List of waypoint dictionaries containing position and orientation
        """
        try:
            waypoints = []
            if not self.vehicle:
                return waypoints

            # Get current vehicle waypoint
            current_waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location())
            if not current_waypoint:
                return waypoints

            # Get vehicle transform for relative calculations
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            vehicle_rotation = vehicle_transform.rotation

            next_waypoint = current_waypoint
            for i in range(num_waypoints):
                # Get next waypoint
                next_waypoints = next_waypoint.next(distance)
                if not next_waypoints:
                    break
                next_waypoint = next_waypoints[0]

                # Calculate relative position to vehicle
                waypoint_location = next_waypoint.transform.location
                relative_x = waypoint_location.x - vehicle_location.x
                relative_y = waypoint_location.y - vehicle_location.y

                # Convert to vehicle's local coordinate system
                yaw_rad = math.radians(-vehicle_rotation.yaw)
                local_x = relative_x * math.cos(yaw_rad) - relative_y * math.sin(yaw_rad)
                local_y = relative_x * math.sin(yaw_rad) + relative_y * math.cos(yaw_rad)

                # Calculate relative heading
                waypoint_yaw = next_waypoint.transform.rotation.yaw
                relative_yaw = math.radians(waypoint_yaw - vehicle_rotation.yaw)
                while relative_yaw > math.pi:
                    relative_yaw -= 2 * math.pi
                while relative_yaw < -math.pi:
                    relative_yaw += 2 * math.pi

                waypoints.append({
                    'x': local_x,
                    'y': local_y,
                    'yaw': relative_yaw,
                    'road_id': next_waypoint.road_id,
                    'lane_id': next_waypoint.lane_id,
                    'distance': math.sqrt(local_x**2 + local_y**2)
                })

                # Debug visualization
                if hasattr(self, 'world') and hasattr(self.world, 'debug'):
                    self.world.debug.draw_point(
                        waypoint_location,
                        size=0.1,
                        color=carla.Color(r=0, g=255, b=0),
                        life_time=0.1
                    )
                    self.world.debug.draw_line(
                        vehicle_location,
                        waypoint_location,
                        thickness=0.1,
                        color=carla.Color(r=0, g=255, b=0),
                        life_time=0.1
                    )

            return waypoints

        except Exception as e:
            print(f"Error in get_waypoints: {e}")
            traceback.print_exc()
            return []

    def get_destination(self):
        """
        Get destination point for the current episode
        """
        try:
            if not hasattr(self, '_destination') or self._destination is None:
                # Get all possible spawn points
                spawn_points = self.world.get_map().get_spawn_points()
                if not spawn_points:
                    return self.vehicle.get_location()

                # Filter spawn points that are too close or too far
                vehicle_location = self.vehicle.get_location()
                valid_points = [p for p in spawn_points 
                              if 50 < vehicle_location.distance(p.location) < 200]

                if not valid_points:
                    valid_points = spawn_points

                # Select random destination
                self._destination = random.choice(valid_points).location

                # Visualize destination
                if hasattr(self.world, 'debug'):
                    self.world.debug.draw_point(
                        self._destination,
                        size=0.5,
                        color=carla.Color(r=255, g=0, b=0),
                        life_time=20.0
                    )

            return self._destination

        except Exception as e:
            print(f"Error in get_destination: {e}")
            traceback.print_exc()
            return self.vehicle.get_location()

    def reset_destination(self):
        """Reset the destination point"""
        if hasattr(self, '_destination'):
            delattr(self, '_destination')

    def get_state(self):
        """Get state with consistent dimensions"""
        try:
            # Initialize state array with zeros
            state_array = np.zeros(21)  # Fixed size state array

            # 1. Process YOLO detections (3 values per object: x, y, depth)
            detections = self.process_yolo_detection(self.front_camera)
            num_objects = min(5, len(detections))  # Consider up to 5 objects
            for i in range(num_objects):
                obj = detections[i]
                # Normalize positions and depth
                state_array[i*3] = obj['position'][0] / IM_WIDTH     # x position
                state_array[i*3 + 1] = obj['position'][1] / IM_HEIGHT  # y position
                state_array[i*3 + 2] = obj['depth'] / 100.0          # depth

            # 2. Vehicle state (6 values)
            if self.vehicle:
                # Get vehicle state
                velocity = self.vehicle.get_velocity()
                transform = self.vehicle.get_transform()
                control = self.vehicle.get_control()

                # Vehicle speed (normalized)
                speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                state_array[15] = speed / 50.0  # Normalize by max speed

                # Vehicle steering
                state_array[16] = control.steer

                # Vehicle acceleration
                acceleration = self.vehicle.get_acceleration()
                accel_magnitude = math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2)
                state_array[17] = accel_magnitude / 10.0

                # Vehicle angular velocity
                angular_velocity = self.vehicle.get_angular_velocity()
                state_array[18] = angular_velocity.z / math.pi

                # Vehicle orientation (normalized to [-1, 1])
                state_array[19] = math.sin(math.radians(transform.rotation.yaw))
                state_array[20] = math.cos(math.radians(transform.rotation.yaw))

            return state_array.astype(np.float32)

        except Exception as e:
            print(f"Error in get_state: {str(e)}")
            return None
    
    def calculate_reward(self):
        reward = 0.0
        done = False
        info = {}

        try:
            # Calculate velocity magnitude
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convert to km/h

            # Get current location and waypoint
            location = self.vehicle.get_location()
            waypoint = self.world.get_map().get_waypoint(location)

            # Base reward for staying alive
            reward += 0.1

            # Speed reward (penalize deviation from target speed)
            target_speed = 30.0  # km/h
            speed_reward = -abs(speed - target_speed) / target_speed
            reward += speed_reward

            # Progress reward (distance to destination)
            destination = self.get_destination()
            distance_to_destination = location.distance(destination)
            progress_reward = -distance_to_destination / 1000.0  # Normalized
            reward += progress_reward

            # Collision penalty
            if len(self.collision_hist) > 0:
                reward -= 100.0
                done = True
                info['termination_reason'] = 'collision'

            # Lane keeping reward
            distance_from_center = location.distance(waypoint.transform.location)
            lane_keeping_reward = -distance_from_center
            reward += lane_keeping_reward

            # Add information to info dict
            info['speed'] = speed
            info['distance_to_destination'] = distance_to_destination
            info['distance_from_center'] = distance_from_center

            return reward, done, info

        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            traceback.print_exc()
            return 0.0, True, {'error': str(e)}
    
    def get_next_waypoints(self, num, distance=10.0):
        """Get next num waypoints ahead of the vehicle"""
        waypoints = []
        current_location = self.vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(current_location)
        next_waypoint = current_waypoint
        
        for _ in range(num):
            next_waypoints = next_waypoint.next(distance)
            if not next_waypoints:
                break
            next_waypoint = next_waypoints[0]
            waypoints.append(next_waypoint)
        
        # Calculate relative position information for each waypoint
        waypoint_info = []
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.get_forward_vector()
        
        for wp in waypoints:
            # Calculate distance from vehicle
            wp_distance = vehicle_location.distance(wp.transform.location)
            
            # Calculate direction vector to waypoint
            direction_vector = wp.transform.location - vehicle_location
            direction_vector = carla.Vector3D(
                direction_vector.x,
                direction_vector.y,
                0.0  # Ignore vertical component
            )
            
            # Normalize vectors
            vehicle_forward_norm = math.sqrt(vehicle_forward.x**2 + vehicle_forward.y**2)
            direction_vector_norm = math.sqrt(direction_vector.x**2 + direction_vector.y**2)
            
            if vehicle_forward_norm == 0 or direction_vector_norm == 0:
                angle = 0.0
            else:
                # Calculate angle between forward vector and direction vector
                cos_theta = (vehicle_forward.x * direction_vector.x + vehicle_forward.y * direction_vector.y) / \
                            (vehicle_forward_norm * direction_vector_norm)
                angle = math.degrees(math.acos(np.clip(cos_theta, -1.0, 1.0)))
                
                # Determine sign using cross product
                cross = vehicle_forward.x * direction_vector.y - vehicle_forward.y * direction_vector.x
                if cross < 0:
                    angle = -angle
            
            waypoint_info.append({
                'distance': wp_distance,
                'angle': math.radians(angle)  # Convert to radians for state representation
            })
        
        return waypoint_info

    def step(self, action):
        try:
            if self.use_mpc:
                # Try to use MPC as the primary controller
                mpc_action = self.mpc_controller.optimize(self.get_state(), self.get_waypoints())
                if mpc_action is not None:
                    # MPC succeeded, use its action
                    throttle, brake, steer = mpc_action
                    self.mpc_failure_count = 0  # Reset failure count
                else:
                    # MPC failed, increment failure count
                    self.mpc_failure_count += 1
                    if self.mpc_failure_count >= self.mpc_failure_threshold:
                        # Switch to RL after too many failures
                        self.use_mpc = False
                        print("Switching to RL due to MPC failures")
                    # Use RL action as fallback
                    throttle = float(np.clip((action[1] + 1) / 2, 0.0, 1.0))
                    steer = float(np.clip(action[0], -1.0, 1.0))
                    brake = 0.0
            else:
                # Use RL as the primary controller
                throttle = float(np.clip((action[1] + 1) / 2, 0.0, 1.0))
                steer = float(np.clip(action[0], -1.0, 1.0))
                brake = 0.0

            # Smooth control changes
            current_control = self.vehicle.get_control()
            smooth_throttle = 0.8 * current_control.throttle + 0.2 * throttle
            smooth_steer = 0.8 * current_control.steer + 0.2 * steer
            if throttle < current_control.throttle:
                brake = min((current_control.throttle - throttle) * 2.0, 1.0)

            # Apply control
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
            info['controller'] = 'MPC' if self.use_mpc else 'RL'

            if not self.use_mpc:
                detections = self.process_yolo_detection(self.front_camera)
                if len(detections) < 3:  # Fewer than 3 objects detected
                    self.use_mpc = True
                    self.mpc_failure_count = 0
                    print("Switching back to MPC")

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
            vehicle_location = self.vehicle.get_location()
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(1.5)
            traffic_manager.global_percentage_speed_difference(-50.0)
            all_spawn_points = self.world.get_map().get_spawn_points()
            nearby_spawn_points = []
            for spawn_point in all_spawn_points:
                if spawn_point.location.distance(vehicle_location) <= spawn_radius:
                    nearby_spawn_points.append(spawn_point)
            if not nearby_spawn_points:
                print(f"No spawn points found within {spawn_radius}m of training vehicle!")
                all_spawn_points.sort(key=lambda p: p.location.distance(vehicle_location))
                nearby_spawn_points = all_spawn_points[:number_of_vehicles]
            print(f"Found {len(nearby_spawn_points)} potential spawn points near training vehicle")
            vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
            vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute('number_of_wheels')) == 4]
            spawned_count = 0
            for spawn_point in nearby_spawn_points:
                if spawned_count >= number_of_vehicles:
                    break
                blueprint = random.choice(vehicle_bps)
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
                if vehicle is not None:
                    self.npc_vehicles.append(vehicle)
                    vehicle.set_autopilot(True)
                    try:
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(0, 10))
                        traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 2.0))
                        spawned_count += 1
                        distance_to_ego = vehicle.get_location().distance(vehicle_location)
                        print(f"Spawned {vehicle.type_id} at {distance_to_ego:.1f}m from training vehicle")
                        debug = self.world.debug
                        if debug:
                            debug.draw_line(
                                vehicle_location,
                                vehicle.get_location(),
                                thickness=0.1,
                                color=carla.Color(r=0, g=255, b=0),
                                life_time=5.0
                            )
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
            try:
                debug = self.world.debug
                if debug:
                    num_points = 36
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
            for controller in self.pedestrian_controllers:
                if controller.is_alive:
                    controller.stop()
                    controller.destroy()
            self.pedestrian_controllers.clear()
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
        env = CarEnv()
        state_dim = env.max_objects * 3 + 6
        action_dim = 2
        print(f"Initializing agent with state_dim={state_dim}, action_dim={action_dim}")
        agent = PPOAgent(state_dim, action_dim)
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
        if 'env' in locals():
            env.cleanup_actors()

def train():
    """Main training loop with improved monitoring and visualization"""
    print(f"Starting training at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using device: {DEVICE}")
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('tensorboard_logs', exist_ok=True)
    training_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'best_reward': float('-inf'),
        'last_save_time': time.time(),
        'total_steps': 0
    }
    try:
        print("Initializing CARLA environment...")
        env = CarEnv()
        print("Waiting for environment to stabilize...")
        time.sleep(2)
        state_dim = env.max_objects * 3 + 6
        action_dim = 2
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        print("Initializing PPO agent...")
        agent = PPOAgent(state_dim, action_dim)
        writer = SummaryWriter(f'tensorboard_logs/training_{int(time.time())}')
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
                done = False
                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    action_result = agent.select_action(state)
                    if action_result is None:
                        print("Failed to select action, ending episode")
                        break
                    action, value, log_prob = action_result
                    next_state, reward, done, info = env.step(action)
                    episode_metrics['reward'] += reward
                    episode_metrics['steps'] += 1
                    training_metrics['total_steps'] += 1
                    if 'speed' in info:
                        episode_metrics['speeds'].append(info['speed'])
                    if 'collision' in info:
                        episode_metrics['collisions'] += 1
                    agent.memory.states.append(state)
                    agent.memory.actions.append(action)
                    agent.memory.rewards.append(reward)
                    agent.memory.values.append(value)
                    agent.memory.log_probs.append(log_prob)
                    agent.memory.masks.append(1 - done)
                    if episode_metrics['steps'] % 100 == 0:
                        current_speed = info.get('speed', 0)
                        print(f"Step {episode_metrics['steps']}: "
                              f"Reward = {episode_metrics['reward']:.2f}, "
                              f"Speed = {current_speed:.2f} km/h")
                        writer.add_scalar('Training/Step_Reward', reward, training_metrics['total_steps'])
                        writer.add_scalar('Training/Current_Speed', current_speed, training_metrics['total_steps'])
                    if len(agent.memory.states) >= UPDATE_TIMESTEP:
                        print(f"\nPerforming policy update at step {episode_metrics['steps']}")
                        agent.update()
                    if done:
                        break
                    state = next_state
                episode_duration = time.time() - episode_start_time
                episode_metrics['avg_speed'] = np.mean(episode_metrics['speeds']) if episode_metrics['speeds'] else 0
                print(f"\nEpisode {episode} Summary:")
                print(f"Total Reward: {episode_metrics['reward']:.2f}")
                print(f"Steps: {episode_metrics['steps']}")
                print(f"Average Speed: {episode_metrics['avg_speed']:.2f} km/h")
                print(f"Collisions: {episode_metrics['collisions']}")
                print(f"Duration: {episode_duration:.2f} seconds")
                print(f"Average Step Time: {episode_duration/episode_metrics['steps']:.3f} seconds")
                training_metrics['episode_rewards'].append(episode_metrics['reward'])
                training_metrics['episode_lengths'].append(episode_metrics['steps'])
                writer.add_scalar('Training/Episode_Reward', episode_metrics['reward'], episode)
                writer.add_scalar('Training/Episode_Length', episode_metrics['steps'], episode)
                writer.add_scalar('Training/Average_Speed', episode_metrics['avg_speed'], episode)
                writer.add_scalar('Training/Collisions', episode_metrics['collisions'], episode)
                if episode_metrics['reward'] > training_metrics['best_reward']:
                    training_metrics['best_reward'] = episode_metrics['reward']
                    best_model_path = os.path.join('checkpoints', 'best_model.pth')
                    agent.save_checkpoint(episode, best_model_path, training_metrics['best_reward'])
                    print(f"New best reward: {episode_metrics['reward']:.2f}")
                if episode % 10 == 0:
                    checkpoint_path = os.path.join('checkpoints', f'checkpoint_{episode}.pth')
                    agent.save_checkpoint(episode, checkpoint_path, training_metrics['best_reward'])
                    print(f"Checkpoint saved at episode {episode}")
                agent.save_checkpoint(episode, latest_checkpoint, training_metrics['best_reward'])
            except Exception as e:
                print(f"Error in episode {episode}: {e}")
                traceback.print_exc()
                try:
                    env.cleanup_actors()
                    env.cleanup_npcs()
                except:
                    pass
                continue
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
        print("\nPerforming final cleanup...")
        try:
            if 'agent' in locals() and 'episode' in locals():
                final_checkpoint_path = os.path.join('checkpoints', 'final_model.pth')
                agent.save_checkpoint(episode, final_checkpoint_path, training_metrics['best_reward'])
                print("Final checkpoint saved")
            if 'env' in locals():
                env.close()
                print("Environment cleaned up")
            if 'writer' in locals():
                writer.close()
                print("TensorBoard writer closed")
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
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.backends.cudnn.deterministic = True
        print("Starting CARLA PPO training with YOLO object detection")
        print(f"Running on: {DEVICE}")
        train()
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