# Standard library imports
import os
import sys
import glob
import time
import random
import math
import weakref
import traceback
import json
from collections import deque
from threading import Thread
from datetime import datetime
from typing import List, Optional, Union, Dict, Tuple
import heapq

# Third-party imports
import numpy as np 
# Deep learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

# Computer vision and visualization
import cv2
import pygame
from pygame import surfarray

# Optimization and control
import casadi as ca

# Other utilities
import requests
from tqdm import tqdm

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
EPISODES = 500
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



from scipy.linalg import solve
from functools import partial

class MPCController:
    def __init__(self, dt=0.1, N=50):
        """Initialize MPC controller with iLQR"""
        self.dt = dt
        self.N = N
        
        # State dimensions [x, y, v, phi, beta]
        self.nx = 5
        self.nu = 3  # [steering, throttle, brake]
        
        # Vehicle parameters
        self.L = 2.9  # Wheelbase
        self.lr = 1.538  # Distance from rear axle to CoM
        
        # Cost weights - Heavily weighted for forward motion and path following
        self.Q = np.diag([100.0, 100.0, 50.0, 20.0, 1.0])  # Increased weights
        self.R = np.diag([1.0, 0.1, 1.0])  # Higher penalty for brake
        
        # Control bounds with minimum throttle
        self.bounds = {
            'steer': [-1.0, 1.0],
            'throttle': [0.1, 1.0],  
            'brake': [0.0, 1.0],
            'v_max': 12.0,
            'v_min': 0.0,  # Don't allow negative velocity
        }
        
        # Safety parameters
        self.safety_distance = 15.0
        self.comfort_brake_distance = 30.0
        
        # Initialize control smoothing
        self.last_controls = None
        self.control_smoothing = 0.2 
        


    def stage_cost(self, state, control, reference, detected_objects):
        """Modified cost function for better tracking and obstacle avoidance"""
        # Path following error
        path_error = state[:2] - reference[:2]
        path_cost = np.sum(path_error**2)
        
        # Velocity tracking with minimum speed requirement
        vel_error = state[2] - max(reference[2], 2.0)  # Minimum speed of 2 m/s
        vel_cost = vel_error**2
        
        # Heading error with wrapping
        heading_error = state[3] - reference[3]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        heading_cost = heading_error**2
        
        # Control cost with preference for throttle over brake
        control_cost = control[0]**2 + 0.1*control[1]**2 + 2.0*control[2]**2
        
        # Additional cost for low speeds
        speed_penalty = 10.0 * np.exp(-state[2])  # Penalize low speeds
        
        # Obstacle avoidance cost
        obstacle_cost = 0.0
        for obj in detected_objects:
            # Convert relative to absolute coordinates
            obj_x = state[0] + obj[0]
            obj_y = state[1] + obj[1]
            
            # Calculate distance to object
            dist = np.sqrt((state[0]-obj_x)**2 + (state[1]-obj_y)**2)
            
            # Sigmoid cost that increases sharply within 5m
            obstacle_cost += 100.0 / (1 + np.exp(0.5*(dist - 5.0)))
        
        return 100.0*path_cost + 50.0*vel_cost + 20.0*heading_cost + control_cost + speed_penalty + obstacle_cost

    def vehicle_dynamics(self, state, controls):
        """Continuous time vehicle dynamics"""
        x, y, v, phi, beta = state
        steering, throttle, brake = controls
        
        # Clip controls
        steering = np.clip(steering, -1.0, 1.0)
        throttle = np.clip(throttle, 0.0, 1.0)
        brake = np.clip(brake, 0.0, 1.0)
        
        # Dynamics
        dx = v * np.cos(phi + beta)
        dy = v * np.sin(phi + beta)
        dphi = v * np.sin(beta) / self.lr
        
        # Simplified acceleration model with air resistance
        dv = 5.0 * throttle - 0.2 * v * v - 10.0 * brake
        
        # Steering dynamics
        dbeta = -0.6 * beta + 0.4 * steering
        
        return np.array([dx, dy, dv, dphi, dbeta])

    def discrete_dynamics(self, state, controls):
        """Discrete time dynamics using Euler integration"""
        return state + self.vehicle_dynamics(state, controls) * self.dt

    def get_jacobians(self, state, control):
        """Compute Jacobians numerically"""
        eps = 1e-6
        fx = np.zeros((self.nx, self.nx))
        fu = np.zeros((self.nx, self.nu))
        
        # Compute state Jacobian
        for i in range(self.nx):
            state_perturbed = state.copy()
            state_perturbed[i] += eps
            fx[:, i] = (self.discrete_dynamics(state_perturbed, control) - 
                       self.discrete_dynamics(state, control)) / eps
            
        # Compute control Jacobian
        for i in range(self.nu):
            control_perturbed = control.copy()
            control_perturbed[i] += eps
            fu[:, i] = (self.discrete_dynamics(state, control_perturbed) - 
                       self.discrete_dynamics(state, control)) / eps
            
        return fx, fu

    def forward_pass(self, x0, u_traj, k, K):
        """Forward pass for iLQR"""
        x_traj = np.zeros((self.N + 1, self.nx))
        x_traj[0] = x0  # Changed from .at[0].set()
        u_new = np.zeros_like(u_traj)

        for t in range(self.N):
            dx = x_traj[t] - x0
            u_new[t] = u_traj[t] + k[t] + K[t] @ dx  # Changed from .at[t].set()
            x_traj[t+1] = self.discrete_dynamics(x_traj[t], u_new[t])  # Changed from .at[t+1].set()

        return x_traj, u_new

    def backward_pass(self, x_traj, u_traj, reference_traj, detected_objects):
        """Backward pass for iLQR"""
        k = np.zeros((self.N, self.nu))
        K = np.zeros((self.N, self.nu, self.nx))

        V_x = np.zeros(self.nx)
        V_xx = np.zeros((self.nx, self.nx))

        for t in range(self.N-1, -1, -1):
            state = x_traj[t]
            control = u_traj[t]

            f_x, f_u = self.get_jacobians(state, control)

            # Compute Q terms
            Q_x = V_x @ f_x
            Q_u = V_x @ f_u
            Q_xx = f_x.T @ V_xx @ f_x
            Q_uu = f_u.T @ V_xx @ f_u + self.R
            Q_ux = f_u.T @ V_xx @ f_x

            # Compute gains
            try:
                k_t = -np.linalg.solve(Q_uu, Q_u)
                K_t = -np.linalg.solve(Q_uu, Q_ux)
            except np.linalg.LinAlgError:
                # If matrix is singular, use pseudo-inverse
                k_t = -np.linalg.pinv(Q_uu) @ Q_u
                K_t = -np.linalg.pinv(Q_uu) @ Q_ux

            k[t] = k_t  # Changed from .at[t].set()
            K[t] = K_t  # Changed from .at[t].set()

            # Update value function derivatives
            V_x = Q_x + K_t.T @ Q_uu @ k_t
            V_xx = Q_xx + K_t.T @ Q_uu @ K_t

        return k, K
    
    def solve(self, current_state, reference_trajectory, detected_objects):
        """Solve MPC problem using iLQR with enhanced safety features"""
        try:
            # Extract relevant state variables [x, y, v, phi, beta]
            if len(current_state) > 5:
                x = current_state[0]
                y = current_state[1]
                v = current_state[2]
                phi = current_state[3]
                beta = current_state[4] if len(current_state) > 4 else 0.0

                reduced_state = np.array([x, y, v, phi, beta])
            else:
                reduced_state = current_state

            # Print current state for debugging
            print(f"Current state: v={reduced_state[2]:.2f} m/s, phi={np.degrees(reduced_state[3]):.2f}°")

            # Enhanced safety check with Time-To-Collision (TTC)
            emergency_brake = False
            for obj in detected_objects:
                rel_x, rel_y = obj[0] - reduced_state[0], obj[1] - reduced_state[1]
                vel = obj[2] if len(obj) > 2 else 0.0  # Object velocity if available

                # Calculate distance and time-to-collision
                distance = np.sqrt(rel_x**2 + rel_y**2)
                closing_speed = reduced_state[2] - vel
                ttc = distance / (closing_speed + 1e-5)  # Avoid division by zero

                # Check both distance and TTC thresholds
                if distance < self.safety_distance or (ttc > 0 and ttc < 2.0):
                    print(f"Emergency brake! TTC: {ttc:.1f}s, Distance: {distance:.1f}m")
                    emergency_brake = True
                    break
                
            if emergency_brake:
                return np.array([0.0, 0.0, 1.0])

            # Get heading to next reference point
            if len(reference_trajectory) > 0:
                target = reference_trajectory[0]
                dx = target[0] - reduced_state[0]
                dy = target[1] - reduced_state[1]
                target_heading = np.arctan2(dy, dx)
                heading_error = target_heading - reduced_state[3]
                heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
                print(f"Target heading: {np.degrees(target_heading):.2f}°, Error: {np.degrees(heading_error):.2f}°")

            # Force initial movement
            if reduced_state[2] < 0.5:  # If speed is less than 0.5 m/s
                print("Forcing initial forward motion")
                return np.array([0.0, 0.4, 0.0])  # Apply significant throttle

            # Initialize trajectories
            x_traj = np.zeros((self.N + 1, self.nx))
            u_traj = np.zeros((self.N, self.nu))
            x_traj[0] = reduced_state

            # Perform iLQR iterations
            for _ in range(5):
                k, K = self.backward_pass(x_traj, u_traj, reference_trajectory, detected_objects)
                x_traj_new, u_traj_new = self.forward_pass(reduced_state, u_traj, k, K)
                x_traj = x_traj_new
                u_traj = u_traj_new

            # Get raw control action
            raw_control = u_traj[0]

            # Apply bounds
            control = np.array([
                np.clip(raw_control[0], self.bounds['steer'][0], self.bounds['steer'][1]),
                np.clip(raw_control[1], self.bounds['throttle'][0], self.bounds['throttle'][1]),
                np.clip(raw_control[2], self.bounds['brake'][0], self.bounds['brake'][1])
            ])

            # Ensure minimum forward motion
            if control[1] < self.bounds['throttle'][0]:
                control[1] = self.bounds['throttle'][0]

            # Don't brake while accelerating
            if control[1] > 0.1:
                control[2] = 0.0

            # Apply dynamic control smoothing based on steering angle
            if self.last_controls is not None:
                current_steer = np.abs(control[0])
                # Reduce smoothing when steering more (dynamic smoothing)
                smooth_factor = self.control_smoothing * (1.0 - current_steer)
                control = smooth_factor * self.last_controls + (1 - smooth_factor) * control

            # Store controls for next iteration
            self.last_controls = control.copy()

            # Print control values for debugging
            print(f"Controls: steer={control[0]:.2f}, throttle={control[1]:.2f}, brake={control[2]:.2f}")

            return control

        except Exception as e:
            print(f"Error in MPC solve: {e}")
            traceback.print_exc()
            return np.array([0.0, 0.3, 0.0])  # Safe fallback with forward motion

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        # Ignore the passed state_dim and calculate our own
        self.max_objects = 5  # Track 5 closest objects
        self.state_dim = (
            self.max_objects * 3 +  # x, y, depth for each object
            2 +                     # lane distance and angle
            2                      # speed and steering
        )
        self.action_dim = action_dim
        
        # Initialize actor-critic with calculated state_dim
        self.actor_critic = ActorCritic(self.state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)
        self.memory = PPOMemory()
        
        # Initialize logging and tracking
        current_time = time.strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'logs/ppo_driving_{current_time}')
        self.training_step = 0
        self.best_reward = float('-inf')
        self.episode_rewards = []
        
        # Create checkpoint directory
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def select_action(self, state):
        """Select action based on state"""
        try:
            if state is None:
                return None, None, None
            
            # Ensure state has correct shape
            if isinstance(state, np.ndarray):
                expected_shape = (self.state_dim,)
                if state.shape != expected_shape:
                    print(f"Warning: Reshaping state from {state.shape} to {expected_shape}")
                    state = state.reshape(expected_shape)
                
                # Convert to tensor and ensure float32
                state_tensor = torch.FloatTensor(state).to(DEVICE)
            elif isinstance(state, torch.Tensor):
                state_tensor = state.float().to(DEVICE)
            else:
                raise ValueError(f"Unexpected state type: {type(state)}")
            
            # Get action from actor-critic
            action, value, log_prob = self.actor_critic.get_action(state_tensor)
            
            return action.cpu().numpy(), value.item(), log_prob.item()
            
        except Exception as e:
            print(f"Error in select_action: {e}")
            print(f"State shape: {state.shape if hasattr(state, 'shape') else 'No shape'}")
            print(f"State type: {type(state)}")
            print(f"State content: {state}")
            print(f"Expected state dim: {self.state_dim}")
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

class GlobalPathPlanner:
    def __init__(self, world, sampling_resolution=2.0):
        self.world = world
        self.sampling_resolution = sampling_resolution
        self.topology = self._get_topology()
        
    def _get_topology(self):
        topology = []
        # Generate waypoints for the entire map
        waypoints = self.world.get_map().generate_waypoints(self.sampling_resolution)
        
        # Create a graph of waypoints
        for waypoint in waypoints:
            connections = []
            # Get next waypoints in all directions
            next_waypoints = waypoint.next(self.sampling_resolution)
            if next_waypoints:
                connections.append(next_waypoints[0])  # Take first option
            
            # Get previous waypoints
            prev_waypoints = waypoint.previous(self.sampling_resolution)
            if prev_waypoints:
                connections.append(prev_waypoints[0])
            
            topology.append({
                'waypoint': waypoint,
                'connections': connections
            })
        return topology

    def a_star_search(self, start, goal):
        """A* search algorithm to find shortest path"""
        print("Starting A* search...")

        class ComparableWaypoint:
            def __init__(self, waypoint, priority):
                self.waypoint = waypoint
                self.priority = priority

            def __lt__(self, other):
                return self.priority < other.priority

        frontier = []
        heapq.heappush(frontier, ComparableWaypoint(start, 0))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        while frontier:
            current = heapq.heappop(frontier).waypoint

            # Check if we're close enough to goal
            if self.heuristic(current.transform.location, goal.transform.location) < 2.0:
                print("Goal found!")
                break

            # Get next possible waypoints using Carla's built-in next waypoint
            next_waypoints = current.next(2.0)  # Get waypoints 2 meters ahead

            for next_wp in next_waypoints:
                # Calculate new cost
                new_cost = cost_so_far[current] + self.heuristic(
                    current.transform.location,
                    next_wp.transform.location
                )

                # If this is a new node or we found a better path
                if next_wp not in cost_so_far or new_cost < cost_so_far[next_wp]:
                    cost_so_far[next_wp] = new_cost
                    priority = new_cost + self.heuristic(
                        next_wp.transform.location,
                        goal.transform.location
                    )
                    heapq.heappush(frontier, ComparableWaypoint(next_wp, priority))
                    came_from[next_wp] = current

        # Reconstruct path
        path = []
        current_wp = current
        while current_wp != start:
            path.append(current_wp)
            current_wp = came_from[current_wp]
        path.append(start)
        path.reverse()

        # Convert to custom Waypoints with velocity information
        path = self._add_velocity_to_path(path)

        print(f"Path found with {len(path)} waypoints")

        # Create dense interpolated path
        dense_path = []
        for i in range(len(path)-1):
            wp1 = path[i]
            wp2 = path[i+1]
            distance = wp1.transform.location.distance(wp2.transform.location)
            num_points = max(2, int(distance // 0.5))  # 0.5m spacing
    
            # Get Euler angles for both rotations
            roll1, pitch1, yaw1 = wp1.transform.rotation.roll, wp1.transform.rotation.pitch, wp1.transform.rotation.yaw
            roll2, pitch2, yaw2 = wp2.transform.rotation.roll, wp2.transform.rotation.pitch, wp2.transform.rotation.yaw
    
            # Ensure the angles are properly wrapped
            yaw1 = np.degrees(np.arctan2(np.sin(np.radians(yaw1)), np.cos(np.radians(yaw1))))
            yaw2 = np.degrees(np.arctan2(np.sin(np.radians(yaw2)), np.cos(np.radians(yaw2))))
            
            # Handle the case where angles cross the 180/-180 boundary
            if abs(yaw2 - yaw1) > 180:
                if yaw2 > yaw1:
                    yaw1 += 360
                else:
                    yaw2 += 360
    
            for t in np.linspace(0, 1, num_points):
                # Interpolate location
                loc = wp1.transform.location + t*(wp2.transform.location - wp1.transform.location)
                
                # Interpolate rotation angles
                roll = roll1 + t*(roll2 - roll1)
                pitch = pitch1 + t*(pitch2 - pitch1)
                yaw = yaw1 + t*(yaw2 - yaw1)
                
                # Wrap yaw angle back to [-180, 180]
                yaw = yaw % 360
                if yaw > 180:
                    yaw -= 360
                
                # Create new rotation object
                rotation = carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
                
                # Create new waypoint with interpolated transform
                interpolated_wp = self.world.get_map().get_waypoint(loc)
                interpolated_wp.transform.rotation = rotation
                dense_path.append(interpolated_wp)

        # Visualization code
        for i in range(len(dense_path)-1):
            # Draw thick blue line between consecutive waypoints
            begin = dense_path[i].transform.location
            end = dense_path[i+1].transform.location

            # Draw line slightly above ground
            begin = begin + carla.Location(z=0.5)
            end = end + carla.Location(z=0.5)

            # Draw a thick blue line that persists
            self.world.debug.draw_line(
                begin, end,
                thickness=0.5,
                color=carla.Color(b=255, g=0, r=0),  # Blue
                life_time=0.0,  # 0.0 means permanent until destroyed
                persistent_lines=True
            )

            # Draw waypoint markers
            self.world.debug.draw_point(
                begin,
                size=0.1,
                color=carla.Color(b=255, g=255, r=255),  # White
                life_time=0.0,
                persistent_lines=True
            )

        # Draw final waypoint
        self.world.debug.draw_point(
            dense_path[-1].transform.location + carla.Location(z=0.5),
            size=0.1,
            color=carla.Color(b=255, g=255, r=255),
            life_time=0.0,
            persistent_lines=True
        )

        # Draw start and end points more prominently
        self.world.debug.draw_point(
            dense_path[0].transform.location + carla.Location(z=1.0),
            size=0.2,
            color=carla.Color(r=0, g=255, b=0),  # Green for start
            life_time=0.0,
            persistent_lines=True
        )

        self.world.debug.draw_point(
            dense_path[-1].transform.location + carla.Location(z=1.0),
            size=0.2,
            color=carla.Color(r=255, g=0, b=0),  # Red for end
            life_time=0.0,
            persistent_lines=True
        )

        return dense_path

    def _add_velocity_to_path(self, path):
        """Add velocity information to path waypoints based on curvature"""
        waypoints = []
        for i in range(len(path)):
            wp = path[i]
            velocity = 20.0  # Default velocity
            
            if i < len(path) - 1:
                # Calculate curvature based on next waypoint
                next_wp = path[i + 1]
                dx = next_wp.transform.location.x - wp.transform.location.x
                dy = next_wp.transform.location.y - wp.transform.location.y
                angle = math.atan2(dy, dx)
                curvature = abs(angle)
                
                # Reduce speed in curves
                velocity = max(5.0, 20.0 - 15.0 * curvature)

            # Create custom Waypoint with velocity
            waypoints.append(
                Waypoint(wp.transform, velocity)
            )
        return waypoints

    def heuristic(self, a, b):
        """Calculate heuristic distance between two points"""
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

class Waypoint:
    def __init__(self, transform, velocity=None):
        self.transform = transform
        self.velocity = velocity if velocity else self._calculate_default_velocity()
        
    def _calculate_default_velocity(self):
        """Calculate default velocity based on road type"""
        # Get Carla map waypoint for this location
        map_wp = carla_map.get_waypoint(self.transform.location)
        if map_wp:
            if map_wp.is_junction:
                return 10.0  # Reduce speed in intersections
            if map_wp.lane_type == carla.LaneType.Driving:
                return 25.0 if map_wp.lane_width > 3.5 else 15.0
        return 20.0  # Default speed
        
    def get_waypoints_from_path(self, path):
        """Convert A* path to waypoints with velocity information"""
        waypoints = []
        for i in range(len(path)):
            # Get current waypoint
            wp = path[i]

            # Calculate desired velocity based on curvature
            velocity = 20.0  # Default velocity
            if i < len(path) - 1:
                # Calculate curvature based on next waypoint
                next_wp = path[i + 1]
                dx = next_wp.transform.location.x - wp.transform.location.x
                dy = next_wp.transform.location.y - wp.transform.location.y
                curvature = abs(np.arctan2(dy, dx))
                # Reduce speed in curves
                velocity = max(5.0, 20.0 - 15.0 * curvature)

            waypoints.append(Waypoint(wp.transform, velocity))

            return waypoints

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
        self.episode_start = datetime.now() 
        self.total_distance = 0.0
        
        # Add missing attributes
        self.front_camera = None
        self.npc_vehicles = []
        self.pedestrians = []
        self.pedestrian_controllers = []

        pygame.init()
        self.display = None
        self.clock = None
        self.init_pygame_display()
        self.map = None
        self.global_path = []
        self.current_path_index = 0
        self.target_destination = None
        self.path_planner = None
        # Camera settings
        self.camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7),
            carla.Rotation(pitch=0)
        )
        
        # Initialize CARLA world first
        self.setup_world()
        
        # Initialize YOLO model
        self._init_yolo()
        self.waypoint_buffer = 5
        self.lookahead_distance = 5.0
        self.path_tolerance = 2.0
        self.min_distance_to_next_wp = 2.0

        # Initialize state dimensions
        self.max_objects = 5
        self.state_dim = self.max_objects * 3 + 4  # objects * features + (x, y, yaw, velocity)
        self.action_dim = 2  # steering and throttle
        
        # Initialize both controllers
        self.mpc = MPCController(dt=0.1, N=50) 
        self.agent = PPOAgent(self.state_dim, self.action_dim)  # Initialize PPO agent
        
        # Controller management
        self.use_mpc = True
        self.mpc_fail_counter = 0
        self.MAX_MPC_FAILURES = 5

        self.reward_params = {
            'collision_penalty': -50.0,
            'lane_deviation_weight': -0.1,
            'min_speed': 5.0,  # km/h
            'target_speed': 10.0,  # km/h
            'speed_weight': 0.1,
            'stuck_penalty': -10.0,
            'alive_bonus': 0.1,
            'distance_weight': 0.5
        }


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

    def should_use_mpc(self, state, detected_objects):
        """Determine whether to use MPC or switch to PPO"""
        try:
            # Get velocity magnitude using vector components
            velocity = self.vehicle.get_velocity()
            vehicle_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h

            # Only switch to PPO in extreme cases
            if self.mpc_fail_counter >= self.MAX_MPC_FAILURES:
                return False

            # Count nearby objects (within 10m)
            nearby_objects = len([obj for obj in detected_objects if obj['depth'] < 10.0])

            # Switch to PPO only in very complex scenarios
            if (nearby_objects > 5 or  # Increased from 3 to 5
                (vehicle_speed > 50 and nearby_objects > 2) or  # Increased speed threshold
                len(self.collision_hist) > 1):  # Allow one minor collision before switching
                return False

            return True

        except Exception as e:
            print(f"Error in should_use_mpc: {e}")
            return True  # Default to MPC on error
    

    def visualize_current_progress(self, current_loc, target_wp):
        """Enhanced visualization of path following status"""
        try:
            # Draw path segment being followed
            for i in range(self.current_path_index, min(self.current_path_index+10, len(self.global_path)-1)):
                start = self.global_path[i].transform.location + carla.Location(z=0.5)
                end = self.global_path[i+1].transform.location + carla.Location(z=0.5)

                self.world.debug.draw_line(
                    start, end,
                    thickness=0.2,
                    color=carla.Color(r=255, g=165, b=0),  # Orange
                    life_time=0.1
                )

            # Draw current target
            self.world.debug.draw_point(
                target_wp.transform.location + carla.Location(z=1.0),
                size=0.2,
                color=carla.Color(r=0, g=255, b=0),
                life_time=0.1
            )

            # Draw vehicle forward vector
            v_transform = self.vehicle.get_transform()
            forward = v_transform.get_forward_vector() * 5.0
            end = v_transform.location + forward + carla.Location(z=0.5)

            self.world.debug.draw_line(
                v_transform.location + carla.Location(z=0.5),
                end,
                thickness=0.1,
                color=carla.Color(r=255, g=0, b=255),
                life_time=0.1
            )

        except Exception as e:
            pass

    # In get_reference_from_global_path:
    def get_reference_from_global_path(self, current_state):
        reference = []
        closest_idx = self.current_path_index
        
        # Look 2 seconds ahead for the first target
        lookahead = int(2.0 / self.mpc.dt)  # 2 seconds
        start_idx = min(closest_idx + lookahead, len(self.global_path)-1)
        
        for i in range(self.mpc.N):
            idx = min(start_idx + i, len(self.global_path)-1)
            wp = self.global_path[idx]
            
            # Calculate direction vector for accurate heading
            if idx < len(self.global_path)-1:
                next_wp = self.global_path[idx+1]
                dx = next_wp.transform.location.x - wp.transform.location.x
                dy = next_wp.transform.location.y - wp.transform.location.y
            else:
                dx = np.cos(wp.transform.rotation.yaw)
                dy = np.sin(wp.transform.rotation.yaw)
                
            target_heading = np.arctan2(dy, dx)
            
            reference.append([
                wp.transform.location.x,
                wp.transform.location.y,
                wp.velocity,  # Use waypoint's velocity
                target_heading,
                0.0
            ])
        
        return np.array(reference)

    def get_next_waypoint(self, current_location, lookahead_distance=10.0):  # Reduced lookahead
        """Get next target waypoint with lane alignment"""
        if not self.global_path:
            return None, 0

        # Find nearest waypoint in path
        closest_idx = 0
        min_dist = float('inf')
        for i, wp in enumerate(self.global_path):
            dist = current_location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Look ahead by fixed number of waypoints
        target_idx = min(closest_idx + 5, len(self.global_path)-1)  # Fixed offset
        return self.global_path[target_idx], target_idx
    
    def apply_control(self, waypoint, current_state):
        """Apply smooth control to follow waypoint"""
        # Calculate speed error
        target_speed = waypoint.velocity
        current_speed = current_state[2]  # Assuming state[2] is velocity
        speed_error = target_speed - current_speed

        # Proportional speed control
        throttle = min(max(speed_error * 0.5, 0.0), 1.0)  # Kp = 0.5
        brake = min(max(-speed_error * 0.5, 0.0), 1.0)

        # Calculate steering
        current_yaw = current_state[3]  # Assuming state[3] is yaw
        target_yaw = waypoint.transform.rotation.yaw * np.pi / 180.0

        # Normalize yaw difference
        yaw_diff = target_yaw - current_yaw
        while yaw_diff > np.pi: yaw_diff -= 2*np.pi
        while yaw_diff < -np.pi: yaw_diff += 2*np.pi

        # Proportional steering control
        steer = np.clip(yaw_diff * 0.8, -1.0, 1.0)  # Kp = 0.8

        return carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
   
    def select_distant_points(self, min_distance=100.0):
        """Select two points at least min_distance apart"""
        spawn_points = self.world.get_map().get_spawn_points()

        # If we already have a spawn point from a previous attempt, use that as start
        if not hasattr(self, '_episode_spawn_point'):
            self._episode_spawn_point = random.choice(spawn_points)
        start_point = self._episode_spawn_point

        # Try to find an end point far enough from the start point
        max_attempts = 50
        for _ in range(max_attempts):
            end_point = random.choice(spawn_points)
            distance = start_point.location.distance(end_point.location)
            if distance >= min_distance:
                return start_point, end_point

        print("Warning: Could not find end point far enough apart")
        return start_point,random.choice(spawn_points)


    def generate_global_path(self):
        """Generate a new global path using A*"""
        try:
            # Get start and end points
            start_wp, end_wp = self.select_distant_points()

            # Draw start and end markers with labels
            self.world.debug.draw_string(
                start_wp.location + carla.Location(z=2.0),
                'START',
                color=carla.Color(r=0, g=255, b=0),
                life_time=20.0
            )
            self.world.debug.draw_string(
                end_wp.location + carla.Location(z=2.0),
                'END',
                color=carla.Color(r=255, g=0, b=0),
                life_time=20.0
            )

            # Get waypoints for start and end
            start_waypoint = self.world.get_map().get_waypoint(start_wp.location)
            end_waypoint = self.world.get_map().get_waypoint(end_wp.location)

            # Calculate A* path
            print("Calculating A* path...")
            raw_path = self.path_planner.a_star_search(start_waypoint, end_waypoint)
            
            # Store the processed path
            self.global_path = raw_path
            
            # Visualize the path if it exists
            if self.global_path:
                print(f"Path processed with {len(self.global_path)} waypoints")
                print("Visualizing path...")
                
                # Draw lines between consecutive waypoints
                for i in range(len(self.global_path)-1):
                    # Current and next waypoint locations
                    current = self.global_path[i].transform.location
                    next_wp = self.global_path[i+1].transform.location

                    # Draw thick blue line
                    self.world.debug.draw_line(
                        current + carla.Location(z=0.5),  # Raise slightly above ground
                        next_wp + carla.Location(z=0.5),
                        thickness=0.5,  # Thicker line
                        color=carla.Color(b=255, g=0, r=0),  # Pure blue
                        life_time=20.0,
                        persistent_lines=True
                    )

                    # Draw waypoint markers
                    self.world.debug.draw_point(
                        current + carla.Location(z=0.5),
                        size=0.1,
                        color=carla.Color(b=255, g=255, r=255),  # White dots
                        life_time=20.0,
                        persistent_lines=True
                    )

                # Draw final waypoint
                self.world.debug.draw_point(
                    self.global_path[-1].transform.location + carla.Location(z=0.5),
                    size=0.1,
                    color=carla.Color(b=255, g=255, r=255),
                    life_time=20.0,
                    persistent_lines=True
                )

                print("Path visualization complete")
            else:
                print("No path found!")

            # Reset path index
            self.current_path_index = 0
            return True

        except Exception as e:
            print(f"Error in generate_global_path: {e}")
            traceback.print_exc()
            return False

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
        self.cleanup_actors()  # Clean up previous actors
        self.cleanup_npcs()    # Clean up NPCs  


        if hasattr(self, '_episode_spawn_point'):
            delattr(self, '_episode_spawn_point')

        # Generate new global path
        self.generate_global_path() 

        # Setup new episode
        if not self.setup_vehicle():  # This will handle vehicle spawning
            raise Exception("Failed to setup vehicle")  

        self.collision_hist = []  # Clear collision history
        self.stuck_time = 0
        self.episode_start = datetime.now()  # Set as datetime object
        self.last_location = None   

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

            # Use the episode's spawn point if it exists
            if hasattr(self, '_episode_spawn_point'):
                spawn_point = self._episode_spawn_point
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle is not None:
                    self.vehicle = vehicle
                    print("Vehicle spawned successfully at episode spawn point")
                else:
                    # If episode spawn point fails, try random points
                    for _ in range(10):
                        spawn_point = random.choice(spawn_points)
                        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                        if vehicle is not None:
                            self.vehicle = vehicle
                            self._episode_spawn_point = spawn_point  # Update episode spawn point
                            print("Vehicle spawned successfully at new spawn point")
                            break
                    else:
                        raise Exception("No valid spawn points found after 10 attempts")
            else:
                # If no episode spawn point exists, try random points
                for _ in range(10):
                    spawn_point = random.choice(spawn_points)
                    vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if vehicle is not None:
                        self.vehicle = vehicle
                        self._episode_spawn_point = spawn_point  # Store the successful spawn point
                        print("Vehicle spawned successfully")
                        break
                else:
                    raise Exception("No valid spawn points found after 10 attempts")

            # Rest of the setup code remains the same...
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
            self.path_planner = GlobalPathPlanner(self.world)
            
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
            self.map = self.world.get_map()
            # Wait for the world to stabilize
            for _ in range(50):
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
        """Get the current state of the vehicle"""
        try:
            # Get vehicle state
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            angular_velocity = self.vehicle.get_angular_velocity()

            # Calculate velocity magnitude
            v = np.sqrt(velocity.x**2 + velocity.y**2)

            # Get yaw angle (phi)
            phi = np.radians(transform.rotation.yaw)

            # Estimate slip angle (beta)
            # Simple estimation based on velocity direction
            beta = np.arctan2(velocity.y, velocity.x) - phi if v > 0.1 else 0.0

            # Return the 5 states needed by MPC
            return np.array([
                transform.location.x,  # x position
                transform.location.y,  # y position
                v,                    # velocity magnitude
                phi,                  # heading angle
                beta                  # slip angle
            ])

        except Exception as e:
            print(f"Error in get_state: {e}")
            traceback.print_exc()
            return np.zeros(5)  # Return safe default state
    
    def get_detected_objects(self):
        """Get detected objects in the environment"""
        detected_objects = []
        
        try:
            # Get all actors in the world
            actor_list = self.world.get_actors()
            vehicle_list = actor_list.filter('vehicle.*')
            
            # Get ego vehicle location
            ego_location = self.vehicle.get_location()
            
            # Get ego vehicle waypoint
            ego_waypoint = self.map.get_waypoint(ego_location)
            
            for vehicle in vehicle_list:
                if vehicle.id != self.vehicle.id:  # Skip ego vehicle
                    vehicle_location = vehicle.get_location()
                    
                    # Get waypoint for other vehicle
                    try:
                        vehicle_waypoint = self.map.get_waypoint(vehicle_location)
                        
                        # Check if vehicle is in same or adjacent lane
                        if vehicle_waypoint.lane_id == ego_waypoint.lane_id:
                            # Calculate relative position
                            rel_x = vehicle_location.x - ego_location.x
                            rel_y = vehicle_location.y - ego_location.y
                            
                            # Get velocity
                            velocity = vehicle.get_velocity()
                            vel = np.sqrt(velocity.x**2 + velocity.y**2)
                            
                            detected_objects.append(np.array([rel_x, rel_y, vel]))
                    except:
                        # If waypoint calculation fails, use distance-based detection
                        distance = ego_location.distance(vehicle_location)
                        if distance < 50.0:  # Only consider vehicles within 50 meters
                            rel_x = vehicle_location.x - ego_location.x
                            rel_y = vehicle_location.y - ego_location.y
                            
                            velocity = vehicle.get_velocity()
                            vel = np.sqrt(velocity.x**2 + velocity.y**2)
                            
                            detected_objects.append(np.array([rel_x, rel_y, vel]))
            
            return detected_objects
            
        except Exception as e:
            print(f"Error in get_detected_objects: {e}")
            traceback.print_exc()
            return []  # Return empty list if there's an error

    def calculate_reward(self):
        """Calculate reward for the current state"""
        try:
            reward = 0.0
            done = False
            info = {}

            # Get current state
            velocity = self.vehicle.get_velocity()
            current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            location = self.vehicle.get_location()

            # Base reward for staying alive
            reward += self.reward_params['alive_bonus']

            # Speed reward
            speed_diff = abs(current_speed - self.reward_params['target_speed'])
            speed_reward = -speed_diff * self.reward_params['speed_weight']
            reward += speed_reward

            # Check if vehicle is stuck
            if current_speed < self.reward_params['min_speed']:
                self.stuck_time += 0.1
                if self.stuck_time > 20.0:
                    done = True
                    reward += self.reward_params['stuck_penalty']
                    info['termination_reason'] = 'stuck'
            else:
                self.stuck_time = 0

            # Distance traveled reward
            if self.last_location is not None:
                distance_traveled = location.distance(self.last_location)
                reward += distance_traveled * self.reward_params['distance_weight']

            # Collision penalty
            if len(self.collision_hist) > 0:
                reward += self.reward_params['collision_penalty']
                done = True
                info['termination_reason'] = 'collision'

            # Path following reward
            if self.global_path and self.current_path_index < len(self.global_path):
                target_wp = self.global_path[self.current_path_index]
                distance_to_path = target_wp.transform.location.distance(location)
                reward -= distance_to_path * 0.1
            else:
                # If we've reached the end of the path
                done = True
                reward += 100.0  # Bonus for completing the path
                info['termination_reason'] = 'path_completed'

            # Lane keeping reward
            waypoint = self.world.get_map().get_waypoint(location)
            distance_from_center = location.distance(waypoint.transform.location)
            reward += distance_from_center * self.reward_params['lane_deviation_weight']

            # Update info dictionary
            info.update({
                'speed': current_speed,
                'reward': reward,
                'distance_from_center': distance_from_center,
                'stuck_time': self.stuck_time
            })

            return reward, done, info

        except Exception as e:
            print(f"Error in calculate_reward: {e}")
            traceback.print_exc()
            return 0.0, True, {'error': str(e)}
    


    def step(self, action=None):
        try:
            # Initialize info dictionary
            info = {
                'controller_used': 'MPC',
                'speed': 0.0,
                'mpc_fail': False,
                'collision': False,
                'distance_traveled': 0.0,
                'lane_deviation': 0.0,
                'throttle': 0.0,
                'steer': 0.0,
                'brake': 0.0,
                'stuck_time': 0.0
            }

            # Get current state and observations
            current_state = self.get_state()
            if current_state is None:
                print("Failed to get state")
                return None, 0, True, info

            # Get current vehicle state
            velocity = self.vehicle.get_velocity()
            current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            info['speed'] = current_speed
            current_loc = self.vehicle.get_location()

            # Check if we have a valid path
            if not self.global_path or self.current_path_index >= len(self.global_path):
                print("No valid path available")
                return current_state, 0, True, info

            try:
                # Get reference trajectory from global path
                next_wp, next_idx = self.get_next_waypoint(self.vehicle.get_location())
                if next_wp:
                    self.current_path_index = next_idx
                reference_trajectory = self.get_reference_from_global_path(current_state)

                # Get detected objects
                detected_objects = self.get_detected_objects()

                # Get MPC control action
                mpc_action = self.mpc.solve(current_state, reference_trajectory, detected_objects)

                if mpc_action is not None:
                    self.mpc_fail_counter = 0
                    # Convert MPC action to CARLA controls
                    steering = float(np.clip(mpc_action[0], -1.0, 1.0))
                    throttle = float(np.clip(mpc_action[1], 0.0, 1.0))
                    brake = float(np.clip(mpc_action[2], 0.0, 1.0))
                else:
                    self.mpc_fail_counter += 1
                    info['mpc_fail'] = True
                    print(f"MPC failed (attempt {self.mpc_fail_counter})")
                    # Safe fallback control
                    steering = 0.0
                    throttle = 0.0
                    brake = 0.3

                # Apply smoothing
                if hasattr(self, 'last_control'):
                    smooth_factor = 0.8
                    throttle = smooth_factor * self.last_control.throttle + (1 - smooth_factor) * throttle
                    steering = smooth_factor * self.last_control.steer + (1 - smooth_factor) * steering

                # Create control command
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=steering,
                    brake=brake,
                    hand_brake=False,
                    reverse=False,
                    manual_gear_shift=False
                )

                # Store for next iteration
                self.last_control = control

                # Update info
                info.update({
                    'throttle': throttle,
                    'steer': steering,
                    'brake': brake
                })

                # Apply control to vehicle
                self.vehicle.apply_control(control)

            except Exception as control_error:
                print(f"Error in control calculation: {control_error}")
                traceback.print_exc()
                return current_state, 0, True, info

            # Visualize current target and progress
            if self.global_path and self.current_path_index < len(self.global_path):
                target_wp = self.global_path[self.current_path_index]
                self.visualize_current_progress(current_loc, target_wp)

            # Update simulation
            for _ in range(4):  # 4 physics steps per control step
                self.world.tick()

            # Check for waypoint progression
            if self.global_path and self.current_path_index < len(self.global_path):
                target_wp = self.global_path[self.current_path_index]
                distance_to_target = current_loc.distance(target_wp.transform.location)
                if distance_to_target < 2.0:  # Within 2 meters of current waypoint
                    self.current_path_index += 1
                    print(f"Reached waypoint {self.current_path_index-1}, moving to next")

            # Calculate reward and check termination conditions
            reward, done, reward_info = self.calculate_reward()
            info.update(reward_info)

            # Update metrics
            if self.last_location is not None:
                info['distance_traveled'] = current_loc.distance(self.last_location)
                self.total_distance += info['distance_traveled']
            self.last_location = current_loc

            # Check for collisions
            if len(self.collision_hist) > 0:
                done = True
                info['collision'] = True
                print("Collision detected!")

            # Calculate lane deviation
            try:
                waypoint = self.map.get_waypoint(current_loc)
                info['lane_deviation'] = current_loc.distance(waypoint.transform.location)
            except:
                info['lane_deviation'] = 0.0

            # Update stuck detection
            if current_speed < self.reward_params['min_speed']:
                self.stuck_time += 0.1
                info['stuck_time'] = self.stuck_time
                if self.stuck_time > 10.0:  # Vehicle stuck for more than 5 seconds
                    print("Vehicle stuck!")
                    done = True
            else:
                self.stuck_time = 0

            # Check maximum episode duration
            current_time = datetime.now()
            if (current_time - self.episode_start).total_seconds() > 300:  # 5 minutes max
                print("Maximum episode duration reached")
                done = True

            return current_state, reward, done, info

        except Exception as e:
            print(f"Critical error in step: {e}")
            traceback.print_exc()
            return None, 0, True, info


    def spawn_npcs(self):
        """Spawn NPC vehicles near and in front of the training vehicle"""
        try:
            number_of_vehicles = 5
            spawn_radius = 40.0
            # Define forward angle range (270° to 90° where 0° is forward)
            min_angle = -90  # 270 degrees
            max_angle = 90   # 90 degrees

            if self.vehicle is None:
                print("Training vehicle not found! Cannot spawn NPCs.")
                return

            # Get training vehicle's location and forward vector
            vehicle_location = self.vehicle.get_location()
            vehicle_transform = self.vehicle.get_transform()
            forward_vector = vehicle_transform.get_forward_vector()

            # Configure traffic manager
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(1.0)
            traffic_manager.global_percentage_speed_difference(50.0)

            # Get all spawn points
            all_spawn_points = self.world.get_map().get_spawn_points()

            # Filter spawn points to only include those within radius and in front of the vehicle
            nearby_spawn_points = []
            for spawn_point in all_spawn_points:
                # Calculate distance
                distance = spawn_point.location.distance(vehicle_location)
                if distance <= spawn_radius:
                    # Calculate angle between forward vector and spawn point
                    to_spawn_vector = carla.Vector3D(
                        x=spawn_point.location.x - vehicle_location.x,
                        y=spawn_point.location.y - vehicle_location.y,
                        z=0
                    )

                    # Calculate angle in degrees
                    angle = math.degrees(math.atan2(to_spawn_vector.y, to_spawn_vector.x) - 
                                      math.atan2(forward_vector.y, forward_vector.x))
                    # Normalize angle to [-180, 180]
                    angle = (angle + 180) % 360 - 180

                    # Check if spawn point is within the desired angle range
                    if min_angle <= angle <= max_angle:
                        nearby_spawn_points.append((spawn_point, distance))  # Store distance for sorting

            # Sort spawn points by distance
            nearby_spawn_points.sort(key=lambda x: x[1])  # Sort by distance
            nearby_spawn_points = [sp[0] for sp in nearby_spawn_points]  # Extract just the spawn points

            if not nearby_spawn_points:
                print(f"No suitable spawn points found in front of the training vehicle within {spawn_radius}m!")
                return

            print(f"Found {len(nearby_spawn_points)} potential spawn points in front of training vehicle")

            # Rest of the spawning logic remains similar
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
                        traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(20, 50))
                        traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(0.5, 1.0))

                        spawned_count += 1

                        # Print spawn information
                        distance_to_ego = vehicle.get_location().distance(vehicle_location)
                        print(f"Spawned {vehicle.type_id} at {distance_to_ego:.1f}m from training vehicle")

                        # Draw debug visualization
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

            print(f"Successfully spawned {spawned_count} vehicles in front of training vehicle")

            # Visualize spawn area (forward arc instead of full circle)
            try:
                debug = self.world.debug
                if debug:
                    num_points = 18  # Number of points to approximate the arc
                    for i in range(num_points + 1):
                        angle_rad = math.radians(min_angle + (max_angle - min_angle) * i / num_points)
                        # Rotate the point by vehicle's rotation
                        vehicle_rotation = vehicle_transform.rotation.yaw
                        total_angle = math.radians(vehicle_rotation) + angle_rad

                        point = carla.Location(
                            x=vehicle_location.x + spawn_radius * math.cos(total_angle),
                            y=vehicle_location.y + spawn_radius * math.sin(total_angle),
                            z=vehicle_location.z
                        )
                        debug.draw_point(
                            point,
                            size=0.1,
                            color=carla.Color(r=0, g=255, b=0),
                            life_time=5.0
                        )
                        if i > 0:
                            prev_angle_rad = math.radians(min_angle + (max_angle - min_angle) * (i-1) / num_points)
                            prev_total_angle = math.radians(vehicle_rotation) + prev_angle_rad
                            prev_point = carla.Location(
                                x=vehicle_location.x + spawn_radius * math.cos(prev_total_angle),
                                y=vehicle_location.y + spawn_radius * math.sin(prev_total_angle),
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
    """Main training loop with hybrid control and improved monitoring"""
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
        'mpc_usage': [],
        'ppo_usage': [],
        'collisions': [],
        'mpc_failures': [],
        'best_reward': float('-inf'),
        'last_save_time': time.time(),
        'total_steps': 0
    }
    
    try:
        # Initialize environment
        print("Initializing CARLA environment...")
        env = CarEnv()
        
        # Calculate state dimension
        state_dim = env.max_objects * 3 + 4  # objects * features + additional features
        action_dim = 2  # steering and throttle
        print(f"State dimension: {state_dim}, Action dimension: {action_dim}")
        
        # Initialize agent
        print("Initializing PPO agent...")
        agent = PPOAgent(state_dim, action_dim)
        writer = SummaryWriter(f'tensorboard_logs/training_{int(time.time())}')
        
        # Load latest checkpoint if exists
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
                'mpc_uses': 0,
                'ppo_uses': 0,
                'collisions': 0,
                'mpc_failures': 0,
                'avg_speed': 0,
                'speeds': [],
                'lane_deviations': []
            }
            
            try:
                print(f"\nEpisode {episode}/{EPISODES}")
                state = env.reset()
                
                if state is None:
                    print("Failed to get initial state, retrying episode")
                    continue
                
                done = False
                while not done:
                    # Handle pygame events
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt
                    
                    # Execute step
                    next_state, reward, done, info = env.step([0, 0])  # Dummy action, will be replaced by controller
                    
                    # Update metrics based on info
                    episode_metrics['reward'] += reward
                    episode_metrics['steps'] += 1
                    training_metrics['total_steps'] += 1
                    
                    if info['controller_used'] == 'MPC':
                        episode_metrics['mpc_uses'] += 1
                    else:
                        episode_metrics['ppo_uses'] += 1
                        
                    if info.get('mpc_fail', False):
                        episode_metrics['mpc_failures'] += 1
                    
                    if info.get('collision', False):
                        episode_metrics['collisions'] += 1
                    
                    if 'speed' in info:
                        episode_metrics['speeds'].append(info['speed'])
                    
                    if 'lane_deviation' in info:
                        episode_metrics['lane_deviations'].append(info['lane_deviation'])
                    
                    # Only store PPO experiences when PPO was used
                    if info['controller_used'] == 'PPO':
                        action, value, log_prob = agent.select_action(state)
                        agent.memory.states.append(state)
                        agent.memory.actions.append(action)
                        agent.memory.rewards.append(reward)
                        agent.memory.values.append(value)
                        agent.memory.log_probs.append(log_prob)
                        agent.memory.masks.append(1 - done)
                    
                    # Update policy if enough steps
                    if len(agent.memory.states) >= UPDATE_TIMESTEP:
                        print(f"\nPerforming policy update at step {episode_metrics['steps']}")
                        agent.update()
                    
                    state = next_state
                    
                    # Print progress
                    if episode_metrics['steps'] % 100 == 0:
                        print(f"Step {episode_metrics['steps']}: "
                              f"Reward = {episode_metrics['reward']:.2f}, "
                              f"MPC/PPO ratio = {episode_metrics['mpc_uses']}/{episode_metrics['ppo_uses']}")
                
                # Episode completion statistics
                episode_duration = time.time() - episode_start_time
                episode_metrics['avg_speed'] = np.mean(episode_metrics['speeds']) if episode_metrics['speeds'] else 0
                episode_metrics['avg_lane_deviation'] = np.mean(episode_metrics['lane_deviations']) if episode_metrics['lane_deviations'] else 0
                
                # Log episode metrics
                print(f"\nEpisode {episode} Summary:")
                print(f"Total Reward: {episode_metrics['reward']:.2f}")
                print(f"Steps: {episode_metrics['steps']}")
                print(f"MPC/PPO Usage: {episode_metrics['mpc_uses']}/{episode_metrics['ppo_uses']}")
                print(f"MPC Failures: {episode_metrics['mpc_failures']}")
                print(f"Collisions: {episode_metrics['collisions']}")
                print(f"Average Speed: {episode_metrics['avg_speed']:.2f} km/h")
                print(f"Average Lane Deviation: {episode_metrics['avg_lane_deviation']:.2f} m")
                print(f"Duration: {episode_duration:.2f} seconds")
                
                # Update training metrics
                training_metrics['episode_rewards'].append(episode_metrics['reward'])
                training_metrics['episode_lengths'].append(episode_metrics['steps'])
                training_metrics['mpc_usage'].append(episode_metrics['mpc_uses'])
                training_metrics['ppo_usage'].append(episode_metrics['ppo_uses'])
                training_metrics['collisions'].append(episode_metrics['collisions'])
                training_metrics['mpc_failures'].append(episode_metrics['mpc_failures'])
                
                # Log to tensorboard
                writer.add_scalar('Training/Episode_Reward', episode_metrics['reward'], episode)
                writer.add_scalar('Training/Episode_Length', episode_metrics['steps'], episode)
                writer.add_scalar('Training/MPC_Usage', episode_metrics['mpc_uses'], episode)
                writer.add_scalar('Training/PPO_Usage', episode_metrics['ppo_uses'], episode)
                writer.add_scalar('Training/MPC_Failures', episode_metrics['mpc_failures'], episode)
                writer.add_scalar('Training/Collisions', episode_metrics['collisions'], episode)
                writer.add_scalar('Training/Average_Speed', episode_metrics['avg_speed'], episode)
                writer.add_scalar('Training/Lane_Deviation', episode_metrics['avg_lane_deviation'], episode)
                
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
            
            # Save training metrics
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
        
        print(f"Starting CARLA Hybrid Control (MPC+PPO) training at {time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        traceback.print_exc()
    finally:
        # Final cleanup
        try:
            pygame.quit()
            print("\nCleaned up pygame")
        except:
            pass
        print("\nProgram terminated")