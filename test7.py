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
import casadi as ca
import json
from datetime import datetime
from typing import List, Optional, Union,Dict
import heapq
from scipy.interpolate import CubicSpline

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


class MPCController:
    def __init__(self, dt=0.1, N=20):
        self.dt = dt
        self.N = N
        self.obstacle_safety_margin = 1.5
        self.obstacle_cost_weight = 1e6
        self.Q = np.diag([15.0, 15.0, 5.0, 5.0])  # Changed to 4x4 for state dimensions
        self.R = np.diag([1.0, 0.5])

        # State and control dimensions
        self.nx = 4  # [x, y, yaw, v] - Removed slip angle
        self.nu = 2  # [steering, throttle]

        # Calculate dimensions
        self.n_states = self.nx
        self.n_controls = self.nu
        self.n_vars = (self.N + 1) * self.n_states + self.N * self.n_controls
        self.n_equality_constraints = (self.N + 1) * self.n_states

        # Vehicle parameters
        self.L = 2.9
        self.bounds = {
            'steer': [-1.0, 1.0],
            'throttle': [-1.0, 1.0],
            'v_max': 15.0,
            'v_min': 0.0,
            'max_yaw_rate': 1.57
        }

        self.safety_distance = 3.0
        self.brake_gain = 10.0
        self.throttle_gain = 5.0

        # Initialize solution guess
        self.x0 = np.zeros(self.n_vars)

        # Setup optimization problem
        self._setup_optimization()

    def _setup_optimization(self):
        """Setup MPC optimization problem using CasADi with corrected vehicle dynamics"""
        try:
            # State variables
            x = ca.SX.sym('x')
            y = ca.SX.sym('y')
            yaw = ca.SX.sym('yaw')
            v = ca.SX.sym('v')
            state = ca.vertcat(x, y, yaw, v)  # 4-dimensional state vector

            # Control variables
            steer = ca.SX.sym('steer')
            throttle = ca.SX.sym('throttle')
            controls = ca.vertcat(steer, throttle)

            # Decision variables
            X = ca.SX.sym('X', self.n_states, self.N + 1)
            U = ca.SX.sym('U', self.n_controls, self.N)

            # Parameters
            P = ca.SX.sym('P', self.n_states + self.n_states * self.N)

            # Vehicle dynamics
            acceleration = ca.if_else(throttle >= 0,
                                 self.throttle_gain * throttle,
                                 self.brake_gain * throttle)

            rhs = ca.vertcat(
                v * ca.cos(yaw),
                v * ca.sin(yaw),
                v * ca.tan(steer) / self.L,
                acceleration - 0.2 * v**2
            )

            # Create integration function
            f = ca.Function('f', [state, controls], [rhs])

            # Initialize constraints and objective
            g = []
            obj = 0

            # Initial state constraint
            g.append(X[:, 0] - P[0:self.n_states])

            # Dynamic constraints and cost
            for k in range(self.N):
                # State error cost
                ref_idx = self.n_states + k * self.n_states
                state_error = X[:, k] - P[ref_idx:ref_idx + self.n_states]
                obj += ca.mtimes(state_error.T, ca.mtimes(self.Q, state_error))

                # Control cost
                obj += ca.mtimes(U[:, k].T, ca.mtimes(self.R, U[:, k]))

                # Next state based on dynamics
                state_next = X[:, k] + self.dt * f(X[:, k], U[:, k])
                g.append(X[:, k + 1] - state_next)

                # Yaw rate constraint
                if k > 0:
                    yaw_rate = (X[2, k] - X[2, k-1]) / self.dt
                    g.append(yaw_rate)

            # Create optimization variables
            opt_vars = ca.vertcat(
                ca.reshape(X, -1, 1),
                ca.reshape(U, -1, 1)
            )

            # Concatenate constraints
            g = ca.vertcat(*g)

            # Create NLP
            nlp = {
                'x': opt_vars,
                'f': obj,
                'g': g,
                'p': P
            }

            # Solver options
            opts = {
                'ipopt.print_level': 0,
                'ipopt.max_iter': 100,
                'ipopt.tol': 1e-4,
                'ipopt.acceptable_tol': 1e-4,
                'print_time': 0,
                'ipopt.warm_start_init_point': 'yes',
                'ipopt.mu_strategy': 'adaptive'
            }

            # Create solver
            self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

            # Store variables
            self.X = X
            self.U = U
            self.f = f
            self.opt_vars = opt_vars
            self.n_constraints = g.size1()

            print("MPC optimization problem setup completed successfully")
            return True

        except Exception as e:
            print(f"Error in MPC setup_optimization: {e}")
            traceback.print_exc()
            return False

    def solve(self, current_state, reference_trajectory, detected_objects):
        """ Solve MPC optimization problem with timeout and emergency handling """
        try:
            # Start timeout timer
            start_time = time.time()
            max_solve_time = 0.05  # 50ms timeout

            # Convert current state to numpy array
            current_state = np.array(current_state[:4], dtype=np.float64).reshape(-1)

            # Ensure reference trajectory has correct dimensions
            if not isinstance(reference_trajectory, np.ndarray):
                reference_trajectory = np.array(reference_trajectory, dtype=np.float64)

            # If reference trajectory is a single point, extend it to N+1 points
            if reference_trajectory.shape[0] == 1:
                reference_trajectory = np.tile(reference_trajectory, (self.N + 1, 1))
            elif reference_trajectory.shape[1] > 4:
                reference_trajectory = reference_trajectory[:, :4]

            # Ensure we have exactly N+1 timesteps
            if reference_trajectory.shape[0] < self.N + 1:
                last_point = reference_trajectory[-1:]
                points_needed = self.N + 1 - reference_trajectory.shape[0]
                extension = np.tile(last_point, (points_needed, 1))
                reference_trajectory = np.vstack([reference_trajectory, extension])
            elif reference_trajectory.shape[0] > self.N + 1:
                reference_trajectory = reference_trajectory[:(self.N + 1), :]

            # Calculate expected parameter vector size
            expected_param_size = self.n_states + self.N * self.n_states

            # Create parameter vector
            param_vector = np.concatenate([
                current_state,
                reference_trajectory[:self.N].reshape(-1)
            ]).astype(np.float64)

            # Verify parameter vector size and adjust if necessary
            if param_vector.size != expected_param_size:
                print(f"Warning: Parameter vector size mismatch. Expected {expected_param_size}, got {param_vector.size}")
                param_vector = param_vector[:expected_param_size]

            # Initialize warm start if needed
            if self.x0 is None:
                self.x0 = np.zeros(self.n_vars)

            # Initialize bounds
            lbx = np.zeros(self.n_vars)
            ubx = np.zeros(self.n_vars)

            # State bounds
            for i in range(self.N + 1):
                lbx[i * self.n_states:(i + 1) * self.n_states] = [-np.inf, -np.inf, -2*np.pi, self.bounds['v_min']]
                ubx[i * self.n_states:(i + 1) * self.n_states] = [np.inf, np.inf, 2*np.pi, self.bounds['v_max']]

            # Control bounds
            control_start = (self.N + 1) * self.n_states
            for i in range(self.N):
                idx = control_start + i * self.n_controls
                lbx[idx:idx + self.n_controls] = [self.bounds['steer'][0], self.bounds['throttle'][0]]
                ubx[idx:idx + self.n_controls] = [self.bounds['steer'][1], self.bounds['throttle'][1]]

            # Constraint bounds
            lbg = np.zeros(self.n_constraints)
            ubg = np.zeros(self.n_constraints)

            # Dynamic constraints (equality)
            lbg[:self.n_equality_constraints] = 0
            ubg[:self.n_equality_constraints] = 0

            # Yaw rate constraints
            if self.N > 1:
                yaw_rate_start = self.n_equality_constraints
                for i in range(self.N - 1):
                    lbg[yaw_rate_start + i] = -self.bounds['max_yaw_rate']
                    ubg[yaw_rate_start + i] = self.bounds['max_yaw_rate']

            try:
                # Solve the optimization problem with timeout check
                sol = self.solver(
                    x0=self.x0,
                    lbx=lbx,
                    ubx=ubx,
                    lbg=lbg,
                    ubg=ubg,
                    p=param_vector
                )

                # Check for timeout
                if time.time() - start_time > max_solve_time:
                    print("MPC solve timeout - applying emergency brake")
                    return np.array([0.0, 0.0, 1.0])  # Full brake with split controls

                # Check if solution was found
                if self.solver.stats()['success']:
                    opt_vars = sol['x'].full().flatten()
                    control_start = (self.N + 1) * self.n_states
                    optimal_control = opt_vars[control_start:control_start + self.n_controls]
                    self.x0 = opt_vars  # Update warm start

                    # Convert throttle/brake output
                    steering = float(np.clip(optimal_control[0], -1.0, 1.0))
                    throttle_brake = float(np.clip(optimal_control[1], -1.0, 1.0))

                    # Split into throttle and brake
                    throttle = max(throttle_brake, 0.0)
                    brake = max(-throttle_brake, 0.0)

                    return np.array([steering, throttle, brake])
                else:
                    print("MPC solver failed to find a solution - applying partial brake")
                    return np.array([0.0, 0.0, 0.8])  # Partial brake with split controls

            except Exception as e:
                print(f"MPC solver error: {e} - applying emergency brake")
                return np.array([0.0, 0.0, 1.0])  # Full brake with split controls

        except Exception as e:
            print(f"Error in MPC solve: {e} - applying emergency brake")
            traceback.print_exc()
            return np.array([0.0, 0.0, 1.0])  # Full brake with split controls
        
    def adjust_for_turns(self, reference_trajectory):
        """Adjust target speed based on the curvature of the reference trajectory."""
        # Calculate curvature and adjust speed
        for i in range(len(reference_trajectory) - 1):
            x1, y1, _, _ = reference_trajectory[i]
            x2, y2, _, _ = reference_trajectory[i + 1]
            curvature = np.arctan2(y2 - y1, x2 - x1)  # Simple curvature calculation
            if abs(curvature) > 0.1:  # Threshold for sharp turns
                # Reduce speed for sharp turns
                reference_trajectory[i + 1][3] *= 0.5  # Reduce target speed by half

    def solve_with_obstacle_avoidance(self, current_state, reference_trajectory, detected_objects):
            """
            Solve MPC optimization problem with obstacle avoidance.
            """
            # Check for obstacles in the same lane
            detected_obstacles = []
            for obj in detected_objects:
                x = current_state[0] + obj['rel_x']
                y = current_state[1] + obj['rel_y']
                radius = obj['radius']
                detected_obstacles.append((x, y, radius))

            return self.solve(current_state, reference_trajectory,  detected_obstacles)


class KalmanFilter:
    """Simple Kalman Filter for obstacle tracking"""
    def __init__(self, initial_position):
        self.dt = 0.1
        self.state = np.array([initial_position[0], initial_position[1], 0, 0])
        self.P = np.eye(4) * 0.1
        
    def update(self, position):
        # Prediction step
        F = np.array([[1, 0, self.dt, 0],
                      [0, 1, 0, self.dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Q = np.eye(4) * 0.01
        
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        
        # Update step
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R = np.eye(2) * 0.1
        y = position - H @ self.state
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        
    def get_velocity(self):
        return self.state[2], self.state[3]

class GlobalPathPlanner:
    def __init__(self, world, sampling_resolution=2.0):
        self.world = world
        self.sampling_resolution = sampling_resolution
        self.topology = self._get_topology()
        
    def _get_topology(self):
        topology = []
        waypoints = self.world.get_map().generate_waypoints(self.sampling_resolution)
        
        for waypoint in waypoints:
            connections = []
            next_waypoints = waypoint.next(self.sampling_resolution)
            if next_waypoints:
                connections.append(next_waypoints[0])
            
            prev_waypoints = waypoint.previous(self.sampling_resolution)
            if prev_waypoints:
                connections.append(prev_waypoints[0])
            
            topology.append({
                'waypoint': waypoint,
                'connections': connections
            })
        return topology
    
    def smooth_path(self, path):
        """Apply cubic spline smoothing to path"""
        if len(path) < 4:
            return path
            
        x = [wp.transform.location.x for wp in path]
        y = [wp.transform.location.y for wp in path]
        
        # Parametrize by cumulative distance
        dist = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2))
        
        # Fit splines
        cs_x = CubicSpline(dist, x)
        cs_y = CubicSpline(dist, y)
        
        # Resample at higher resolution
        new_dist = np.linspace(0, dist[-1], 2*len(dist))
        new_x = cs_x(new_dist)
        new_y = cs_y(new_dist)
        
        # Create new waypoints
        smoothed_waypoints = []
        for x, y in zip(new_x, new_y):
            loc = carla.Location(x=x, y=y)
            wp = self.world.get_map().get_waypoint(loc)
            smoothed_waypoints.append(wp)
            
        return smoothed_waypoints

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

            if self.heuristic(current.transform.location, goal.transform.location) < 2.0:
                print("Goal found!")
                break

            next_waypoints = current.next(2.0)

            for next_wp in next_waypoints:
                new_cost = cost_so_far[current] + self.heuristic(
                    current.transform.location,
                    next_wp.transform.location
                )

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

        # Apply path smoothing
        smoothed_path = self.smooth_path(path)
        
        # Convert to custom Waypoints with velocity information
        final_path = self._add_velocity_to_path(smoothed_path)

        print(f"Path found with {len(final_path)} waypoints")

        # Visualization code
        for i in range(len(final_path)-1):
            begin = final_path[i].transform.location
            end = final_path[i+1].transform.location

            begin = begin + carla.Location(z=0.5)
            end = end + carla.Location(z=0.5)

            self.world.debug.draw_line(
                begin, end,
                thickness=0.5,
                color=carla.Color(b=255, g=0, r=0),
                life_time=0.0,
                persistent_lines=True
            )

            self.world.debug.draw_point(
                begin,
                size=0.1,
                color=carla.Color(b=255, g=255, r=255),
                life_time=0.0,
                persistent_lines=True
            )

        # Draw final waypoint and endpoints
        self.world.debug.draw_point(
            final_path[-1].transform.location + carla.Location(z=0.5),
            size=0.1,
            color=carla.Color(b=255, g=255, r=255),
            life_time=0.0,
            persistent_lines=True
        )

        self.world.debug.draw_point(
            final_path[0].transform.location + carla.Location(z=1.0),
            size=0.2,
            color=carla.Color(r=0, g=255, b=0),
            life_time=0.0,
            persistent_lines=True
        )

        self.world.debug.draw_point(
            final_path[-1].transform.location + carla.Location(z=1.0),
            size=0.2,
            color=carla.Color(r=255, g=0, b=0),
            life_time=0.0,
            persistent_lines=True
        )

        return final_path

    def _add_velocity_to_path(self, path):
        """Add velocity information to path waypoints based on curvature"""
        waypoints = []
        for i in range(len(path)):
            wp = path[i]
            velocity = 20.0
            
            if i < len(path) - 1:
                next_wp = path[i + 1]
                dx = next_wp.transform.location.x - wp.transform.location.x
                dy = next_wp.transform.location.y - wp.transform.location.y
                angle = math.atan2(dy, dx)
                curvature = abs(angle)
                
                velocity = max(5.0, 20.0 - 15.0 * curvature)

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

from sklearn.cluster import DBSCAN

def transform_points(points, vehicle_transform):
    """Transform points from world to vehicle coordinates"""
    # Create rotation matrix
    yaw = np.radians(vehicle_transform.rotation.yaw)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Translate and rotate points
    points = points - np.array([
        vehicle_transform.location.x,
        vehicle_transform.location.y,
        vehicle_transform.location.z
    ])
    return np.dot(points, R.T)

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
        self.lidar = None
        self.tracked_obstacles = {}
        self.detected_obstacles = []
        
        # Initialize point cloud parameters
        self.lidar_points_per_cloud = 120000  # Match with LiDAR sensor setting
        self.points_per_channel = 3750  # Typical value for 32 channel LiDAR
        # Initialize YOLO model
        self._init_yolo()
        self.waypoint_buffer = 5
        self.lookahead_distance = 5.0
        self.path_tolerance = 2.0
        self.min_distance_to_next_wp = 2.0

        # Initialize state dimensions
        self.state_dim = 5  # (x, y, v, heading, slip angle)

        # Initialize MPC controller
        self.mpc = MPCController(dt=0.1, N=20)

        # Reward parameters
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

        
    def process_lidar(self, weak_self, point_cloud):
        """Process LiDAR point cloud data and track obstacles using Kalman filtering"""
        self = weak_self()
        if self is None:
            return

        try:
            # Convert point cloud to numpy array safely
            raw_data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))

            # Ensure the total size is divisible by 4 (x, y, z, intensity)
            valid_points = len(raw_data) - (len(raw_data) % 4)
            points = raw_data[:valid_points]

            # Reshape with exact dimensions
            points = points.reshape(-1, 4)  # Let numpy calculate the first dimension

            # Extract coordinates (x, y, z)
            points_xyz = points[:, :3]

            # Transform to vehicle coordinates
            vehicle_transform = self.vehicle.get_transform()
            points_xyz = transform_points(points_xyz, vehicle_transform)

            # Filter points by height
            height_min, height_max = -1.0, 2.0
            mask = (points_xyz[:, 2] >= height_min) & (points_xyz[:, 2] <= height_max)
            points_filtered = points_xyz[mask]

            if len(points_filtered) > 0:
                # Cluster points using DBSCAN
                clustering = DBSCAN(
                    eps=1.5,          # 1.5 meters radius
                    min_samples=5,    # Minimum 5 points to form a cluster
                    n_jobs=-1         # Use all CPU cores
                ).fit(points_filtered[:, :2])  # Only cluster on x,y coordinates

                labels = clustering.labels_

                # Store current detections for tracking
                current_detections = set()

                # Process clusters
                self.detected_obstacles = []
                unique_labels = np.unique(labels)

                for label in unique_labels:
                    if label == -1:  # Skip noise points
                        continue

                    # Get cluster points
                    cluster_mask = labels == label
                    cluster_points = points_filtered[cluster_mask]

                    # Calculate cluster properties
                    centroid = np.mean(cluster_points[:, :2], axis=0)
                    radius = np.max(np.linalg.norm(cluster_points[:, :2] - centroid, axis=1))
                    height = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])

                    # Generate unique ID for the obstacle based on position
                    obstacle_id = hash(tuple(map(float, centroid)))  # Ensure hashable values
                    current_detections.add(obstacle_id)

                    # Initialize tracked_obstacles if not exists
                    if not hasattr(self, 'tracked_obstacles'):
                        self.tracked_obstacles = {}

                    # Update or initialize Kalman filter
                    if obstacle_id in self.tracked_obstacles:
                        self.tracked_obstacles[obstacle_id].update(centroid)
                    else:
                        self.tracked_obstacles[obstacle_id] = KalmanFilter(centroid)

                    # Get velocity estimate from Kalman filter
                    vx, vy = self.tracked_obstacles[obstacle_id].get_velocity()

                    # Store obstacle information
                    self.detected_obstacles.append({
                        'id': obstacle_id,
                        'position': [float(centroid[0]), float(centroid[1])],
                        'x': float(centroid[0]),
                        'y': float(centroid[1]),
                        'radius': float(radius),
                        'height': float(height),
                        'vx': float(vx),
                        'vy': float(vy)
                    })

                # Remove tracks for obstacles that weren't detected
                if hasattr(self, 'tracked_obstacles'):
                    for obstacle_id in list(self.tracked_obstacles.keys()):
                        if obstacle_id not in current_detections:
                            del self.tracked_obstacles[obstacle_id]

        except Exception as e:
            print(f"Error processing LiDAR data: {e}")
            if hasattr(self, 'world') and hasattr(self.world, 'debug'):
                # Draw debug sphere at vehicle location to indicate error
                try:
                    debug = self.world.debug
                    vehicle_location = self.vehicle.get_location()
                    debug.draw_point(
                        vehicle_location,
                        size=0.1,
                        color=carla.Color(r=255, g=0, b=0),
                        life_time=0.1
                    )
                except:
                    pass  # Ignore debug visualization errors

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


    def get_reference_from_global_path(self, current_state):
        """Get smoothed reference trajectory from global path using cubic splines"""
        if not self.global_path:
            return np.zeros((self.mpc.N, 4))

        # Get current position and yaw
        current_pos = np.array([current_state[0], current_state[1]])
        current_yaw = current_state[2]

        # Find nearest waypoint ahead of vehicle
        closest_idx = self.current_path_index
        min_dist = float('inf')

        # Search forward from current index
        search_range = 20  # Look 20 waypoints ahead
        end_idx = min(len(self.global_path), self.current_path_index + search_range)

        for i in range(self.current_path_index, end_idx):
            wp = self.global_path[i]
            dist = np.linalg.norm(current_pos - [wp.transform.location.x, wp.transform.location.y])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        self.current_path_index = closest_idx

        # Get points for spline fitting
        num_spline_points = min(5, len(self.global_path) - closest_idx)
        waypoints = self.global_path[closest_idx:closest_idx + num_spline_points]

        # Extract x, y coordinates including current position
        x = [current_pos[0]] + [wp.transform.location.x for wp in waypoints]
        y = [current_pos[1]] + [wp.transform.location.y for wp in waypoints]

        # Create spline parameterized by cumulative distance
        diffs = np.diff(x)**2 + np.diff(y)**2
        distances = np.concatenate(([0], np.cumsum(np.sqrt(diffs))))

        # Fit cubic splines
        cs_x = CubicSpline(distances, x)
        cs_y = CubicSpline(distances, y)

        # Generate N reference points
        reference_points = []
        total_distance = distances[-1]
        for i in range(self.mpc.N):
            # Interpolate position along the spline
            dist = (i / self.mpc.N) * total_distance
            x_ref = cs_x(dist)
            y_ref = cs_y(dist)

            # Get corresponding waypoint for yaw and velocity
            wp_idx = min(closest_idx + i, len(self.global_path)-1)
            wp = self.global_path[wp_idx]
            target_yaw = np.radians(wp.transform.rotation.yaw)

            reference_points.append([
                x_ref,
                y_ref,
                target_yaw,
                wp.velocity
            ])

        return np.array(reference_points)

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
                    # Check fusion source and handle different detection types
                    if obj.get('fusion_source') == 'lidar_only':
                        # For LiDAR-only detections, draw a circle with minimum radius
                        center_x = int(obj['position'][0] * IM_WIDTH / 640)
                        center_y = int(obj['position'][1] * IM_HEIGHT / 640)
                        
                        # Ensure minimum radius and proper line width
                        base_radius = obj.get('radius', 20) * min(IM_WIDTH, IM_HEIGHT) / 640
                        line_width = 2
                        radius = max(int(base_radius), line_width + 1)  # Ensure radius is greater than line width
                        
                        try:
                            pygame.draw.circle(surface, (255, 0, 0), (center_x, center_y), radius, min(line_width, radius - 1))
                        except ValueError:
                            # Fallback to filled circle if line drawing fails
                            pygame.draw.circle(surface, (255, 0, 0), (center_x, center_y), max(radius, 3), 0)
                    else:
                        # For YOLO or fused detections with bounding boxes
                        if 'bbox_width' in obj and 'bbox_height' in obj:
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
                            color = (0, 255, 0) if obj.get('fusion_source') == 'lidar+yolo' else (0, 255, 255)
                            pygame.draw.rect(surface, color, (x1, y1, x2-x1, y2-y1), 2)
                        else:
                            # Fallback for detections without bounding box information
                            center_x = int(obj['position'][0] * IM_WIDTH / 640)
                            center_y = int(obj['position'][1] * IM_HEIGHT / 640)
                            
                            # Ensure minimum radius and proper line width
                            base_radius = obj.get('radius', 20) * min(IM_WIDTH, IM_HEIGHT) / 640
                            line_width = 2
                            radius = max(int(base_radius), line_width + 1)
                            
                            try:
                                pygame.draw.circle(surface, (0, 255, 255), (center_x, center_y), radius, min(line_width, radius - 1))
                            except ValueError:
                                # Fallback to filled circle if line drawing fails
                                pygame.draw.circle(surface, (0, 255, 255), (center_x, center_y), max(radius, 3), 0)

                    # Draw label with appropriate positioning
                    if pygame.font.get_init():
                        font = pygame.font.Font(None, 24)
                        # Create label based on available information
                        label_parts = []
                        if 'class_name' in obj:
                            label_parts.append(obj['class_name'])
                        if 'depth' in obj:
                            label_parts.append(f"{obj['depth']:.1f}m")
                        if 'fusion_source' in obj:
                            label_parts.append(f"({obj['fusion_source']})")
                        
                        label = " ".join(label_parts)
                        text = font.render(label, True, (255, 255, 255))
                        
                        # Position label above the detection
                        text_x = int(obj['position'][0] * IM_WIDTH / 640)
                        text_y = int(obj['position'][1] * IM_HEIGHT / 640) - 20
                        surface.blit(text, (text_x, text_y))

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

        # Clear the previous path data
        self.global_path = []  # Clear the previous path
        self.current_path_index = 0  # Reset path index
        self.target_destination = None  # Clear target destination

        # Clear spawn point from previous episode
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
        """Spawn and setup the ego vehicle with camera and LiDAR sensors"""
        try:
            print("Starting vehicle setup...")  

            # Get the blueprint library
            blueprint_library = self.world.get_blueprint_library()
            print("Got blueprint library")  

            # Get the vehicle blueprint - Add error handling for blueprint
            vehicle_bp = None
            try:
                vehicle_bp = blueprint_library.filter('model3')[0]
                print("Got vehicle blueprint")
            except IndexError:
                print("model3 blueprint not found, trying alternative vehicle models...")
                available_vehicles = blueprint_library.filter('vehicle.*')
                if available_vehicles:
                    vehicle_bp = random.choice(available_vehicles)
                    print(f"Using alternative vehicle blueprint: {vehicle_bp.id}")
                else:
                    raise Exception("No vehicle blueprints available")

            # Get spawn points with validation
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points found")    

            # Initialize vehicle as None
            self.vehicle = None
            spawn_attempts = 0
            max_spawn_attempts = 20  # Increased from 10 to 20

            # Try spawning at episode spawn point first if it exists
            if hasattr(self, '_episode_spawn_point'):
                vehicle = self.world.try_spawn_actor(vehicle_bp, self._episode_spawn_point)
                if vehicle is not None:
                    self.vehicle = vehicle
                    print("Vehicle spawned successfully at episode spawn point")
                else:
                    print("Failed to spawn at episode spawn point, will try random points")

            # If no vehicle yet, try random spawn points
            while self.vehicle is None and spawn_attempts < max_spawn_attempts:
                spawn_point = random.choice(spawn_points)
                # Add small random offset to avoid exact spawn point collisions
                spawn_point.location.x += random.uniform(-1.0, 1.0)
                spawn_point.location.y += random.uniform(-1.0, 1.0)

                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle is not None:
                    self.vehicle = vehicle
                    self._episode_spawn_point = spawn_point
                    print(f"Vehicle spawned successfully after {spawn_attempts + 1} attempts")
                    break
                
                spawn_attempts += 1
                time.sleep(0.1)  # Add small delay between spawn attempts

            if self.vehicle is None:
                raise Exception(f"Failed to spawn vehicle after {max_spawn_attempts} attempts")

            # Set up sensors with better error handling and validation
            sensors_setup_complete = self._setup_sensors(blueprint_library)
            if not sensors_setup_complete:
                raise Exception("Failed to set up sensors")

            return True

        except Exception as e:
            print(f"Error setting up vehicle: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup_actors()
            return False

    def _setup_sensors(self, blueprint_library):
        """Helper method to set up sensors with better error handling"""
        try:
            # Camera setup
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            if camera_bp is None:
                raise Exception("Camera blueprint not found")

            camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
            camera_bp.set_attribute('fov', '90')
            print("Camera blueprint configured")

            # Spawn camera with more robust transform
            self.camera_transform = carla.Transform(
                carla.Location(x=2.0, y=0.0, z=1.4),
                carla.Rotation(pitch=-15.0)
            )

            self.camera = self.world.spawn_actor(
                camera_bp,
                self.camera_transform,
                attach_to=self.vehicle
            )

            if self.camera is None:
                raise Exception("Failed to spawn camera")
            print("Camera spawned successfully")

            # Set up camera callback
            weak_self = weakref.ref(self)
            self.camera.listen(lambda image: self._process_image(weak_self, image))
            print("Camera callback set up")

            # Set up collision sensor
            collision_bp = blueprint_library.find('sensor.other.collision')
            if collision_bp is None:
                raise Exception("Collision sensor blueprint not found")

            self.collision_sensor = self.world.spawn_actor(
                collision_bp,
                carla.Transform(),
                attach_to=self.vehicle
            )

            if self.collision_sensor is None:
                raise Exception("Failed to spawn collision sensor")
            print("Collision sensor spawned")

            # Set up collision callback
            self.collision_sensor.listen(lambda event: self.collision_hist.append(event))
            print("Collision callback set up")

            # Set up LiDAR sensor
            print("Setting up LiDAR sensor...")
            lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
            if lidar_bp is None:
                raise Exception("LiDAR blueprint not found")

            # Configure only core LiDAR parameters
            lidar_bp.set_attribute('channels', '32')
            lidar_bp.set_attribute('points_per_second', '120000')
            lidar_bp.set_attribute('rotation_frequency', '20.0')
            lidar_bp.set_attribute('range', '50.0')
            lidar_bp.set_attribute('upper_fov', '10.0')
            lidar_bp.set_attribute('lower_fov', '-30.0')
            lidar_bp.set_attribute('sensor_tick', '0.05')

            # Initialize tracked_obstacles dictionary
            if not hasattr(self, 'tracked_obstacles'):
                self.tracked_obstacles = {}

            # Spawn LiDAR with optimal positioning
            lidar_transform = carla.Transform(
                carla.Location(x=0.8, y=0.0, z=1.7),
                carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
            )

            self.lidar = self.world.spawn_actor(
                lidar_bp,
                lidar_transform,
                attach_to=self.vehicle
            )

            if self.lidar is None:
                raise Exception("Failed to spawn LiDAR")
            print("LiDAR spawned successfully")

            # Set up LiDAR callback
            def lidar_callback(point_cloud, weak_self=weak_self):
                self = weak_self()
                if self is not None:
                    try:
                        raw_data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
                        print(f"Raw LiDAR data size: {len(raw_data)}")
                        self.process_lidar(weak_self, point_cloud)
                    except Exception as e:
                        print(f"Error in LiDAR callback: {e}")

            self.lidar.listen(lidar_callback)
            print("LiDAR callback set up")

            # Wait for sensors to initialize
            sensor_init_timeout = 15
            start_time = time.time()

            while time.time() - start_time < sensor_init_timeout:
                if (hasattr(self, 'front_camera') and self.front_camera is not None and
                    hasattr(self, 'detected_obstacles')):
                    print("All sensors initialized successfully")
                    return True
                time.sleep(0.1)
                self.world.tick()

            raise Exception("Sensor initialization timeout")

        except Exception as e:
            print(f"Error setting up sensors: {e}")
            return False


    def cleanup_actors(self):
        """Clean up all spawned actors including LiDAR"""
        try:
            print("Starting cleanup of actors...")

            # Clean up LiDAR sensor
            if hasattr(self, 'lidar') and self.lidar and self.lidar.is_alive:
                self.lidar.destroy()
                self.lidar = None
                print("LiDAR sensor destroyed")

            # Clean up other sensors and vehicle
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
            self.detected_obstacles = []  # Clear detected obstacles list

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
        """Process image with YOLO and fuse with LiDAR data"""
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
    
            yolo_detections = []
            if len(pred[0]):
                # Process YOLO detections
                for *xyxy, conf, cls in pred[0]:
                    x1, y1, x2, y2 = map(float, xyxy)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    depth, depth_confidence = calculate_depth([x1, y1, x2, y2], int(cls))
    
                    yolo_detections.append({
                        'position': [center_x, center_y],  # Store position as list
                        'x': center_x,
                        'y': center_y,
                        'depth': depth,
                        'depth_confidence': depth_confidence,
                        'class': int(cls),
                        'class_name': self.yolo_model.names[int(cls)],
                        'confidence': float(conf),
                        'bbox_width': x2 - x1,
                        'bbox_height': y2 - y1,
                        'radius': max(x2 - x1, y2 - y1) / 2  # Estimate radius from bbox
                    })
    
            # Fuse YOLO detections with LiDAR data
            fused_objects = []
    
            for yolo_obj in yolo_detections:
                matched = False
                # Find matching LiDAR cluster
                for lidar_obj in self.detected_obstacles:
                    dx = yolo_obj['x'] - lidar_obj['x']
                    dy = yolo_obj['y'] - lidar_obj['y']
                    if math.sqrt(dx**2 + dy**2) < 2.0:  # 2.0m threshold for matching
                        # Create fused object with LiDAR position and YOLO classification
                        fused_obj = {
                            'position': [lidar_obj['x'], lidar_obj['y']],  # Add position list
                            'x': lidar_obj['x'],
                            'y': lidar_obj['y'],
                            'radius': lidar_obj['radius'],
                            'depth': lidar_obj['depth'] if 'depth' in lidar_obj else yolo_obj['depth'],
                            'class': yolo_obj['class'],
                            'class_name': yolo_obj['class_name'],
                            'confidence': yolo_obj['confidence'],
                            'fusion_source': 'lidar+yolo'
                        }
                        fused_objects.append(fused_obj)
                        matched = True
                        break
                    
                if not matched:
                    # Use YOLO-only detection if no LiDAR match found
                    yolo_obj['fusion_source'] = 'yolo_only'
                    fused_objects.append(yolo_obj)
    
            # Add remaining LiDAR obstacles that weren't matched
            for lidar_obj in self.detected_obstacles:
                matched = False
                for fused_obj in fused_objects:
                    dx = fused_obj['x'] - lidar_obj['x']
                    dy = fused_obj['y'] - lidar_obj['y']
                    if math.sqrt(dx**2 + dy**2) < 2.0:
                        matched = True
                        break
                    
                if not matched:
                    lidar_obj['fusion_source'] = 'lidar_only'
                    lidar_obj['class'] = -1  # Unknown class
                    lidar_obj['class_name'] = 'unknown'
                    lidar_obj['confidence'] = 1.0  # High confidence in obstacle presence
                    lidar_obj['position'] = [lidar_obj['x'], lidar_obj['y']]  # Add position list
                    fused_objects.append(lidar_obj)
    
            # Sort by depth and confidence
            fused_objects.sort(key=lambda x: x.get('depth', float('inf')) * (1 - x['confidence']))
            return fused_objects[:self.max_objects]
    
        except Exception as e:
            print(f"Error in YOLO detection and fusion: {e}")
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
        """Get complete state information including camera, LiDAR, and vehicle data"""
        if self.front_camera is None:
            return None

        # Initialize state array
        state_array = []

        # 1. Process YOLO detections (max_objects * 3 features)
        detections = self.process_yolo_detection(self.front_camera)
        for obj in detections:
            # Get position values with default of 0
            x_pos = obj.get('position', [0, 0])[0] / IM_WIDTH
            y_pos = obj.get('position', [0, 0])[1] / IM_HEIGHT

            # Handle case where depth might not be available
            depth = obj.get('depth', 0) / 100.0 if 'depth' in obj else 0.0

            state_array.extend([x_pos, y_pos, depth])

        # Pad if fewer than max objects
        remaining_objects = self.max_objects - len(detections)
        state_array.extend([0.0] * (remaining_objects * 3))

        # 2. Get waypoint information
        waypoint_info = self.get_waypoint_info()
        state_array.extend([
            waypoint_info.get('distance', 0) / 50.0,      # distance normalized
            waypoint_info.get('angle', 0) / math.pi       # angle normalized
        ])

        # 3. Get vehicle state
        vehicle_state = self.get_vehicle_state()
        state_array.extend([
            vehicle_state.get('speed', 0) / 50.0,         # speed normalized
            vehicle_state.get('steering', 0)              # steering already normalized
        ])

        # 4. Add LiDAR obstacle information
        lidar_state = []
        if hasattr(self, 'detected_obstacles'):
            for obstacle in self.detected_obstacles[:5]:  # Limit to 5 closest obstacles
                pos = obstacle.get('position', [0, 0])
                lidar_state.extend([
                    pos[0] / 50.0,                        # x position normalized by LiDAR range
                    pos[1] / 50.0,                        # y position normalized by LiDAR range
                    obstacle.get('radius', 0) / 10.0      # radius normalized by expected max size
                ])

            # Pad if fewer than 5 obstacles
            remaining = 5 - len(self.detected_obstacles)
            lidar_state.extend([0.0] * (remaining * 3))

        # Combine camera-based state with LiDAR state
        return np.concatenate([
            np.array(state_array, dtype=np.float16),
            np.array(lidar_state, dtype=np.float16)
        ])
    
    def get_detected_objects(self):
        """Get detected objects in the environment"""
        detected_objects = []

        try:
            # Get all actors in the world
            if not hasattr(self, 'world') or self.world is None:
                return []

            actor_list = self.world.get_actors()
            if actor_list is None:
                return []

            vehicle_list = actor_list.filter('vehicle.*')

            # Get ego vehicle location
            if not hasattr(self, 'vehicle') or self.vehicle is None:
                return []

            ego_location = self.vehicle.get_location()
            if ego_location is None:
                return []

            # Use distance-based detection as primary method instead of waypoints
            for vehicle in vehicle_list:
                if vehicle.id != self.vehicle.id:  # Skip ego vehicle
                    try:
                        vehicle_location = vehicle.get_location()
                        if vehicle_location is None:
                            continue

                        # Calculate distance
                        distance = ego_location.distance(vehicle_location)

                        # Only consider vehicles within 50 meters
                        if distance < 50.0:
                            # Calculate relative position
                            rel_x = vehicle_location.x - ego_location.x
                            rel_y = vehicle_location.y - ego_location.y

                            # Get velocity with error handling
                            try:
                                velocity = vehicle.get_velocity()
                                vel = np.sqrt(velocity.x**2 + velocity.y**2) if velocity is not None else 0.0
                            except:
                                vel = 0.0

                            # Optional: Add lane information if map is available
                            in_same_lane = False
                            if hasattr(self, 'map') and self.map is not None:
                                try:
                                    ego_waypoint = self.map.get_waypoint(ego_location)
                                    vehicle_waypoint = self.map.get_waypoint(vehicle_location)
                                    if ego_waypoint and vehicle_waypoint:
                                        in_same_lane = (ego_waypoint.lane_id == vehicle_waypoint.lane_id)
                                except:
                                    pass
                                
                            # Add additional information about lane status
                            detected_objects.append(np.array([
                                rel_x,
                                rel_y,
                                vel,
                                float(in_same_lane)  # Add lane information as a flag
                            ]))
                    except Exception as vehicle_error:
                        continue  # Skip this vehicle if there's an error
                    
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
    

    def step(self):
        try:
            # Initialize info dictionary
            info = {
                'speed': 0.0,
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
                mpc_action = self.mpc.solve_with_obstacle_avoidance(current_state, reference_trajectory, detected_objects)

                if mpc_action is not None:
                    # Convert MPC action to CARLA controls
                    steering = float(np.clip(mpc_action[0], -1.0, 1.0))
                    throttle = float(np.clip(mpc_action[1], 0.0, 1.0))
                    brake = float(np.clip(mpc_action[2], 0.0, 1.0))
                else:
                    print("MPC solution failed")
                    # Safe fallback control
                    steering = 0.0
                    throttle = 0.0
                    brake = 0.3

                # Apply smoothing
                if hasattr(self, 'last_control'):
                    smooth_factor = 0.8
                    throttle = smooth_factor * self.last_control.throttle + (1 - smooth_factor) * throttle
                    steering = smooth_factor * self.last_control.steer + (1 - smooth_factor) * steering

                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=steering,
                    brake=brake,
                    hand_brake=False
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

                # Visualize current target and progress
                if self.global_path and self.current_path_index < len(self.global_path):
                    target_wp = self.global_path[self.current_path_index]
                    self.visualize_current_progress(current_loc, target_wp)

                # Update simulation
                for _ in range(4):  # 4 physics steps per control step
                    self.world.tick()

                # Calculate reward and check termination conditions
                reward, done, reward_info = self.calculate_reward()
                info.update(reward_info)

                return current_state, reward, done, info

            except Exception as control_error:
                print(f"Error in control calculation: {control_error}")
                traceback.print_exc()
                return current_state, 0, True, info

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


if __name__ == "__main__":
    try:
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        print(f"Starting CARLA MPC Control at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create and run environment
        env = CarEnv()
        state = env.reset()
        
        while True:
            state, reward, done, info = env.step()
            
            if done:
                state = env.reset()
                
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