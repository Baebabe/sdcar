import pygame
import carla
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import math
from tqdm import tqdm

# Configuration
DT = 0.1
PREDICTION_STEPS = 20
CONTROL_STEPS = 5
ILQR_ITERATIONS = 30
REGULARIZATION = 1.0
SPEED_TARGET = 8.0  # m/s

# Neural Network Dynamics Model
class DynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 5)
        
        # Initialize with pretrained weights
        pretrained = torch.load('model/net_bicycle_model_100ms_20000_v4.model')
        self.load_state_dict(pretrained.state_dict())
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return self.fc3(x)

# CARLA Environment Class
class CarlaEnv:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.actor_list = []
        
        # Setup world
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        self.world.apply_settings(settings)
        
    def reset(self):
        self.destroy_actors()
        
        # Spawn vehicle
        vehicle_bp = self.blueprint_lib.filter('model3')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)
        
        # Attach camera
        camera_bp = self.blueprint_lib.find('sensor.camera.rgb')
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        
        return self.get_state()
    
    def get_state(self):
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        accel = self.vehicle.get_acceleration()
        angular_vel = self.vehicle.get_angular_velocity()
        
        return np.array([
            transform.location.x,
            transform.location.y,
            math.radians(transform.rotation.yaw),
            velocity.x,
            velocity.y,
            math.radians(angular_vel.z)
        ])
    
    def apply_control(self, control):
        carla_control = carla.VehicleControl(
            throttle=float(control[0]),
            steer=float(control[1]),
            brake=float(control[2])
        )
        self.vehicle.apply_control(carla_control)
        self.world.tick()
    
    def destroy_actors(self):
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

# MPC Controller
class MPCController:
    def __init__(self, model_path):
        self.model = DynamicsModel().eval()
        self.state_dim = 6
        self.control_dim = 3
        
    def predict(self, state, controls):
        states = [torch.FloatTensor(state)]
        for control in controls:
            x = torch.cat([states[-1], torch.FloatTensor(control)])
            dx = self.model(x)
            states.append(states[-1] + dx * DT)
        return torch.stack(states[1:])
    
    def ilqr(self, initial_state, reference_path):
        # Initialize controls
        controls = np.zeros((PREDICTION_STEPS, self.control_dim))
        states = self.predict(initial_state, controls).detach().numpy()
        
        for _ in range(ILQR_ITERATIONS):
            # Forward pass
            states = self.predict(initial_state, controls)
            cost = self.calculate_cost(states, reference_path)
            
            # Backward pass
            k, K = self.backward_pass(states, controls, reference_path)
            
            # Update controls
            controls += k + K @ (states - states).flatten()
            controls = np.clip(controls, [-1, 0, 0], [1, 1, 1])
        
        return controls[:CONTROL_STEPS]
    
    def calculate_cost(self, states, reference):
        position_cost = np.linalg.norm(states[:, :2] - reference[:PREDICTION_STEPS, :2], axis=1)
        speed_cost = (states[:, 3] - SPEED_TARGET)**2
        control_cost = np.linalg.norm(states[:, 6:9], axis=1)
        return np.sum(position_cost + speed_cost + 0.1*control_cost)
    
    def backward_pass(self, states, controls, reference):
        # Simplified backward pass implementation
        k = np.random.randn(*controls.shape) * 0.1
        K = np.random.randn(controls.shape[0], controls.shape[1], states.shape[1]) * 0.01
        return k, K

# Main Loop
def main():
    env = CarlaEnv()
    mpc = MPCController('net_bicycle_model_100ms_20000_v4.model')
    
    try:
        state = env.reset()
        reference_path = generate_reference_path(state)
        
        while True:
            # Get reference path (replace with actual path planning)
            reference_path = update_reference_path(state, reference_path)
            
            # Compute MPC controls
            controls = mpc.ilqr(state, reference_path)
            
            # Apply first control step
            env.apply_control(controls[0])
            
            # Update state
            state = env.get_state()
            
    finally:
        env.destroy_actors()

def generate_reference_path(initial_state):
    # Simple straight path generator
    path = np.zeros((PREDICTION_STEPS, 2))
    path[:, 0] = initial_state[0] + np.arange(PREDICTION_STEPS) * SPEED_TARGET * DT
    path[:, 1] = initial_state[1]
    return path

def update_reference_path(current_state, previous_path):
    # Shift path forward and append new point
    new_path = np.roll(previous_path, -1, axis=0)
    new_path[-1] = [previous_path[-1,0] + SPEED_TARGET*DT, previous_path[-1,1]]
    return new_path

if __name__ == '__main__':
    main()