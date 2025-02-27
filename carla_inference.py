
# import glob
# import os
# import sys
# import time
# import random
# import math
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.applications import Xception
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model, load_model
# import signal

# # Add the CARLA Python API
# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     print("CARLA egg file not found. Please check your CARLA installation path.")
#     sys.exit(1)

# import carla

# # Constants - must match training exactly
# IM_WIDTH = 320
# IM_HEIGHT = 240
# LEARNING_RATE = 0.001

# class DQNAgent:
#     def __init__(self):
#         self.model = None
#         # Debug flags
#         self.last_prediction_time = time.time()
#         self.prediction_count = 0
        
#         # Set up GPU memory growth
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             try:
#                 for gpu in gpus:
#                     tf.config.experimental.set_memory_growth(gpu, True)
#             except RuntimeError as e:
#                 print(f"GPU setup error: {e}")

#     def load_model(self, model_path):
#         try:
#             print(f"Loading model from {model_path}...")
#             self.model = load_model(model_path)
#             print("\nModel summary:")
#             self.model.summary()
            
#             # Perform test prediction
#             print("\nPerforming test prediction...")
#             dummy_input = np.random.random((1, IM_HEIGHT, IM_WIDTH, 3))
#             test_pred = self.model.predict(dummy_input, verbose=0)
#             print("Test prediction shape:", test_pred.shape)
#             print("Test prediction values:", test_pred)
#             print("Test prediction action:", np.argmax(test_pred[0]))
#             return True
            
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             return False

#     def predict_action(self, state):
#         """Separate method for prediction with debugging"""
#         if state is None:
#             print("Warning: Received None state")
#             return 1  # Default to straight

#         current_time = time.time()
#         should_print = (current_time - self.last_prediction_time) >= 2.0

#         if should_print:
#             print(f"\nState shape: {state.shape}")
#             print(f"State range: {state.min():.2f} to {state.max():.2f}")

#         # Normalize state (keep as is since we're already normalizing in the main loop)
#         normalized_state = state / 255.0
#         state_expanded = np.expand_dims(normalized_state, axis=0)

#         # Get prediction - now using softmax, values will sum to 1
#         predictions = self.model.predict(state_expanded, verbose=0)[0]
#         action = np.argmax(predictions)

#         if should_print:
#             # Updated printing to show probabilities
#             print(f"Action probabilities:")
#             print(f"Left: {predictions[0]:.3%}")
#             print(f"Straight: {predictions[1]:.3%}")
#             print(f"Right: {predictions[2]:.3%}")
#             print(f"Selected action: {action}")
#             self.last_prediction_time = current_time

#         return action
# class CarlaEnv:
#     def __init__(self):
#         print("Initializing CARLA environment...")
#         self.client = carla.Client("localhost", 2000)
#         self.client.set_timeout(10.0)
#         self.world = self.client.get_world()
#         self.blueprint_library = self.world.get_blueprint_library()
#         self.model_3 = self.blueprint_library.filter("model3")[0]
#         self.actor_list = []
#         self.front_camera = None
#         self.vehicle = None
#         self.camera = None
#         self.window_name = "CARLA Camera Feed"
#         self.display_initialized = False
#         self.last_control_time = time.time()
        
#         # Define multiple spawn points as fallbacks
#         self.SPAWN_POINTS = [
#             carla.Transform(
#                 carla.Location(x=170.0, y=-10.0, z=0.5),  # Increased z coordinate
#                 carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
#             ),
#             carla.Transform(
#                 carla.Location(x=170.0, y=-15.0, z=0.5),
#                 carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
#             ),
#             carla.Transform(
#                 carla.Location(x=170.0, y=-5.0, z=0.5),
#                 carla.Rotation(pitch=0.0, yaw=90.0, roll=0.0)
#             )
#         ]

#     def clear_spawn_point(self, location, radius=10.0):  # Increased radius
#         """Clear all actors within a radius of the spawn point."""
#         print(f"Clearing spawn area around {location}")
#         actors = self.world.get_actors()
#         cleared_count = 0
        
#         for actor in actors:
#             if hasattr(actor, 'type_id') and (actor.type_id.startswith('vehicle') or actor.type_id.startswith('walker')):
#                 if hasattr(actor, 'get_location'):
#                     actor_loc = actor.get_location()
#                     if actor_loc.distance(location) < radius:
#                         print(f"Removing {actor.type_id} at {actor_loc}")
#                         actor.destroy()
#                         cleared_count += 1
        
#         print(f"Cleared {cleared_count} actors from spawn area")
#         time.sleep(1.0)  # Increased wait time after clearing
    
#     def process_img(self, image):
#         array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
#         array = np.reshape(array, (image.height, image.width, 4))
#         array = array[:, :, :3]  # Remove alpha channel
        
#         # Print image statistics periodically
#         if not hasattr(self, 'last_img_debug') or time.time() - self.last_img_debug > 5.0:
#             print(f"\nImage statistics:")
#             print(f"Shape: {array.shape}")
#             print(f"Range: [{array.min()}, {array.max()}]")
#             print(f"Mean: {array.mean():.2f}")
#             print(f"Std: {array.std():.2f}")
#             self.last_img_debug = time.time()
        
#         self.front_camera = array
        
#         # Add debug info to display
#         debug_img = array.copy()
#         if hasattr(self, 'vehicle') and self.vehicle:
#             v = self.vehicle.get_velocity()
#             kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
#             loc = self.vehicle.get_location()
#             cv2.putText(debug_img, f"Speed: {kmh:.1f} km/h", (10, 30), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#             cv2.putText(debug_img, f"Pos: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})", 
#                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
#         cv2.imshow(self.window_name, debug_img)
#         cv2.waitKey(1)
#         if not self.display_initialized:
#             self.display_initialized = True

#     def step(self, action, predictions=None):
#         try:
#             current_time = time.time()
#             should_print = (current_time - self.last_control_time) >= 1.0

#             control = carla.VehicleControl()

#             # More granular control based on prediction confidence
#             if action == 0:    # Left
#                 control.throttle = 0.5
#                 control.steer = -0.5
#             elif action == 1:  # Straight
#                 control.throttle = 0.7
#                 control.steer = 0
#             elif action == 2:  # Right
#                 control.throttle = 0.5
#                 control.steer = 0.5

#             if should_print:
#                 print(f"\nControl Update:")
#                 print(f"Action: {action}")
#                 if predictions is not None:
#                     print(f"Action probabilities:")
#                     print(f"Left: {predictions[0]:.1%}")
#                     print(f"Straight: {predictions[1]:.1%}")
#                     print(f"Right: {predictions[2]:.1%}")
#                 print(f"Applied control - Throttle: {control.throttle:.2f}, Steering: {control.steer:.2f}")

#                 v = self.vehicle.get_velocity()
#                 kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
#                 loc = self.vehicle.get_location()
#                 print(f"Speed: {kmh:.1f} km/h")
#                 print(f"Position: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
#                 self.last_control_time = current_time

#             self.vehicle.apply_control(control)

#         except Exception as e:
#             print(f"Error in step: {e}")
    
#     def setup_car(self):
#         try:
#             print("\nSetting up vehicle...")
            
#             # Try each spawn point until successful
#             for spawn_point in self.SPAWN_POINTS:
#                 try:
#                     # Clear spawn area
#                     self.clear_spawn_point(spawn_point.location)
                    
#                     # Check if spawn point is clear
#                     if self.world.get_spectator().get_transform().location.distance(spawn_point.location) < 2.0:
#                         print("Spawn point occupied by spectator, trying next point...")
#                         continue
                    
#                     # Try to spawn vehicle
#                     print(f"Attempting to spawn vehicle at: {spawn_point.location}")
#                     self.vehicle = self.world.try_spawn_actor(self.model_3, spawn_point)
                    
#                     if self.vehicle is not None:
#                         self.actor_list.append(self.vehicle)
#                         print("Vehicle spawned successfully")
#                         break
#                     else:
#                         print("Spawn attempt failed, trying next point...")
                
#                 except Exception as e:
#                     print(f"Error during spawn attempt: {e}")
#                     continue
            
#             if self.vehicle is None:
#                 raise RuntimeError("Failed to spawn vehicle at any of the designated spawn points")
            
#             # Set up camera
#             print("Setting up camera...")
#             self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
#             self.rgb_cam.set_attribute("image_size_x", f"{IM_WIDTH}")
#             self.rgb_cam.set_attribute("image_size_y", f"{IM_HEIGHT}")
#             self.rgb_cam.set_attribute("fov", "110")
            
#             transform = carla.Transform(carla.Location(x=2.5, z=0.7))
#             self.camera = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
#             self.actor_list.append(self.camera)
#             self.camera.listen(lambda data: self.process_img(data))
#             print("Camera setup complete")
            
#             # Print initial state
#             location = self.vehicle.get_location()
#             rotation = self.vehicle.get_transform().rotation
#             print(f"Initial position: x={location.x:.1f}, y={location.y:.1f}, z={location.z:.1f}")
#             print(f"Initial rotation: pitch={rotation.pitch:.1f}, yaw={rotation.yaw:.1f}, roll={rotation.roll:.1f}")
            
#         except Exception as e:
#             print(f"Error in setup_car: {e}")
#             self.cleanup()
#             raise

#     def cleanup(self):
#         print("\nStarting cleanup...")
#         try:
#             for actor in self.actor_list:
#                 if actor is not None and actor.is_alive:
#                     actor.destroy()
#             self.actor_list.clear()
            
#             if self.display_initialized:
#                 cv2.destroyWindow(self.window_name)
#                 cv2.waitKey(1)
            
#             cv2.destroyAllWindows()
#             for i in range(5):
#                 cv2.waitKey(1)
            
#             print("Cleanup complete")
#         except Exception as e:
#             print(f"Error during cleanup: {e}")


# def main():
#     env = None
#     try:
#         print("\n=== Starting Autonomous Driving Test ===")
        
#         # Create and load agent
#         print("\nInitializing DQN agent...")
#         agent = DQNAgent()
#         model_path = 'models/checkpoint_latest.h5'
#         if not os.path.exists(model_path):
#             print(f"Error: Model file not found at: {model_path}")
#             return
        
#         if not agent.load_model(model_path):
#             print("Failed to load model properly")
#             return
        
#         # Test model with random input
#         print("\nTesting model with random inputs...")
#         for _ in range(5):
#             test_input = np.random.random((1, IM_HEIGHT, IM_WIDTH, 3))
#             pred = agent.model.predict(test_input, verbose=0)[0]
#             print(f"Action probabilities:")
#             print(f"Left: {pred[0]:.1%}")
#             print(f"Straight: {pred[1]:.1%}")
#             print(f"Right: {pred[2]:.1%}")
#             print(f"Selected action: {np.argmax(pred)}\n")
            
#         # Set up environment
#         print("\nInitializing CARLA environment...")
#         env = CarlaEnv()
#         env.setup_car()
        
#         print("\nWaiting for camera initialization...")
#         start_time = time.time()
#         while env.front_camera is None:
#             if time.time() - start_time > 10:
#                 raise TimeoutError("Camera initialization timeout")
#             time.sleep(0.1)
        
#         print("\nStarting autonomous driving loop...")
#         while True:
#             if env.front_camera is None:
#                 print("Warning: No camera feed")
#                 continue
                
#             # Get current state and normalize
#             current_state = env.front_camera / 255.0
            
#             # Get prediction
#             predictions = agent.model.predict(
#                 np.expand_dims(current_state, axis=0),
#                 verbose=0
#             )[0]
#             action = np.argmax(predictions)
            
#             # Step environment with predictions for debugging
#             env.step(action, predictions)
            
#             # Small delay to prevent GPU overload
#             time.sleep(0.1)
            
#             if cv2.getWindowProperty(env.window_name, cv2.WND_PROP_VISIBLE) < 1:
#                 print("\nDisplay window closed, exiting...")
#                 break

#     except KeyboardInterrupt:
#         print('\nManual interrupt received')
#     except Exception as e:
#         print(f"\nAn error occurred: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         if env is not None:
#             env.cleanup()

# if __name__ == "__main__":
#     main()

import os
import sys
import glob
import time
import math
import numpy as np
import cv2
import torch
import argparse
from collections import deque

# Try to import CARLA
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("CARLA egg file not found. Please check your CARLA installation.")
    sys.exit(1)

import carla

# Constants (matching training)
IM_WIDTH = 320
IM_HEIGHT = 240
LATENT_DIM = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        
        # Calculate dimensions
        h1, w1 = conv2d_output_size(IM_HEIGHT, IM_WIDTH, 4, 2, 1)
        h2, w2 = conv2d_output_size(h1, w1, 4, 2, 1)
        h3, w3 = conv2d_output_size(h2, w2, 4, 2, 1)
        h4, w4 = conv2d_output_size(h3, w3, 4, 2, 1)
        
        self.final_h = h4
        self.final_w = w4
        self.conv_output_size = 256 * h4 * w4

        # Keep the same exact architecture as in training
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        # These were missing in the previous version
        self.fc_mu = torch.nn.Linear(self.conv_output_size, LATENT_DIM)
        self.fc_var = torch.nn.Linear(self.conv_output_size, LATENT_DIM)
        
        self.decoder_input = torch.nn.Linear(LATENT_DIM, self.conv_output_size)
        
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 256, self.final_h, self.final_w)
        return self.decoder(z)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def conv2d_output_size(h_in, w_in, kernel_size, stride, padding=0):
    h_out = (h_in + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    w_out = (w_in + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    return h_out, w_out

class ActorCritic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Actor network (matching the checkpoint architecture)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, action_dim),  # 2 outputs for steering and throttle
            torch.nn.Tanh()  # Bound outputs to [-1, 1]
        )
        
        # Critic network (matching the checkpoint architecture)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)  # Single value output
        )
    
    def forward(self, state):
        action_mean = self.actor(state)
        value = self.critic(state)
        return action_mean, value
    
    def get_action(self, state):
        with torch.no_grad():
            action_mean = self.actor(state)
            value = self.critic(state)
            return action_mean, value

class InferenceEnv:
    def __init__(self):
        print("\nInitializing CARLA environment for inference...")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        
        self.actor_list = []
        self.front_camera = None
        self.collision_hist = []
        self.window_name = "Autonomous Driving"
        
        # Metrics
        self.total_distance = 0
        self.prev_location = None
        self.collision_count = 0
        self.avg_speed = deque(maxlen=50)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
    def setup(self):
        # Spawn point (same as training)
        spawn_point = carla.Transform(
            carla.Location(x=96.0, y=-6.5, z=0.5),
            carla.Rotation(pitch=0.0, yaw=180.0, roll=0.0)
        )
        
        try:
            self.vehicle = self.world.spawn_actor(self.model_3, spawn_point)
            self.actor_list.append(self.vehicle)
            print("Vehicle spawned successfully")
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
            return False
        
        # Setup camera
        camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
        camera_bp.set_attribute('fov', '110')
        
        camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        self.camera.listen(lambda data: self._process_img(data))
        
        # Setup collision sensor
        col_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        return True
    
    def _process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.array(image.raw_data)
        array = array.reshape((IM_HEIGHT, IM_WIDTH, 4))
        array = array[:, :, :3]
        self.front_camera = array
        
        # Add telemetry to display
        debug_img = array.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get vehicle state
        v = self.vehicle.get_velocity()
        kmh = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)
        self.avg_speed.append(kmh)
        
        # Update total distance
        current_loc = self.vehicle.get_location()
        if self.prev_location:
            self.total_distance += current_loc.distance(self.prev_location)
        self.prev_location = current_loc
        
        # Display metrics
        cv2.putText(debug_img, f"Speed: {kmh:.1f} km/h", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Avg Speed: {np.mean(self.avg_speed):.1f} km/h", (10, 60), font, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Distance: {self.total_distance:.1f}m", (10, 90), font, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Collisions: {self.collision_count}", (10, 120), font, 0.7, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, debug_img)
        cv2.waitKey(1)
    
    def _on_collision(self, event):
        self.collision_hist.append(event)
        self.collision_count += 1
        print(f"\nCollision detected! Total collisions: {self.collision_count}")
    
    def get_state(self):
        if self.front_camera is None:
            return None
            
        v = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)  # km/h
        
        vehicle_location = self.vehicle.get_location()
        current_waypoint = self.world.get_map().get_waypoint(vehicle_location)
        
        dist = vehicle_location.distance(current_waypoint.transform.location)
        forward_vector = self.vehicle.get_transform().get_forward_vector()
        wp_vector = current_waypoint.transform.get_forward_vector()
        dot = forward_vector.x * wp_vector.x + forward_vector.y * wp_vector.y
        cross = forward_vector.x * wp_vector.y - forward_vector.y * wp_vector.x
        orientation = math.atan2(cross, dot)
        
        return {
            'image': self.front_camera,
            'speed': speed / 20.0,
            'throttle': self.vehicle.get_control().throttle,
            'steer': self.vehicle.get_control().steer,
            'waypoint_dist': dist,
            'waypoint_angle': orientation
        }
    
    def step(self, action):
        steer, throttle = action
        
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=0.0
        )
        
        self.vehicle.apply_control(control)
    
    def cleanup(self):
        print("\nCleaning up...")
        for actor in self.actor_list:
            if actor is not None and actor.is_alive:
                actor.destroy()
        self.actor_list.clear()
        
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        print("Cleanup complete")

