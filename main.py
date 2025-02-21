# import os
# import sys
# import numpy as np
# from tqdm import tqdm
# import pygame
# import torch
# from model_def import Net_v4
# from carla_env import CarEnv
# from mpc_controller import MPCController

# # Add Net_v4 to main module namespace
# sys.modules['__main__'].Net_v4 = Net_v4

# def main():
#     try:
#         # Create environment
#         env = CarEnv()
        
#         # Create MPC controller
#         controller = MPCController(
#             model_path="model/net_bicycle_model_100ms_20000_v4.model",
#             horizon=50,
#             dt=0.1
#         )
        
#         # Main loop
#         state, waypoints = env.reset()
        
#         with tqdm(total=2000, desc="Simulation") as pbar:
#             for i in range(2000):
#                 try:
#                     # Get control action from MPC
#                     # Pass the world object for visualization
#                     control = controller.solve(
#                         state=state,
#                         waypoints=waypoints,
#                         world=env.world if hasattr(env, 'world') else None
#                     )
                    
#                     # Apply control to environment
#                     state, waypoints, done, _ = env.step(control)
                    
#                     # Update progress bar
#                     pbar.update(1)
#                     pbar.set_postfix({
#                         'speed': f"{state[2]*3.6:.1f}km/h",
#                         'steering': f"{control[0]:.2f}"
#                     })
                    
#                     if done:
#                         print("\nCollision detected! Resetting environment...")
#                         state, waypoints = env.reset()
                        
#                     # Handle pygame events
#                     for event in pygame.event.get():
#                         if event.type == pygame.QUIT:
#                             return
#                         elif event.type == pygame.KEYDOWN:
#                             if event.key == pygame.K_ESCAPE:
#                                 return
                            
#                 except KeyboardInterrupt:
#                     print("\nSimulation interrupted by user")
#                     break
#                 except Exception as e:
#                     print(f"\nError during simulation step {i}: {str(e)}")
#                     break
                    
#     except Exception as e:
#         print(f"Error: {str(e)}")
        
#     finally:
#         # Cleanup
#         if 'env' in locals():
#             env.close()

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print('\nCancelled by user.')
#     except Exception as e:
#         print(f'Error occurred: {e}')
#     finally:
#         pygame.quit()



import os
import sys
import time
import numpy as np
from tqdm import tqdm
import pygame
import carla
import random

# Pygame window settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FPS = 30

class CameraManager:
    def __init__(self, parent_actor, world):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self._world = world
        
        # Set up camera blueprint
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        blueprint.set_attribute('image_size_x', str(WINDOW_WIDTH))
        blueprint.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        blueprint.set_attribute('fov', '110')
        
        # Find camera spawn point (behind and above the vehicle)
        spawn_point = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        
        # Spawn camera
        self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=self._parent)
        
        # Setup callback for camera data
        self.sensor.listen(self._parse_image)
    
    def _parse_image(self, image):
        """Convert CARLA raw image to Pygame surface"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def render(self, display):
        """Render camera image to Pygame display"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    
    def destroy(self):
        """Clean up camera sensor"""
        if self.sensor is not None:
            self.sensor.destroy()

def connect_to_carla(retries=10, timeout=5.0):
    """Attempt to connect to CARLA with retries"""
    for attempt in range(retries):
        try:
            print(f"Attempting to connect to CARLA (Attempt {attempt + 1}/{retries})")
            client = carla.Client('localhost', 2000)
            client.set_timeout(timeout)
            world = client.get_world()
            print("Successfully connected to CARLA")
            return client, world
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
    raise ConnectionError("Failed to connect to CARLA after multiple attempts")

def find_spawn_points(world, min_distance=30.0):
    """Find suitable spawn points with minimum distance between them"""
    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) < 2:
        raise ValueError("Not enough spawn points found!")
    
    for _ in range(50):
        start_point = random.choice(spawn_points)
        end_point = random.choice(spawn_points)
        distance = start_point.location.distance(end_point.location)
        if distance >= min_distance:
            return start_point, end_point
    return spawn_points[0], spawn_points[1]

def main():
    try:
        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("CARLA Navigation")
        display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()
        
        # Connect to CARLA
        print("Connecting to CARLA...")
        client, world = connect_to_carla()
        
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        world.apply_settings(settings)
        
        # Find suitable spawn points
        print("Finding suitable spawn points...")
        start_point, end_point = find_spawn_points(world)
        print(f"Start point: {start_point.location}")
        print(f"End point: {end_point.location}")
        
        # Spawn vehicle
        print("Spawning vehicle...")
        blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
        vehicle = None
        camera = None
        
        try:
            vehicle = world.spawn_actor(blueprint, start_point)
            print("Vehicle spawned successfully")
            
            # Set up camera
            camera = CameraManager(vehicle, world)
            
            # Allow everything to settle
            world.tick()
            time.sleep(0.5)
            
            # Create and setup navigation controller
            from navigation_controller import NavigationController
            controller = NavigationController()
            
            # Plan path
            print("Planning path...")
            success = controller.set_path(world, start_point.location, end_point.location)
            
            if not success:
                print("Failed to plan path!")
                return
            
            print(f"Path planned with {len(controller.waypoints)} waypoints")
            
            # Main simulation loop
            with tqdm(total=2000, desc="Navigation") as pbar:
                for i in range(2000):
                    try:
                        # Tick the world
                        world.tick()
                        
                        # Update Pygame display
                        display.fill((0, 0, 0))
                        camera.render(display)
                        pygame.display.flip()
                        
                        # Get and apply control
                        control = controller.get_control(vehicle, world)
                        vehicle.apply_control(control)
                        
                        # Update progress
                        speed = controller._get_speed(vehicle)
                        if controller.waypoints:
                            progress = (len(controller.visited_waypoints) / 
                                      len(controller.waypoints)) * 100
                        else:
                            progress = 0
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'speed': f"{speed:.1f}km/h",
                            'progress': f"{progress:.1f}%"
                        })
                        
                        # Check if destination reached
                        if (controller.current_waypoint_index >= 
                            len(controller.waypoints) - 1):
                            print("\nDestination reached!")
                            time.sleep(2)  # Show final state briefly
                            break
                        
                        # Handle pygame events
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                return
                            elif event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_ESCAPE:
                                    return
                        
                        clock.tick(FPS)
                        
                    except KeyboardInterrupt:
                        print("\nNavigation interrupted by user")
                        break
                    except Exception as e:
                        print(f"\nError during navigation step {i}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        break
                        
        finally:
            print("Cleaning up...")
            if camera is not None:
                camera.destroy()
            if vehicle is not None:
                vehicle.destroy()
            print("Cleanup complete")
            
            # Restore original settings
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        pygame.quit()
        print("Pygame quit")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user.')
    except Exception as e:
        print(f'Error occurred: {e}')
        import traceback
        traceback.print_exc()