
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import pygame
import carla
import random
import math
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

class LidarManager:
    def __init__(self, parent_actor, world, controller):
        self.sensor = None
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.points = []  # Store current points for visualization
        self.surface = None  # Pygame surface for visualization
        
        # Set up LiDAR blueprint with improved settings
        blueprint = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        blueprint.set_attribute('range', '50')
        blueprint.set_attribute('rotation_frequency', '20')  # Increased from 10
        blueprint.set_attribute('channels', '64')  # Increased from 32
        blueprint.set_attribute('points_per_second', '100000')  # Increased from 56000
        blueprint.set_attribute('upper_fov', '15.0')
        blueprint.set_attribute('lower_fov', '-25.0')
        
        # Find LiDAR spawn point (above the vehicle)
        spawn_point = carla.Transform(carla.Location(z=2.5))
        
        # Spawn LiDAR
        self.sensor = world.spawn_actor(blueprint, spawn_point, attach_to=self._parent)
        
        # Setup callback for LiDAR data
        self.sensor.listen(self._parse_lidar)
    
    def _parse_lidar(self, data):
        """Parse LiDAR data, update obstacles, and prepare visualization"""
        points = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        
        # Store points for visualization
        self.points = points
        
        # Update obstacles for controller
        obstacles = [carla.Location(x=float(point[0]), y=float(point[1]), z=float(point[2])) 
                    for point in points]
        self._controller.update_obstacles(obstacles)
        
        # Create visualization surface
        self._create_lidar_visualization()
    
    def _create_lidar_visualization(self):
        """Create a visualization of the LiDAR points"""
        if len(self.points) == 0:
            return
            
        # Create pygame surface
        WIDTH = 200  # Size of visualization window
        HEIGHT = 200
        SCALE = 4  # Scale factor for point visualization
        
        # Create surface if it doesn't exist
        if self.surface is None:
            self.surface = pygame.Surface((WIDTH, HEIGHT))
        
        self.surface.fill((0, 0, 0))  # Clear surface
        
        # Transform points to top-down view
        points = self.points.copy()
        
        # Calculate point colors based on height (z-value)
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        z_range = max(z_max - z_min, 1)  # Avoid division by zero
        
        # Draw points on surface
        for point in points:
            x, y, z = point
            
            # Transform to screen coordinates (top-down view)
            screen_x = int(WIDTH/2 + x * SCALE)
            screen_y = int(HEIGHT/2 - y * SCALE)
            
            if 0 <= screen_x < WIDTH and 0 <= screen_y < HEIGHT:
                # Color based on height (blue to red)
                z_normalized = (z - z_min) / z_range
                color = (int(255 * z_normalized), 0, int(255 * (1 - z_normalized)))
                
                # Draw point
                pygame.draw.circle(self.surface, color, (screen_x, screen_y), 1)
    
    def render(self, display):
        """Render LiDAR visualization to display"""
        if self.surface is not None:
            # Position the LiDAR visualization in the top-right corner
            display.blit(self.surface, (display.get_width() - self.surface.get_width() - 10, 10))
    
    def destroy(self):
        """Clean up LiDAR sensor"""
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

def spawn_strategic_npcs(world, player_vehicle, close_npcs=5, far_npcs=15):
    """
    Spawn NPCs with some specifically placed in front of the player vehicle
    """
    blueprints = world.get_blueprint_library().filter('vehicle.*')
    
    # Filter for cars (no bikes/motorcycles)
    car_blueprints = [
        bp for bp in blueprints 
        if any(car_type in bp.id.lower() 
               for car_type in ['car', 'tesla', 'audi', 'bmw', 'mercedes', 'toyota', 'ford'])
    ]
    
    if not car_blueprints:
        print("Warning: Using all vehicle blueprints as fallback")
        car_blueprints = blueprints
    
    spawn_points = world.get_map().get_spawn_points()
    
    # Get player's transform
    player_transform = player_vehicle.get_transform()
    player_location = player_transform.location
    player_forward = player_transform.get_forward_vector()
    
    vehicles = []
    
    # First, spawn NPCs close to the player
    close_spawn_points = []
    for spawn_point in spawn_points:
        # Calculate vector from player to spawn point
        to_spawn_x = spawn_point.location.x - player_location.x
        to_spawn_y = spawn_point.location.y - player_location.y
        to_spawn_z = spawn_point.location.z - player_location.z
        
        # Calculate distance using pythagorean theorem
        distance = math.sqrt(to_spawn_x**2 + to_spawn_y**2 + to_spawn_z**2)
        
        # Calculate dot product manually
        dot_product = (to_spawn_x * player_forward.x + 
                      to_spawn_y * player_forward.y + 
                      to_spawn_z * player_forward.z)
        
        # Check if point is in front of player and within distance range
        if (distance < 30.0 and distance > 10.0 and  # Between 10m and 30m
            dot_product > 0):                        # In front of player
            close_spawn_points.append(spawn_point)
    
    # Spawn close NPCs
    print(f"\nSpawning {close_npcs} NPCs near player...")
    random.shuffle(close_spawn_points)
    for i in range(min(close_npcs, len(close_spawn_points))):
        try:
            blueprint = random.choice(car_blueprints)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
            
            vehicle = world.spawn_actor(blueprint, close_spawn_points[i])
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            print(f"Spawned close NPC {i+1}/{close_npcs}")
            
            # Give time for physics to settle
            world.tick(2.0)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed to spawn close NPC: {e}")
            continue
    
    # Then spawn far NPCs randomly in the map
    print(f"\nSpawning {far_npcs} NPCs around the map...")
    random.shuffle(spawn_points)
    for i in range(far_npcs):
        try:
            blueprint = random.choice(car_blueprints)
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', random.choice(blueprint.get_attribute('color').recommended_values))
            
            # Try to find a spawn point not too close to player
            spawn_point = None
            for point in spawn_points:
                # Calculate distance manually
                dx = point.location.x - player_location.x
                dy = point.location.y - player_location.y
                dz = point.location.z - player_location.z
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if distance > 50.0:  # More than 50m away
                    spawn_point = point
                    spawn_points.remove(point)
                    break
            
            if spawn_point is None:
                print("Couldn't find suitable spawn point for far NPC")
                continue
                
            vehicle = world.spawn_actor(blueprint, spawn_point)
            vehicle.set_autopilot(True)
            vehicles.append(vehicle)
            print(f"Spawned far NPC {i+1}/{far_npcs}")
            
            # Give time for physics to settle
            world.tick(2.0)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Failed to spawn far NPC: {e}")
            continue
    
    print(f"\nSuccessfully spawned {len(vehicles)} NPCs total")
    return vehicles


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
        
        # Set synchronous mode with enhanced physics settings
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / FPS
        settings.substepping = True  # Enable physics substepping
        settings.max_substep_delta_time = 0.01  # Maximum physics substep size
        settings.max_substeps = 10  # Maximum number of substeps
        world.apply_settings(settings)
        
        # Wait for the world to stabilize
        print("Waiting for world to stabilize...")
        for _ in range(20):
            world.tick(2.0)
            time.sleep(0.1)
        
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
        lidar = None
        
        try:
            vehicle = world.spawn_actor(blueprint, start_point)
            print("Vehicle spawned successfully")
            
            # Set up camera
            camera = CameraManager(vehicle, world)
            
            # Set up LiDAR
            from navigation_controller import NavigationController
            controller = NavigationController()
            lidar = LidarManager(vehicle, world, controller)
            
            # Allow everything to settle
            world.tick()
            time.sleep(0.5)
            
            # Plan path
            print("Planning path...")
            success = controller.set_path(world, start_point.location, end_point.location)
            
            if not success:
                print("Failed to plan path!")
                return
            
            print(f"Path planned with {len(controller.waypoints)} waypoints")
            
            # Spawn NPC vehicles
            print("Spawning NPC vehicles...")
            npcs = spawn_strategic_npcs(world, vehicle, close_npcs=5, far_npcs=15)
            print(f"Spawned {len(npcs)} total NPCs")
            # Main simulation loop
            with tqdm(total=2000, desc="Navigation") as pbar:
                for i in range(2000):
                    try:
                        # Tick the world with correct parameter
                        start_time = time.time()
                        while True:
                            try:
                                world.tick(2.0)
                                break
                            except RuntimeError as e:
                                if time.time() - start_time > 10.0:  # Overall timeout
                                    raise
                                time.sleep(0.1)
                                continue
                        
                        # Update Pygame display
                        display.fill((0, 0, 0))
                        camera.render(display)
                        if lidar is not None:
                            lidar.render(display)
                        pygame.display.flip()
                        
                        # Get and apply control
                        control = controller.get_control(vehicle, world)
                        vehicle.apply_control(control)
                        
                        # Debug output
                        print(f"Control commands - Throttle: {control.throttle:.2f}, "
                              f"Brake: {control.brake:.2f}, "
                              f"Steer: {control.steer:.2f}")
                        
                        # Debug vehicle state
                        velocity = vehicle.get_velocity()
                        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
                        print(f"Vehicle speed: {speed:.2f} km/h")
                        
                        # Update progress
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
                            time.sleep(2)
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
                        if "time-out" in str(e).lower():
                            print("Attempting to recover from timeout...")
                            time.sleep(1.0)  # Give the simulator time to recover
                            continue
                        else:
                            print(f"Unexpected error: {str(e)}")
                            break
                        
        finally:
            print("Cleaning up...")
            try:
                if lidar is not None:
                    lidar.destroy()
                if camera is not None:
                    camera.destroy()
                if vehicle is not None:
                    vehicle.destroy()
                
                # Destroy NPC vehicles
                if 'npcs' in locals() and npcs:
                    for npc in npcs:
                        if npc is not None and npc.is_alive:
                            try:
                                npc.set_autopilot(False)  # Disable autopilot before destroying
                                npc.destroy()
                            except:
                                pass
                
                # Restore original settings
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                settings.substepping = False  # Disable physics substepping
                world.apply_settings(settings)
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
                
            print("Cleanup complete")
                
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