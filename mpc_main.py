import glob
import os
import sys
import random
import time
import numpy as np
import pygame
import carla
from mpc import MPCController

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

class CarlaEnv:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        # Set up display
        self.display_width = 1280
        self.display_height = 720
        self.display = pygame.display.set_mode(
            (self.display_width, self.display_height),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption('CARLA MPC Controller')
        
        # Initialize CARLA client
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        
        # Get world and map
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)
        
        # Initialize important variables
        self.vehicle = None
        self.camera = None
        self.camera_surface = None
        self.controller = None
        self.clock = pygame.time.Clock()
        
        # Camera setup parameters
        self.camera_transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8),
            carla.Rotation(pitch=-15)
        )
        
    def setup(self):
        """Setup vehicle, sensors and controller"""
        try:
            # Clear existing actors
            self.clear_actors()
            
            # Get random spawn point
            spawn_points = self.map.get_spawn_points()
            start_point = random.choice(spawn_points)
            
            # Spawn vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            self.vehicle = self.world.spawn_actor(vehicle_bp, start_point)
            
            # Setup camera
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(self.display_width))
            camera_bp.set_attribute('image_size_y', str(self.display_height))
            camera_bp.set_attribute('fov', '110')
            
            self.camera = self.world.spawn_actor(
                camera_bp,
                self.camera_transform,
                attach_to=self.vehicle
            )
            self.camera.listen(self.process_image)
            
            # Initialize MPC controller
            self.controller = MPCController()
            
            # Get random destination point
            end_point = random.choice(spawn_points)
            while end_point.location.distance(start_point.location) < 100:
                end_point = random.choice(spawn_points)
            
            # Set path using A*
            success = self.controller.set_path(
                self.world,
                start_point.location,
                end_point.location
            )
            
            if not success:
                print("Failed to find path!")
                return False
            
            return True
            
        except Exception as e:
            print(f"Setup failed: {str(e)}")
            return False
    
    def process_image(self, image):
        """Process camera image"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    def clear_actors(self):
        """Destroy all actors"""
        if self.camera is not None:
            self.camera.destroy()
            self.camera = None
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
    
    def render(self):
        """Render the Pygame display"""
        if self.camera_surface is not None:
            self.display.blit(self.camera_surface, (0, 0))
        
        # Render additional info
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        
        # Display vehicle info
        if self.vehicle:
            vel = self.vehicle.get_velocity()
            speed = 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)  # km/h
            
            speed_text = font.render(
                f'Speed: {speed:.1f} km/h', 
                True, 
                (255, 255, 255)
            )
            self.display.blit(speed_text, (10, 10))
        
        pygame.display.flip()
    
    def run(self):
        """Main game loop"""
        try:
            running = True
            while running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_r:
                            # Reset simulation
                            self.setup()
                
                # Get control input from MPC
                if self.vehicle and self.controller:
                    control = self.controller.get_control(self.vehicle, self.world)
                    self.vehicle.apply_control(control)
                
                # Tick world
                self.world.tick()
                
                # Render
                self.render()
                
                # Maintain fps
                self.clock.tick(20)
        
        finally:
            # Cleanup
            self.clear_actors()
            pygame.quit()
            
            # Disable synchronous mode
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

def main():
    """Main function"""
    try:
        env = CarlaEnv()
        if env.setup():
            env.run()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()

