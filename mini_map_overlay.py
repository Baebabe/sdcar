import sys
import os
import glob
import argparse
import logging
import math
import random
import numpy as np
import cv2
import pygame

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass
  
import carla
from carla import ColorConverter as cc

# Import CARLA Agents
from agents.navigation.basic_agent import BasicAgent
from agents.navigation.roaming_agent import RoamingAgent

class CarlaSimulation:
    def __init__(self, host='localhost', port=2000, width=1280, height=720, filter_pattern='vehicle.*'):
        # CARLA Connection
        self.client = self._connect_to_carla(host, port)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        # Simulation Parameters
        self.width = width
        self.height = height
        self.filter_pattern = filter_pattern

        # Game Objects
        self.player = None
        self.agent = None
        self.mini_map_camera = None
        self.main_camera = None

        # Images
        self.mini_map_image = None
        self.main_camera_image = None

        # Pygame Setup
        pygame.init()
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Simulation with Mini-Map")
        self.clock = pygame.time.Clock()

    def _connect_to_carla(self, host, port):
        try:
            client = carla.Client(host, port)
            client.set_timeout(5.0)
            print("[INFO] Connected to CARLA server.")
            return client
        except Exception as e:
            print(f"[ERROR] Could not connect to CARLA: {e}")
            sys.exit(1)

    def spawn_player(self):
        # Spawn player vehicle
        blueprint = random.choice(self.world.get_blueprint_library().filter(self.filter_pattern))
        blueprint.set_attribute('role_name', 'hero')
        spawn_points = self.map.get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        
        self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        if not self.player:
            raise RuntimeError("Failed to spawn player vehicle")

        # Setup agent
        self.agent = BasicAgent(self.player)
        self.agent.set_destination(spawn_point.location)

    def setup_cameras(self):
        # Mini-map camera (top-down view)
        mini_map_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        mini_map_bp.set_attribute('image_size_x', '256')
        mini_map_bp.set_attribute('image_size_y', '256')
        mini_map_bp.set_attribute('fov', '90')
        
        map_location = carla.Location(x=self.player.get_location().x, 
                                      y=self.player.get_location().y, 
                                      z=50)
        mini_map_transform = carla.Transform(map_location, carla.Rotation(pitch=-90))
        self.mini_map_camera = self.world.spawn_actor(mini_map_bp, mini_map_transform, attach_to=self.player)

        # Main camera
        main_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        main_camera_bp.set_attribute('image_size_x', str(self.width))
        main_camera_bp.set_attribute('image_size_y', str(self.height))
        
        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.main_camera = self.world.spawn_actor(main_camera_bp, camera_transform, attach_to=self.player)

    def process_images(self):
        def mini_map_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            self.mini_map_image = array[:, :, :3]

        def main_camera_callback(image):
            image.convert(cc.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            self.main_camera_image = array[:, :, :3]

        self.mini_map_camera.listen(mini_map_callback)
        self.main_camera.listen(main_camera_callback)

    def render(self):
        if self.main_camera_image is not None:
            # Convert main camera image to pygame surface
            main_surface = pygame.surfarray.make_surface(self.main_camera_image.swapaxes(0, 1))
            self.display.blit(main_surface, (0, 0))

        if self.mini_map_image is not None:
            # Convert mini map to pygame surface and render in top-right corner
            mini_map_surface = pygame.surfarray.make_surface(self.mini_map_image.swapaxes(0, 1))
            mini_map_surface = pygame.transform.scale(mini_map_surface, (256, 256))
            self.display.blit(mini_map_surface, (self.width - 266, 10))

        pygame.display.flip()

    def run(self):
        try:
            self.spawn_player()
            self.setup_cameras()
            self.process_images()

            while True:
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return

                # Agent control
                control = self.agent.run_step()
                self.player.apply_control(control)

                # Render
                self.render()
                self.clock.tick(60)

        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            # Cleanup
            if self.player:
                self.player.destroy()
            if self.mini_map_camera:
                self.mini_map_camera.destroy()
            if self.main_camera:
                self.main_camera.destroy()
            pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='CARLA Mini-Map Simulation')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--width', type=int, default=1280, help='Display width')
    parser.add_argument('--height', type=int, default=720, help='Display height')
    parser.add_argument('--filter', default='vehicle.*', help='Vehicle filter pattern')
    
    args = parser.parse_args()

    simulation = CarlaSimulation(
        host=args.host, 
        port=args.port, 
        width=args.width, 
        height=args.height, 
        filter_pattern=args.filter
    )
    simulation.run()

if __name__ == '__main__':
    main()