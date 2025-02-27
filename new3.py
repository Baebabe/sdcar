import os
import sys
import time
import numpy as np
from tqdm import tqdm
import pygame
import carla
import random
import math
from sklearn.cluster import DBSCAN

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
        if (distance < 50.0 and distance > 30.0 and  # Between 30m and 50m
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

class VehicleDetector:
    def __init__(self, parent_actor, world, controller, detection_distance=40.0, detection_angle=160.0):
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.debug = True
        self.detection_distance = detection_distance 
        self.detection_angle = detection_angle  
        self.min_distance = float('inf')
        
        # Multiple sensor configuration
        self.sensors = []
        self._setup_sensors()
        
        # Visualization parameters
        self.width = 375
        self.height = 375
        self.scale = (self.height * 0.7) / self.detection_distance
        self.surface = None
        
        # Detection history for stability
        self.detection_history = []
        self.history_length = 5  # Keep track of last 5 frames
        
        print(f"Improved Vehicle Detector initialized with {detection_distance}m range and {detection_angle}° angle")
    
    def _setup_sensors(self):
        """Setup multiple sensors for redundant detection"""
        try:
            # Main forward-facing radar
            radar_bp = self._world.get_blueprint_library().find('sensor.other.radar')
            radar_bp.set_attribute('horizontal_fov', str(self.detection_angle))
            radar_bp.set_attribute('vertical_fov', '20')
            radar_bp.set_attribute('range', str(self.detection_distance))
            radar_location = carla.Transform(carla.Location(x=2.0, z=1.0))
            radar = self._world.spawn_actor(radar_bp, radar_location, attach_to=self._parent)
            radar.listen(lambda data: self._radar_callback(data))
            self.sensors.append(radar)
            
            # Semantic LIDAR for additional detection
            # lidar_bp = self._world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp = self._world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
            lidar_bp.set_attribute('range', str(self.detection_distance))
            lidar_bp.set_attribute('rotation_frequency', '20')
            lidar_bp.set_attribute('channels', '32')
            lidar_bp.set_attribute('points_per_second', '90000')
            lidar_location = carla.Transform(carla.Location(x=0.0, z=2.0))
            lidar = self._world.spawn_actor(lidar_bp, lidar_location, attach_to=self._parent)
            lidar.listen(lambda data: self._lidar_callback(data))
            self.sensors.append(lidar)
            
            print("Successfully set up multiple detection sensors")
            
        except Exception as e:
            print(f"Error setting up sensors: {e}")
    
    def _radar_callback(self, radar_data):
        """Process radar data for vehicle detection"""
        try:
            points = []
            for detection in radar_data:
                # Get the detection coordinates in world space
                distance = detection.depth
                if distance < self.detection_distance:
                    azimuth = math.degrees(detection.azimuth)
                    if abs(azimuth) < self.detection_angle / 2:
                        points.append({
                            'distance': distance,
                            'azimuth': azimuth,
                            'altitude': math.degrees(detection.altitude),
                            'velocity': detection.velocity
                        })
            
            # Store radar detections
            self.radar_points = points
            
        except Exception as e:
            print(f"Error in radar callback: {e}")
    
    def _lidar_callback(self, lidar_data):
        """Process semantic LIDAR data for vehicle detection"""
        try:
            # Filter points that belong to vehicles (semantic tag 10 in CARLA)
            vehicle_points = [point for point in lidar_data 
                            if point.object_tag == 10 
                            and point.distance < self.detection_distance]
            
            # Store LIDAR detections
            self.lidar_points = vehicle_points
            
        except Exception as e:
            print(f"Error in LIDAR callback: {e}")
    
    def detect_vehicles(self):
        
        try:
            detected_vehicles = set()
            
            # 1. Direct object detection from world
            all_vehicles = self._world.get_actors().filter('vehicle.*')
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            ego_right = self._calculate_right_vector(ego_transform.rotation)
            
            for vehicle in all_vehicles:
                # Skip self
                if vehicle.id == self._parent.id:
                    continue
                    
                vehicle_location = vehicle.get_location()
                distance = ego_location.distance(vehicle_location)
                
                if distance <= self.detection_distance:
                    # Calculate relative position and angle
                    relative_loc = vehicle_location - ego_location
                    forward_dot = (relative_loc.x * ego_forward.x + 
                                 relative_loc.y * ego_forward.y)
                    right_dot = (relative_loc.x * ego_right.x + 
                               relative_loc.y * ego_right.y)
                    
                    angle = math.degrees(math.atan2(right_dot, forward_dot))
                    
                    # Check if vehicle is within detection angle
                    if abs(angle) < self.detection_angle / 2:
                        detected_vehicles.add((vehicle, distance))
                        
                        if self.debug:
                            self._draw_debug_vehicle(vehicle, distance)
            
            # 2. Sensor fusion - combine with radar detection
            if hasattr(self, 'radar_points'):
                for point in self.radar_points:
                    # Filter out static objects
                    if point['velocity'] > 0.5:
                        self._add_potential_vehicle_location(point, detected_vehicles)
            
            # 3. Sensor fusion - combine with lidar detection
            if hasattr(self, 'lidar_points'):
                for point in self.lidar_points:
                    self._add_potential_vehicle_location(point, detected_vehicles)
            
            # 4. Update detection history
            self.detection_history.append(detected_vehicles)
            if len(self.detection_history) > self.history_length:
                self.detection_history.pop(0)
            
            # 5. Update minimum distance tracking
            self.min_distance = (min((d for _, d in detected_vehicles))
                               if detected_vehicles else float('inf'))
            
            # 6. Get stable detections and update controller
            stable_detections = self._get_stable_detections()
            self._controller.update_obstacles(
                [v.get_location() for v, _ in stable_detections]
            )
            
            # 7. Update visualization if needed
            self._create_visualization(stable_detections)
            
            return stable_detections
            
        except Exception as e:
            print(f"Error in vehicle detection: {e}")
            import traceback
            traceback.print_exc()
            return set()
    
    def _add_potential_vehicle_location(self, point, detected_vehicles):
        """Add potential vehicle location from sensor data"""
        try:
            # Convert sensor point to world location
            if hasattr(point, 'distance'):  # Radar point
                distance = point['distance']
                azimuth = math.radians(point['azimuth'])
                altitude = math.radians(point['altitude'])
                
                x = distance * math.cos(azimuth) * math.cos(altitude)
                y = distance * math.sin(azimuth) * math.cos(altitude)
                z = distance * math.sin(altitude)
                
            else:  # LIDAR point
                x, y, z = point.point
                distance = point.distance
            
            location = carla.Location(x=x, y=y, z=z)
            
            # Check if there's already a vehicle detected at this location
            for vehicle, _ in detected_vehicles:
                if vehicle.get_location().distance(location) < 2.0:  # 2m threshold
                    return
            
            # Add as potential detection
            detected_vehicles.add((None, distance))
            
        except Exception as e:
            print(f"Error adding potential vehicle: {e}")
    
    def _get_stable_detections(self):
        """Get detections that have been stable across multiple frames"""
        if not self.detection_history:
            return set()
        
        # Count how many times each vehicle appears in history
        vehicle_counts = {}
        for detections in self.detection_history:
            for vehicle, distance in detections:
                if vehicle:  # Only count actual vehicles, not sensor-only detections
                    vehicle_id = vehicle.id
                    vehicle_counts[vehicle_id] = vehicle_counts.get(vehicle_id, 0) + 1
        
        # Return vehicles that appear in majority of frames
        threshold = self.history_length * 0.6  # 60% of frames
        stable_vehicles = {v_id for v_id, count in vehicle_counts.items() 
                         if count >= threshold}
        
        # Get the most recent detection for stable vehicles
        latest_detections = self.detection_history[-1]
        return {(v, d) for v, d in latest_detections 
                if v and v.id in stable_vehicles}
    
    def _calculate_right_vector(self, rotation):
        """Calculate right vector from rotation"""
        yaw = math.radians(rotation.yaw)
        return carla.Vector3D(x=math.cos(yaw + math.pi/2),
                            y=math.sin(yaw + math.pi/2),
                            z=0.0)
    
    def _draw_debug_vehicle(self, vehicle, distance):
        """Draw enhanced debug visualization for detected vehicle"""
        try:
            if self.debug:
                # Draw bounding box
                vehicle_bbox = vehicle.bounding_box
                vehicle_transform = vehicle.get_transform()
                
                # Color based on distance
                if distance < 10:
                    color = carla.Color(255, 0, 0)  # Red for very close
                elif distance < 20:
                    color = carla.Color(255, 165, 0)  # Orange for intermediate
                else:
                    color = carla.Color(0, 255, 0)  # Green for far
                
                self._world.debug.draw_box(
                    box=vehicle_bbox,
                    rotation=vehicle_transform.rotation,
                    thickness=0.5,
                    color=color,
                    life_time=0.1
                )
                
                # Draw distance and velocity info
                velocity = vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
                
                info_text = f"{distance:.1f}m, {speed:.1f}km/h"
                self._world.debug.draw_string(
                    vehicle.get_location() + carla.Location(z=2.0),
                    info_text,
                    color=color,
                    life_time=0.1
                )
                
        except Exception as e:
            print(f"Error in debug drawing: {e}")
    
    def _get_relative_velocity(self, other_vehicle):
        """Calculate relative velocity between ego vehicle and other vehicle"""
        try:
            ego_vel = self._parent.get_velocity()
            other_vel = other_vehicle.get_velocity()

            # Convert to numpy arrays for easier calculation
            ego_vel_array = np.array([ego_vel.x, ego_vel.y])
            other_vel_array = np.array([other_vel.x, other_vel.y])

            # Calculate relative velocity
            rel_vel = np.linalg.norm(ego_vel_array - other_vel_array)

            # Determine if approaching
            ego_fwd = self._parent.get_transform().get_forward_vector()
            ego_fwd_array = np.array([ego_fwd.x, ego_fwd.y])

            # If dot product is positive, we're approaching
            if np.dot(ego_vel_array - other_vel_array, ego_fwd_array) > 0:
                return rel_vel
            return -rel_vel

        except Exception as e:
            print(f"Error calculating relative velocity: {e}")
            return 0.0

    def _create_visualization(self, detected_vehicles):
        try:
            if self.surface is None:
                self.surface = pygame.Surface((self.width, self.height))
                self.font = pygame.font.Font(None, 24)
                self.small_font = pygame.font.Font(None, 20)

            # Fill background with a premium dark gradient
            gradient = pygame.Surface((self.width, self.height))
            for y in range(self.height):
                alpha = y / self.height
                color = (20, 22, 30, int(255 * (0.95 - alpha * 0.3)))
                pygame.draw.line(gradient, color, (0, y), (self.width, y))
            self.surface.blit(gradient, (0, 0))

            # Constants for visualization
            center_x = self.width // 2
            center_y = int(self.height * 0.8)

            # Draw sensor range indicators with premium styling
            ranges = [10, 20, 30, 40]
            for range_dist in ranges:
                radius = int(range_dist * self.scale)
                points = []
                for angle in range(0, 360, 2):
                    rad = math.radians(angle)
                    x = center_x + radius * math.sin(rad)
                    y = center_y - radius * math.cos(rad)
                    points.append((x, y))

                # Draw circles with fade effect
                alpha = 100 - (range_dist / 40) * 60
                circle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                pygame.draw.lines(circle_surface, (0, 255, 255, int(alpha)), False, points, 1)
                self.surface.blit(circle_surface, (0, 0))

                # Premium distance labels
                label = self.small_font.render(f"{range_dist}m", True, (0, 255, 255, int(alpha)))
                self.surface.blit(label, (center_x + radius + 5, center_y - 10))

            def draw_realistic_car(surface, x, y, width, height, color, is_ego=False):
                """Draw a more realistic car shape with corrected dimensions"""
                # Scale down size for more realistic proportions
                width = int(width * 0.8)
                height = int(height * 0.8)

                # Create a surface for the car with alpha channel
                car_surface = pygame.Surface((width * 2, height * 2), pygame.SRCALPHA)
                center_x = width
                center_y = height

                # Main body shape points (more curved)
                body_points = [
                    (center_x - width//2, center_y),                    # Left bottom
                    (center_x - width//2, center_y - height*0.7),      # Left middle
                    (center_x - width//2.2, center_y - height*0.8),    # Left front curve
                    (center_x - width//3, center_y - height*0.85),     # Left hood
                    (center_x, center_y - height*0.9),                 # Front center
                    (center_x + width//3, center_y - height*0.85),     # Right hood
                    (center_x + width//2.2, center_y - height*0.8),    # Right front curve
                    (center_x + width//2, center_y - height*0.7),      # Right middle
                    (center_x + width//2, center_y),                   # Right bottom
                ]

                # Draw car body
                pygame.draw.polygon(car_surface, color, body_points)

                # Windshield (more curved)
                windshield_points = [
                    (center_x - width//3, center_y - height*0.85),     # Left bottom
                    (center_x - width//3.5, center_y - height*0.95),   # Left top
                    (center_x + width//3.5, center_y - height*0.95),   # Right top
                    (center_x + width//3, center_y - height*0.85),     # Right bottom
                ]
                pygame.draw.polygon(car_surface, (*color[:3], 150), windshield_points)

                # Wheels (more elliptical)
                wheel_width = width // 6
                wheel_height = height // 8

                # Front wheels (black ellipses)
                pygame.draw.ellipse(car_surface, (30, 30, 30),
                                  (center_x - width//2 - wheel_width//4,
                                   center_y - wheel_height//2,
                                   wheel_width, wheel_height))
                pygame.draw.ellipse(car_surface, (30, 30, 30),
                                  (center_x + width//2 - wheel_width*3//4,
                                   center_y - wheel_height//2,
                                   wheel_width, wheel_height))

                # Wheel well highlights (curved lines instead of arcs)
                # Left wheel well
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x - width//2 - wheel_width//4, center_y - wheel_height),
                               (center_x - width//2 + wheel_width, center_y - wheel_height), 2)

                # Right wheel well
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x + width//2 - wheel_width*3//4, center_y - wheel_height),
                               (center_x + width//2 + wheel_width//4, center_y - wheel_height), 2)

                # Add detail lines for body contour
                pygame.draw.line(car_surface, (*color[:3], 200),
                               (center_x - width//3, center_y - height*0.75),
                               (center_x + width//3, center_y - height*0.75), 1)

                # Add headlights or taillights
                light_width = width // 8
                light_height = height // 12

                if is_ego:
                    # Headlights (bright yellow)
                    light_color = (255, 255, 200, 200)
                else:
                    # Taillights (red)
                    light_color = (255, 50, 50, 200)

                # Left light
                pygame.draw.ellipse(car_surface, light_color,
                                  (center_x - width//2.5,
                                   center_y - height*0.85,
                                   light_width, light_height))

                # Right light
                pygame.draw.ellipse(car_surface, light_color,
                                  (center_x + width//2.5 - light_width,
                                   center_y - height*0.85,
                                   light_width, light_height))

                # Blit the car onto the main surface
                surface.blit(car_surface, (x - width, y - height))

            # Draw ego vehicle (smaller size)
            draw_realistic_car(self.surface, center_x, center_y, 24, 40, (0, 255, 255), True)

            # Draw detection cone with gradient
            fov_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            fov_radius = self.detection_distance * self.scale

            # Create detection cone
            fov_points = [(center_x, center_y)]
            for angle in range(-int(self.detection_angle/2), int(self.detection_angle/2)):
                rad = math.radians(angle)
                x = center_x + fov_radius * math.sin(rad)
                y = center_y - fov_radius * math.cos(rad)
                fov_points.append((x, y))
            fov_points.append((center_x, center_y))

            # Draw cone with gradient
            pygame.draw.polygon(fov_surface, (0, 255, 255, 20), fov_points)
            self.surface.blit(fov_surface, (0, 0))

            # Draw detected vehicles
            vehicle_colors = {
                'alert': (255, 50, 50),
                'warning': (255, 165, 0),
                'safe': (0, 255, 255)
            }

            detected_info = []

            # First draw the connection lines
            line_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

            for vehicle, distance in detected_vehicles:
                if vehicle is None or distance > 40:  # Only process vehicles within 40m radius
                    continue

                try:
                    ego_transform = self._parent.get_transform()
                    ego_forward = ego_transform.get_forward_vector()
                    ego_right = self._calculate_right_vector(ego_transform.rotation)

                    vehicle_location = vehicle.get_location()
                    ego_location = self._parent.get_location()

                    rel_loc = vehicle_location - ego_location
                    forward_dot = (rel_loc.x * ego_forward.x + rel_loc.y * ego_forward.y)
                    right_dot = (rel_loc.x * ego_right.x + rel_loc.y * ego_right.y)

                    screen_x = center_x + right_dot * self.scale
                    screen_y = center_y - forward_dot * self.scale

                    # Determine color based on distance
                    if distance < 10:
                        color = vehicle_colors['alert']
                        status = 'ALERT'
                    elif distance < 20:
                        color = vehicle_colors['warning']
                        status = 'WARNING'
                    else:
                        color = vehicle_colors['safe']
                        status = 'TRACKED'

                    # Draw connection line with gradient effect
                    for i in range(3):
                        alpha = 150 - (i * 40)
                        pygame.draw.line(line_surface, (*color[:3], alpha),
                                       (center_x, center_y),
                                       (screen_x, screen_y), 1 + i)

                    # Calculate relative velocity
                    rel_velocity = self._get_relative_velocity(vehicle)

                    detected_info.append({
                        'distance': distance,
                        'velocity': rel_velocity,
                        'status': status,
                        'color': color,
                        'position': (screen_x, screen_y)
                    })

                except Exception as e:
                    print(f"Error processing vehicle: {e}")

            # Blit all connection lines
            self.surface.blit(line_surface, (0, 0))

            # Then draw all vehicles
            for info in detected_info:
                draw_realistic_car(self.surface, 
                                 int(info['position'][0]), 
                                 int(info['position'][1]), 
                                 20, 32,  # Smaller size for other vehicles
                                 info['color'])

            # Draw premium status panel
            panel_height = 130
            panel_surface = pygame.Surface((self.width, panel_height), pygame.SRCALPHA)

            # Panel background with gradient
            for y in range(panel_height):
                alpha = 200 - (y / panel_height) * 100
                pygame.draw.line(panel_surface, (20, 22, 30, int(alpha)),
                               (0, y), (self.width, y))

            # Status header
            header = self.font.render("Vehicle Detection Status", True, (0, 255, 255))
            panel_surface.blit(header, (15, 10))

            # Detection information
            y_offset = 40
            for i, info in enumerate(sorted(detected_info, key=lambda x: x['distance'])):
                if i < 4:
                    status_text = f"{info['status']}: {info['distance']:.1f}m"
                    if abs(info['velocity']) > 0.1:
                        status_text += f" | Speed: {info['velocity']:.1f} m/s"

                    pygame.draw.circle(panel_surface, info['color'], (20, y_offset + 7), 4)

                    shadow = self.small_font.render(status_text, True, (0, 0, 0))
                    text = self.small_font.render(status_text, True, info['color'])
                    panel_surface.blit(shadow, (32, y_offset + 1))
                    panel_surface.blit(text, (30, y_offset))
                    y_offset += 22

            self.surface.blit(panel_surface, (0, 0))

            # Footer with system status
            timestamp = time.strftime("%H:%M:%S", time.localtime())
            stats_text = f"System Active | Time: {timestamp} | Vehicles Detected: {len(detected_info)}"
            footer_surface = pygame.Surface((self.width, 30), pygame.SRCALPHA)
            pygame.draw.rect(footer_surface, (20, 22, 30, 200), (0, 0, self.width, 30))
            stats = self.small_font.render(stats_text, True, (0, 255, 255))
            footer_surface.blit(stats, (15, 8))
            self.surface.blit(footer_surface, (0, self.height - 30))

        except Exception as e:
            print(f"Error in visualization: {e}")
            import traceback
            traceback.print_exc()


    def render(self, display):
        """Render the detection visualization"""
        if self.surface is not None:
            display.blit(self.surface, (display.get_width() - self.width - 20, 20))
    
    def destroy(self):
        """Clean up sensors and resources"""
        try:
            for sensor in self.sensors:
                if sensor is not None:
                    sensor.destroy()
            self.sensors.clear()
        except Exception as e:
            print(f"Error destroying sensors: {e}")


class SafetyController:
    def __init__(self, parent_actor, world, controller):
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.debug = True
        
        # Increased detection parameters
        self.detection_distance = 50.0  # Increased from 40m to 50m
        self.lane_width = 3.5
        self.last_detected = False

        self.recovery_mode = False
        self.recovery_start_time = None
        self.max_recovery_time = 3.0  # seconds
        self.min_speed_threshold = 0.5  # m/s
        self.last_brake_time = None
        self.brake_cooldown = 1.0  # seconds
        
        # New braking parameters
        self.time_to_collision_threshold = 3.0  # seconds
        self.min_safe_distance = 10.0  # meters
        self.emergency_brake_distance = 5.0  # meters
        
        # Create collision sensor
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        print("Enhanced Safety controller initialized")

    def _on_collision(self, event):
        """Collision event handler"""
        print("!!! COLLISION DETECTED !!!")
        self._emergency_stop()

    def _calculate_time_to_collision(self, ego_velocity, other_velocity, distance):
        """Calculate time to collision based on relative velocity"""
        # Get velocity magnitudes
        ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        other_speed = math.sqrt(other_velocity.x**2 + other_velocity.y**2)
        
        # If both vehicles are stationary or moving at same speed
        if abs(ego_speed - other_speed) < 0.1:
            return float('inf') if distance > self.emergency_brake_distance else 0
            
        # Calculate time to collision
        relative_speed = ego_speed - other_speed
        if relative_speed > 0:  # Only if we're moving faster than the other vehicle
            return distance / relative_speed
        return float('inf')


    def _is_in_same_lane(self, ego_location, ego_forward, other_location):
        """
        More accurate lane detection that handles curves by checking waypoints and road curvature
        """
        try:
            # Get the current waypoint of ego vehicle
            ego_waypoint = self._world.get_map().get_waypoint(ego_location)
            other_waypoint = self._world.get_map().get_waypoint(other_location)

            if not ego_waypoint or not other_waypoint:
                return True, 0  # Conservative approach if waypoints can't be found

            # Check if vehicles are on the same road and in the same lane
            same_road = (ego_waypoint.road_id == other_waypoint.road_id)
            same_lane = (ego_waypoint.lane_id == other_waypoint.lane_id)

            # Vector to other vehicle
            to_other = other_location - ego_location

            # Project onto forward direction for distance calculation
            forward_dist = (to_other.x * ego_forward.x + to_other.y * ego_forward.y)

            if forward_dist <= 0:  # Vehicle is behind
                return False, forward_dist

            # If they're on different roads/lanes, no need for further checks
            if not (same_road and same_lane):
                return False, forward_dist

            # For vehicles on the same lane, check if they're within reasonable distance
            # Get a series of waypoints ahead to account for road curvature
            next_waypoints = ego_waypoint.next(forward_dist)
            if not next_waypoints:
                return True, forward_dist  # Conservative approach if can't get waypoints

            # Find the closest waypoint to the other vehicle
            min_dist = float('inf')
            closest_wp = None

            for wp in next_waypoints:
                dist = wp.transform.location.distance(other_location)
                if dist < min_dist:
                    min_dist = dist
                    closest_wp = wp

            # If no waypoint found within reasonable distance, vehicles aren't in same lane
            if not closest_wp:
                return False, forward_dist

            # Check lateral distance from the predicted path
            # Use a more generous threshold in curves
            road_curvature = abs(ego_waypoint.transform.rotation.yaw - closest_wp.transform.rotation.yaw)
            lateral_threshold = self.lane_width * (0.5 + (road_curvature / 90.0) * 0.3)  # Increase threshold in curves

            is_in_lane = min_dist < lateral_threshold

            if self.debug and forward_dist > 0:
                color = carla.Color(0, 255, 0) if is_in_lane else carla.Color(128, 128, 128)
                self._world.debug.draw_line(
                    ego_location,
                    other_location,
                    thickness=0.1,
                    color=color,
                    life_time=0.1
                )
                # Draw predicted path for debugging
                for wp in next_waypoints:
                    self._world.debug.draw_point(
                        wp.transform.location,
                        size=0.1,
                        color=carla.Color(0, 0, 255),
                        life_time=0.1
                    )

            return is_in_lane, forward_dist

        except Exception as e:
            print(f"Lane check error: {str(e)}")
            return True, 0  # Conservative approach - assume in lane if error
    
    def check_safety(self):
        """Enhanced safety checking with predictive braking"""
        try:
            all_vehicles = self._world.get_actors().filter('vehicle.*')
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            ego_velocity = self._parent.get_velocity()
            
            detected_vehicles = []
            min_distance = float('inf')
            min_ttc = float('inf')  # Time to collision
            
            # Current speed in km/h
            speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
            
            for vehicle in all_vehicles:
                if vehicle.id == self._parent.id:
                    continue
                    
                vehicle_location = vehicle.get_location()
                distance = ego_location.distance(vehicle_location)
                
                # Check vehicles within extended detection range
                if distance < self.detection_distance:
                    is_in_lane, forward_dist = self._is_in_same_lane(ego_location, ego_forward, vehicle_location)
                    
                    if is_in_lane:
                        other_velocity = vehicle.get_velocity()
                        ttc = self._calculate_time_to_collision(ego_velocity, other_velocity, forward_dist)
                        
                        detected_vehicles.append((vehicle, distance, ttc))
                        min_distance = min(min_distance, distance)
                        min_ttc = min(min_ttc, ttc)
                        
                        if self.debug:
                            self._world.debug.draw_box(
                                vehicle.bounding_box,
                                vehicle.get_transform().rotation,
                                thickness=0.5,
                                color=carla.Color(255, 0, 0, 255),
                                life_time=0.1
                            )
                            self._world.debug.draw_string(
                                vehicle_location + carla.Location(z=2.0),
                                f'!!! {distance:.1f}m, TTC: {ttc:.1f}s !!!',
                                color=carla.Color(255, 0, 0, 255),
                                life_time=0.1
                            )
            
            if detected_vehicles:
                print(f"\nVehicle detected - Distance: {min_distance:.1f}m, TTC: {min_ttc:.1f}s, Speed: {speed:.1f} km/h")
                
                # Progressive braking based on TTC and distance
                if min_distance < self.emergency_brake_distance or min_ttc < 1.0:
                    print(f"!!! EMERGENCY STOP !!! Distance: {min_distance:.1f}m, TTC: {min_ttc:.1f}s")
                    self._emergency_stop()
                elif min_distance < self.min_safe_distance or min_ttc < 2.0:
                    print(f"!! HARD BRAKING !! Distance: {min_distance:.1f}m, TTC: {min_ttc:.1f}s")
                    self._hard_brake()
                elif min_ttc < self.time_to_collision_threshold:
                    print(f"! CAUTION BRAKING ! Distance: {min_distance:.1f}m, TTC: {min_ttc:.1f}s")
                    self._cautious_brake(min_ttc)
                
                self._maintain_path()
                self.last_detected = True
            else:
                if self.last_detected:
                    print("Path clear - resuming normal operation")
                    self._resume_path()
                self.last_detected = False
            
            # Update controller with obstacles
            self._controller.update_obstacles([v[0].get_location() for v in detected_vehicles])
            
        except Exception as e:
            print(f"Error in safety check: {str(e)}")
            import traceback
            traceback.print_exc()
            self._emergency_stop()

    def _emergency_stop(self):
        """Maximum braking force"""
        control = carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True,
            steer=0.0
        )
        self._parent.apply_control(control)

    def _hard_brake(self):
        """Strong braking with steering control"""
        control = carla.VehicleControl(
            throttle=0.0,
            brake=0.6,
            hand_brake=False
        )
        self._parent.apply_control(control)

    def _cautious_brake(self, ttc):
        """Progressive braking based on time to collision"""
        # Brake intensity increases as TTC decreases
        brake_intensity = min(0.4, 1.5 / ttc)
        control = carla.VehicleControl(
            throttle=0.0,
            brake=brake_intensity,
            hand_brake=False
        )
        self._parent.apply_control(control)

    def _maintain_path(self):
        """Enhanced path maintenance during braking"""
        try:
            if not self._controller:
                return
                
            current_time = time.time()
            
            # Check if we need to enter recovery mode
            if not self.recovery_mode:
                velocity = self._parent.get_velocity()
                speed = math.sqrt(velocity.x**2 + velocity.y**2)
                
                if speed < self.min_speed_threshold:
                    self.recovery_mode = True
                    self.recovery_start_time = current_time
                    print("Entering path recovery mode")
            
            # Handle recovery mode
            if self.recovery_mode:
                if current_time - self.recovery_start_time > self.max_recovery_time:
                    self.recovery_mode = False
                    print("Exiting recovery mode - timeout")
                else:
                    # Force path recovery
                    control = self._controller.force_path_recovery(self._parent)
                    self._parent.apply_control(control)
                    return
            
            # Normal path maintenance
            target_wp = self._controller.waypoints[self._controller.current_waypoint_index]
            vehicle_transform = self._parent.get_transform()
            steer = self._controller._calculate_steering(vehicle_transform, target_wp.transform)
            
            # Get current control and maintain brake while updating steering
            control = self._parent.get_control()
            control.steer = steer
            self._parent.apply_control(control)
            
        except Exception as e:
            print(f"Error in path maintenance: {str(e)}")

    def _resume_path(self):
        """Improved path resumption after braking"""
        try:
            current_time = time.time()
            
            # Check if we're still in brake cooldown
            if self.last_brake_time and current_time - self.last_brake_time < self.brake_cooldown:
                # Maintain current path but release brake gradually
                control = self._parent.get_control()
                control.brake *= 0.5  # Gradually release brake
                self._parent.apply_control(control)
                return
                
            # Reset recovery mode
            self.recovery_mode = False
            self.recovery_start_time = None
            
            # Reset controller state
            self._controller._reset_control_state()
            
            # Resume normal control
            control = self._controller.get_control(self._parent, self._world)
            self._parent.apply_control(control)
            
        except Exception as e:
            print(f"Error resuming path: {str(e)}")

    def destroy(self):
        """Clean up sensors"""
        if hasattr(self, 'collision_sensor'):
            self.collision_sensor.destroy()


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
        detector = None
        safety_controller = None
        
        try:
            vehicle = world.spawn_actor(blueprint, start_point)
            print("Vehicle spawned successfully")
            
            # Set up camera
            camera = CameraManager(vehicle, world)
            
            # Set up Vehicle Detector and Controllers
            from navigation_controller import NavigationController
            controller = NavigationController()
            detector = VehicleDetector(vehicle, world, controller)
            safety_controller = SafetyController(vehicle, world, controller)
            
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
            with tqdm(total=5000, desc="Navigation") as pbar:
                # Inside the main simulation loop
                try:
                    while True:
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
                            if detector is not None:
                                detector.detect_vehicles()
                                detector.render(display)
                            pygame.display.flip()

                            # Safety check before applying control
                            safety_controller.check_safety()

                            # Apply the updated safety control logic
                            if safety_controller.last_detected:
                                # Apply emergency brake
                                vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0, steer=0))
                                print("Safety stop active")
                            else:
                                # Resume normal path following
                                control = controller.get_control(vehicle, world)
                                vehicle.apply_control(control)
                                # Debug output for normal operation
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
                                'progress': f"{progress:.1f}%",
                                'safety': 'ACTIVE' if safety_controller.last_detected else 'OK'
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

                        except Exception as e:
                            print(f"\nError during simulation step: {str(e)}")
                            if "time-out" in str(e).lower():
                                print("Attempting to recover from timeout...")
                                time.sleep(1.0)  # Give the simulator time to recover
                                continue
                            else:
                                raise
                
                except KeyboardInterrupt:
                    print("\nNavigation interrupted by user")
                except Exception as e:
                    print(f"Unexpected error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                        
        finally:
            print("Cleaning up...")
            try:
                if detector is not None:
                    detector.destroy()
                if camera is not None:
                    camera.destroy()
                if safety_controller is not None:
                    safety_controller.destroy()
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

