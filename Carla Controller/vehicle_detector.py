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