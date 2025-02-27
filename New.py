class VehicleDetector:
    def __init__(self, parent_actor, world, controller, detection_distance=30.0, detection_angle=120.0, mode='frontal'):
        """
        mode: 'frontal' detects vehicles only in front of the ego vehicle within detection_angle;
              'all' detects every vehicle within detection_distance.
        """
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.debug = True
        self.mode = mode
        self.detection_distance = detection_distance  # meters
        self.detection_angle = detection_angle        # degrees (used in 'frontal' mode)
        self.min_distance = float('inf')
        
        # Visualization parameters
        self.width = 320
        self.height = 320
        # Scale factor: adjust so that detection_distance fits nicely in the view.
        self.scale = (self.height * 0.6) / self.detection_distance  
        self.surface = None
        
        print(f"Improved Vehicle detection system initialized (mode: {self.mode})")

    def detect_vehicles(self):
        """Detect vehicles robustly and update the controller with obstacle locations."""
        try:
            # Retrieve all vehicles from the world
            all_vehicles = self._world.get_actors().filter('vehicle.*')
            ego_transform = self._parent.get_transform()
            ego_location = ego_transform.location
            ego_forward = ego_transform.get_forward_vector()
            # Compute ego vehicle heading (in radians)
            ego_heading = math.atan2(ego_forward.y, ego_forward.x)
            
            detected_vehicles = []
            self.min_distance = float('inf')
            
            for vehicle in all_vehicles:
                if vehicle.id == self._parent.id:
                    continue  # skip self

                vehicle_location = vehicle.get_location()
                distance = ego_location.distance(vehicle_location)
                if distance > self.detection_distance:
                    continue  # outside detection range

                # Compute relative vector from ego to vehicle
                rel_vec = vehicle_location - ego_location
                dx = rel_vec.x
                dy = rel_vec.y
                # Rotate coordinates so that ego's forward direction is along positive X axis.
                local_x = dx * math.cos(-ego_heading) - dy * math.sin(-ego_heading)
                local_y = dx * math.sin(-ego_heading) + dy * math.cos(-ego_heading)
                
                # In frontal mode, only accept vehicles in front (local_x > 0) and within the detection cone.
                if self.mode == 'frontal':
                    if local_x <= 0:
                        continue
                    angle_deg = math.degrees(math.atan2(abs(local_y), local_x))
                    if angle_deg > self.detection_angle / 2.0:
                        continue
                
                detected_vehicles.append({
                    'vehicle': vehicle,
                    'distance': distance,
                    'local_pos': (local_x, local_y)
                })
                self.min_distance = min(self.min_distance, distance)
                
                if self.debug:
                    # Draw a debug box around the detected vehicle
                    vehicle_bbox = vehicle.bounding_box
                    vehicle_transform = vehicle.get_transform()
                    self._world.debug.draw_box(
                        box=vehicle_bbox,
                        rotation=vehicle_transform.rotation,
                        thickness=0.5,
                        color=carla.Color(255, 0, 0, 255),
                        life_time=0.1
                    )
                    # Draw distance text above the vehicle
                    self._world.debug.draw_string(
                        vehicle_location + carla.Location(z=2.0),
                        f'{distance:.1f}m',
                        color=carla.Color(255, 255, 255, 255),
                        life_time=0.1
                    )
            
            # Update controller with the world locations of detected obstacles
            obstacle_locations = [d['vehicle'].get_location() for d in detected_vehicles]
            self._controller.update_obstacles(obstacle_locations)
            
            if self.debug and detected_vehicles:
                print(f"Detected {len(detected_vehicles)} vehicles. Closest: {self.min_distance:.1f}m")
            
            # Update the visualization HUD
            self._create_visualization(detected_vehicles)
            
        except Exception as e:
            print(f"Error in improved vehicle detection: {str(e)}")
            import traceback
            traceback.print_exc()

    def _create_visualization(self, detected_vehicles):
        """Create an enhanced HUD similar to self-driving car interfaces."""
        if self.surface is None:
            self.surface = pygame.Surface((self.width, self.height))
            self.font = pygame.font.Font(None, 20)
        
        # Fill background with a dark tone
        self.surface.fill((20, 20, 20))
        
        # Draw grid lines for reference
        grid_color = (50, 50, 50)
        for x in range(0, self.width, 20):
            pygame.draw.line(self.surface, grid_color, (x, 0), (x, self.height))
        for y in range(0, self.height, 20):
            pygame.draw.line(self.surface, grid_color, (0, y), (self.width, y))
        
        # Set the ego vehicle's position near the bottom center of the display.
        center_x = self.width // 2
        center_y = int(self.height * 0.8)
        
        # Draw range rings to indicate distance
        num_rings = 3
        for ring in range(1, num_rings + 1):
            ring_distance = (self.detection_distance / num_rings) * ring
            ring_radius = ring_distance * self.scale
            pygame.draw.circle(self.surface, (80, 80, 80), (center_x, center_y), int(ring_radius), 1)
            # Label each ring with its distance value.
            label = self.font.render(f"{ring_distance:.0f}m", True, (150, 150, 150))
            self.surface.blit(label, (center_x + int(ring_radius) - 30, center_y - 20))
        
        # Draw the ego vehicle as a triangle (pointing upward).
        ego_size = 10
        tip = (center_x, center_y - ego_size)
        bottom_left = (center_x - ego_size // 2, center_y + ego_size // 2)
        bottom_right = (center_x + ego_size // 2, center_y + ego_size // 2)
        pygame.draw.polygon(self.surface, (0, 255, 0), [tip, bottom_left, bottom_right])
        
        # Draw detected vehicles
        for d in detected_vehicles:
            local_x, local_y = d['local_pos']
            # In the ego coordinate system, local_x is forward distance and local_y is lateral offset.
            # Convert these into screen coordinates:
            screen_x = int(center_x + local_y * self.scale)
            screen_y = int(center_y - local_x * self.scale)
            
            # Only display markers that fall within the HUD.
            if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
                # Color coding: red if very close, orange for intermediate, white for farther away.
                if d['distance'] < 10:
                    color = (255, 0, 0)
                elif d['distance'] < 20:
                    color = (255, 165, 0)
                else:
                    color = (255, 255, 255)
                pygame.draw.circle(self.surface, color, (screen_x, screen_y), 6)
                # Display the distance next to the marker.
                text = self.font.render(f"{d['distance']:.1f}m", True, color)
                self.surface.blit(text, (screen_x + 8, screen_y - 8))
        
        # Optionally, draw the detection zone arc if in frontal mode.
        if self.mode == 'frontal':
            arc_radius = self.detection_distance * self.scale
            arc_rect = pygame.Rect(center_x - arc_radius, center_y - arc_radius, arc_radius * 2, arc_radius * 2)
            start_angle = math.radians(90 + self.detection_angle / 2)
            end_angle = math.radians(90 - self.detection_angle / 2)
            pygame.draw.arc(self.surface, (0, 255, 0), arc_rect, start_angle, end_angle, 2)

    def render(self, display):
        """Render the HUD on the given display surface."""
        if self.surface is not None:
            # Position the HUD in the top-right corner.
            display.blit(self.surface, (display.get_width() - self.width - 20, 20))
    
    def destroy(self):
        """Clean up any resources if needed."""
        pass






















# import carla
# import math
# import numpy as np
# from queue import PriorityQueue
# import time
# import random
# import pygame
# from tqdm import tqdm

# class NavigationController:
#     def __init__(self):
#         # Control parameters
#         self.max_steer = 0.7
#         self.target_speed = 30.0  # km/h

#         # Controller gains - increased for more responsive control
#         self.k_p_lateral = 1.5    # Increased from 0.9
#         self.k_p_heading = 1.0    # Increased from 0.5
#         self.k_p_speed = 1.0      # Increased from 0.5

#         # Path tracking
#         self.waypoints = []
#         self.visited_waypoints = set()
#         self.current_waypoint_index = 0
#         self.waypoint_distance_threshold = 2.0
        
#         # A* parameters
#         self.waypoint_distance = 2.0
#         self.max_search_dist = 200.0
        
#         # Visualization
#         self.debug_lifetime = 0.1
#         self.path_visualization_done = False

#         # Obstacle detection
#         self.obstacles = []

#     def _heuristic(self, waypoint, goal_waypoint):
#         """A* heuristic: straight-line distance to goal"""
#         return waypoint.transform.location.distance(goal_waypoint.transform.location)

#     def set_path(self, world, start_location, end_location):
#         """Generate shortest path using A* algorithm"""
#         try:
#             self.world = world
#             self.map = world.get_map()
            
#             # Convert locations to waypoints
#             start_waypoint = self.map.get_waypoint(start_location)
#             end_waypoint = self.map.get_waypoint(end_location)
            
#             print(f"Planning path using A* from {start_waypoint.transform.location} to {end_waypoint.transform.location}")
            
#             # Find path using A*
#             path, distance = self._find_path_astar(start_waypoint, end_waypoint)
            
#             if not path:
#                 print("No path found!")
#                 return False
            
#             self.waypoints = path
#             self.current_waypoint_index = 0
#             self.visited_waypoints.clear()
            
#             print(f"Path found with {len(path)} waypoints and distance {distance:.1f} meters")
            
#             # Visualize the path and exploration
#             self._visualize_complete_path(world, path, distance)
            
#             return True
            
#         except Exception as e:
#             print(f"Path planning failed: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             return False

#     def _find_path_astar(self, start_wp, end_wp):
#         """A* algorithm implementation for finding shortest path"""
#         # Priority queue for A* (f_score, counter, waypoint)
#         counter = 0  # Unique identifier for comparing waypoints
#         open_set = PriorityQueue()
#         start_f_score = self._heuristic(start_wp, end_wp)
#         open_set.put((start_f_score, counter, start_wp))
        
#         # For reconstructing path
#         came_from = {}
        
#         # g_score: cost from start to current node
#         g_score = {}
#         g_score[start_wp] = 0
        
#         # f_score: estimated total cost from start to goal through current node
#         f_score = {}
#         f_score[start_wp] = start_f_score
        
#         # Keep track of explored paths for visualization
#         explored_paths = []
        
#         # Set for tracking nodes in open_set
#         open_set_hash = {start_wp}
        
#         while not open_set.empty():
#             current = open_set.get()[2]
#             open_set_hash.remove(current)
            
#             # Check if we reached the destination
#             if current.transform.location.distance(end_wp.transform.location) < self.waypoint_distance:
#                 # Reconstruct path
#                 path = []
#                 temp_current = current
#                 while temp_current in came_from:
#                     path.append(temp_current)
#                     temp_current = came_from[temp_current]
#                 path.append(start_wp)
#                 path.reverse()
                
#                 # Calculate total distance
#                 total_distance = g_score[current]
                
#                 # Visualize explored paths
#                 self._visualize_exploration(explored_paths)
                
#                 return path, total_distance
            
#             # Get next waypoints
#             next_wps = current.next(self.waypoint_distance)
            
#             for next_wp in next_wps:
#                 # Calculate tentative g_score
#                 tentative_g_score = g_score[current] + current.transform.location.distance(
#                     next_wp.transform.location)
                
#                 if next_wp not in g_score or tentative_g_score < g_score[next_wp]:
#                     # This is a better path, record it
#                     came_from[next_wp] = current
#                     g_score[next_wp] = tentative_g_score
#                     f_score[next_wp] = tentative_g_score + self._heuristic(next_wp, end_wp)
                    
#                     if next_wp not in open_set_hash:
#                         counter += 1  # Increment counter for unique comparison
#                         open_set.put((f_score[next_wp], counter, next_wp))
#                         open_set_hash.add(next_wp)
                        
#                         # Store for visualization
#                         explored_paths.append((current, next_wp))
                        
#                         # Real-time visualization of exploration
#                         self.world.debug.draw_line(
#                             current.transform.location + carla.Location(z=0.5),
#                             next_wp.transform.location + carla.Location(z=0.5),
#                             thickness=0.1,
#                             color=carla.Color(64, 64, 255),  # Light blue for exploration
#                             life_time=0.1
#                         )
        
#         return None, float('inf')

#     def get_control(self, vehicle, world=None):
#         """Calculate control commands to follow path"""
#         try:
#             if not self.waypoints:
#                 print("No waypoints available!")
#                 return carla.VehicleControl(throttle=0, steer=0, brake=1.0)
                
#             # Get current vehicle state
#             vehicle_transform = vehicle.get_transform()
#             vehicle_loc = vehicle_transform.location
#             current_speed = self._get_speed(vehicle)
            
#             # Debug current state
#             print(f"Current location: x={vehicle_loc.x:.2f}, y={vehicle_loc.y:.2f}")
#             print(f"Current speed: {current_speed:.2f} km/h")
            
#             # Update current waypoint index based on distance
#             self._update_waypoint_progress(vehicle_loc)
            
#             # Get target waypoint
#             target_wp = self.waypoints[self.current_waypoint_index]
            
#             # Calculate steering
#             steer = self._calculate_steering(vehicle_transform, target_wp.transform)
            
#             # Calculate throttle and brake with more aggressive values
#             target_speed = self.target_speed
#             error = target_speed - current_speed
            
#             if error > 0:
#                 throttle = min(abs(error) * self.k_p_speed * 2.0, 1.0)  # More aggressive throttle
#                 brake = 0.0
#             else:
#                 throttle = 0.0
#                 brake = min(abs(error) * self.k_p_speed, 1.0)
            
#             # Debug control outputs
#             print(f"Target speed: {target_speed:.2f} km/h")
#             print(f"Speed error: {error:.2f} km/h")
#             print(f"Control outputs - Throttle: {throttle:.2f}, Brake: {brake:.2f}, Steer: {steer:.2f}")
            
#             # Ensure minimum throttle when starting from stop
#             if current_speed < 0.1 and not brake:
#                 throttle = max(throttle, 0.3)  # Minimum throttle to overcome inertia
            
#             control = carla.VehicleControl(
#                 throttle=float(throttle),
#                 steer=float(steer),
#                 brake=float(brake),
#                 hand_brake=False,
#                 reverse=False,
#                 manual_gear_shift=False,
#                 gear=1
#             )
            
#             return control
            
#         except Exception as e:
#             print(f"Error in get_control: {str(e)}")
#             return carla.VehicleControl(throttle=0, steer=0, brake=1.0)

#     def _calculate_steering(self, vehicle_transform, waypoint_transform):
#         """Calculate steering angle based on lateral and heading error"""
#         # Get relative location
#         wp_loc = waypoint_transform.location
#         veh_loc = vehicle_transform.location
        
#         # Transform to vehicle's coordinate system
#         dx = wp_loc.x - veh_loc.x
#         dy = wp_loc.y - veh_loc.y
        
#         # Calculate errors
#         vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
#         cos_yaw = math.cos(vehicle_yaw)
#         sin_yaw = math.sin(vehicle_yaw)
        
#         lateral_error = -dx * sin_yaw + dy * cos_yaw
        
#         # Heading error
#         waypoint_yaw = math.radians(waypoint_transform.rotation.yaw)
#         heading_error = math.atan2(math.sin(waypoint_yaw - vehicle_yaw), 
#                                  math.cos(waypoint_yaw - vehicle_yaw))
        
#         # Combine errors
#         steering = (self.k_p_lateral * lateral_error + 
#                    self.k_p_heading * heading_error)
        
#         return np.clip(steering, -self.max_steer, self.max_steer)

    # def _speed_control(self, current_speed):
    #     """Aggressive speed control with guaranteed stopping for obstacles"""
    #     min_distance = float('inf')
    #     if self.obstacles:
    #         vehicle_loc = self._parent.get_location()
    #         for obstacle in self.obstacles:
    #             distance = vehicle_loc.distance(obstacle)
    #             if distance < min_distance:
    #                 min_distance = distance

    #     # Very aggressive stopping distances
    #     emergency_distance = 8.0   # Emergency brake if closer than this
    #     critical_distance = 10.0   # Start strong braking
    #     safe_distance = 20.0   # Start slowing down

    #     # Debug output
    #     if min_distance < float('inf'):
    #         print(f"\nVEHICLE DISTANCE: {min_distance:.1f}m")
    #         if min_distance < emergency_distance:
    #             print("!!! EMERGENCY STOP !!!")
    #         elif min_distance < critical_distance:
    #             print("!! STRONG BRAKING !!")
    #         elif min_distance < safe_distance:
    #             print("! Slowing Down !")

    #     # Collision avoidance logic
    #     if min_distance < emergency_distance:
    #         # Emergency stop - maximum brake
    #         return 0.0, 1.0

    #     elif min_distance < critical_distance:
    #         # Strong braking - proportional to how close we are to emergency distance
    #         brake_intensity = 0.9 * (1 - (min_distance - emergency_distance) / 
    #                                (critical_distance - emergency_distance))
    #         return 0.0, brake_intensity

    #     elif min_distance < safe_distance:
    #         # Gradual speed reduction
    #         target_speed = self.target_speed * (min_distance - critical_distance) / \
    #                       (safe_distance - critical_distance)
    #         target_speed = max(5.0, min(target_speed, self.target_speed))
    #     else:
    #         target_speed = self.target_speed

    #     # Normal speed control
    #     error = target_speed - current_speed
    #     if error > 0:
    #         throttle = min(abs(error) * self.k_p_speed, 0.75)
    #         brake = 0.0
    #     else:
    #         throttle = 0.0
    #         brake = min(abs(error) * self.k_p_speed, 1.0)

    #     return throttle, brake

#     def _get_speed(self, vehicle):
#         """Get current speed in km/h"""
#         vel = vehicle.get_velocity()
#         return 3.6 * math.sqrt(vel.x**2 + vel.y**2)

#     # 3. Path Tracking Functions
#     def _update_waypoint_progress(self, vehicle_location):
#         """Update progress along waypoints"""
#         if self.current_waypoint_index >= len(self.waypoints):
#             return
            
#         current_wp = self.waypoints[self.current_waypoint_index]
#         distance = vehicle_location.distance(current_wp.transform.location)
        
#         if distance < self.waypoint_distance_threshold:
#             self.visited_waypoints.add(self.current_waypoint_index)
#             self.current_waypoint_index = min(self.current_waypoint_index + 1, 
#                                             len(self.waypoints) - 1)

#     # 4. Visualization Functions
#     def _visualize(self, world, vehicle):
#         """Visualize real-time progress along the A* path"""
#         if not self.waypoints:
#             return
            
#         # First time visualization of complete path with A* results
#         if not self.path_visualization_done:
#             self._visualize_complete_path(world, self.waypoints, 
#                                        self._calculate_total_distance(self.waypoints))
#             self.path_visualization_done = True
            
#         # Draw current progress on path
#         for i in range(len(self.waypoints) - 1):
#             start = self.waypoints[i].transform.location
#             end = self.waypoints[i + 1].transform.location
            
#             # Color based on whether waypoint has been visited
#             if i < self.current_waypoint_index:
#                 color = carla.Color(0, 255, 0)  # Green for visited
#             else:
#                 color = carla.Color(255, 0, 0)  # Red for upcoming
                
#             world.debug.draw_line(
#                 start + carla.Location(z=0.5),
#                 end + carla.Location(z=0.5),
#                 thickness=0.1,
#                 color=color,
#                 life_time=self.debug_lifetime
#             )
        
#         # Draw current target waypoint
#         if self.current_waypoint_index < len(self.waypoints):
#             target = self.waypoints[self.current_waypoint_index]
#             world.debug.draw_point(
#                 target.transform.location + carla.Location(z=1.0),
#                 size=0.1,
#                 color=carla.Color(0, 255, 255),  # Cyan
#                 life_time=self.debug_lifetime
#             )
            
#         # Draw progress percentage and current metrics
#         progress = (len(self.visited_waypoints) / len(self.waypoints)) * 100
#         current_loc = vehicle.get_location()
#         distance_to_target = current_loc.distance(
#             self.waypoints[self.current_waypoint_index].transform.location
#         )
        
#         # Draw progress information
#         info_text = [
#             f"Progress: {progress:.1f}%",
#             f"Distance to next waypoint: {distance_to_target:.1f}m",
#             f"Waypoints remaining: {len(self.waypoints) - self.current_waypoint_index}"
#         ]
        
#         for i, text in enumerate(info_text):
#             world.debug.draw_string(
#                 current_loc + carla.Location(z=2.0 + i * 0.5),  # Stack text vertically
#                 text,
#                 color=carla.Color(255, 255, 255),
#                 life_time=self.debug_lifetime
#             )

#     def _visualize_exploration(self, explored_paths):
#         """Visualize all explored paths"""
#         for start_wp, end_wp in explored_paths:
#             self.world.debug.draw_line(
#                 start_wp.transform.location + carla.Location(z=0.5),
#                 end_wp.transform.location + carla.Location(z=0.5),
#                 thickness=0.1,
#                 color=carla.Color(64, 64, 255),  # Light blue
#                 life_time=0.0
#             )

#     def _visualize_complete_path(self, world, path, total_distance):
#         """Visualize the complete planned path with metrics"""
#         if not path:
#             return
            
#         # Draw complete path
#         for i in range(len(path) - 1):
#             start = path[i].transform.location
#             end = path[i + 1].transform.location
            
#             # Draw path line
#             world.debug.draw_line(
#                 start + carla.Location(z=0.5),
#                 end + carla.Location(z=0.5),
#                 thickness=0.3,
#                 color=carla.Color(0, 255, 0),  # Green
#                 life_time=0.0
#             )
            
#             # Draw waypoint markers
#             world.debug.draw_point(
#                 start + carla.Location(z=0.5),
#                 size=0.1,
#                 color=carla.Color(255, 255, 0),  # Yellow
#                 life_time=0.0
#             )
        
#         # Draw start and end points
#         world.debug.draw_point(
#             path[0].transform.location + carla.Location(z=1.0),
#             size=0.2,
#             color=carla.Color(0, 255, 0),  # Green for start
#             life_time=0.0
#         )
        
#         world.debug.draw_point(
#             path[-1].transform.location + carla.Location(z=1.0),
#             size=0.2,
#             color=carla.Color(255, 0, 0),  # Red for end
#             life_time=0.0
#         )

#         # Draw distance markers and path info
#         for i in range(len(path) - 1):
#             current_distance = path[0].transform.location.distance(path[i].transform.location)
#             if int(current_distance) % 10 == 0:  # Every 10 meters
#                 world.debug.draw_string(
#                     path[i].transform.location + carla.Location(z=2.0),
#                     f"{int(current_distance)}m",
#                     color=carla.Color(255, 255, 255),
#                     life_time=0.0
#                 )

#         # Draw path information
#         info_location = path[0].transform.location + carla.Location(z=3.0)
#         world.debug.draw_string(
#             info_location,
#             f"Path Length: {total_distance:.1f}m",
#             color=carla.Color(0, 255, 0),
#             life_time=0.0
#         )

#     def _calculate_total_distance(self, path):
#         """Calculate total path distance"""
#         total_distance = 0
#         for i in range(len(path) - 1):
#             total_distance += path[i].transform.location.distance(
#                 path[i + 1].transform.location
#             )
#         return total_distance

#     def update_obstacles(self, obstacles):
#         """Update list of detected obstacles"""
#         self.obstacles = obstacles









# class SafetyController:
#     def __init__(self, parent_actor, world, controller):
#         self._parent = parent_actor
#         self._world = world
#         self._controller = controller
#         self.debug = True
        
#         # Detection parameters
#         self.detection_distance = 40.0  # Look ahead distance
#         self.lane_width = 3.5          # Standard lane width in meters
#         self.last_detected = False
        
#         # Create collision sensor
#         blueprint = world.get_blueprint_library().find('sensor.other.collision')
#         self.collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
#         self.collision_sensor.listen(lambda event: self._on_collision(event))
        
#         print("Safety controller initialized with improved lane detection")

    # def _on_collision(self, event):
    #     """Collision event handler"""
    #     print("!!! COLLISION DETECTED !!!")
    #     self._emergency_stop()

#     def _calculate_right_vector(self, forward_vector, rotation):
#         """Calculate right vector from forward vector and rotation"""
#         # Right vector is perpendicular to forward vector
#         # In a right-handed coordinate system, right = forward rotated 90 degrees clockwise
#         pitch = math.radians(rotation.pitch)
#         yaw = math.radians(rotation.yaw)
        
#         # Calculate right vector
#         right_x = math.cos(yaw)
#         right_y = math.sin(yaw)
        
#         return carla.Vector3D(x=right_x, y=right_y, z=0.0)

#     def _is_in_same_lane(self, ego_location, ego_forward, other_location):
#         """Simplified lane detection using lateral distance"""
#         try:
#             # Vector to other vehicle
#             to_other = other_location - ego_location
            
#             # Project onto forward direction to get longitudinal distance
#             forward_dist = (to_other.x * ego_forward.x + to_other.y * ego_forward.y)
            
#             if forward_dist <= 0:  # Vehicle is behind
#                 return False
                
#             # Calculate lateral offset using cross product
#             # Cross product magnitude gives lateral distance
#             lateral_dist = abs(ego_forward.x * to_other.y - ego_forward.y * to_other.x)
            
#             # Check if within lane bounds (using 80% of lane width to be conservative)
#             is_in_lane = lateral_dist < (self.lane_width * 0.8)
            
#             if self.debug and forward_dist > 0:
#                 color = carla.Color(0, 255, 0) if is_in_lane else carla.Color(128, 128, 128)
#                 self._world.debug.draw_line(
#                     ego_location,
#                     other_location,
#                     thickness=0.1,
#                     color=color,
#                     life_time=0.1
#                 )
            
#             return is_in_lane
            
#         except Exception as e:
#             print(f"Lane check error: {str(e)}")
#             return True  # Default to true for safety

#     def check_safety(self):
#         """Check for vehicles in same lane"""
#         try:
#             all_vehicles = self._world.get_actors().filter('vehicle.*')
#             ego_transform = self._parent.get_transform()
#             ego_location = ego_transform.location
#             ego_forward = ego_transform.get_forward_vector()
            
#             detected_vehicles = []
#             min_distance = float('inf')
            
#             # Get current speed
#             velocity = self._parent.get_velocity()
#             speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
            
#             for vehicle in all_vehicles:
#                 if vehicle.id == self._parent.id:
#                     continue
                    
#                 vehicle_location = vehicle.get_location()
#                 distance = ego_location.distance(vehicle_location)
                
#                 # Only check vehicles within detection range
#                 if distance < self.detection_distance:
#                     if self._is_in_same_lane(ego_location, ego_forward, vehicle_location):
#                         detected_vehicles.append((vehicle, distance))
#                         min_distance = min(min_distance, distance)
                        
#                         if self.debug:
#                             # Draw warning box
#                             self._world.debug.draw_box(
#                                 vehicle.bounding_box,
#                                 vehicle.get_transform().rotation,
#                                 thickness=0.5,
#                                 color=carla.Color(255, 0, 0, 255),
#                                 life_time=0.1
#                             )
                            
#                             # Draw warning text
#                             self._world.debug.draw_string(
#                                 vehicle_location + carla.Location(z=2.0),
#                                 f'!!! {distance:.1f}m !!!',
#                                 color=carla.Color(255, 0, 0, 255),
#                                 life_time=0.1
#                             )
            
#             if detected_vehicles:
#                 print(f"\nSame-lane vehicle detected at {min_distance:.1f}m, Speed: {speed:.1f} km/h")
                
#                 # Very conservative braking distances
#                 # emergency_stop_distance = max(15.0, speed * 0.7)  # Increased from 0.5 to 0.7 seconds
#                 emergency_stop_distance = max(5.0, speed * 0.4)

#                 if min_distance < emergency_stop_distance:
#                     print(f"!!! EMERGENCY STOP !!! Vehicle at {min_distance:.1f}m")
#                     self._emergency_stop()
#                 elif min_distance < emergency_stop_distance * 1.5:
#                     print(f"!! HARD BRAKING !! Vehicle at {min_distance:.1f}m")
#                     self._hard_brake()
#                 else:
#                     print(f"! Slowing ! Vehicle at {min_distance:.1f}m")
#                     self._slow_down()
                
#                 self.last_detected = True
#             else:
#                 if self.last_detected:
#                     print("No same-lane vehicles detected")
#                 self.last_detected = False
            
#             # Update controller
#             self._controller.update_obstacles([v[0].get_location() for v in detected_vehicles])
            
#         except Exception as e:
#             print(f"Error in safety check: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             self._emergency_stop()


#     def _emergency_stop(self):
#         """Force emergency stop"""
#         control = carla.VehicleControl(
#             throttle=0.0,
#             brake=1.0,
#             hand_brake=True,
#             steer=0.0  # Keep straight while stopping
#         )
#         self._parent.apply_control(control)

#     def _hard_brake(self):
#         """Apply strong braking"""
#         control = carla.VehicleControl(
#             throttle=0.0,
#             brake=0.8,
#             hand_brake=False,
#             steer=0.0  # Keep straight while braking
#         )
#         self._parent.apply_control(control)

#     def _slow_down(self):
#         """Gradual speed reduction"""
#         control = carla.VehicleControl(
#             throttle=0.0,
#             brake=0.5,
#             hand_brake=False
#         )
#         self._parent.apply_control(control)

    # def destroy(self):
    #     """Clean up sensors"""
    #     if hasattr(self, 'collision_sensor'):
    #         self.collision_sensor.destroy()








# def main():
#     try:
#         # Initialize Pygame
#         pygame.init()
#         pygame.display.set_caption("CARLA Navigation")
#         display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
#         clock = pygame.time.Clock()
        
#         # Connect to CARLA
#         print("Connecting to CARLA...")
#         client, world = connect_to_carla()
        
#         # Set synchronous mode with enhanced physics settings
#         settings = world.get_settings()
#         settings.synchronous_mode = True
#         settings.fixed_delta_seconds = 1.0 / FPS
#         settings.substepping = True  # Enable physics substepping
#         settings.max_substep_delta_time = 0.01  # Maximum physics substep size
#         settings.max_substeps = 10  # Maximum number of substeps
#         world.apply_settings(settings)
        
#         # Wait for the world to stabilize
#         print("Waiting for world to stabilize...")
#         for _ in range(20):
#             world.tick(2.0)
#             time.sleep(0.1)
        
#         # Find suitable spawn points
#         print("Finding suitable spawn points...")
#         start_point, end_point = find_spawn_points(world)
#         print(f"Start point: {start_point.location}")
#         print(f"End point: {end_point.location}")
        
#         # Spawn vehicle
#         print("Spawning vehicle...")
#         blueprint = world.get_blueprint_library().find('vehicle.tesla.model3')
#         vehicle = None
#         camera = None
#         detector = None
#         safety_controller = None
        
#         try:
#             vehicle = world.spawn_actor(blueprint, start_point)
#             print("Vehicle spawned successfully")
            
#             # Set up camera
#             camera = CameraManager(vehicle, world)
            
#             # Set up Vehicle Detector and Controllers
#             from navigation_controller import NavigationController
#             controller = NavigationController()
#             detector = VehicleDetector(vehicle, world, controller)
#             safety_controller = SafetyController(vehicle, world, controller)
            
#             # Allow everything to settle
#             world.tick()
#             time.sleep(0.5)
            
#             # Plan path
#             print("Planning path...")
#             success = controller.set_path(world, start_point.location, end_point.location)
            
#             if not success:
#                 print("Failed to plan path!")
#                 return
            
#             print(f"Path planned with {len(controller.waypoints)} waypoints")
            
#             # Spawn NPC vehicles
#             print("Spawning NPC vehicles...")
#             npcs = spawn_strategic_npcs(world, vehicle, close_npcs=5, far_npcs=15)
#             print(f"Spawned {len(npcs)} total NPCs")
            
#             # Main simulation loop
#             with tqdm(total=5000, desc="Navigation") as pbar:
#                 try:
#                     while True:
#                         try:
#                             # Tick the world with correct parameter
#                             start_time = time.time()
#                             while True:
#                                 try:
#                                     world.tick(2.0)
#                                     break
#                                 except RuntimeError as e:
#                                     if time.time() - start_time > 10.0:  # Overall timeout
#                                         raise
#                                     time.sleep(0.1)
#                                     continue
                            
#                             # Update Pygame display
#                             display.fill((0, 0, 0))
#                             camera.render(display)
#                             if detector is not None:
#                                 detector.detect_vehicles()
#                                 detector.render(display)
#                             pygame.display.flip()
                            
#                             # Safety check before applying control
#                             safety_controller.check_safety()
        
#                             # Correctly pass the vehicle parameter to get_control
#                             if not safety_controller.last_detected:  # Only apply if no obstacles detected
#                                 control = controller.get_control(vehicle, world)  # Pass both vehicle and world
#                                 vehicle.apply_control(control)
#                                 # Debug output for normal operation
#                                 print(f"Control commands - Throttle: {control.throttle:.2f}, "
#                                       f"Brake: {control.brake:.2f}, "
#                                       f"Steer: {control.steer:.2f}")
#                             else:
#                                 print("Safety controller activated - stopping vehicle")
                            
#                             # Debug vehicle state
#                             velocity = vehicle.get_velocity()
#                             speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)  # km/h
#                             print(f"Vehicle speed: {speed:.2f} km/h")
                            
#                             # Update progress
#                             if controller.waypoints:
#                                 progress = (len(controller.visited_waypoints) / 
#                                           len(controller.waypoints)) * 100
#                             else:
#                                 progress = 0
                            
#                             pbar.update(1)
#                             pbar.set_postfix({
#                                 'speed': f"{speed:.1f}km/h",
#                                 'progress': f"{progress:.1f}%",
#                                 'safety': 'ACTIVE' if safety_controller.last_detected else 'OK'
#                             })
                            
#                             # Check if destination reached
#                             if (controller.current_waypoint_index >= 
#                                 len(controller.waypoints) - 1):
#                                 print("\nDestination reached!")
#                                 time.sleep(2)
#                                 break
                            
#                             # Handle pygame events
#                             for event in pygame.event.get():
#                                 if event.type == pygame.QUIT:
#                                     return
#                                 elif event.type == pygame.KEYDOWN:
#                                     if event.key == pygame.K_ESCAPE:
#                                         return
                            
#                             clock.tick(FPS)
                            
#                         except Exception as e:
#                             print(f"\nError during simulation step: {str(e)}")
#                             if "time-out" in str(e).lower():
#                                 print("Attempting to recover from timeout...")
#                                 time.sleep(1.0)  # Give the simulator time to recover
#                                 continue
#                             else:
#                                 raise
                
#                 except KeyboardInterrupt:
#                     print("\nNavigation interrupted by user")
#                 except Exception as e:
#                     print(f"Unexpected error: {str(e)}")
#                     import traceback
#                     traceback.print_exc()
                        
#         finally:
#             print("Cleaning up...")
#             try:
#                 if detector is not None:
#                     detector.destroy()
#                 if camera is not None:
#                     camera.destroy()
#                 if safety_controller is not None:
#                     safety_controller.destroy()
#                 if vehicle is not None:
#                     vehicle.destroy()
                
#                 # Destroy NPC vehicles
#                 if 'npcs' in locals() and npcs:
#                     for npc in npcs:
#                         if npc is not None and npc.is_alive:
#                             try:
#                                 npc.set_autopilot(False)  # Disable autopilot before destroying
#                                 npc.destroy()
#                             except:
#                                 pass
                
#                 # Restore original settings
#                 settings = world.get_settings()
#                 settings.synchronous_mode = False
#                 settings.fixed_delta_seconds = None
#                 settings.substepping = False  # Disable physics substepping
#                 world.apply_settings(settings)
                
#             except Exception as e:
#                 print(f"Error during cleanup: {e}")
                
#             print("Cleanup complete")
                
#     except Exception as e:
#         print(f"Error: {str(e)}")
#         import traceback
#         traceback.print_exc()
        
#     finally:
#         pygame.quit()
#         print("Pygame quit")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print('\nCancelled by user.')
#     except Exception as e:
#         print(f'Error occurred: {e}')
#         import traceback
#         traceback.print_exc()










    def _speed_control(self, current_speed):
        """Aggressive speed control with guaranteed stopping for obstacles"""
        min_distance = float('inf')
        if self.obstacles:
            vehicle_loc = self._parent.get_location()
            for obstacle in self.obstacles:
                distance = vehicle_loc.distance(obstacle)
                if distance < min_distance:
                    min_distance = distance

        # Very aggressive stopping distances
        emergency_distance = 3.0   # Emergency brake if closer than this
        critical_distance = 6.0   # Start strong braking
        safe_distance = 10.0   # Start slowing down

        # Debug output
        if min_distance < float('inf'):
            print(f"\nVEHICLE DISTANCE: {min_distance:.1f}m")
            if min_distance < emergency_distance:
                print("!!! EMERGENCY STOP !!!")
            elif min_distance < critical_distance:
                print("!! STRONG BRAKING !!")
            elif min_distance < safe_distance:
                print("! Slowing Down !")

        # Collision avoidance logic
        if min_distance < emergency_distance:
            # Emergency stop - maximum brake
            return 0.0, 1.0

        elif min_distance < critical_distance:
            # Strong braking - proportional to how close we are to emergency distance
            brake_intensity = 0.9 * (1 - (min_distance - emergency_distance) / 
                                   (critical_distance - emergency_distance))
            return 0.0, brake_intensity

        elif min_distance < safe_distance:
            # Gradual speed reduction
            target_speed = self.target_speed * (min_distance - critical_distance) / \
                          (safe_distance - critical_distance)
            target_speed = max(5.0, min(target_speed, self.target_speed))
        else:
            target_speed = self.target_speed

        # Normal speed control
        error = target_speed - current_speed
        if error > 0:
            throttle = min(abs(error) * self.k_p_speed, 0.75)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(error) * self.k_p_speed, 1.0)

        return throttle, brake
    


















        # def _is_in_same_lane(self, ego_location, ego_forward, other_location):
    #     """More accurate lane detection with additional safety checks"""
    #     try:
    #         # Vector to other vehicle
    #         to_other = other_location - ego_location
            
    #         # Project onto forward direction
    #         forward_dist = (to_other.x * ego_forward.x + to_other.y * ego_forward.y)
            
    #         if forward_dist <= 0:  # Vehicle is behind
    #             return False, forward_dist
                
    #         # Calculate lateral offset
    #         lateral_dist = abs(ego_forward.x * to_other.y - ego_forward.y * to_other.x)
            
    #         # Stricter lane bounds (50% of lane width)
    #         is_in_lane = lateral_dist < (self.lane_width * 0.5)
            
    #         if self.debug and forward_dist > 0:
    #             color = carla.Color(0, 255, 0) if is_in_lane else carla.Color(128, 128, 128)
    #             self._world.debug.draw_line(
    #                 ego_location,
    #                 other_location,
    #                 thickness=0.1,
    #                 color=color,
    #                 life_time=0.1
    #             )
            
    #         return is_in_lane, forward_dist
            
    #     except Exception as e:
    #         print(f"Lane check error: {str(e)}")
    #         return True, 0  # Conservative approach - assume in lane if error
