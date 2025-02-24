import carla
import math
import numpy as np
from queue import PriorityQueue
import time

class NavigationController:
    def __init__(self):
        # Control parameters
        self.max_steer = 0.7
        self.target_speed = 30.0  # km/h

        # Adjusted controller gains for smoother control
        self.k_p_lateral = 0.9    # Reduced from 1.5
        self.k_p_heading = 0.8    # Reduced from 1.0
        self.k_p_speed = 1.0      

        # Path tracking
        self.waypoints = []
        self.visited_waypoints = set()
        self.current_waypoint_index = 0
        self.waypoint_distance_threshold = 2.0

        # A* parameters
        self.waypoint_distance = 2.0
        self.max_search_dist = 200.0

        # Store reference to vehicle for speed calculations
        self._parent = None
        self.last_control = None
        
        # Visualization
        self.debug_lifetime = 0.1
        self.path_visualization_done = False
        
        # Obstacle detection
        self.obstacles = []
        
        # World reference for visualization
        self.world = None
        self.map = None

    def _heuristic(self, waypoint, goal_waypoint):
        """A* heuristic: straight-line distance to goal"""
        return waypoint.transform.location.distance(goal_waypoint.transform.location)

    def set_path(self, world, start_location, end_location):
        """Generate shortest path using A* algorithm"""
        try:
            self.world = world
            self.map = world.get_map()
            
            # Convert locations to waypoints
            start_waypoint = self.map.get_waypoint(start_location)
            end_waypoint = self.map.get_waypoint(end_location)
            
            print(f"Planning path using A* from {start_waypoint.transform.location} to {end_waypoint.transform.location}")
            
            # Find path using A*
            path, distance = self._find_path_astar(start_waypoint, end_waypoint)
            
            if not path:
                print("No path found!")
                return False
            
            self.waypoints = path
            self.current_waypoint_index = 0
            self.visited_waypoints.clear()
            
            print(f"Path found with {len(path)} waypoints and distance {distance:.1f} meters")
            
            # Visualize the path and exploration
            self._visualize_complete_path(world, path, distance)
            
            return True
            
        except Exception as e:
            print(f"Path planning failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _find_path_astar(self, start_wp, end_wp):
        """A* algorithm implementation for finding shortest path"""
        # Priority queue for A* (f_score, counter, waypoint)
        counter = 0  # Unique identifier for comparing waypoints
        open_set = PriorityQueue()
        start_f_score = self._heuristic(start_wp, end_wp)
        open_set.put((start_f_score, counter, start_wp))
        
        # For reconstructing path
        came_from = {}
        
        # g_score: cost from start to current node
        g_score = {}
        g_score[start_wp] = 0
        
        # f_score: estimated total cost from start to goal through current node
        f_score = {}
        f_score[start_wp] = start_f_score
        
        # Keep track of explored paths for visualization
        explored_paths = []
        
        # Set for tracking nodes in open_set
        open_set_hash = {start_wp}
        
        while not open_set.empty():
            current = open_set.get()[2]
            open_set_hash.remove(current)
            
            # Check if we reached the destination
            if current.transform.location.distance(end_wp.transform.location) < self.waypoint_distance:
                # Reconstruct path
                path = []
                temp_current = current
                while temp_current in came_from:
                    path.append(temp_current)
                    temp_current = came_from[temp_current]
                path.append(start_wp)
                path.reverse()
                
                # Calculate total distance
                total_distance = g_score[current]
                
                # Visualize explored paths
                self._visualize_exploration(explored_paths)
                
                return path, total_distance
            
            # Get next waypoints
            next_wps = current.next(self.waypoint_distance)
            
            for next_wp in next_wps:
                # Calculate tentative g_score
                tentative_g_score = g_score[current] + current.transform.location.distance(
                    next_wp.transform.location)
                
                if next_wp not in g_score or tentative_g_score < g_score[next_wp]:
                    # This is a better path, record it
                    came_from[next_wp] = current
                    g_score[next_wp] = tentative_g_score
                    f_score[next_wp] = tentative_g_score + self._heuristic(next_wp, end_wp)
                    
                    if next_wp not in open_set_hash:
                        counter += 1  # Increment counter for unique comparison
                        open_set.put((f_score[next_wp], counter, next_wp))
                        open_set_hash.add(next_wp)
                        
                        # Store for visualization
                        explored_paths.append((current, next_wp))
                        
                        # Real-time visualization of exploration
                        self.world.debug.draw_line(
                            current.transform.location + carla.Location(z=0.5),
                            next_wp.transform.location + carla.Location(z=0.5),
                            thickness=0.1,
                            color=carla.Color(64, 64, 255),  # Light blue for exploration
                            life_time=0.1
                        )
        
        return None, float('inf')

    def get_control(self, vehicle, world=None):
        """Calculate control commands to follow path - simplified for strict path following"""
        try:
            if not self.waypoints:
                print("No waypoints available!")
                return carla.VehicleControl(throttle=0, steer=0, brake=1.0)
            
            self._parent = vehicle
            # Get current vehicle state
            vehicle_transform = vehicle.get_transform()
            vehicle_loc = vehicle_transform.location
            current_speed = self._get_speed(vehicle)
            
            # Update the world reference if provided
            if world is not None and self.world is None:
                self.world = world
            
            # Update current waypoint index based on distance
            self._update_waypoint_progress(vehicle_loc)
            
            # Get target waypoint
            target_wp = self.waypoints[self.current_waypoint_index]
            
            # Calculate distance to current waypoint
            distance_to_waypoint = vehicle_loc.distance(target_wp.transform.location)
            
            # Calculate steering
            steer = self._calculate_steering(vehicle_transform, target_wp.transform)
            
            # Speed control based on distance to waypoint
            target_speed = self.target_speed
            if distance_to_waypoint > 5.0:  # If far from waypoint
                target_speed *= 0.7  # Reduce speed to 70%
            
            error = target_speed - current_speed
            
            # Calculate throttle and brake
            if error > 0:
                throttle = min(abs(error) * self.k_p_speed, 0.75)
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(abs(error) * self.k_p_speed, 0.75)
            
            # Ensure minimum throttle when starting from stop
            if current_speed < 0.1 and not brake:
                throttle = max(throttle, 0.3)  # Minimum throttle to overcome inertia
            
            # Gradual steering changes for stability
            if self.last_control:
                max_steer_change = 0.1
                steer = np.clip(
                    steer,
                    self.last_control.steer - max_steer_change,
                    self.last_control.steer + max_steer_change
                )
            
            # Debug output
            print(f"\nPath following status:")
            print(f"Current waypoint index: {self.current_waypoint_index}/{len(self.waypoints)-1}")
            print(f"Distance to waypoint: {distance_to_waypoint:.2f}m")
            print(f"Current speed: {current_speed:.2f}km/h")
            print(f"Target speed: {target_speed:.2f}km/h")
            print(f"Controls - Throttle: {throttle:.2f}, Brake: {brake:.2f}, Steer: {steer:.2f}")
            
            # Visualize the path if world is available
            if self.world is not None:
                self._visualize(self.world, vehicle)
            
            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
                gear=1
            )
            
            self.last_control = control
            return control
            
        except Exception as e:
            print(f"Error in get_control: {str(e)}")
            import traceback
            traceback.print_exc()
            return carla.VehicleControl(throttle=0, steer=0, brake=1.0)

    def _calculate_steering(self, vehicle_transform, waypoint_transform):
        """Calculate steering angle with strict path following"""
        try:
            # Current vehicle state
            veh_loc = vehicle_transform.location
            vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
            current_speed = self._get_speed(self._parent)

            # Shorter lookahead for stricter path following
            base_lookahead = max(2.0, min(current_speed * 0.2, 5.0))  # Reduced lookahead

            # Calculate path curvature but with less influence
            curvature = self._estimate_path_curvature()

            # Minimal reduction of lookahead in turns
            lookahead_distance = base_lookahead * (1.0 - 0.1 * abs(curvature))  # Reduced influence of curvature

            # Use fewer preview points closer to vehicle
            preview_points = self._get_preview_points(lookahead_distance)

            # Calculate weighted steering based on preview points
            total_steering = 0
            total_weight = 0

            for i, target_loc in enumerate(preview_points):
                # Higher weight on closest point
                weight = 1.0 / (i + 1.1)  # More emphasis on immediate path

                # Convert target to vehicle's coordinate system
                dx = target_loc.x - veh_loc.x
                dy = target_loc.y - veh_loc.y

                # Transform target into vehicle's coordinate system
                cos_yaw = math.cos(vehicle_yaw)
                sin_yaw = math.sin(vehicle_yaw)

                target_x = dx * cos_yaw + dy * sin_yaw
                target_y = -dx * sin_yaw + dy * cos_yaw

                # Calculate angle and immediate path curvature
                angle = math.atan2(target_y, target_x)
                if abs(target_x) > 0.01:
                    point_curvature = 2.0 * target_y / (target_x * target_x + target_y * target_y)
                else:
                    point_curvature = 0.0

                # Adjust steering components to focus on immediate path following
                point_steering = (
                    0.4 * point_curvature +  # Reduced influence of curvature
                    0.6 * self.k_p_lateral * (target_y / lookahead_distance) +  # Increased cross-track correction
                    0.3 * self.k_p_heading * angle  # Moderate heading correction
                )

                total_steering += point_steering * weight
                total_weight += weight

            # Calculate final steering
            if total_weight > 0:
                steering = total_steering / total_weight
            else:
                steering = 0.0

            # More conservative speed-based steering adjustments
            speed_factor = min(current_speed / 30.0, 1.0)
            max_steer_change = 0.12 * (1.0 - 0.4 * speed_factor)  # More conservative rate limiting

            # Apply steering limits
            if self.last_control:
                steering = np.clip(
                    steering,
                    self.last_control.steer - max_steer_change,
                    self.last_control.steer + max_steer_change
                )

            # Reduced additional steering in turns
            max_steer = self.max_steer * (1.0 + 0.1 * abs(curvature))  # Minimal increase in turns
            steering = np.clip(steering, -max_steer, max_steer)

            return steering

        except Exception as e:
            print(f"Error in steering calculation: {str(e)}")
            return 0.0

    def _estimate_path_curvature(self):
        """Estimate the curvature of the upcoming path section"""
        try:
            if self.current_waypoint_index >= len(self.waypoints) - 2:
                return 0.0

            # Get three consecutive waypoints
            p1 = self.waypoints[self.current_waypoint_index].transform.location
            p2 = self.waypoints[self.current_waypoint_index + 1].transform.location
            p3 = self.waypoints[self.current_waypoint_index + 2].transform.location

            # Calculate vectors
            v1 = np.array([p2.x - p1.x, p2.y - p1.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])

            # Calculate angle between vectors
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)

            if norms < 1e-6:
                return 0.0

            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angle = np.arccos(cos_angle)

            # Normalize curvature to [-1, 1] range
            curvature = angle / np.pi

            return curvature

        except Exception as e:
            print(f"Error calculating curvature: {str(e)}")
            return 0.0

    def _get_preview_points(self, base_lookahead):
        """Get preview points with closer spacing for stricter path following"""
        preview_points = []
        current_idx = self.current_waypoint_index

        # Get points with closer spacing
        distances = [base_lookahead * mult for mult in [0.8, 1.0, 1.2]]  # Closer spacing of preview points

        for dist in distances:
            # Find waypoint at approximately this distance
            accumulated_dist = 0
            idx = current_idx

            while idx < len(self.waypoints) - 1:
                wp1 = self.waypoints[idx].transform.location
                wp2 = self.waypoints[idx + 1].transform.location
                segment_dist = wp1.distance(wp2)

                if accumulated_dist + segment_dist >= dist:
                    # Interpolate point at exact distance
                    remaining = dist - accumulated_dist
                    fraction = remaining / segment_dist
                    x = wp1.x + fraction * (wp2.x - wp1.x)
                    y = wp1.y + fraction * (wp2.y - wp1.y)
                    z = wp1.z + fraction * (wp2.z - wp1.z)
                    preview_points.append(carla.Location(x, y, z))
                    break
                
                accumulated_dist += segment_dist
                idx += 1

            if idx >= len(self.waypoints) - 1:
                preview_points.append(self.waypoints[-1].transform.location)

        return preview_points

    def _update_waypoint_progress(self, vehicle_location):
        """Update progress along waypoints with improved safety stop handling"""
        if self.current_waypoint_index >= len(self.waypoints):
            return

        current_wp = self.waypoints[self.current_waypoint_index]
        distance = vehicle_location.distance(current_wp.transform.location)

        # Only update waypoint if we're close enough AND moving forward
        if distance < self.waypoint_distance_threshold:
            # Check if we're actually moving towards the next waypoint
            if self.last_control and self.last_control.throttle > 0:
                self.visited_waypoints.add(self.current_waypoint_index)
                self.current_waypoint_index = min(self.current_waypoint_index + 1, 
                                                len(self.waypoints) - 1)


    def _reset_control_state(self):
        """Reset control state after aggressive braking"""
        self.last_control = None
        # Reset any accumulated steering
        return carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0)

    def force_path_recovery(self, vehicle):
        """Force vehicle back to path after deviation"""
        if not self.waypoints:
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

        try:
            # Get current vehicle state
            vehicle_transform = vehicle.get_transform()
            vehicle_loc = vehicle_transform.location

            # Find closest waypoint on path
            min_dist = float('inf')
            closest_idx = self.current_waypoint_index

            # Look at all remaining waypoints
            for i in range(self.current_waypoint_index, len(self.waypoints)):
                wp_loc = self.waypoints[i].transform.location
                dist = vehicle_loc.distance(wp_loc)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            # Update waypoint index if we found a closer one
            if closest_idx != self.current_waypoint_index:
                print(f"Path recovery - updating waypoint index from {self.current_waypoint_index} to {closest_idx}")
                self.current_waypoint_index = closest_idx

            # Get target waypoint
            target_wp = self.waypoints[self.current_waypoint_index]

            # Calculate strong corrective steering
            steer = self._calculate_steering(vehicle_transform, target_wp.transform)

            # Apply stronger steering correction for recovery
            steer *= 1.2  # Increase steering response

            # Get current speed
            current_speed = self._get_speed(vehicle)

            # Determine throttle and brake
            if current_speed < 5.0:  # Very slow or stopped
                throttle = 0.3  # Gentle acceleration
                brake = 0.0
            else:
                # Maintain moderate speed during recovery
                target_speed = min(20.0, self.target_speed * 0.5)  # Reduced speed
                speed_error = target_speed - current_speed

                if speed_error > 0:
                    throttle = min(0.5, speed_error * self.k_p_speed)
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = min(0.5, -speed_error * self.k_p_speed)

            # Create recovery control
            control = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake),
                hand_brake=False
            )

            self.last_control = control
            return control

        except Exception as e:
            print(f"Error in path recovery: {str(e)}")
            return self._reset_control_state()
    
    def _get_speed(self, vehicle):
        """Get current speed in km/h"""
        vel = vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x**2 + vel.y**2)
    
    # New Visualization Functions
    def _visualize(self, world, vehicle):
        """Visualize real-time progress along the A* path"""
        if not self.waypoints:
            return
            
        # First time visualization of complete path with A* results
        if not self.path_visualization_done:
            self._visualize_complete_path(world, self.waypoints, 
                                        self._calculate_total_distance(self.waypoints))
            self.path_visualization_done = True
            
        # Draw current progress on path
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i].transform.location
            end = self.waypoints[i + 1].transform.location
            
            # Color based on whether waypoint has been visited
            if i < self.current_waypoint_index:
                color = carla.Color(0, 255, 0)  # Green for visited
            else:
                color = carla.Color(255, 0, 0)  # Red for upcoming
                
            world.debug.draw_line(
                start + carla.Location(z=0.5),
                end + carla.Location(z=0.5),
                thickness=0.1,
                color=color,
                life_time=self.debug_lifetime
            )
        
        # Draw current target waypoint
        if self.current_waypoint_index < len(self.waypoints):
            target = self.waypoints[self.current_waypoint_index]
            world.debug.draw_point(
                target.transform.location + carla.Location(z=1.0),
                size=0.1,
                color=carla.Color(0, 255, 255),  # Cyan
                life_time=self.debug_lifetime
            )
            
        # Draw progress percentage and current metrics
        progress = (len(self.visited_waypoints) / len(self.waypoints)) * 100 if self.waypoints else 0
        current_loc = vehicle.get_location()
        distance_to_target = current_loc.distance(
            self.waypoints[self.current_waypoint_index].transform.location
        ) if self.current_waypoint_index < len(self.waypoints) else 0
        
        # Draw progress information
        info_text = [
            f"Progress: {progress:.1f}%",
            f"Distance to next waypoint: {distance_to_target:.1f}m",
            f"Waypoints remaining: {len(self.waypoints) - self.current_waypoint_index}"
        ]
        
        for i, text in enumerate(info_text):
            world.debug.draw_string(
                current_loc + carla.Location(z=2.0 + i * 0.5),  # Stack text vertically
                text,
                color=carla.Color(255, 255, 255),
                life_time=self.debug_lifetime
            )

    def _visualize_exploration(self, explored_paths):
        """Visualize all explored paths"""
        if self.world is None:
            return
            
        for start_wp, end_wp in explored_paths:
            self.world.debug.draw_line(
                start_wp.transform.location + carla.Location(z=0.5),
                end_wp.transform.location + carla.Location(z=0.5),
                thickness=0.1,
                color=carla.Color(173, 216, 230),  # Light blue
                life_time=0.0
            )

    def _visualize_complete_path(self, world, path, total_distance):
        """Visualize the complete planned path with metrics"""
        if not path:
            return
            
        # Draw complete path
        for i in range(len(path) - 1):
            start = path[i].transform.location
            end = path[i + 1].transform.location
            
            # Draw path line
            world.debug.draw_line(
                start + carla.Location(z=0.5),
                end + carla.Location(z=0.5),
                thickness=0.3,
                color=carla.Color(0, 255, 0),  # Green
                life_time=0.0
            )
            
            # Draw waypoint markers
            world.debug.draw_point(
                start + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(255, 255, 0),  # Yellow
                life_time=0.0
            )
        
        # Draw start and end points
        world.debug.draw_point(
            path[0].transform.location + carla.Location(z=1.0),
            size=0.2,
            color=carla.Color(0, 255, 0),  # Green for start
            life_time=0.0
        )
        
        world.debug.draw_point(
            path[-1].transform.location + carla.Location(z=1.0),
            size=0.2,
            color=carla.Color(255, 0, 0),  # Red for end
            life_time=0.0
        )

        # Draw distance markers and path info
        for i in range(len(path) - 1):
            current_distance = path[0].transform.location.distance(path[i].transform.location)
            if int(current_distance) % 10 == 0:  # Every 10 meters
                world.debug.draw_string(
                    path[i].transform.location + carla.Location(z=2.0),
                    f"{int(current_distance)}m",
                    color=carla.Color(255, 255, 255),
                    life_time=0.0
                )

        # Draw path information
        info_location = path[0].transform.location + carla.Location(z=3.0)
        world.debug.draw_string(
            info_location,
            f"Path Length: {total_distance:.1f}m",
            color=carla.Color(0, 255, 0),
            life_time=0.0
        )

    def _calculate_total_distance(self, path):
        """Calculate total path distance"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += path[i].transform.location.distance(
                path[i + 1].transform.location
            )
        return total_distance

    def update_obstacles(self, obstacles):
        """Update list of detected obstacles"""
        self.obstacles = obstacles