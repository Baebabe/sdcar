import carla
import math
import numpy as np
from queue import PriorityQueue
from collections import defaultdict

class NavigationController:
    def __init__(self):
        # Control parameters
        self.max_steer = 0.7
        self.target_speed = 30.0  # km/h
        
        # Controller gains
        self.k_p_lateral = 0.9
        self.k_p_heading = 0.5
        self.k_p_speed = 0.5
        
        # Path tracking
        self.waypoints = []
        self.visited_waypoints = set()
        self.current_waypoint_index = 0
        self.waypoint_distance_threshold = 2.0
        
        # A* parameters
        self.waypoint_distance = 2.0
        self.max_search_dist = 200.0
        
        # Visualization
        self.debug_lifetime = 0.1
        self.path_visualization_done = False

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
        """Calculate control commands to follow path"""
        try:
            if not self.waypoints:
                return carla.VehicleControl(throttle=0, steer=0, brake=1.0)
                
            vehicle_transform = vehicle.get_transform()
            current_speed = self._get_speed(vehicle)
            
            self._update_waypoint_progress(vehicle.get_location())
            
            target_wp = self.waypoints[self.current_waypoint_index]
            
            steer = self._calculate_steering(vehicle_transform, target_wp.transform)
            throttle, brake = self._speed_control(current_speed)
            
            if world is not None:
                self._visualize(world, vehicle)
                
            return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            
        except Exception as e:
            print(f"Control calculation failed: {str(e)}")
            return carla.VehicleControl(throttle=0, steer=0, brake=1.0)

    def _calculate_steering(self, vehicle_transform, waypoint_transform):
        """Calculate steering angle based on lateral and heading error"""
        # Get relative location
        wp_loc = waypoint_transform.location
        veh_loc = vehicle_transform.location
        
        # Transform to vehicle's coordinate system
        dx = wp_loc.x - veh_loc.x
        dy = wp_loc.y - veh_loc.y
        
        # Calculate errors
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        cos_yaw = math.cos(vehicle_yaw)
        sin_yaw = math.sin(vehicle_yaw)
        
        lateral_error = -dx * sin_yaw + dy * cos_yaw
        
        # Heading error
        waypoint_yaw = math.radians(waypoint_transform.rotation.yaw)
        heading_error = math.atan2(math.sin(waypoint_yaw - vehicle_yaw), 
                                 math.cos(waypoint_yaw - vehicle_yaw))
        
        # Combine errors
        steering = (self.k_p_lateral * lateral_error + 
                   self.k_p_heading * heading_error)
        
        return np.clip(steering, -self.max_steer, self.max_steer)

    def _speed_control(self, current_speed):
        """Calculate throttle and brake commands"""
        error = self.target_speed - current_speed
        
        if error > 0:
            throttle = min(abs(error) * self.k_p_speed, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(abs(error) * self.k_p_speed, 1.0)
            
        return throttle, brake

    def _get_speed(self, vehicle):
        """Get current speed in km/h"""
        vel = vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x**2 + vel.y**2)

    # 3. Path Tracking Functions
    def _update_waypoint_progress(self, vehicle_location):
        """Update progress along waypoints"""
        if self.current_waypoint_index >= len(self.waypoints):
            return
            
        current_wp = self.waypoints[self.current_waypoint_index]
        distance = vehicle_location.distance(current_wp.transform.location)
        
        if distance < self.waypoint_distance_threshold:
            self.visited_waypoints.add(self.current_waypoint_index)
            self.current_waypoint_index = min(self.current_waypoint_index + 1, 
                                            len(self.waypoints) - 1)

    # 4. Visualization Functions
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
        progress = (len(self.visited_waypoints) / len(self.waypoints)) * 100
        current_loc = vehicle.get_location()
        distance_to_target = current_loc.distance(
            self.waypoints[self.current_waypoint_index].transform.location
        )
        
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
        for start_wp, end_wp in explored_paths:
            self.world.debug.draw_line(
                start_wp.transform.location + carla.Location(z=0.5),
                end_wp.transform.location + carla.Location(z=0.5),
                thickness=0.1,
                color=carla.Color(64, 64, 255),  # Light blue
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
    def get_control(self, vehicle, world=None):
        """Calculate control commands to follow path"""
        try:
            if not self.waypoints:
                return carla.VehicleControl(throttle=0, steer=0, brake=1.0)
                
            # Get current vehicle state
            vehicle_transform = vehicle.get_transform()
            vehicle_loc = vehicle_transform.location
            current_speed = self._get_speed(vehicle)
            
            # Update current waypoint index based on distance
            self._update_waypoint_progress(vehicle_loc)
            
            # Get target waypoint
            target_wp = self.waypoints[self.current_waypoint_index]
            
            # Calculate steering
            steer = self._calculate_steering(vehicle_transform, target_wp.transform)
            
            # Calculate throttle and brake
            throttle, brake = self._speed_control(current_speed)
            
            # Visualize if debugging
            if world is not None:
                self._visualize(world, vehicle)
                
            return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            
        except Exception as e:
            print(f"Control calculation failed: {str(e)}")
            return carla.VehicleControl(throttle=0, steer=0, brake=1.0)

    def _calculate_total_distance(self, path):
        """Calculate total path distance"""
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += path[i].transform.location.distance(
                path[i + 1].transform.location
            )
        return total_distance