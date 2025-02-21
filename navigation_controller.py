import carla
import math
import numpy as np

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
        
        # Visualization
        self.debug_lifetime = 0.1

    def set_path(self, world, start_location, end_location):
        """Generate path between start and end locations"""
        try:
            # Get map
            self.world = world
            self.map = world.get_map()
            
            # Convert locations to waypoints
            start_waypoint = self.map.get_waypoint(start_location)
            end_waypoint = self.map.get_waypoint(end_location)
            
            print(f"Planning path from {start_waypoint.transform.location} to {end_waypoint.transform.location}")
            
            # Generate path using A* like approach
            path = []
            current = start_waypoint
            
            while current and current.transform.location.distance(end_waypoint.transform.location) > 2.0:
                path.append(current)
                
                # Get next waypoints
                next_wps = current.next(2.0)  # Get waypoints 2 meters ahead
                if not next_wps:
                    break
                
                # Find the waypoint that gets us closest to the destination
                current = min(next_wps, 
                            key=lambda wp: wp.transform.location.distance(end_waypoint.transform.location))
                
                # Avoid infinite loops
                if len(path) > 1000:  # Safety limit
                    print("Path too long, stopping path generation")
                    break
            
            # Add final waypoint
            if end_waypoint not in path:
                path.append(end_waypoint)
            
            self.waypoints = path
            self.current_waypoint_index = 0
            self.visited_waypoints.clear()
            
            print(f"Path planned successfully with {len(path)} waypoints")
            
            # Visualize the complete path
            self._visualize_complete_path(world)
            
            return True
            
        except Exception as e:
            print(f"Path planning failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _visualize_complete_path(self, world):
        """Visualize the complete planned path"""
        if not self.waypoints:
            return
            
        # Draw complete path
        for i in range(len(self.waypoints) - 1):
            start = self.waypoints[i].transform.location
            end = self.waypoints[i + 1].transform.location
            
            # Draw path line
            world.debug.draw_line(
                start + carla.Location(z=0.5),
                end + carla.Location(z=0.5),
                thickness=0.2,
                color=carla.Color(255, 0, 0),  # Red for initial path
                life_time=0.0  # Persist until updated
            )
            
            # Draw waypoint markers
            world.debug.draw_point(
                start + carla.Location(z=0.5),
                size=0.1,
                color=carla.Color(0, 255, 255),  # Cyan for waypoints
                life_time=0.0
            )
        
        # Draw end point
        world.debug.draw_point(
            self.waypoints[-1].transform.location + carla.Location(z=0.5),
            size=0.2,
            color=carla.Color(0, 255, 0),  # Green for endpoint
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
    
    def _update_waypoint_progress(self, vehicle_location):
        """Update progress along waypoints"""
        if self.current_waypoint_index >= len(self.waypoints):
            return
            
        # Check if current waypoint is reached
        current_wp = self.waypoints[self.current_waypoint_index]
        distance = vehicle_location.distance(current_wp.transform.location)
        
        if distance < self.waypoint_distance_threshold:
            self.visited_waypoints.add(self.current_waypoint_index)
            self.current_waypoint_index = min(self.current_waypoint_index + 1, 
                                            len(self.waypoints) - 1)
    
    def _calculate_steering(self, vehicle_transform, waypoint_transform):
        """Calculate steering angle"""
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
        """Simple speed control"""
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
    
    def _visualize(self, world, vehicle):
        """Visualize path and progress"""
        if not self.waypoints:
            return
            
        # Draw complete path
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
            
        # Draw progress percentage
        progress = (len(self.visited_waypoints) / len(self.waypoints)) * 100
        world.debug.draw_string(
            vehicle.get_location() + carla.Location(z=2.0),
            f"Progress: {progress:.1f}%",
            color=carla.Color(255, 255, 255),
            life_time=self.debug_lifetime
        )