import os
import sys
import time
import numpy as np
import pygame
import carla
import random
import math

class SafetyController:
    def __init__(self, parent_actor, world, controller):
        self._parent = parent_actor
        self._world = world
        self._controller = controller
        self.debug = True
        
        # Adjusted detection parameters
        self.detection_distance = 50.0  # Distance to detect vehicles ahead
        self.lane_width = 3.5
        self.last_detected = False

        self.recovery_mode = False
        self.recovery_start_time = None
        self.max_recovery_time = 3.0  # seconds
        self.min_speed_threshold = 0.5  # m/s
        self.last_brake_time = None
        self.brake_cooldown = 1.0  # seconds
        
        # Modified braking parameters for more gradual approach
        self.time_to_collision_threshold = 5.0
        self.min_safe_distance = 15.0
        self.emergency_brake_distance = 7.0
        
        # New gradual braking parameters
        self.target_stop_distance = 5.0  # meters - we want to stop this far from obstacles
        self.deceleration_start_distance = 40.0  # Start slowing down from this distance
        self.deceleration_profile = [ # (distance, brake_intensity)
            (30.0, 0.05),
            (20.0, 0.15),
            (15.0, 0.3),
            (10.0, 0.4),
            (8.0, 0.6),
            (5.0, 1.0)
        ]
        
        # UPDATED: Traffic light parameters with shorter distances
        self.traffic_light_detection_distance = 70.0  # Keep detection range high
        self.traffic_light_detection_angle = 45.0  # Degrees - cone of vision for traffic light detection
        self.yellow_light_threshold = 3.0  # seconds - if we can't clear intersection in this time, stop
        self.traffic_light_slowdown_distance = 25.0  # REDUCED from 35m to 25m - start slowing at this distance
        self.traffic_light_stop_distance = 3.0  # REDUCED from 5m to 3m - stop this far from the traffic light
        self.override_timeout = 10.0  # seconds - maximum time to override controller for traffic lights
        self.last_tl_brake_time = None
        self.tl_override_start_time = None
        self.is_tl_override_active = False
        self.green_light_resume_attempts = 0  # NEW: Count resume attempts
        self.max_green_light_resume_attempts = 3  # NEW: Maximum resume attempts
        self.last_green_light_time = None  # NEW: Track when we last saw a green light
        self.green_light_grace_period = 0.5  # NEW: Grace period after green light detection (seconds)
        self.last_tl_state = None  # NEW: Keep track of last traffic light state
        
        # Create collision sensor
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        print("Enhanced Safety controller initialized with improved traffic light detection")

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
    
    def _get_traffic_light_state(self):
        """
        Improved traffic light detection with multiple methods
        Returns: 
            - light_state: carla.TrafficLightState or None
            - distance: float distance to light or float('inf')
            - light_actor: carla.TrafficLight or None
            - light_location: carla.Location or None
        """
        try:
            ego_location = self._parent.get_location()
            ego_transform = self._parent.get_transform()
            ego_forward = ego_transform.get_forward_vector()
            ego_waypoint = self._world.get_map().get_waypoint(ego_location)
            
            if not ego_waypoint:
                return None, float('inf'), None, None
                
            # Method 1: Check if vehicle is at traffic light (CARLA API)
            if ego_waypoint.is_junction:
                # If we're already in a junction, check if we have a traffic light
                traffic_light = self._parent.get_traffic_light()
                if traffic_light:
                    if hasattr(traffic_light, 'get_state'):
                        light_location = traffic_light.get_location()
                        distance = ego_location.distance(light_location)
                        return traffic_light.get_state(), distance, traffic_light, light_location
                    else:
                        print(f"Warning: Found an object that is not a proper traffic light: {type(traffic_light).__name__}")
                        return None, float('inf'), None, None

            
            # Method 2: Check traffic light state of our current lane
            if hasattr(ego_waypoint, 'get_traffic_light'):
                traffic_light = ego_waypoint.get_traffic_light()
                if traffic_light:
                    if traffic_light:
                        if hasattr(traffic_light, 'get_state'):
                            light_location = traffic_light.get_location()
                            distance = ego_location.distance(light_location)
                            return traffic_light.get_state(), distance, traffic_light, light_location
                        else:
                            print(f"Warning: Found an object that is not a proper traffic light: {type(traffic_light).__name__}")
                            return None, float('inf'), None, None

            
            # Method 3: Follow waypoints ahead to detect traffic lights
            waypoints_ahead = []
            next_wp = ego_waypoint
            
            # Collect waypoints ahead
            distance_accumulated = 0
            while distance_accumulated < self.traffic_light_detection_distance:
                next_wps = next_wp.next(2.0)  # 2-meter steps
                if not next_wps:
                    break
                next_wp = next_wps[0]
                waypoints_ahead.append(next_wp)
                distance_accumulated += 2.0
                
                # Check if this waypoint has a traffic light
                if hasattr(next_wp, 'get_traffic_light'):
                    traffic_light = next_wp.get_traffic_light()
                    if traffic_light:
                        if traffic_light:
                            if hasattr(traffic_light, 'get_state'):
                                light_location = traffic_light.get_location()
                                distance = ego_location.distance(light_location)
                                return traffic_light.get_state(), distance, traffic_light, light_location
                            else:
                                print(f"Warning: Found an object that is not a proper traffic light: {type(traffic_light).__name__}")
                                return None, float('inf'), None, None

                
                # Check if waypoint is at a junction with traffic light
                if next_wp.is_junction:
                    # CARLA 0.9.8 approach: Find traffic lights near the junction
                    traffic_lights = self._world.get_actors().filter('traffic.traffic_light*')
                    min_distance = float('inf')
                    closest_light = None
                    junction_loc = next_wp.transform.location
                    
                    for light in traffic_lights:
                        light_loc = light.get_location()
                        distance = light_loc.distance(junction_loc)
                        
                        # Check if the light is within reasonable distance of junction and in our field of view
                        if distance < 20.0:  # Within reasonable distance of junction
                            # Check if the light is in our field of view
                            to_light = light_loc - ego_location
                            norm = math.sqrt(to_light.x**2 + to_light.y**2 + to_light.z**2)
                            to_light_normalized = carla.Vector3D(
                                to_light.x / norm,
                                to_light.y / norm,
                                to_light.z / norm
                            )
                            
                            # Calculate dot product with forward vector (cosine of angle)
                            forward_dot = ego_forward.x * to_light_normalized.x + ego_forward.y * to_light_normalized.y
                            angle = math.acos(max(-1.0, min(1.0, forward_dot))) * 180 / math.pi
                            
                            # Check if light is within our detection angle
                            if angle < self.traffic_light_detection_angle and distance < min_distance:
                                min_distance = distance
                                closest_light = light
                    
                    if closest_light:
                        # Calculate distance to traffic light
                        light_loc = closest_light.get_location()
                        light_distance = ego_location.distance(light_loc)
                        
                        # Draw debug visualization
                        if self.debug:
                            # Draw debug info for traffic light
                            color = carla.Color(255, 255, 0)  # Yellow by default
                            if closest_light.get_state() == carla.TrafficLightState.Red:
                                color = carla.Color(255, 0, 0)  # Red
                            elif closest_light.get_state() == carla.TrafficLightState.Green:
                                color = carla.Color(0, 255, 0)  # Green
                            
                            self._world.debug.draw_line(
                                ego_location,
                                light_loc,
                                thickness=0.2,
                                color=color,
                                life_time=0.1
                            )
                            self._world.debug.draw_point(
                                light_loc,
                                size=0.2,
                                color=color,
                                life_time=0.1
                            )
                            self._world.debug.draw_string(
                                light_loc + carla.Location(z=2.0),
                                f'Traffic Light: {closest_light.get_state()}, {light_distance:.1f}m',
                                color=color,
                                life_time=0.1
                            )
                        
                        return closest_light.get_state(), light_distance, closest_light, light_loc
            
            # Method 4: Direct search for nearby traffic lights in our path
            traffic_lights = self._world.get_actors().filter('traffic.traffic_light*')
            min_distance = float('inf')
            closest_light = None
            closest_light_loc = None
            
            for light in traffic_lights:
                light_loc = light.get_location()
                distance = ego_location.distance(light_loc)
                
                # Only check lights within detection distance
                if distance < self.traffic_light_detection_distance:
                    # Check if the light is in our field of view
                    to_light = light_loc - ego_location
                    norm = math.sqrt(to_light.x**2 + to_light.y**2 + to_light.z**2)
                    to_light_normalized = carla.Vector3D(
                        to_light.x / norm,
                        to_light.y / norm,
                        to_light.z / norm
                    )
                    
                    # Calculate dot product with forward vector (cosine of angle)
                    forward_dot = ego_forward.x * to_light_normalized.x + ego_forward.y * to_light_normalized.y
                    angle = math.acos(max(-1.0, min(1.0, forward_dot))) * 180 / math.pi
                    
                    # Check if light is within our detection angle and closer than any previously found
                    if angle < self.traffic_light_detection_angle and distance < min_distance:
                        # Verify this light affects our lane by checking nearby waypoints
                        light_waypoint = self._world.get_map().get_waypoint(light_loc)
                        if light_waypoint and light_waypoint.road_id == ego_waypoint.road_id:
                            min_distance = distance
                            closest_light = light
                            closest_light_loc = light_loc
            
            if closest_light:
                if self.debug:
                    # Draw debug info for traffic light
                    color = carla.Color(255, 255, 0)  # Yellow by default
                    if closest_light.get_state() == carla.TrafficLightState.Red:
                        color = carla.Color(255, 0, 0)  # Red
                    elif closest_light.get_state() == carla.TrafficLightState.Green:
                        color = carla.Color(0, 255, 0)  # Green
                    
                    self._world.debug.draw_line(
                        ego_location,
                        closest_light_loc,
                        thickness=0.2,
                        color=color,
                        life_time=0.1
                    )
                    self._world.debug.draw_point(
                        closest_light_loc,
                        size=0.2,
                        color=color,
                        life_time=0.1
                    )
                    self._world.debug.draw_string(
                        closest_light_loc + carla.Location(z=2.0),
                        f'Traffic Light: {closest_light.get_state()}, {min_distance:.1f}m',
                        color=color,
                        life_time=0.1
                    )
                
                return closest_light.get_state(), min_distance, closest_light, closest_light_loc
            
            return None, float('inf'), None, None
            
        except Exception as e:
            print(f"Error detecting traffic light: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, float('inf'), None, None
    
    def _handle_traffic_light(self):
        """Enhanced traffic light handling with more robust stopping and resuming"""
        light_state, distance, light_actor, light_location = self._get_traffic_light_state()
        current_time = time.time()
        
        # NEW: Store the last traffic light state for comparison
        previous_state = self.last_tl_state
        self.last_tl_state = light_state
        
        if light_state is None or distance == float('inf'):
            # Reset traffic light override if we're no longer detecting a light
            if self.is_tl_override_active:
                if (current_time - self.tl_override_start_time > self.override_timeout):
                    print("Releasing traffic light override (timeout)")
                    self.is_tl_override_active = False
                    self.tl_override_start_time = None
                    self._force_resume_path()  # UPDATED: Use force resume instead
            return False
        
        # Get current speed
        ego_velocity = self._parent.get_velocity()
        speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)
        speed_kmh = 3.6 * speed  # Convert to km/h for display
        
        # Convert to carla.TrafficLightState object if needed
        if isinstance(light_state, str):
            if light_state == "Red":
                light_state = carla.TrafficLightState.Red
            elif light_state == "Yellow":
                light_state = carla.TrafficLightState.Yellow
            elif light_state == "Green":
                light_state = carla.TrafficLightState.Green
        
        # Handle different light states
        if light_state == carla.TrafficLightState.Red:
            # Red light - we should come to a stop
            if distance < self.traffic_light_slowdown_distance:
                # NEW: Reset green light counter
                self.green_light_resume_attempts = 0
                self.last_green_light_time = None
                
                # Calculate stopping distance
                stopping_distance = distance - self.traffic_light_stop_distance
                
                # Activate traffic light override
                if not self.is_tl_override_active:
                    self.is_tl_override_active = True
                    self.tl_override_start_time = time.time()
                    print(f"\n!!! RED LIGHT DETECTED - Distance: {distance:.1f}m, Speed: {speed_kmh:.1f} km/h !!!")
                
                if stopping_distance <= 0:
                    # We've reached or passed the stopping point
                    self._emergency_stop()
                    print(f"RED LIGHT STOP - Distance: {distance:.1f}m")
                else:
                    # Apply gradual braking based on distance
                    brake_intensity = self._calculate_brake_intensity(stopping_distance)
                    
                    # Increase braking intensity for red lights
                    brake_intensity = min(1.0, brake_intensity * 1.5)
                    
                    print(f"RED LIGHT BRAKING - Distance: {distance:.1f}m, Brake: {brake_intensity:.2f}")
                    control = carla.VehicleControl(
                        throttle=0.0,
                        brake=brake_intensity,
                        steer=self._maintain_path_steer(),  # Keep steering while braking
                        hand_brake=False
                    )
                    self._parent.apply_control(control)
                    self.last_tl_brake_time = time.time()
                return True
        
        elif light_state == carla.TrafficLightState.Yellow:
            # Yellow light handling (same as before)
            time_to_light = distance / max(speed, 0.1)  # Avoid division by zero
            
            if time_to_light > self.yellow_light_threshold:
                # We can't clear the intersection in time, stop
                if distance < self.traffic_light_slowdown_distance:
                    # NEW: Reset green light counter
                    self.green_light_resume_attempts = 0
                    self.last_green_light_time = None
                    
                    # Activate traffic light override
                    if not self.is_tl_override_active:
                        self.is_tl_override_active = True
                        self.tl_override_start_time = time.time()
                        print(f"\n!!! YELLOW LIGHT DETECTED (stopping) - Distance: {distance:.1f}m, Time: {time_to_light:.1f}s !!!")
                    
                    # Calculate stopping distance
                    stopping_distance = distance - self.traffic_light_stop_distance
                    
                    if stopping_distance <= 0:
                        # We've reached or passed the stopping point
                        self._emergency_stop()
                        print(f"YELLOW LIGHT STOP - Distance: {distance:.1f}m")
                    else:
                        # Apply gradual braking based on distance
                        brake_intensity = self._calculate_brake_intensity(stopping_distance)
                        print(f"YELLOW LIGHT BRAKING - Distance: {distance:.1f}m, Brake: {brake_intensity:.2f}")
                        control = carla.VehicleControl(
                            throttle=0.0,
                            brake=brake_intensity,
                            steer=self._maintain_path_steer(),  # Keep steering while braking
                            hand_brake=False
                        )
                        self._parent.apply_control(control)
                        self.last_tl_brake_time = time.time()
                    return True
            else:
                # We can clear the intersection, proceed with caution
                if self.is_tl_override_active:
                    self.is_tl_override_active = False
                    self.tl_override_start_time = None
                print(f"YELLOW LIGHT (proceeding) - Distance: {distance:.1f}m, Time: {time_to_light:.1f}s")
                return False
        
        elif light_state == carla.TrafficLightState.Green:
            # UPDATED: Improved green light handling
            
            # Track when we first see the green light
            if previous_state != carla.TrafficLightState.Green:
                self.last_green_light_time = current_time
                print(f"\n!!! GREEN LIGHT DETECTED - Distance: {distance:.1f}m !!!")
            
            # If we were stopped for a red/yellow light and now it's green
            if self.is_tl_override_active:
                # Wait for a short grace period before trying to resume
                if self.last_green_light_time and (current_time - self.last_green_light_time) >= self.green_light_grace_period:
                    # Increment resume attempt counter
                    self.green_light_resume_attempts += 1
                    
                    print(f"GREEN LIGHT - Resuming operation (attempt {self.green_light_resume_attempts})")
                    
                    # Force resume (apply extra throttle to get moving)
                    self._force_resume_path()
                    
                    # Only release override after max attempts or if we're moving
                    if speed > 2.0 or self.green_light_resume_attempts >= self.max_green_light_resume_attempts:
                        self.is_tl_override_active = False
                        self.tl_override_start_time = None
                        self.green_light_resume_attempts = 0
                        print(f"GREEN LIGHT - Successfully resumed normal operation")
                    
                    # Even if we're still in override mode, return False to allow the controller to work
                    return False
            else:
                # Already moving - no need to do anything special
                self.green_light_resume_attempts = 0
            
            return False
        
        # Reset override for unknown light state
        if self.is_tl_override_active:
            self.is_tl_override_active = False
            self.tl_override_start_time = None
        
        return False
    
    def _force_resume_path(self):
        """Force vehicle to resume movement after stopping for a traffic light"""
        try:
            # Reset brake and recovery state
            self.last_brake_time = None
            self.recovery_mode = False
            self.recovery_start_time = None
            
            # Apply a strong initial acceleration
            control = carla.VehicleControl(
                throttle=0.7,  # Apply strong throttle
                brake=0.0,
                steer=self._maintain_path_steer(),
                hand_brake=False
            )
            self._parent.apply_control(control)
            
            # Reset controller state if method exists
            if hasattr(self._controller, '_reset_control_state'):
                self._controller._reset_control_state()
                
        except Exception as e:
            print(f"Error forcing resume: {str(e)}")
    
    def _maintain_path_steer(self):
        """Get steering value to maintain path while braking"""
        try:
            if not self._controller:
                return 0.0
                
            if hasattr(self._controller, 'waypoints') and hasattr(self._controller, 'current_waypoint_index'):
                if self._controller.current_waypoint_index < len(self._controller.waypoints):
                    target_wp = self._controller.waypoints[self._controller.current_waypoint_index]
                    vehicle_transform = self._parent.get_transform()
                    
                    if hasattr(self._controller, '_calculate_steering'):
                        return self._controller._calculate_steering(vehicle_transform, target_wp.transform)
            
            return 0.0
        
        except Exception as e:
            print(f"Error calculating steering: {str(e)}")
            return 0.0
    
    def _calculate_brake_intensity(self, distance):
        """Calculate brake intensity based on distance to obstacle/traffic light"""
        # Find the appropriate braking level from the deceleration profile
        for dist, brake in self.deceleration_profile:
            if distance <= dist:
                return brake
        
        return 0.0  # No braking needed
    
    def _apply_gradual_braking(self, distance, ttc):
        """Apply gradual braking based on distance and time to collision"""
        # Emergency stop for very close obstacles
        if distance < 3.0 or ttc < 0.5:
            print(f"!!! EMERGENCY STOP !!! Distance: {distance:.1f}m, TTC: {ttc:.1f}s")
            self._emergency_stop()
            return True
        
        # Calculate the appropriate brake intensity
        brake_intensity = self._calculate_brake_intensity(distance)
        
        # If TTC is very small, increase braking
        if ttc < 2.0:
            brake_intensity = max(brake_intensity, 0.8)
        elif ttc < 3.0:
            brake_intensity = max(brake_intensity, 0.5)
        
        if brake_intensity > 0:
            ego_velocity = self._parent.get_velocity()
            speed = 3.6 * math.sqrt(ego_velocity.x**2 + ego_velocity.y**2)  # km/h
            
            print(f"Gradual braking - Distance: {distance:.1f}m, TTC: {ttc:.1f}s, Speed: {speed:.1f} km/h, Brake: {brake_intensity:.2f}")
            
            control = carla.VehicleControl(
                throttle=0.0,
                brake=brake_intensity,
                steer=self._maintain_path_steer(),  # Keep steering while braking
                hand_brake=False
            )
            self._parent.apply_control(control)
            return True
        
        return False
    
    def check_safety(self):
        """Enhanced safety checking with prioritized traffic light handling"""
        try:
            # First check traffic lights - THIS MUST TAKE PRIORITY
            if self._handle_traffic_light():
                # If we're handling a traffic light, skip other checks
                return
            
            # Only check for other vehicles if we're not dealing with a traffic light
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
                
                # Check vehicles within detection range
                if distance < self.detection_distance:
                    is_in_lane, forward_dist = self._is_in_same_lane(ego_location, ego_forward, vehicle_location)
                    
                    if is_in_lane and forward_dist > 0:  # Only consider vehicles ahead
                        other_velocity = vehicle.get_velocity()
                        ttc = self._calculate_time_to_collision(ego_velocity, other_velocity, forward_dist)
                        
                        detected_vehicles.append((vehicle, forward_dist, ttc))
                        min_distance = min(min_distance, forward_dist)
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
                                f'!!! {forward_dist:.1f}m, TTC: {ttc:.1f}s !!!',
                                color=carla.Color(255, 0, 0, 255),
                                life_time=0.1
                            )
            
            if detected_vehicles:
                print(f"\nVehicle detected - Distance: {min_distance:.1f}m, TTC: {min_ttc:.1f}s, Speed: {speed:.1f} km/h")
                
                # Apply gradual braking based on distance and TTC
                if self._apply_gradual_braking(min_distance, min_ttc):
                    self.last_detected = True
                else:
                    # Not braking, proceed normally
                    if self.last_detected:
                        print("Path clear - resuming normal operation")
                        self._resume_path()
                    self.last_detected = False
            else:
                if self.last_detected:
                    print("Path clear - resuming normal operation")
                    self._resume_path()
                self.last_detected = False
            
            # Update controller with obstacles
            if self._controller and hasattr(self._controller, 'update_obstacles'):
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
            steer=self._maintain_path_steer()  # Keep steering while emergency braking
        )
        self._parent.apply_control(control)
        self.last_brake_time = time.time()

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
                    if hasattr(self._controller, 'force_path_recovery'):
                        control = self._controller.force_path_recovery(self._parent)
                        self._parent.apply_control(control)
                    return
            
            # Normal path maintenance
            if hasattr(self._controller, 'waypoints') and hasattr(self._controller, 'current_waypoint_index'):
                if self._controller.current_waypoint_index < len(self._controller.waypoints):
                    target_wp = self._controller.waypoints[self._controller.current_waypoint_index]
                    vehicle_transform = self._parent.get_transform()
                    
                    if hasattr(self._controller, '_calculate_steering'):
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
            
            # Reset controller state if method exists
            if hasattr(self._controller, '_reset_control_state'):
                self._controller._reset_control_state()
            
            # Resume normal control
            if hasattr(self._controller, 'get_control'):
                control = self._controller.get_control(self._parent, self._world)
                self._parent.apply_control(control)
            
        except Exception as e:
            print(f"Error resuming path: {str(e)}")

    def destroy(self):
        """Clean up sensors"""
        if hasattr(self, 'collision_sensor'):
            self.collision_sensor.destroy()