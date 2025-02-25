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
