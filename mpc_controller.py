import sys
import os
import torch
import numpy as np
import casadi as ca
from model_def import Net_v4
import glob
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla


class MPCController:
    def __init__(self, model_path, horizon=8, dt=0.1):
        # Increased numerical stability parameters
        self.eps = 1e-2  # Increased from 1e-4
        self.state_bounds_slack = 0.2  # Increased from 0.1
        self.min_velocity = 0.5  # Minimum velocity to prevent singularities
        
        # Controller parameters
        self.horizon = horizon
        self.dt = dt
        self.lr_mean = 1.538900111477258
        self.wp_ahead = 10
        self.min_speed = 5.0
        self.max_speed = 10.0
        
        # Boundary values for waypoint processing
        self.max_position = 1000.0  # Maximum allowed position value
        
        # Load neural network model
        self.load_nn_model(model_path)
        
        # Setup optimization problem
        self.setup_mpc()
        self.prev_sol = None
        self.solve_status = None
        self.solve_iter = 0
        
        # Add solution history for warm starting
        self.solution_history = []
        self.max_history = 5

    def process_waypoints(self, waypoints):
        """Process waypoints into parameter matrix with bounds checking and interpolation"""
        wp_params = np.zeros((2, self.horizon))
        
        try:
            # Handle empty waypoints case
            if not waypoints:
                return wp_params
            
            # Process available waypoints
            for i in range(min(self.horizon, len(waypoints))):
                wp = waypoints[i]
                # Ensure waypoint locations are within bounds
                x = np.clip(wp.transform.location.x, -self.max_position, self.max_position)
                y = np.clip(wp.transform.location.y, -self.max_position, self.max_position)
                wp_params[0, i] = x
                wp_params[1, i] = y
            
            # If not enough waypoints, interpolate or extrapolate remaining points
            if len(waypoints) < self.horizon:
                last_wp = waypoints[-1]
                last_x = last_wp.transform.location.x
                last_y = last_wp.transform.location.y
                
                if len(waypoints) >= 2:
                    # Use last two waypoints to determine direction
                    second_last_wp = waypoints[-2]
                    dx = last_x - second_last_wp.transform.location.x
                    dy = last_y - second_last_wp.transform.location.y
                    # Normalize direction vector
                    dist = np.sqrt(dx*dx + dy*dy) + self.eps
                    dx = dx / dist
                    dy = dy / dist
                else:
                    # Default to continuing in current direction
                    dx = np.cos(last_wp.transform.rotation.yaw * np.pi / 180.0)
                    dy = np.sin(last_wp.transform.rotation.yaw * np.pi / 180.0)
                
                # Extrapolate remaining waypoints
                for i in range(len(waypoints), self.horizon):
                    steps = i - len(waypoints) + 1
                    x = last_x + steps * dx * 2.0  # 2.0 meters between extrapolated points
                    y = last_y + steps * dy * 2.0
                    wp_params[0, i] = np.clip(x, -self.max_position, self.max_position)
                    wp_params[1, i] = np.clip(y, -self.max_position, self.max_position)
            
            return wp_params
            
        except Exception as e:
            print(f"Waypoint processing error: {str(e)}")
            # Return zero matrix in case of error
            return np.zeros((2, self.horizon))

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π] range using CasADi"""
        return ca.arctan2(ca.sin(angle), ca.cos(angle))

    def setup_mpc(self):
        nx = 5  # [x, y, v, phi, beta]
        nu = 3  # [steering, throttle, brake]
        
        # Create optimization variables
        x = ca.SX.sym('x', nx)
        u = ca.SX.sym('u', nu)
        
        # Reference waypoints as parameters
        wp_ref = ca.SX.sym('wp', 2, self.horizon)
        
        # System dynamics
        xdot = self.system_dynamics(x, u)
        x_next = x + self.dt * xdot
        
        # Create discrete dynamics function
        self.F = ca.Function('F', [x, u], [x_next])
        
        # Start with empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        g = []
        lbg = []
        ubg = []
        J = 0
        
        # Initial state
        Xk = ca.SX.sym('X0', nx)
        w += [Xk]
        lbw += [-np.inf, -np.inf, 0, -2*np.pi, -np.pi/2]
        ubw += [np.inf, np.inf, self.max_speed, 2*np.pi, np.pi/2]
        w0 += [0]*nx
        
        # For storing states and controls
        states = [Xk]
        controls = []
        
        # Formulate the NLP
        for k in range(self.horizon):
            # Control at time k
            Uk = ca.SX.sym(f'U_{k}', nu)
            w += [Uk]
            lbw += [-0.7, 0, 0]  # steering, throttle, brake bounds
            ubw += [0.7, 1, 1]
            w0 += [0, 0.5, 0]
            controls += [Uk]
            
            # State at k+1
            Xk_next = self.F(Xk, Uk)
            Xk = ca.SX.sym(f'X_{k+1}', nx)
            
            # Add slack variables for state constraints
            slack = ca.SX.sym(f'slack_{k}', nx)
            w += [slack]
            lbw += [0] * nx
            ubw += [self.state_bounds_slack] * nx
            w0 += [0] * nx
            
            # State bounds with slack
            w += [Xk]
            lbw += [-np.inf, -np.inf, 0, -2*np.pi, -np.pi/2]
            ubw += [np.inf, np.inf, self.max_speed, 2*np.pi, np.pi/2]
            w0 += [0]*nx
            states += [Xk]
            
            # Add dynamics constraint with slack
            g += [Xk_next - Xk - slack]
            lbg += [0]*nx
            ubg += [0]*nx
            
            # Penalize slack variables
            J += 1000 * ca.sumsqr(slack)
            
            # Small regularization term for states for stability
            J += 0.001 * ca.sumsqr(Xk)
            
            # Current waypoint reference
            wp_k = wp_ref[:, k]
            
            # Position tracking cost with improved numerical stability
            pos_error = ca.vertcat(Xk[0] - wp_k[0], Xk[1] - wp_k[1])
            J += 1000 * ca.fmax(ca.norm_2(pos_error), self.eps)
            
            # Velocity cost (higher in curves)
            if k < self.horizon-1:
                wp_next = wp_ref[:, k+1]
                path_delta = wp_next - wp_k
                curvature = ca.fmax(ca.norm_2(path_delta), self.eps)
                target_speed = ca.if_else(curvature > 2.0, self.min_speed, self.max_speed)
                speed_error = Xk[2] - target_speed
                J += 100 * ca.fmax(ca.fabs(speed_error), self.eps)
            
            # Heading cost with normalization
            if k < self.horizon-1:
                wp_next = wp_ref[:, k+1]
                path_delta = wp_next - wp_k
                path_heading = ca.arctan2(path_delta[1], path_delta[0])
                heading_error = self.normalize_angle(Xk[3] - path_heading)
                J += 500 * ca.fmax(ca.fabs(heading_error), self.eps)
            
            # Control costs with improved numerical stability
            J += 10 * ca.fmax(ca.fabs(Uk[0]), self.eps)  # steering
            J += 1 * ca.fmax(ca.fabs(Uk[1]), self.eps)   # throttle
            J += 1 * ca.fmax(ca.fabs(Uk[2]), self.eps)   # brake
            
            # Control rate cost
            if k > 0:
                delta_steering = Uk[0] - controls[k-1][0]
                J += 100 * ca.fmax(ca.fabs(delta_steering), self.eps)
            
            Xk = Xk_next
        
        opts = {
            'ipopt': {
                'print_level': 0,
                'max_iter': 100,
                'tol': 1e-2,  # Relaxed tolerance
                'acceptable_tol': 1e-1,  # Relaxed acceptable tolerance
                'mu_strategy': 'monotone',  # Changed from adaptive
                'hessian_approximation': 'limited-memory',
                'max_cpu_time': 0.2,
                'bound_push': 0.1,  # Increased from 0.01
                'bound_frac': 0.1,  # Increased from 0.01
                'warm_start_init_point': 'yes',
                'warm_start_bound_push': 1e-2,
                'warm_start_bound_frac': 1e-2,
                'warm_start_mult_bound_push': 1e-2,
                'linear_solver': 'mumps',
                'nlp_scaling_method': 'equilibration-based',
                'bound_relax_factor': 1e-4,
                'honor_original_bounds': 'no',  # Allow slight bound violations
                'check_derivatives_for_naninf': 'yes'
            },
            'print_time': False
        }
        
        nlp = {
            'x': ca.vertcat(*w),
            'f': J,
            'g': ca.vertcat(*g),
            'p': wp_ref
        }
        
        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
        
        # Save variables
        self.nx = nx
        self.nu = nu
        self.w0 = w0
        self.lbw = lbw
        self.ubw = ubw
        self.lbg = lbg
        self.ubg = ubg

    def initialize_trajectory(self, state, waypoints):
        """Initialize solution with a simple straight-line prediction"""
        w0 = []
        
        # Initial state
        w0 += list(state)
        
        for k in range(self.horizon):
            # Control (neutral)
            w0 += [0.0, 0.5, 0.0]
            
            # Slack variables
            w0 += [0.0] * self.nx
            
            # Next state (simple forward prediction)
            next_state = list(state)
            if k < len(waypoints):
                wp = waypoints[k]
                next_state[0] = wp.transform.location.x
                next_state[1] = wp.transform.location.y
            w0 += next_state
            
        return w0

    def get_safe_fallback_control(self):
        """Return safe control values when solver fails"""
        if self.prev_sol is not None and len(self.prev_sol) > self.nx + self.nu:
            # Use previous solution but with reduced throttle and added brake
            u = self.prev_sol[self.nx:self.nx+self.nu].copy()
            u[0] *= 0.8  # Reduce steering
            u[1] *= 0.5  # Reduce throttle
            u[2] = max(0.3, u[2])  # Add some brake
            return u
        return np.array([0.0, 0.0, 0.3])  # Zero steering, some brake

    def smooth_normalize_angle(self, angle):
        """Smooth angle normalization"""
        return ca.arctan2(self.smooth_sin(angle), self.smooth_cos(angle))
    
    def smooth_sin(self, x):
        """Numerically stable sine approximation"""
        return ca.sin(x) / (1 + self.eps * ca.fabs(x))
    
    def smooth_cos(self, x):
        """Numerically stable cosine approximation"""
        return ca.cos(x) / (1 + self.eps * ca.fabs(x))
    
    def smooth_clamp(self, x, min_val, max_val):
        """Smooth clamping function"""
        return min_val + (max_val - min_val) * (1 / (1 + ca.exp(-10 * (x - min_val)/(max_val - min_val))))

    def solve(self, state, waypoints, world=None):
        """Enhanced solver with better error handling and warm starting"""
        try:
            # Ensure state values are within reasonable bounds
            state = np.clip(state, [
                -1000, -1000,  # x, y
                0, -2*np.pi,   # v, phi
                -np.pi/2       # beta
            ], [
                1000, 1000,    # x, y
                self.max_speed, 2*np.pi,  # v, phi
                np.pi/2        # beta
            ])
            
            # Process waypoints with bounds checking
            wp_params = self.process_waypoints(waypoints)
            
            # Initialize or update warm start
            if self.prev_sol is None:
                self.w0 = self.initialize_trajectory(state, waypoints)
                initial_guess = self.w0
            else:
                initial_guess = self.get_warm_start(state)
            
            # Solve with multiple attempts if needed
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    sol = self.solver(
                        x0=initial_guess,
                        lbx=self.lbw,
                        ubx=self.ubw,
                        lbg=self.lbg,
                        ubg=self.ubg,
                        p=wp_params
                    )
                    
                    if sol['success']:
                        # Update solution history
                        solution = sol['x'].full().flatten()
                        if not np.any(np.isnan(solution)):
                            self.update_solution_history(solution)
                            self.prev_sol = solution
                            u0 = solution[self.nx:self.nx+self.nu]
                            self.solve_status = f"Success (attempt {attempt+1})"
                            break
                    
                    # If failed, modify initial guess and retry
                    initial_guess = self.modify_initial_guess(initial_guess)
                    
                except Exception as e:
                    print(f"Attempt {attempt+1} failed: {str(e)}")
                    if attempt == max_attempts - 1:
                        return self.get_safe_fallback_control()
            
            # Visualize if world object provided
            if world is not None:
                self.visualize_prediction(self.prev_sol, waypoints, world)
            
            return u0
            
        except Exception as e:
            print(f"MPC solve failed: {str(e)}")
            return self.get_safe_fallback_control()

    def update_solution_history(self, solution):
        """Maintain a history of successful solutions"""
        self.solution_history.append(solution)
        if len(self.solution_history) > self.max_history:
            self.solution_history.pop(0)

    def get_warm_start(self, current_state):
        """Generate warm start from solution history"""
        if not self.solution_history:
            return self.initialize_trajectory(current_state, [])
            
        # Average recent successful solutions
        avg_solution = np.mean(self.solution_history, axis=0)
        
        # Update initial state
        avg_solution[:self.nx] = current_state
        
        return avg_solution

    def modify_initial_guess(self, guess):
        """Modify initial guess if solver fails"""
        # Add small random perturbation to break symmetry
        perturbation = np.random.normal(0, 0.01, size=len(guess))
        return guess + perturbation

    def system_dynamics(self, state, control):
        """Improved system dynamics with better numerical stability"""
        x, y, v, phi, beta = state[0], state[1], state[2], state[3], state[4]
        
        # Ensure minimum velocity
        v = ca.fmax(v, self.min_velocity)
        
        # Normalize angles with smooth approximation
        phi = self.smooth_normalize_angle(phi)
        beta = self.smooth_normalize_angle(beta)
        
        # Control inputs with smoothing
        steering = self.smooth_clamp(control[0], -0.7, 0.7)
        throttle = self.smooth_clamp(control[1], 0.0, 1.0)
        brake = self.smooth_clamp(control[2], 0.0, 1.0)
        
        # Smooth trigonometric functions
        cos_phi_beta = self.smooth_cos(phi + beta)
        sin_phi_beta = self.smooth_sin(phi + beta)
        
        # Basic dynamics
        dx = v * cos_phi_beta
        dy = v * sin_phi_beta
        dphi = v * self.smooth_sin(beta) / (self.lr_mean + self.eps)
        
        # Neural network prediction with smoothing
        nn_input = ca.vertcat(
            ca.sqrt(v + self.eps),
            self.smooth_cos(beta),
            self.smooth_sin(beta),
            steering,
            throttle,
            brake
        )
        
        accel = self.nn_forward(nn_input)
        
        # Add damping terms
        accel_long_damped = accel[0] - 0.05 * v
        accel_lat_damped = accel[1] - 0.05 * beta
        
        return ca.vertcat(dx, dy, accel_long_damped, dphi, accel_lat_damped)


    def nn_forward(self, x):
        """Neural network forward pass with improved numerical stability"""
        # Increase epsilon for activation functions
        act_eps = 0.01  # Increased from self.eps
        
        # Add clipping for inputs to prevent extreme values
        x_clipped = ca.fmin(ca.fmax(x, -10), 10)
        
        # Layer computations with improved stability
        h1_pre = ca.mtimes(ca.DM(self.w1), x_clipped)
        h1 = ca.tanh(h1_pre + act_eps)
        
        h2_pre = ca.mtimes(ca.DM(self.w2), h1)
        h2 = ca.tanh(h2_pre + act_eps)
        
        y_raw = ca.mtimes(ca.DM(self.w3), h2)
        
        # Clip outputs to reasonable acceleration ranges
        y_clipped = ca.fmin(ca.fmax(y_raw, -3), 3)
        
        return y_clipped

    def load_nn_model(self, model_path):
        """Load the pytorch model weights"""
        try:
            model = torch.load(model_path, map_location='cpu')
            self.w1 = model.hidden1.weight.detach().numpy()
            self.w2 = model.hidden2.weight.detach().numpy()
            self.w3 = model.predict.weight.detach().numpy()
            print("Model loaded successfully")
            print(f"Weight shapes: w1={self.w1.shape}, w2={self.w2.shape}, w3={self.w3.shape}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def visualize_prediction(self, solution, waypoints, world):
        """Visualize MPC prediction and waypoints with status"""
        # Add solver status visualization
        status_text = f"MPC Status: {self.solve_status} (iter {self.solve_iter})"
        world.debug.draw_string(
            carla.Location(x=5, y=5, z=2),
            status_text,
            color=carla.Color(255, 255, 255),
            life_time=0.1
        )
        
        # Draw predicted trajectory
        try:
            for i in range(self.horizon):
                idx = self.nx + i * (self.nx + self.nu + self.nx)  # Account for initial state and slack variables
                
                # Skip visualization if we have NaN values
                if np.isnan(solution[idx]) or np.isnan(solution[idx + 1]):
                    continue
                    
                current = carla.Location(x=solution[idx], y=solution[idx + 1], z=1.0)
                
                if i < self.horizon - 1:
                    next_idx = self.nx + (i + 1) * (self.nx + self.nu + self.nx)
                    
                    # Skip if next point has NaN
                    if np.isnan(solution[next_idx]) or np.isnan(solution[next_idx + 1]):
                        continue
                        
                    next_point = carla.Location(x=solution[next_idx], 
                                              y=solution[next_idx + 1], z=1.0)
                    
                    # Draw prediction in green
                    world.debug.draw_line(
                        current, next_point,
                        thickness=0.1,
                        color=carla.Color(0, 255, 0),
                        life_time=0.1
                    )
                    
                    # Draw velocity vector (with NaN check)
                    velocity = solution[idx + 2]  # v component of state
                    heading = solution[idx + 3]  # phi component of state
                    if not np.isnan(velocity) and not np.isnan(heading):
                        vel_vector_end = carla.Location(
                            x=current.x + velocity * np.cos(heading),
                            y=current.y + velocity * np.sin(heading),
                            z=1.0
                        )
                        world.debug.draw_arrow(
                            current, vel_vector_end,
                            thickness=0.1,
                            arrow_size=0.1,
                            color=carla.Color(255, 0, 0),
                            life_time=0.1
                        )
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Draw waypoints with enhanced visualization
        try:
            for i, wp in enumerate(waypoints):
                if i >= self.horizon:
                    break
                    
                # Color coding for waypoints
                if i == 0:
                    color = carla.Color(255, 0, 0)  # Red (current target)
                    size = 0.2  # Larger size for current target
                elif i == len(waypoints) - 1:
                    color = carla.Color(255, 255, 0)  # Yellow (final)
                    size = 0.15
                else:
                    color = carla.Color(0, 0, 255)  # Blue (future)
                    size = 0.1
                
                # Draw waypoint
                loc = wp.transform.location
                world.debug.draw_point(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0),
                    size=size,
                    color=color,
                    life_time=0.1
                )
                
                # Draw waypoint connections
                if i < len(waypoints) - 1:
                    next_wp = waypoints[i + 1].transform.location
                    world.debug.draw_line(
                        carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0),
                        carla.Location(x=next_wp.x, y=next_wp.y, z=next_wp.z + 1.0),
                        thickness=0.05,
                        color=carla.Color(255, 255, 255),  # White
                        life_time=0.1
                    )
                
                # Draw waypoint orientation
                orientation_end = carla.Location(
                    x=loc.x + np.cos(wp.transform.rotation.yaw * np.pi / 180.0),
                    y=loc.y + np.sin(wp.transform.rotation.yaw * np.pi / 180.0),
                    z=loc.z + 1.0
                )
                world.debug.draw_arrow(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z + 1.0),
                    orientation_end,
                    thickness=0.05,
                    arrow_size=0.1,
                    color=color,
                    life_time=0.1
                )
        except Exception as e:
            print(f"Waypoint visualization error: {e}")
        
        # Draw additional information
        try:
            if self.nx < len(solution) and self.nx + 2 < len(solution):
                info_text = [
                    f"Prediction Horizon: {self.horizon}",
                    f"Current Speed: {solution[2]:.2f} m/s",
                    f"Steering: {solution[self.nx]:.2f}",
                    f"Throttle: {solution[self.nx + 1]:.2f}",
                    f"Brake: {solution[self.nx + 2]:.2f}"
                ]
                
                for i, text in enumerate(info_text):
                    world.debug.draw_string(
                        carla.Location(x=5, y=5 + (i + 1) * 0.5, z=2),
                        text,
                        color=carla.Color(255, 255, 255),
                        life_time=0.1
                    )
        except Exception as e:
            print(f"Info visualization error: {e}")