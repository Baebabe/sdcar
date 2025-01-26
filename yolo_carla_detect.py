import glob
import os
import sys
import shutil
from pathlib import Path

# Modified CARLA path for 0.9.8
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import time
import torch
import carla
import weakref
import random
import cv2

# Add YOLOv5 to path
yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# Now import YOLOv5 modules
try:
    from models.experimental import attempt_load
    from utils.general import non_max_suppression, scale_coords
    from utils.torch_utils import select_device
except ImportError:
    print("Error importing YOLOv5 modules. Make sure YOLOv5 is properly installed.")
    sys.exit(1)
try:
    import pygame
    from pygame.locals import (K_ESCAPE, K_SPACE, K_a, K_UP, K_DOWN,
                             K_LEFT, K_RIGHT, K_d, K_s, K_w, K_m)
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 640
VIEW_HEIGHT = 480
VIEW_FOV = 90

# At the top of your file, after imports
def check_yolo_setup():
    """
    Check if YOLOv5 is properly set up
    """
    yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5s.pt')
    
    if not os.path.exists(yolov5_path):
        raise RuntimeError(f"YOLOv5 directory not found at {yolov5_path}. Please clone YOLOv5 repository.")
    
    if not os.path.exists(weights_path):
        raise RuntimeError(f"YOLOv5 weights not found at {weights_path}. Please download yolov5s.pt.")
    

def scale_coords(img_shape, coords, shapes):
    """
    Rescale coords (xyxy) from img_shape to shapes
    """
    if torch.is_tensor(coords):
        coords = coords.clone()
    else:
        coords = torch.tensor(coords)
        
    gain = min(img_shape[0] / shapes[0], img_shape[1] / shapes[1])  # gain = old / new
    pad = (img_shape[1] - shapes[1] * gain) / 2, (img_shape[0] - shapes[0] * gain) / 2  # wh padding
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(min=0, max=shapes[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(min=0, max=shapes[0])  # y1, y2
    
    return coords

def load_model():
    try:
        print("Loading YOLOv5 model...")
        
        # Add YOLOv5 directory to Python path
        yolov5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5s.pt')
        
        if yolov5_path not in sys.path:
            sys.path.append(yolov5_path)
            print(f"Added YOLOv5 path: {yolov5_path}")
            
        print(f"Loading weights from: {weights_path}")
        
        # Import needed modules
        from models.experimental import attempt_load
        from utils.general import non_max_suppression, scale_coords
        
        # Load model with correct parameters
        model = attempt_load(weights_path, device=device)  # Changed from map_location to device
        
        # Set model parameters
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # NMS IoU threshold
        model.classes = None  # All classes
        model.eval()  # Set in evaluation mode
        
        # Print model information
        print(f"Model loaded successfully!")
        if hasattr(model, 'names'):
            print(f"Available classes: {model.names}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None

class BasicSynchronousClient(object):
    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None
        self.display = None
        self.image = None
        self.capture = True
        self.log = False
        self.pose = []

        # Change to FPP camera transform
        self.camera_transform = carla.Transform(
            carla.Location(x=1.6, z=1.7),  # Position slightly above and forward from car's center
            carla.Rotation(pitch=0)         # Looking straight ahead
        )

        self.camera_options = {
            'width': '640',
            'height': '480',
            'fov': '90' 
        }

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        """
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')

        # Set camera attributes for better visibility
        camera_bp.set_attribute('image_size_x', self.camera_options['width'])
        camera_bp.set_attribute('image_size_y', self.camera_options['height'])
        camera_bp.set_attribute('fov', self.camera_options['fov'])

        # Attach camera to car with better positioning
        self.camera = self.world.spawn_actor(camera_bp, 
                                           self.camera_transform, 
                                           attach_to=self.car)

        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))
    def set_camera_view(self, view_type='fpp'):
        """
        Change camera view between different perspectives
        """
        if self.camera:
            self.camera.destroy()
        
        if view_type == 'fpp':
            # First Person Perspective
            self.camera_transform = carla.Transform(
                carla.Location(x=1.6, z=1.7),
                carla.Rotation(pitch=0)
            )
        elif view_type == 'tpp':
            # Third Person Perspective
            self.camera_transform = carla.Transform(
                carla.Location(x=-5.5, z=2.8),
                carla.Rotation(pitch=-15)
            )
        elif view_type == 'hood':
            # Hood view
            self.camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),
                carla.Rotation(pitch=0)
            )
        
        # Setup camera with new transform
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', self.camera_options['width'])
        camera_bp.set_attribute('image_size_y', self.camera_options['height'])
        camera_bp.set_attribute('fov', self.camera_options['fov'])
        
        self.camera = self.world.spawn_actor(
            camera_bp,
            self.camera_transform,
            attach_to=self.car
        )
        
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))
    def camera_blueprint(self):
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controlled and additional traffic.
        """
        try:
            # Spawn the player vehicle (Tesla Model 3)
            car_bp = self.world.get_blueprint_library().filter('model3')[0]
            location = random.choice(self.world.get_map().get_spawn_points())
            self.car = self.world.spawn_actor(car_bp, location)

            # Spawn some AI-controlled vehicles
            vehicle_bps = self.world.get_blueprint_library().filter('vehicle.*')
            spawn_points = self.world.get_map().get_spawn_points()

            # Spawn 20 vehicles randomly around the map
            for _ in range(20):
                try:
                    bp = random.choice(vehicle_bps)
                    spawn_point = random.choice(spawn_points)
                    vehicle = self.world.spawn_actor(bp, spawn_point)
                    vehicle.set_autopilot(True)  # Enable autopilot
                except:
                    continue

            # Spawn pedestrians
            self.spawn_pedestrians(150)  # Spawn 50 pedestrians

            print("Vehicles and pedestrians spawned successfully!")

        except Exception as e:
            print(f"Error spawning vehicles and pedestrians: {e}")

    def spawn_pedestrians(self, number_of_pedestrians=150):
        """
        Spawn pedestrians in the world
        """
        try:
            # Get pedestrian blueprints
            walker_bp = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            spawn_points = []

            # Get spawn points
            for i in range(number_of_pedestrians):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)

            # Spawn the walker objects
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(walker_bp)
                batch.append(carla.command.SpawnActor(walker_bp, spawn_point))

            # Apply the batch
            results = self.client.apply_batch_sync(batch, True)

            # Get walker controllers
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            batch = []
            for result in results:
                if result.error:
                    continue
                batch.append(carla.command.SpawnActor(walker_controller_bp, 
                                                    carla.Transform(), 
                                                    result.actor_id))

            # Apply the batch
            results = self.client.apply_batch_sync(batch, True)

            # Start walker controllers
            for result in results:
                if result.error:
                    continue
                actor = self.world.get_actor(result.actor_id)
                actor.start()
                actor.go_to_location(self.world.get_random_location_from_navigation())

            print(f"Spawned {len(results)} pedestrians")

        except Exception as e:
            print(f"Error spawning pedestrians: {e}")

    def control(self, car):
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        # Add view switching controls
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.set_camera_view('fpp')
                elif event.key == pygame.K_2:
                    self.set_camera_view('tpp')
                elif event.key == pygame.K_3:
                    self.set_camera_view('hood')

        control = car.get_control()
        control.throttle = 0
        if keys[K_w] or keys[K_UP]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s] or keys[K_DOWN]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a] or keys[K_LEFT]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d] or keys[K_RIGHT]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        self = weak_self()
        if self.capture and self:
            self.image = img
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """
        if self.image is not None:
            try:
                # Convert CARLA raw image to numpy array
                array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (self.image.height, self.image.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]  # BGR to RGB
                array = array.copy()

                try:
                    # Prepare image for YOLO
                    img = cv2.resize(array, (640, 640))
                    img = img.transpose((2, 0, 1))  # HWC to CHW
                    img = np.ascontiguousarray(img)

                    # Convert to torch tensor
                    img = torch.from_numpy(img).to(device)
                    img = img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    with torch.no_grad():
                        pred = model(img)
                        if isinstance(pred, tuple):
                            pred = pred[0]  # Take first element if tuple

                    # NMS (Non-Maximum Suppression)
                    det = non_max_suppression(pred,
                                            conf_thres=0.25,    # Confidence threshold
                                            iou_thres=0.45,     # NMS IOU threshold
                                            classes=None,       # Filter by class
                                            agnostic=False,     # NMS class-agnostic
                                            max_det=300)[0]     # Maximum detections, take first element

                    # Process detections
                    if len(det):  # if detections exist
                        # Rescale boxes from img_size to original size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], array.shape).round()

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            try:
                                # Convert tensor values to Python scalars
                                bbox = [int(x.item()) for x in xyxy]  # Bounding box coordinates
                                conf_val = float(conf.item())         # Confidence value
                                cls_val = int(cls.item())            # Class ID

                                # Get class name
                                class_name = model.names[cls_val] if hasattr(model, 'names') else f'class_{cls_val}'

                                # Draw bounding box
                                cv2.rectangle(array, 
                                            (bbox[0], bbox[1]), 
                                            (bbox[2], bbox[3]), 
                                            (0, 255, 0), 
                                            2)

                                # Create label with class name and confidence
                                label = f'{class_name} {conf_val:.2f}'

                                # Get text size for background rectangle
                                (text_width, text_height), _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                                # Draw filled background rectangle for text
                                cv2.rectangle(array, 
                                            (bbox[0], bbox[1] - text_height - 4), 
                                            (bbox[0] + text_width, bbox[1]), 
                                            (0, 255, 0), 
                                            -1)

                                # Put text
                                cv2.putText(array, 
                                          label,
                                          (bbox[0], bbox[1] - 2),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          0.6,
                                          (0, 0, 0),
                                          2)

                                print(f"Detected: {label} at coordinates {bbox}")

                            except Exception as e:
                                print(f"Error processing individual detection: {e}")
                                continue

                except Exception as e:
                    print(f"Error in detection: {e}")
                    import traceback
                    traceback.print_exc()

                # Convert to pygame surface
                surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                display.blit(surface, (0, 0))

            except Exception as e:
                print(f"Error in image processing: {e}")
    
    def setup_world(self):
        """
        Configure world settings for better visualization
        """
        # Get the world settings
        settings = self.world.get_settings()

        # Configure settings
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # Set weather for better visibility
        weather = carla.WeatherParameters(
            cloudiness=10.0,
            precipitation=0.0,
            sun_altitude_angle=90.0
        )
        self.world.set_weather(weather)


    def game_loop(self):
        try:
            pygame.init()
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()

            # Setup world first
            self.setup_world()

            # Then spawn vehicles and camera
            self.setup_car()
            self.set_camera_view('fpp')  # Start with FPP view

            self.display = pygame.display.set_mode(
                (int(self.camera_options['width']), int(self.camera_options['height'])),
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )

            pygame_clock = pygame.time.Clock()

            while True:
                self.world.tick()
                self.capture = True
                pygame_clock.tick_busy_loop(20)
                self.render(self.display)
                pygame.display.flip()
                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            if self.camera:
                self.camera.destroy()
            if self.car:
                self.car.destroy()
            pygame.quit()
if __name__ == '__main__':
    try:
        # Check YOLOv5 setup first
        check_yolo_setup()
        
        # Initialize device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model
        model = load_model()
        if model is None:
            print("Failed to load model. Exiting...")
            sys.exit(1)
            
        # Create and run client
        client = BasicSynchronousClient()
        client.game_loop()
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print('EXIT')