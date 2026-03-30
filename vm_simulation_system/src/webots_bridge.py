#!/usr/bin/env python3
"""
Enhanced Webots Bridge for UR3 System.

Provides an interface to the Webots simulation environment, handling
initialization, device control, and state queries.
"""

import os
import sys
import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

WEBOTS_HOME = "/opt/webots"
CONTROLLER_BASE = os.path.join(WEBOTS_HOME, "lib", "controller")

py_ver = f"{sys.version_info.major}{sys.version_info.minor}"
python_lib_folder = f"python{py_ver}"
LIB_PATH = os.path.join(CONTROLLER_BASE, python_lib_folder)

if os.path.exists(LIB_PATH):
    if LIB_PATH not in sys.path:
        sys.path.insert(0, LIB_PATH)
        print(f"Found Webots Library for Python {py_ver}: {LIB_PATH}")
else:
    print(f"Auto-detected path {LIB_PATH} does not exist. Attempting fallbacks.")
    for ver in ["python38", "python39", "python36", "python37", "python27"]:
        fallback_path = os.path.join(CONTROLLER_BASE, ver)
        if os.path.exists(fallback_path):
            sys.path.insert(0, fallback_path)
            print(f"Using fallback Webots path: {fallback_path}")
            break

try:
    from controller import Supervisor, Robot
    from scipy.spatial.transform import Rotation as Rot
    WEBOTS_AVAILABLE = True
    print("Webots Controller Module Imported Successfully.")
except ImportError as e:
    WEBOTS_AVAILABLE = False
    print(f"Webots controller unavailable: {e}")
    print("Operating in mock mode.")

try:
    import rospy
    from std_msgs.msg import Int8
    from integrator.msg import BlockPose
    from integrator.srv import SupervisorGrabService, SupervisorPositionService
    from integrator.srv import SimImageCameraService, SimDepthCameraService
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    class MockROS: pass
    BlockPose = MockROS
    Image = MockROS

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

class WebotsSupervisor:
    """Manages scene interactions via the Webots Supervisor API."""
    def __init__(self, simulation: bool = True, world_file: str = "Environmentnewww.wbt", robot_instance=None):
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsSupervisor')
        self.supervisor = robot_instance
        self.number_of_blocks = 5
        self.timestep = 16 
        self.ur3e_position = [0.69, 0.74, 0]
        self.ur3e_rotation = None
        
        if not self.supervisor and not simulation and WEBOTS_AVAILABLE:
            self._init_webots_supervisor()
        elif simulation or not WEBOTS_AVAILABLE:
            self._init_mock_supervisor()
        else:
            self.timestep = int(self.supervisor.getBasicTimeStep())
            self._setup_nodes() 

    def _setup_nodes(self):
        """Initializes pointers to relevant scene nodes."""
        self.ur3e_rotation = Rot.from_rotvec(-(np.pi / 2) * np.array([1.0, 0.0, 0.0]))
        self.blocks = []
        for i in range(self.number_of_blocks):
            block = self.supervisor.getFromDef(f"block{i}")
            if block: self.blocks.append(block)
        self.end_effector = self.supervisor.getFromDef("gps")

    def _init_webots_supervisor(self):
        try:
            self.supervisor = Supervisor()
            self.timestep = int(self.supervisor.getBasicTimeStep())
            self._setup_nodes()
            self.logger.info("Webots supervisor initialized.")
        except Exception as e:
            self.logger.error(f"Failed to init supervisor: {e}")
            self._init_mock_supervisor()
            
    def _init_mock_supervisor(self):
        self.supervisor = None
        self.blocks = []
        self.end_effector = None
        
        for i in range(self.number_of_blocks):
            mock_block = {
                'id': i,
                'position': [np.random.uniform(-0.5, 0.5), 
                           np.random.uniform(-0.5, 0.5),
                           np.random.uniform(0.7, 0.9)],
                'rotation': [0, 0, np.random.uniform(0, 2*np.pi)]
            }
            self.blocks.append(mock_block)
            
        self.logger.info(f"Mock supervisor initialized with {len(self.blocks)} blocks.")
        
    def _init_ros_services(self):
        try:
            if not rospy.get_node_uri():
                rospy.init_node('webots_supervisor', anonymous=True)
                
            self.grab_service = rospy.Service(
                'supervisor_grab_service', 
                SupervisorGrabService, 
                self._handle_grab_request
            )
            
            self.position_service = rospy.Service(
                'supervisor_position_service',
                SupervisorPositionService,
                self._handle_position_request  
            )
            
            self.logger.info("ROS services initialized.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ROS services: {e}")
            
    def step(self) -> bool:
        if self.supervisor:
            return self.supervisor.step(self.timestep) != -1
        time.sleep(self.timestep / 1000.0)  
        return True
            
    def get_block_poses(self) -> List[Dict[str, Any]]:
        """Retrieves current position and orientation of target blocks."""
        block_poses = []
        
        if self.supervisor and hasattr(self.supervisor, 'getFromDef'):
            for i, block in enumerate(self.blocks):
                if block:
                    try:
                        position = block.getPosition()
                        rotation = block.getOrientation()
                        
                        block_poses.append({
                            'id': i,
                            'position': list(position) if position else [0, 0, 0],
                            'rotation': list(rotation) if rotation else [1, 0, 0, 0, 1, 0, 0, 0, 1],
                            'timestamp': time.time()
                        })
                    except Exception as e:
                        self.logger.warning(f"Failed to get pose for block {i}: {e}")
        else:
            for i, block in enumerate(self.blocks):
                if isinstance(block, dict):
                    block_poses.append({
                        'id': i,
                        'position': block['position'],
                        'rotation': block['rotation'] + [1, 0, 0, 0, 1, 0],  
                        'timestamp': time.time()
                    })
                    
        return block_poses
        
    def set_block_pose(self, block_id: int, position: List[float], 
                      rotation: Optional[List[float]] = None) -> bool:
        """Sets the position and rotation of a specified block node."""
        if block_id >= len(self.blocks):
            self.logger.error(f"Block ID {block_id} out of range.")
            return False
            
        if self.supervisor and hasattr(self.supervisor, 'getFromDef'):
            block = self.blocks[block_id]
            if block:
                try:
                    block.getField('translation').setSFVec3f(position)
                    if rotation:
                        block.getField('rotation').setSFRotation(rotation + [1.0])  
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to set block {block_id} pose: {e}")
                    return False
        else:
            if isinstance(self.blocks[block_id], dict):
                self.blocks[block_id]['position'] = position
                if rotation:
                    self.blocks[block_id]['rotation'] = rotation
                return True
                
        return False
        
    def get_robot_state(self) -> Dict[str, Any]:
        """Collects the simulated robot state and end effector orientation."""
        robot_state = {
            'position': self.ur3e_position.copy(),
            'rotation': [0, 0, 0],
            'joint_angles': [0.0] * 6,
            'end_effector_pose': [0, 0, 0, 0, 0, 0],
            'timestamp': time.time()
        }
        
        if self.supervisor and self.end_effector:
            try:
                ee_pos = self.end_effector.getPosition()
                if ee_pos:
                    robot_state['end_effector_pose'][:3] = list(ee_pos)
                    
                ee_rot = self.end_effector.getOrientation()
                if ee_rot:
                    rot_matrix = np.array(ee_rot).reshape(3, 3)
                    if WEBOTS_AVAILABLE:
                        euler = Rot.from_matrix(rot_matrix).as_euler('xyz')
                        robot_state['end_effector_pose'][3:] = list(euler)
                        
            except Exception as e:
                self.logger.warning(f"Failed to get robot state: {e}")
                
        return robot_state
        
    def reset_simulation(self) -> bool:
        if self.supervisor:
            try:
                self.supervisor.simulationReset()
                return True
            except Exception as e:
                self.logger.error(f"Failed to reset simulation: {e}")
                return False
        else:
            for block in self.blocks:
                if isinstance(block, dict):
                    block['position'] = [
                        np.random.uniform(-0.5, 0.5),
                        np.random.uniform(-0.5, 0.5), 
                        np.random.uniform(0.7, 0.9)
                    ]
                    block['rotation'] = [0, 0, np.random.uniform(0, 2*np.pi)]
            return True
            
    def _handle_grab_request(self, request):
        return True
        
    def _handle_position_request(self, request):
        return self.get_robot_state()


class WebotsCamera:
    """Manages virtual camera devices connected to the Webots simulation."""
    OUTPUT_WIDTH  = 640
    OUTPUT_HEIGHT = 360

    def __init__(self, simulation: bool = True, robot_instance=None):
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsCamera')
        self.robot = robot_instance 
        
        self.timestep = 4
        self.image_width  = 1280
        self.image_height = 720

        if self.robot and not simulation:
            self._setup_devices()
        else:
            self._init_mock_camera()

    def _setup_devices(self):
        self.camera = self.robot.getDevice('realsense_color')
        self.depth_camera = self.robot.getDevice('realsense_range')
        if self.camera:
            self.camera.enable(self.timestep)
            self.image_width = self.camera.getWidth()
            self.image_height = self.camera.getHeight()
            self.logger.info(f"Webots RGB camera native resolution: {self.image_width}x{self.image_height}")
        if self.depth_camera:
            self.depth_camera.enable(self.timestep)

    def _init_mock_camera(self):
        self.camera = None
        self.depth_camera = None
        
    def _init_ros_services(self):
        try:
            if ROS_AVAILABLE:
                if not rospy.get_node_uri():
                    rospy.init_node('webots_camera', anonymous=True)
                    
                self.bridge = CvBridge()
                
                self.image_service = rospy.Service(
                    'image_camera_service',
                    SimImageCameraService,
                    self._handle_image_request
                )
                
                self.depth_service = rospy.Service(
                    'depth_camera_service', 
                    SimDepthCameraService,
                    self._handle_depth_request
                )
                
                self.logger.info("Camera ROS services initialized.")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize camera ROS services: {e}")
            
    def capture_rgb_image(self) -> Optional[np.ndarray]:
        if self.camera and WEBOTS_AVAILABLE:
            try:
                image_data = self.camera.getImageArray()
                if image_data:
                    image = np.array(image_data, dtype=np.uint8)
                    image = np.rot90(image, k=3)
                    image = np.fliplr(image)  
                    if OPENCV_AVAILABLE:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if image.shape[1] != self.OUTPUT_WIDTH or image.shape[0] != self.OUTPUT_HEIGHT:
                        image = cv2.resize(image, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT),
                                           interpolation=cv2.INTER_AREA)
                    return image
            except Exception as e:
                self.logger.error(f"Failed to capture RGB image: {e}")
        return np.random.randint(0, 255, (self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)

    def capture_depth_image(self) -> Optional[np.ndarray]:
        if self.depth_camera and WEBOTS_AVAILABLE:
            try:
                depth_data = self.depth_camera.getRangeImageArray()
                if depth_data:
                    depth = np.array(depth_data, dtype=np.float32)
                    depth = np.rot90(depth, k=3)
                    depth = np.fliplr(depth)  
                    if depth.shape[1] != self.OUTPUT_WIDTH or depth.shape[0] != self.OUTPUT_HEIGHT:
                        depth = cv2.resize(depth, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT),
                                           interpolation=cv2.INTER_LINEAR)
                    return depth
            except Exception as e:
                self.logger.error(f"Failed to capture depth image: {e}")
        return np.random.uniform(0.1, 2.0, (self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH)).astype(np.float32)
        
    def capture_rgbd(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rgb_image = self.capture_rgb_image()
        depth_image = self.capture_depth_image()
        return rgb_image, depth_image
        
    def _handle_image_request(self, request):
        rgb_image = self.capture_rgb_image()
        if rgb_image is not None and ROS_AVAILABLE:
            try:
                ros_image = self.bridge.cv2_to_imgmsg(rgb_image, "rgb8")
                return ros_image
            except Exception as e:
                self.logger.error(f"Failed to convert image to ROS message: {e}")
        return None
        
    def _handle_depth_request(self, request):
        depth_image = self.capture_depth_image()
        if depth_image is not None and ROS_AVAILABLE:
            try:
                ros_depth = self.bridge.cv2_to_imgmsg(depth_image, "32FC1")
                return ros_depth
            except Exception as e:
                self.logger.error(f"Failed to convert depth to ROS message: {e}")
        return None


class WebotsBridge:
    """Primary entry point initializing and unifying Webots control structures."""
    def __init__(self, simulation: bool = True, world_file: str = "Environmentnewww.wbt"):
        self.simulation = simulation
        self.logger = logging.getLogger('WebotsBridge')
        
        self.shared_robot = None
        if not simulation and WEBOTS_AVAILABLE:
            try:
                self.shared_robot = Supervisor()
                self.logger.info("Shared Webots Supervisor created.")
            except Exception as e:
                self.logger.error(f"Could not create Supervisor: {e}")

        self.supervisor = WebotsSupervisor(simulation, world_file, robot_instance=self.shared_robot)
        self.camera = WebotsCamera(simulation, robot_instance=self.shared_robot)
        
        self.logger.info(f"Webots bridge initialized (simulation={simulation})")

    def step(self) -> bool:
        return self.supervisor.step()
 
    def get_block_poses(self) -> List[Dict[str, Any]]:
        return self.supervisor.get_block_poses()
        
    def get_robot_state(self) -> Dict[str, Any]:
        return self.supervisor.get_robot_state()
        
    def capture_images(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.camera.capture_rgbd()
        
    def reset_simulation(self) -> bool:
        return self.supervisor.reset_simulation()
        
    def set_block_pose(self, block_id: int, position: List[float], 
                      rotation: Optional[List[float]] = None) -> bool:
        return self.supervisor.set_block_pose(block_id, position, rotation)


def create_webots_bridge(config: Optional[Dict[str, Any]] = None,
                        simulation: bool = True) -> WebotsBridge:
    """Factory configuration function for initializing a WebotsBridge."""
    world_file = "Environmentnewww.wbt"
    if config and 'world_file' in config:
        world_file = config['world_file']
        
    return WebotsBridge(simulation=simulation, world_file=world_file)