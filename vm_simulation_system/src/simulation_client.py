#!/usr/bin/env python3
"""
VM Simulation Client for UR3 System.

Handles communication between the GPU Inference server, the simulated Webots environment,
and real hardware when configured. Manages training episodes, the curriculum progression,
and domain randomization.
"""

import socket
import json
import numpy as np
import math
import cv2
import time
import threading
import yaml
import argparse
import base64
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

try:
    import roslib; roslib.load_manifest('robotiq_2f_gripper_control')
    from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as RobotiqOutput
    from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_input  as RobotiqInput
    ROBOTIQ_AVAILABLE = True
except Exception:
    ROBOTIQ_AVAILABLE = False

try:
    import actionlib
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectoryPoint
    ACTIONLIB_AVAILABLE = True
except ImportError:
    ACTIONLIB_AVAILABLE = False

try:
    import rospy
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float32MultiArray, Bool, Empty
    from geometry_msgs.msg import Pose, PoseStamped
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from cv_bridge import CvBridge, CvBridgeError
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    class CvBridge: pass
    class Image: pass
    class JointState: pass
    class Float32MultiArray: pass
    class Bool: pass
    class Empty: pass
    class Pose: pass
    class PoseStamped: pass
    class MockRospy:
        def init_node(self, *args, **kwargs): pass
        def loginfo(self, msg): print(f"[INFO] {msg}")
        def logwarn(self, msg): print(f"[WARN] {msg}")
        def logerr(self, msg): print(f"[ERROR] {msg}")
        def is_shutdown(self): return False
        def Rate(self, hz): return self
        def sleep(self): time.sleep(0.1)
        def Subscriber(self, *args, **kwargs): pass
        def Publisher(self, *args, **kwargs):
            class MockPub:
                def publish(self, *args): pass
            return MockPub()
        class ROSInterruptException(Exception): pass
    if 'rospy' not in locals():
        rospy = MockRospy()

from enhanced_robot_controller import create_robot_system
from enhanced_camera_handler import EnhancedCameraHandler
from webots_bridge import WebotsBridge


class CurriculumManager:
    """
    Performance-gated training curriculum manager. 
    Advances simulation difficulty based on successful grasps by the model.
    """

    PLATFORM_CENTER_X  = -0.67
    PLATFORM_CENTER_Z  = 0.85065
    PLATFORM_HALF_SIZE_X = 0.1475   
    PLATFORM_HALF_SIZE_Z = 0.0925  

    PHASE_CONFIG = [
        (0.000, 0.000,  60,  0.85, 20),
        (0.005, 0.015, 200,  0.75, 40),
        (0.015, 0.035, 250,  0.65, 50),
        (0.035, 0.070, 300,  0.55, 50),
        (0.070, 0.115, 9999, 0.00, 50),
    ]

    def __init__(self, state_file="config/curriculum_state.json"):
        self.state_file = Path(state_file)
        self.phase = 4
        self.episodes_in_phase = 0
        self.episode = 0
        self.ai_recent_results = deque(maxlen=50)
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                import json
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.phase = state.get('phase', 0)
                self.episodes_in_phase = state.get('episodes_in_phase', 0)
                self.episode = state.get('episode', 0)
                
                ai_window = self.PHASE_CONFIG[self.phase][4]
                self.ai_recent_results = deque(state.get('ai_recent_results', []), maxlen=ai_window)
                print(f"[CURRICULUM] Resumed from Phase {self.phase}, Episode {self.episode}")
            except Exception as e:
                print(f"[CURRICULUM] Could not load state: {e}")

    def _save_state(self):
        try:
            import json
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({
                    'phase': self.phase,
                    'episodes_in_phase': self.episodes_in_phase,
                    'episode': self.episode,
                    'ai_recent_results': list(self.ai_recent_results)
                }, f)
        except Exception:
            pass

    def record_result(self, success: bool, mode: str):
        """Records AI outcomes for evaluating phase advancement."""
        self.episodes_in_phase += 1
        if mode == 'exploit':
            self.ai_recent_results.append(success)
        self._save_state()

    def update(self, episode: int):
        self.episode = episode
        self._save_state()

    def get_ai_success_rate(self) -> float:
        if not self.ai_recent_results:
            return 0.0
        return sum(self.ai_recent_results) / len(self.ai_recent_results)

    def check_phase_advance(self) -> bool:
        """Determines if the success rate threshold is met to advance difficulty."""
        if self.phase >= len(self.PHASE_CONFIG) - 1:
            return False

        r_min, r_max, min_eps, threshold, ai_window = self.PHASE_CONFIG[self.phase]
        self.ai_recent_results = deque(self.ai_recent_results, maxlen=ai_window)

        enough_episodes = self.episodes_in_phase >= min_eps
        ai_attempts     = len(self.ai_recent_results)
        ai_rate         = self.get_ai_success_rate()
        ai_mastered     = ai_attempts >= 20 and ai_rate >= threshold

        if enough_episodes and ai_mastered:
            old_phase = self.phase
            self.phase += 1
            self.episodes_in_phase = 0
            self.ai_recent_results.clear()

            r_new_min, r_new_max, _, new_threshold, new_window = self.PHASE_CONFIG[self.phase]
            print(f"[CURRICULUM] Phase {old_phase} -> {self.phase} | "
                  f"AI mastery: {ai_rate*100:.1f}% over {ai_attempts} attempts")
            print(f"[CURRICULUM] New radius: {r_new_min*100:.1f}-{r_new_max*100:.1f}cm | "
                  f"Next target: {new_threshold*100:.0f}% over {new_window} AI attempts")
            return True
        return False

    def get_spawn_radius(self) -> tuple:
        r_min, r_max, _, _, _ = self.PHASE_CONFIG[self.phase]
        return r_min, r_max

    def get_spawn_position(self) -> tuple:
        r_min, r_max = self.get_spawn_radius()

        cx = self.PLATFORM_CENTER_X
        cz = self.PLATFORM_CENTER_Z
        half_x = 0.1475   
        half_z = 0.0925  

        if self.phase == 4:
            spawn_x = np.random.uniform(cx - half_x, cx + half_x)
            spawn_z = np.random.uniform(cz - half_z, cz + half_z)
        else:
            if r_max < 0.001:
                spawn_x, spawn_z = cx, cz
            else:
                while True:
                    sample_x = np.random.uniform(cx - half_x, cx + half_x)
                    sample_z = np.random.uniform(cz - half_z, cz + half_z)
                    dist = np.sqrt((sample_x - cx)**2 + (sample_z - cz)**2)
                    
                    if r_min <= dist <= r_max:
                        spawn_x, spawn_z = sample_x, sample_z
                        break

        ai_rate = self.get_ai_success_rate()
        ai_window = self.PHASE_CONFIG[self.phase][4]
        print(f"[CURRICULUM] Episode {self.episode} | Phase {self.phase} | "
              f"Spawn: ({spawn_x:.3f}, {spawn_z:.3f}) | "
              f"AI rate: {ai_rate*100:.1f}% ({len(self.ai_recent_results)}/{ai_window} attempts)")

        return (spawn_x, None, spawn_z)

    def _get_phase_number(self) -> int:
        return self.phase


class SimulationClient:
    """Main client coordinating data transfer between simulation/hardware and server."""

    def __init__(self, config_path: str = "config/network_config.yaml",
                 mode: str = 'inference', real_robot: bool = False):
        self.mode = mode
        self.real_robot = real_robot
        self.config = self._load_config(config_path)

        if ROS_AVAILABLE:
            rospy.init_node('ur3_simulation_client', anonymous=True)
            rospy.loginfo(f"UR3 Client started (Mode: {mode}, Real robot: {real_robot})")

        self.host_socket = None
        self.connected = False
        self.connection_lock = threading.Lock()
        self.bridge = CvBridge() if ROS_AVAILABLE else None
        
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_rgb_b64 = None
        self.latest_depth_b64 = None
        self.latest_joint_states = {'names': [], 'positions': [0]*6}
        
        self.curriculum = CurriculumManager()
        self.episode_count = self.curriculum.episode
        self.episode_active = False
        self.last_grasp_mode = 'explore'

        self.inference_mode = 'normal'
        self.cycle_episodes_per_phase = 10   
        self.fixed_phase = 0    
        self._cycle_phase = 0    
        self._cycle_count_in_phase = 0    
        self._nan_reset_pending = False  

        if real_robot:
            self.webots_bridge = None
            self._init_realsense()
            self._init_real_robot_motion()
            self._init_robotiq_gripper()
        else:
            self.webots_bridge = WebotsBridge(simulation=False)

            self.robot_controller, self.gripper_controller, self.motion_planner = \
                create_robot_system(
                    config_path="config/robot_config.yaml",
                    simulation=True,
                    webots_bridge=self.webots_bridge
                )

            self.camera_handler = EnhancedCameraHandler(
                config_path="config/camera_config.yaml",
                simulation=True,
                camera_type="simulation",
                webots_bridge=self.webots_bridge
            )

            if ROS_AVAILABLE:
                self._setup_ros_interface()

    def _init_realsense(self):
        if not REALSENSE_AVAILABLE:
            rospy.logerr("[REAL] pyrealsense2 not installed. Required for real mode.")
            raise RuntimeError("pyrealsense2 required for --real mode")

        self._rs_pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 360, rs.format.z16,  30)
        profile = self._rs_pipeline.start(cfg)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._rs_depth_scale = depth_sensor.get_depth_scale()
        self._rs_align = rs.align(rs.stream.color)

        self._rs_spatial = rs.spatial_filter()
        self._rs_temporal = rs.temporal_filter()
        self._rs_holefill = rs.hole_filling_filter()

        for _ in range(30):
            self._rs_pipeline.wait_for_frames()
        rospy.loginfo("[REAL] RealSense D455 ready")

    def _capture_realsense(self):
        frames = self._rs_pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self._rs_align.process(frames)
        c_frame = aligned.get_color_frame()
        d_frame = aligned.get_depth_frame()
        if not c_frame or not d_frame:
            return None, None
            
        d_frame = self._rs_spatial.process(d_frame)
        d_frame = self._rs_temporal.process(d_frame)
        d_frame = self._rs_holefill.process(d_frame)
        
        rgb = np.asanyarray(c_frame.get_data())
        depth = np.asanyarray(d_frame.get_data()).astype(np.float32) * self._rs_depth_scale
        return rgb, depth

    def _init_real_robot_motion(self):
        self.robot_controller, self.gripper_controller, self.motion_planner = \
            create_robot_system(
                config_path="config/robot_config.yaml",
                simulation=False,   
                webots_bridge=None
            )

        self._traj_client = None
        if not ACTIONLIB_AVAILABLE:
            rospy.logwarn("[REAL] actionlib not available - motion will be stubbed")
            return

        UR3_ACTION = '/scaled_pos_joint_traj_controller/follow_joint_trajectory'
        self._traj_client = actionlib.SimpleActionClient(UR3_ACTION, FollowJointTrajectoryAction)
        connected = self._traj_client.wait_for_server(timeout=rospy.Duration(10.0))
        if connected:
            rospy.loginfo("[REAL] UR3e trajectory action server connected")
        else:
            rospy.logwarn("[REAL] Could not connect to trajectory server.")
            self._traj_client = None

        rospy.Subscriber('/joint_states', JointState, self._real_joint_state_cb)

    def _real_joint_state_cb(self, msg):
        NAMES = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                 'wrist_1_joint','wrist_2_joint','wrist_3_joint']
        positions = dict(zip(msg.name, msg.position))
        joints = [positions.get(n, 0.0) for n in NAMES]
        self.robot_controller.joints_state = joints

    def _send_real_joints(self, joints: List[float], duration: float):
        NAMES = ['shoulder_pan_joint','shoulder_lift_joint','elbow_joint',
                 'wrist_1_joint','wrist_2_joint','wrist_3_joint']

        wait_timeout = duration * 25.0 + 15.0

        if self._traj_client is None:
            rospy.logwarn(f"[REAL STUB] Move joints {[round(j,3) for j in joints]} (duration {duration:.1f}s)")
            rospy.sleep(duration)
            return True

        goal = FollowJointTrajectoryGoal()
        from trajectory_msgs.msg import JointTrajectory
        import actionlib
        traj = JointTrajectory()
        traj.joint_names = NAMES
        pt = JointTrajectoryPoint()
        pt.positions = joints
        pt.velocities = [0.0] * 6
        pt.time_from_start = rospy.Duration(duration)
        traj.points = [pt]
        goal.trajectory = traj

        self._traj_client.send_goal(goal)
        finished = self._traj_client.wait_for_result(timeout=rospy.Duration(wait_timeout))

        if not finished:
            rospy.logerr(f"[REAL] Trajectory timed out. Cancelling goal.")
            self._traj_client.cancel_goal()
            return False

        state = self._traj_client.get_state()
        if state != actionlib.GoalStatus.SUCCEEDED:
            rospy.logwarn(f"[REAL] Trajectory ended with action state {state}.")
            return False

        return True

    def _init_robotiq_gripper(self):
        self._gripper_ready = False
        self._gripper_status = None
        self._gripper_pub = None

        if not ROBOTIQ_AVAILABLE or not ROS_AVAILABLE:
            rospy.logwarn("[REAL] Robotiq package not found. Gripper stubbed.")
            return

        self._gripper_pub = rospy.Publisher(
            'Robotiq2FGripperRobotOutput',
            RobotiqOutput.Robotiq2FGripper_robot_output,
            queue_size=10,
            latch=True)   

        rospy.Subscriber(
            'Robotiq2FGripperRobotInput',
            RobotiqInput.Robotiq2FGripper_robot_input,
            self._gripper_status_cb)

        rospy.sleep(2.0)
        rospy.loginfo("[REAL] Gripper publisher registered")

        rospy.loginfo("[REAL] Resetting gripper...")
        cmd = RobotiqOutput.Robotiq2FGripper_robot_output()
        cmd.rACT = 0
        for _ in range(10):          
            self._gripper_pub.publish(cmd)
            rospy.sleep(0.1)
        rospy.sleep(0.5)             

        rospy.loginfo("[REAL] Activating gripper...")
        cmd = RobotiqOutput.Robotiq2FGripper_robot_output()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rSP = 255
        cmd.rFR = 150
        for _ in range(10):          
            self._gripper_pub.publish(cmd)
            rospy.sleep(0.1)

        deadline = time.time() + 8.0
        while not self._gripper_ready and time.time() < deadline:
            rospy.sleep(0.1)

        if self._gripper_ready:
            rospy.loginfo("[REAL] Robotiq 2F activated.")
        else:
            rospy.logwarn("[REAL] Gripper activation timed out.")
            self._gripper_ready = True
    
    def _gripper_status_cb(self, msg):
        self._gripper_status = msg
        if msg.gACT == 1 and msg.gSTA == 3:
            self._gripper_ready = True

    def _gripper_open(self):
        if not getattr(self, '_gripper_ready', False) or self._gripper_pub is None:
            rospy.loginfo("[GRIPPER STUB] Open")
            return
        cmd = RobotiqOutput.Robotiq2FGripper_robot_output()
        cmd.rACT = 1; cmd.rGTO = 1; cmd.rSP = 255; cmd.rFR = 150; cmd.rPR = 0
        self._gripper_pub.publish(cmd)
        rospy.sleep(1.0)

    def _gripper_close(self):
        if not getattr(self, '_gripper_ready', False) or self._gripper_pub is None:
            rospy.loginfo("[GRIPPER STUB] Close")
            return
        cmd = RobotiqOutput.Robotiq2FGripper_robot_output()
        cmd.rACT = 1; cmd.rGTO = 1; cmd.rSP = 255; cmd.rFR = 150; cmd.rPR = 255
        self._gripper_pub.publish(cmd)
        rospy.sleep(1.5)

    def _gripper_reactivate(self):
        if not getattr(self, '_gripper_ready', False) or self._gripper_pub is None:
            return
        cmd = RobotiqOutput.Robotiq2FGripper_robot_output()
        cmd.rACT = 1; cmd.rGTO = 1; cmd.rSP = 255; cmd.rFR = 150
        self._gripper_pub.publish(cmd)
        rospy.sleep(0.5)

    def _gripper_grasped(self) -> bool:
        if self._gripper_status is None:
            return False
        return self._gripper_status.gOBJ in (1, 2)

    def _execute_real_grasp(self, prediction: Dict):
        import math as _m

        raw_pose = list(prediction['pose'])
        rospy.loginfo(f"[REAL] Network output: {raw_pose}")

        raw_pose[0] = raw_pose[0]   
        raw_pose[2] = raw_pose[2]  
        rospy.loginfo(f"[REAL] Rescaled x={raw_pose[0]:.4f}  z={raw_pose[2]:.4f}")

        pose = raw_pose.copy()
        pose[0] = float(np.clip(pose[0], -0.862, -0.578))   
        pose[1] = float(np.clip(pose[1],  0.420,  0.460))   
        pose[2] = float(np.clip(pose[2],  0.65,  0.972))   
        if any(abs(raw_pose[i] - pose[i]) > 0.001 for i in range(3)):
            rospy.loginfo(f"[REAL CLAMP] {raw_pose[:3]} -> {pose[:3]}")

        x, y, z = pose[0], pose[1], pose[2]
        yaw = pose[5]

        ik_x, ik_y, _ = self.robot_controller.transform_real_to_ur3(x, y, z)

        PLATFORM_Z = 0.068   
        FLOOR_MARGIN = 0.050   
        GRIPPER_OFF = 0.129   
        HOVER_OFF = 0.08    

        target_z = PLATFORM_Z + FLOOR_MARGIN        
        grasp_z  = target_z   + GRIPPER_OFF         
        hover_z  = grasp_z    + HOVER_OFF           
        safe_z   = hover_z    + 0.05                

        rospy.loginfo(f"[REAL] Height: platform={PLATFORM_Z:.3f} grasp={grasp_z:.3f} hover={hover_z:.3f} safe={safe_z:.3f}")

        WRIST_ANGLE = _m.pi   

        def _wrap(a):
            return min(abs(a), 2 * _m.pi - abs(a))

        def solve_wp(tx, ty, tz):
            R_down = np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]])
            cy, sy = _m.cos(yaw), _m.sin(yaw)
            R_yaw = np.array([[cy, -sy, 0], [sy,  cy, 0], [ 0,   0, 1]])
            T = np.eye(4)
            T[:3, 3] = [tx, ty, tz]
            T[:3, :3] = R_yaw @ R_down

            sols = self.robot_controller._solve_ik_analytical(T)
            if not sols:
                return None

            J0_SAFE_MIN = math.radians(35)    
            J0_SAFE_MAX = math.radians(95)    
            J0_FIND_MIN = math.radians(30)    
            J0_FIND_MAX = math.radians(130)   
            valid = [s for s in sols if J0_FIND_MIN < s[0] < J0_FIND_MAX and s[1] < 0.0 and s[2] > 0.0]
            
            if not valid:
                rospy.logerr(f"[REAL] No elbow-down solution in range.")
                return None

            cur = np.array(self.robot_controller.joints_state)
            best, best_s = None, float('inf')
            for s in valid:
                sc = (np.linalg.norm(np.array(s) - cur)
                      + 20.0 * _wrap(s[1] - cur[1])   
                      + 20.0 * _wrap(s[2] - cur[2])   
                      + 50.0 * _wrap(s[3] - cur[3])   
                      + 50.0 * _wrap(s[4] - cur[4]))  
                if sc < best_s:
                    best_s, best = sc, s

            best = list(best)
            best[0] = float(np.clip(best[0], J0_SAFE_MIN, J0_SAFE_MAX))
            best[5] = WRIST_ANGLE   
            return best

        rospy.loginfo(f"[REAL] IK target: ({ik_x:.3f}, {ik_y:.3f})")

        j_safe = solve_wp(ik_x, ik_y, safe_z)
        j_hover = solve_wp(ik_x, ik_y, hover_z)
        j_grasp = solve_wp(ik_x, ik_y, grasp_z)

        if not all([j_safe, j_hover, j_grasp]):
            rospy.logerr("[REAL] IK failed — skipping grasp")
            return

        HOME = list(self.robot_controller.get_home_joints(simulation=False))
        HOME[5] = WRIST_ANGLE   

        JOINT_SPEED = 0.5   
        MIN_DURATION = 1.5   

        def duration_for(j_from, j_to):
            dist = np.linalg.norm(np.array(j_to) - np.array(j_from))
            return float(max(MIN_DURATION, dist / JOINT_SPEED))

        cur = list(self.robot_controller.joints_state)

        rospy.loginfo("[REAL] Re-activating gripper...")
        self._gripper_reactivate()
        self._gripper_open()

        rospy.loginfo("[REAL] Moving to Hover")
        self._send_real_joints(j_hover, duration_for(j_safe, j_hover))

        HOME_BASE_ANGLE = _m.pi / 2   
        HOME_WRIST_ANGLE = _m.pi       
        base_delta = j_hover[0] - HOME_BASE_ANGLE
        wrist_compensated = HOME_WRIST_ANGLE + base_delta

        j_hover_comp = list(j_hover)
        j_hover_comp[5] = wrist_compensated
        rospy.loginfo(f"[REAL] Hover wrist compensated")
        self._send_real_joints(j_hover_comp, duration_for(j_hover, j_hover_comp))

        j_grasp_comp = list(j_grasp)
        j_grasp_comp[5] = wrist_compensated
        rospy.loginfo("[REAL] Descending")
        self._send_real_joints(j_grasp_comp, duration_for(j_hover_comp, j_grasp_comp))
        rospy.sleep(0.5)   

        rospy.loginfo("[REAL] Closing gripper")
        self._gripper_close()
        rospy.sleep(1.0)   

        success = self._gripper_grasped()
        status_msg = "SUCCESS" if success else "FAIL"
        rospy.loginfo(f"[REAL] Grasp {status_msg}")

        rospy.loginfo("[REAL] Lifting straight up")
        self._send_real_joints(j_safe, duration_for(j_grasp_comp, j_safe))

        rospy.loginfo("[REAL] Returning home")
        self._send_real_joints(HOME, duration_for(j_safe, HOME))

        rospy.sleep(1.0)
        self._gripper_open()
        rospy.loginfo("[REAL] Cycle complete")

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {'network': {'host_ip': '192.168.1.133', 'host_port': 8888}}

    def _setup_ros_interface(self):
        self.rgb_sub = rospy.Subscriber('/camera/image_raw', Image, self._rgb_callback)
        self.depth_sub = rospy.Subscriber('/camera/depth/image_raw', Image, self._depth_callback)
        self.joint_state_sub = rospy.Subscriber('/ur3/joint_states', JointState, self._joint_state_callback)

    def _rgb_callback(self, msg):
        self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def _depth_callback(self, msg):
        self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")

    def _joint_state_callback(self, msg):
        self.latest_joint_states = {'positions': list(msg.position)}

    def connect_to_host(self) -> bool:
        host_ip = self.config['network']['host_ip']
        host_port = self.config['network']['host_port']
        try:
            with self.connection_lock:
                self.host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.host_socket.settimeout(30)
                self.host_socket.connect((host_ip, host_port))
                self.connected = True
            rospy.loginfo(f"Connected to GPU server at {host_ip}:{host_port}")
            return True
        except Exception as e:
            self.connected = False
            return False

    def _send_camera_data_to_host(self):
        try:
            if self.latest_rgb_image is None:
                return

            data_dir = Path("~/catkin_ws/src/vm_simulation_system/data").expanduser()
            data_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = data_dir / "latest_camera_view.jpg"
            cv2.imwrite(str(save_path), self.latest_rgb_image)

            _, rgb_enc = cv2.imencode('.jpg', self.latest_rgb_image)
            self.latest_rgb_b64 = base64.b64encode(rgb_enc).decode('utf-8')

            if self.latest_depth_image is not None:
                depth_mm = (self.latest_depth_image * 1000).astype(np.uint16)
                h, w = depth_mm.shape
                header = np.array([h, w], dtype=np.uint32).tobytes()
                self.latest_depth_b64 = base64.b64encode(header + depth_mm.tobytes()).decode('utf-8')
            else:
                self.latest_depth_b64 = ""

            payload  = {
                'type':   'camera_data',
                'data':   {'rgb': self.latest_rgb_b64, 'depth': self.latest_depth_b64},
                'mode':   self.mode,
                'source': 'real' if self.real_robot else 'simulation'
            }
            response = self._send_message_to_host(payload)
            if response and response.get('type') == 'grasp_prediction':
                if self.real_robot:
                    self._execute_real_grasp(response)
                else:
                    self._execute_grasp_prediction(response)
        except Exception as e:
            rospy.logerr(f"Camera Send Error: {e}")

    def _send_message_to_host(self, message: Dict) -> Optional[Dict]:
        if not self.connected:
            return None
        try:
            with self.connection_lock:
                data = json.dumps(message).encode('utf-8')
                self.host_socket.sendall(len(data).to_bytes(4, byteorder='big'))
                self.host_socket.sendall(data)

                header = self.host_socket.recv(4)
                if not header:
                    return None
                resp_size = int.from_bytes(header, byteorder='big')

                resp_data = b''
                while len(resp_data) < resp_size:
                    chunk = self.host_socket.recv(min(resp_size - len(resp_data), 4096))
                    if not chunk:
                        break
                    resp_data += chunk
                return json.loads(resp_data.decode('utf-8'))
        except:
            self.connected = False
            return None

    def _calculate_shaped_reward(self, success: bool, closest_dist: float) -> float:
        WRIST_OFFSET = 0.060
        CUTOFF_DIST = 0.20
        DECAY = 15.0

        finger_dist = max(0.0, closest_dist - WRIST_OFFSET)

        if success:
            reward = 1.0
        elif closest_dist > CUTOFF_DIST:
            reward = 0.0
        else:
            reward = 0.8 * math.exp(-DECAY * finger_dist)

        print(f"[REWARD] ClosestDist: {closest_dist:.4f}m | FingerDist: {finger_dist:.4f}m | Success: {success} | Reward: {reward:.4f}")
        return float(reward)

    def _generate_guided_random_grasp(self) -> List[float]:
        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            duck_node = supervisor.getFromDef("TARGET_OBJECT")
            if not duck_node:
                rospy.logwarn("Target Object not found!")
                return [-0.685, 0.44, 0.55, 3.14, 0, 0]

            d_pos = np.array(duck_node.getPosition())

            if np.any(np.isnan(d_pos)):
                if not self._nan_reset_pending:
                    print("[NaN GUARD] Object position returned NaN. Triggering one-time sim reset.")
                    self._nan_reset_pending = True
                    self._reset_simulation_for_nan()
                else:
                    print("[NaN GUARD] NaN still present after reset; skipping episode.")
                return [-0.685, 0.44, 0.55, 3.14, 0.0, 0.0]

            ROBOT_BASE_X = -0.685
            ROBOT_BASE_Z =  0.47235

            dx = d_pos[0] - ROBOT_BASE_X
            dz = d_pos[2] - ROBOT_BASE_Z

            dist_to_obj = np.sqrt(dx**2 + dz**2)
            angle_to_obj = np.arctan2(dz, dx)

            REACH_OFFSET = -0.105
            SHIFT_OFFSET =  0.1  
            HEIGHT_OFFSET =  0.02

            final_dist = dist_to_obj + REACH_OFFSET

            target_x = ROBOT_BASE_X + (final_dist * np.cos(angle_to_obj)) - (SHIFT_OFFSET * np.sin(angle_to_obj))
            target_z = ROBOT_BASE_Z + (final_dist * np.sin(angle_to_obj)) + (SHIFT_OFFSET * np.cos(angle_to_obj))
            target_y = d_pos[1] + HEIGHT_OFFSET

            x = target_x + np.random.uniform(-0.005, 0.005)
            y = target_y + np.random.uniform(0.005,  0.005)
            z = target_z + np.random.uniform(-0.005, 0.005)

            yaw = angle_to_obj

            print(f"[CLIENT] Target: {x:.3f}, {y:.3f}, {z:.3f} | Reach Adj: {REACH_OFFSET}")
            return [float(x), float(y), float(z), 3.14, 0.0, float(yaw)]

        except Exception as e:
            rospy.logerr(f"Guided Random Error: {e}")
            return [-0.685, 0.44, 0.55, 0, 0, 0]

    def _execute_grasp_prediction(self, prediction: Dict):
        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            duck_node = supervisor.getFromDef("TARGET_OBJECT")
            initial_y = 0.0
            node_found = False
            self.last_grasp_mode = prediction.get('mode', 'exploit')

            if duck_node:
                initial_y = duck_node.getPosition()[1]
                node_found = True

                if math.isnan(initial_y) or any(math.isnan(v) for v in duck_node.getPosition()):
                    if not self._nan_reset_pending:
                        print("[NaN GUARD] NaN detected before grasp. Triggering one-time sim reset.")
                        self._nan_reset_pending = True
                        self._reset_simulation_for_nan()
                    else:
                        print("[NaN GUARD] NaN still present after reset; skipping episode.")
                    self.end_current_episode(False)
                    self.robot_controller.home_position()
                    time.sleep(2.0)
                    self.start_new_episode()
                    return

            current_state = {'rgb': self.latest_rgb_b64, 'depth': self.latest_depth_b64}
            mode = prediction.get('mode', 'unknown')

            if mode == 'explore':
                pose = self._generate_guided_random_grasp()
            else:
                raw_pose = list(prediction['pose'])
                print(f"[AI PREDICTION] Network output: {raw_pose}")
                pose = raw_pose.copy()
                pose[0] = float(np.clip(pose[0], -0.90, -0.50))   
                pose[1] = float(np.clip(pose[1], 0.483, 0.490))   
                pose[2] = float(np.clip(pose[2],  0.70,  1.05))   
                if raw_pose[0] != pose[0] or raw_pose[1] != pose[1] or raw_pose[2] != pose[2]:
                    print(f"[AI CLAMP] Clamped: [{raw_pose[0]:.3f},{raw_pose[1]:.3f},{raw_pose[2]:.3f}] -> [{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}]")

            self.robot_controller._closest_approach_dist = 9999.0
            self.robot_controller.execute_grasp(pose)

            closest_dist = getattr(self.robot_controller, '_closest_approach_dist', 9999.0)

            success = False
            if node_found:
                final_y = duck_node.getPosition()[1]
                lift_delta = final_y - initial_y
                REQUIRED_LIFT = 0.023
                success = lift_delta > REQUIRED_LIFT
                status = "SUCCESS" if success else "FAIL"
                print(f"[RESULT] {status}. Lifted {lift_delta:.4f}m (Threshold: {REQUIRED_LIFT}m)")
                if not math.isnan(lift_delta):
                    self._nan_reset_pending = False
            else:
                print("[RESULT] FAIL. Object not found.")

            reward = self._calculate_shaped_reward(success, closest_dist)

            self.webots_bridge.step()
            self.camera_handler.update_from_webots()

            if self.camera_handler.current_rgb_frame is not None and self.mode != 'inference':
                _, r_enc = cv2.imencode('.jpg', self.camera_handler.current_rgb_frame)
                depth_mm = (self.camera_handler.current_depth_frame * 1000).astype(np.uint16)
                h, w = depth_mm.shape
                header = np.array([h, w], dtype=np.uint32).tobytes()
                d_b64 = base64.b64encode(header + depth_mm.tobytes()).decode('utf-8')

                next_state = {
                    'rgb':   base64.b64encode(r_enc).decode('utf-8'),
                    'depth': d_b64
                }

                network_action = [
                    pose[0],   
                    pose[1],   
                    pose[2],   
                    3.14,      
                    0.0,       
                    pose[5],   
                ]

                obj_pos = [0.0, 0.0]
                if duck_node:
                    dp = duck_node.getPosition()
                    obj_pos = [float(dp[0]), float(dp[2])]  

                self._send_message_to_host({
                    'type': 'training_data',
                    'source': 'real' if self.real_robot else 'simulation',
                    'data': {
                        'state':      current_state,
                        'action':     network_action,
                        'reward':     reward,
                        'next_state': next_state,
                        'done':       True,
                        'mode':       mode,       
                        'object_pos': obj_pos,    
                    }
                })

            self.end_current_episode(success)
            self.robot_controller.home_position()
            time.sleep(2.0)
            self.start_new_episode()

        except Exception as e:
            rospy.logerr(f"Grasp Execution Error: {e}")

    def _reset_simulation_for_nan(self):
        """Resets the Webots environment to recover from NaN position errors."""
        try:
            print("[NaN GUARD] Resetting simulation to recover lost object...")
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            supervisor.simulationReset()
            time.sleep(1.0)
            self.robot_controller.home_position()
            time.sleep(2.0)

            obj_node = supervisor.getFromDef("TARGET_OBJECT")
            if obj_node:
                cx = CurriculumManager.PLATFORM_CENTER_X
                cz = CurriculumManager.PLATFORM_CENTER_Z
                position_field = obj_node.getField("translation")
                if position_field:
                    position_field.setSFVec3f([cx, 0.461, cz])
                rotation_field = obj_node.getField("rotation")
                if rotation_field:
                    rotation_field.setSFRotation([0.0, 1.0, 0.0, 0.0])
                obj_node.resetPhysics()
                print(f"[NaN GUARD] Object re-spawned at centre ({cx:.3f}, 0.461, {cz:.3f})")
            else:
                print("[NaN GUARD] TARGET_OBJECT not found during NaN recovery.")

        except Exception as e:
            rospy.logerr(f"[NaN GUARD] Reset failed: {e}")

    def _randomize_domain(self):
        """Applies visual randomization (colors, textures, lighting) to bridge the Sim2Real gap."""
        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            import random

            target_node = supervisor.getFromDef("TARGET_OBJECT")
            if target_node:
                r, g, b = random.random(), random.random(), random.random()
                target_node.getField("baseColor").setSFColor([r, g, b])

            floor_mat = supervisor.getFromDef("FLOOR_MATERIAL")           
            platform_mat = supervisor.getFromDef("PLATFORM_MATERIAL")     
            floor_tex = supervisor.getFromDef("PLATFORM_TEXTURE")
            tex_transform = supervisor.getFromDef("PLATFORM_TEX_TRANSFORM")

            if floor_mat and floor_tex and platform_mat:
                import os
                import math
                tex_dir = os.path.expanduser("~/catkin_ws/src/vm_simulation_system/Webots/protos/textures")
                url_field = floor_tex.getField("url")

                tr, tg, tb = random.uniform(0.1, 0.8), random.uniform(0.1, 0.8), random.uniform(0.1, 0.8)
                if floor_mat.getTypeName() == "Material":
                    floor_mat.getField("diffuseColor").setSFColor([tr, tg, tb])
                elif floor_mat.getTypeName() == "PBRAppearance":
                    floor_mat.getField("baseColor").setSFColor([tr, tg, tb])

                if random.random() < 0.25: 
                    plate_images = ["plate1.jpg", "plate2.jpg", "plate3.jpg", "plate4.jpg"] 
                    chosen_img = random.choice(plate_images)
                    img_path = os.path.join(tex_dir, chosen_img)
                    
                    if url_field.getCount() == 0:
                        url_field.insertMFString(0, img_path)
                    else:
                        url_field.setMFString(0, img_path)
                        
                    if tex_transform:
                        rotations = [0.0, math.pi/2, math.pi, 3*math.pi/2]
                        tex_transform.getField("rotation").setSFFloat(random.choice(rotations))
                        
                    if platform_mat.getTypeName() == "Material":
                        platform_mat.getField("diffuseColor").setSFColor([1.0, 1.0, 1.0])
                    elif platform_mat.getTypeName() == "PBRAppearance":
                        platform_mat.getField("baseColor").setSFColor([1.0, 1.0, 1.0])
                else:
                    if url_field.getCount() > 0:
                        url_field.removeMF(0)
                        
                    if tex_transform:
                        tex_transform.getField("rotation").setSFFloat(0.0)
                        
                    pr, pg, pb = random.uniform(0.1, 0.8), random.uniform(0.1, 0.8), random.uniform(0.1, 0.8)
                    
                    if platform_mat.getTypeName() == "Material":
                        platform_mat.getField("diffuseColor").setSFColor([pr, pg, pb])
                    elif platform_mat.getTypeName() == "PBRAppearance":
                        platform_mat.getField("baseColor").setSFColor([pr, pg, pb])

            light_node = supervisor.getFromDef("MAIN_LIGHT")
            if light_node:
                intensity = random.uniform(0.4, 2.5)
                ambient = random.uniform(0.1, 0.8)
                light_node.getField("intensity").setSFFloat(intensity)
                light_node.getField("ambientIntensity").setSFFloat(ambient)

                lr, lg, lb = random.uniform(0.8, 1.0), random.uniform(0.8, 1.0), random.uniform(0.8, 1.0)
                light_node.getField("color").setSFColor([lr, lg, lb])

                if light_node.getTypeName() == "DirectionalLight":
                    dir_x = random.uniform(-0.8, 0.8)
                    dir_y = random.uniform(-1.0, -0.4) 
                    dir_z = random.uniform(-0.8, 0.8)
                    light_node.getField("direction").setSFVec3f([dir_x, dir_y, dir_z])

        except Exception as e:
            print(f"[DOMAIN RAND] Skipping randomization (error): {e}")

    def start_new_episode(self):
        self.episode_count += 1
        self.episode_active = True
        self.curriculum.update(self.episode_count)

        if not self.real_robot:
            self._randomize_domain()

        self._spawn_object_at_curriculum_position()
        self._send_message_to_host({'type': 'episode_start', 'episode': self.episode_count})

    def _spawn_object_at_curriculum_position(self):
        try:
            supervisor = self.webots_bridge.supervisor
            if hasattr(supervisor, 'supervisor'):
                supervisor = supervisor.supervisor

            obj_node = supervisor.getFromDef("TARGET_OBJECT")
            if obj_node is None:
                rospy.logwarn("[CURRICULUM] TARGET_OBJECT not found, skipping spawn.")
                return

            if self.mode == 'inference' and self.inference_mode == 'free':
                pos = obj_node.getPosition()
                print(f"[INFERENCE/free] Object left at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
                return

            if self.mode == 'inference' and self.inference_mode == 'cycle':
                spawn_x, spawn_z = self._get_spawn_for_phase(self._cycle_phase)
            elif self.mode == 'inference' and self.inference_mode == 'phase':
                spawn_x, spawn_z = self._get_spawn_for_phase(self.fixed_phase)
            else:
                spawn_x, _, spawn_z = self.curriculum.get_spawn_position()

            spawn_y = 0.461  
            
            position_field = obj_node.getField("translation")
            if position_field:
                position_field.setSFVec3f([spawn_x, spawn_y, spawn_z])

            rotation_field = obj_node.getField("rotation")
            if rotation_field:
                rotation_field.setSFRotation([0.0, 1.0, 0.0, 0.0])

            obj_node.resetPhysics()

        except Exception as e:
            rospy.logerr(f"[CURRICULUM] Spawn error: {e}")

    def _get_spawn_for_phase(self, phase_index: int) -> tuple:
        cfg = CurriculumManager.PHASE_CONFIG
        phase_index = max(0, min(phase_index, len(cfg) - 1))
        r_min, r_max, _, _, _ = cfg[phase_index]
        cx = CurriculumManager.PLATFORM_CENTER_X
        cz = CurriculumManager.PLATFORM_CENTER_Z
        half_x = CurriculumManager.PLATFORM_HALF_SIZE_X
        half_z = CurriculumManager.PLATFORM_HALF_SIZE_Z

        if r_max < 0.001:
            print(f"[INFERENCE] Phase {phase_index} spawn: ({cx:.3f}, {cz:.3f}) | static centre")
            return (cx, cz)

        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(r_min, r_max)
        sx = np.clip(cx + radius * np.cos(angle), cx - half_x, cx + half_x)
        sz = np.clip(cz + radius * np.sin(angle), cz - half_z, cz + half_z)
        print(f"[INFERENCE] Phase {phase_index} spawn: ({sx:.3f}, {sz:.3f}) | radius {radius*100:.1f}cm")
        return (sx, sz)

    def end_current_episode(self, success):
        self.episode_active = False
        mode = getattr(self, 'last_grasp_mode', 'explore')

        if self.mode != 'inference':
            self.curriculum.record_result(success, mode)
            advanced = self.curriculum.check_phase_advance()
            if advanced:
                print(f"[CURRICULUM] Phase advanced to {self.curriculum.phase}! Resetting epsilon.")
                self._send_message_to_host({
                    'type':  'reset_epsilon',
                    'value': 0.4
                })

        if self.mode == 'inference' and self.inference_mode == 'cycle':
            self._cycle_count_in_phase += 1
            if self._cycle_count_in_phase >= self.cycle_episodes_per_phase:
                self._cycle_count_in_phase = 0
                num_phases = len(CurriculumManager.PHASE_CONFIG)
                self._cycle_phase = (self._cycle_phase + 1) % num_phases
                print(f"[INFERENCE/cycle] Moving to Phase {self._cycle_phase} ({self.cycle_episodes_per_phase} episodes completed)")

        status = "Success" if success else "Fail"
        if self.mode == 'inference':
            print(f"[INFERENCE] {status} Episode {self.episode_count} complete")
        else:
            ai_rate = self.curriculum.get_ai_success_rate()
            ai_attempts = len(self.curriculum.ai_recent_results)
            ai_window = self.curriculum.PHASE_CONFIG[self.curriculum.phase][4]
            print(f"[EPISODE] {status} Ep {self.episode_count} | Phase {self.curriculum.phase} | AI: {ai_rate*100:.1f}% ({ai_attempts}/{ai_window}) | Mode: {mode}")
        
        self._send_message_to_host({'type': 'episode_end', 'success': success})

    def run_simulation_loop(self, max_episodes: int = None):
        if not self.connect_to_host():
            return

        if not self.real_robot:
            self.robot_controller.home_position()
            time.sleep(2.0)
            self.start_new_episode()
        else:
            rospy.loginfo('[REAL] Moving to home position before starting...')
            home = self.robot_controller.get_home_joints(simulation=False)
            self._send_real_joints(home, duration=4.0)
            rospy.sleep(1.0)
            self._gripper_open()
            rospy.loginfo('[REAL] Ready — waiting for GPU server predictions')

        if self.real_robot:
            FLUSH_FRAMES = 10   
            SETTLE_SLEEP = 1.0  

            while not rospy.is_shutdown():
                if max_episodes is not None and self.episode_count > max_episodes:
                    print(f"[CLIENT] Reached {max_episodes} episodes. Stopping.")
                    break

                if not self.connected:
                    break

                rospy.loginfo("[REAL] Flushing camera buffer before capture...")
                for _ in range(FLUSH_FRAMES):
                    self._capture_realsense()
                rospy.sleep(SETTLE_SLEEP)

                rgb, depth = self._capture_realsense()
                if rgb is None:
                    rospy.logwarn("[REAL] Camera returned None — retrying")
                    continue
                self.latest_rgb_image = rgb
                self.latest_depth_image = depth

                rospy.loginfo("[REAL] Captured settled frame — sending to GPU server")
                self._send_camera_data_to_host()

        else:
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                if max_episodes is not None and self.episode_count > max_episodes:
                    print(f"[CLIENT] Reached {max_episodes} episodes. Stopping.")
                    break

                self.webots_bridge.step()
                self.camera_handler.update_from_webots()
                self.latest_rgb_image = self.camera_handler.current_rgb_frame
                self.latest_depth_image = self.camera_handler.current_depth_frame

                if self.latest_rgb_image is not None and self.connected:
                    self._send_camera_data_to_host()
                rate.sleep()


def main():
    parser = argparse.ArgumentParser(
        description='UR3 Simulation Client',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--mode', type=str, default='training',
                        help='training | inference')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Max episodes to run (training). Omit for infinite.')
    parser.add_argument('--real', action='store_true',
                        help='Run on real UR3e. Forces inference mode.')

    inf_group = parser.add_argument_group(
        'Inference sub-modes',
        'These flags only take effect when --mode inference is set.\n'
        'Only one may be used at a time.'
    )
    inf_group.add_argument('--cycle', type=int, default=None, metavar='N',
                           help='Cycle through all curriculum phases, N episodes per phase.')
    inf_group.add_argument('--free', action='store_true',
                           help='No automatic spawning. Place the object manually.')
    inf_group.add_argument('--phase', type=int, default=None, metavar='N',
                           help='Lock to a specific curriculum phase (0–4).')

    args = parser.parse_args()

    mode = 'inference' if args.real else args.mode

    inf_flags = [args.cycle is not None, args.free, args.phase is not None]
    if sum(inf_flags) > 1:
        parser.error("Only one of --cycle, --free, --phase may be used at a time.")
    if any(inf_flags) and mode != 'inference':
        parser.error("--cycle / --free / --phase require --mode inference.")

    client = SimulationClient(mode=mode, real_robot=args.real)

    if mode == 'inference':
        if args.cycle is not None:
            client.inference_mode = 'cycle'
            client.cycle_episodes_per_phase = args.cycle
            num_phases = len(CurriculumManager.PHASE_CONFIG)
            print(f"[INFERENCE] Mode: CYCLE | {args.cycle} episodes x {num_phases} phases")
        elif args.free:
            client.inference_mode = 'free'
            print("[INFERENCE] Mode: FREE | Place the object manually each episode")
        elif args.phase is not None:
            max_phase = len(CurriculumManager.PHASE_CONFIG) - 1
            if not 0 <= args.phase <= max_phase:
                parser.error(f"--phase must be between 0 and {max_phase}.")
            client.inference_mode = 'phase'
            client.fixed_phase = args.phase
            cfg = CurriculumManager.PHASE_CONFIG[args.phase]
            print(f"[INFERENCE] Mode: PHASE {args.phase} | radius {cfg[0]*100:.1f}-{cfg[1]*100:.1f}cm")
        else:
            client.inference_mode = 'normal'
            print("[INFERENCE] Mode: NORMAL | Following curriculum as usual")

    client.run_simulation_loop(max_episodes=None if args.real else args.episodes)


if __name__ == "__main__":
    main()