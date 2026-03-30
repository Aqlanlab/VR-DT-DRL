#!/usr/bin/env python3
"""
Enhanced Robot Controller for UR3 System.

Integrates with Webots hardware bridge and provides robust motion planning.
Handles coordinate mismatches between simulated and real environments,
linear descent calculations, and maintains an optimal "tucked" posture.
"""

import numpy as np
import time
import yaml
import math
import logging
import sys
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from math import pi, sin, cos, acos, atan2, sqrt

try:
    import rospy
    import actionlib
    from geometry_msgs.msg import Pose, JointState
    from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Bool
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available, operating in standalone Simulation mode.")

try:
    from scipy.spatial.transform import Rotation as Rot
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    class MockRotation:
        @staticmethod
        def from_euler(seq, angles): return MockRotation()
        @staticmethod
        def from_matrix(matrix): return MockRotation()
        def as_matrix(self): return np.eye(3)
        def as_euler(self, seq): return [0, 0, 0]
    Rot = MockRotation

class UR3KinematicsController:
    """
    Controller for Webots UR3 and Real Hardware.
    """

    # Shared "elbow-up" home poses configuration.
    _HOME_JOINTS_SIM = [
        0.0,
        math.radians(-105.18),
        math.radians( 102.93),
        math.radians( -87.75),
        math.radians( -90.05),
        3.0,
    ]
    
    _HOME_JOINTS_REAL = [
        math.pi / 2,
        math.radians(-105.18),
        math.radians( 102.93),
        math.radians( -87.75),
        math.radians( -90.05),
        math.pi,
    ]

    @classmethod
    def get_home_joints(cls, simulation: bool = True) -> list:
        """Returns the appropriate home joint angles for sim or real robot."""
        return list(cls._HOME_JOINTS_SIM if simulation else cls._HOME_JOINTS_REAL)

    HOME_JOINTS = _HOME_JOINTS_SIM

    def __init__(self, config_path: str = "config/robot_config.yaml", 
                 simulation: bool = True, 
                 robot_instance: Any = None,
                 webots_bridge: Any = None):
                 
        self.logger = logging.getLogger('UR3Controller')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.is_sim = simulation
        self.webots_bridge = webots_bridge
        self.gripper: Optional['GripperController'] = None
        
        self.d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]
        self.a = [0, -0.24365, -0.21325, 0, 0, 0]
        self.alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]

        self.joints_state = [0.0] * 6
        self.motors = []

        if self.is_sim and robot_instance:
            self._setup_webots_hardware(robot_instance)
        
        if ROS_AVAILABLE:
            self._setup_ros_interface()

    def _setup_webots_hardware(self, robot_instance):
        motor_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
                       "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        for name in motor_names:
            motor = robot_instance.getDevice(name)
            if motor:
                motor.setPosition(float('inf')) 
                motor.setVelocity(1.0)
                self.motors.append(motor)
            else:
                self.logger.error(f"Motor {name} NOT FOUND!")

    def _setup_ros_interface(self):
        self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self._joint_state_callback)

    def _joint_state_callback(self, msg):
        if len(msg.position) >= 6:
            self.joints_state = list(msg.position[:6])

    def _wait_step(self, duration: float):
        """Steps the simulation for a deterministic number of simulation intervals."""
        if self.webots_bridge:
            n = max(1, int(duration * 80))
            for _ in range(n):
                self.webots_bridge.step()
        else:
            time.sleep(duration)

    def _axis_angle_to_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = axis / np.linalg.norm(axis)
        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)

    def transform_webots_to_ur3(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Converts Webots world coordinates to UR3 base frame (Simulation)."""
        base_pos_world = np.array([-0.685, 0.372, 0.47235])
        axis = np.array([-0.57735, -0.57735, -0.577351])
        angle = 2.0944
        R_wb = self._axis_angle_to_rotation(axis, angle)

        P_object_world = np.array([x, y, z])
        P_diff_world = P_object_world - base_pos_world
        P_local = R_wb.T @ P_diff_world
        return P_local[0], P_local[1], P_local[2]

    def transform_real_to_ur3(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Converts world coordinates to UR3 base frame for real hardware."""
        base_pos_world = np.array([-0.685, 0.372, 0.47235])
        axis_wb = np.array([-0.57735, -0.57735, -0.577351])
        R_wb = self._axis_angle_to_rotation(axis_wb, 2.0944)
        
        R_Ym90 = self._axis_angle_to_rotation(np.array([0.0, 1.0, 0.0]), -math.pi / 2)
        R_real = R_Ym90 @ R_wb

        P_diff = np.array([x, y, z]) - base_pos_world
        P_local = R_real.T @ P_diff
        return P_local[0], P_local[1], P_local[2]

    def _solve_ik_analytical(self, T_target: np.ndarray) -> List[List[float]]:
        solutions = []
        try:
            P_tcp = T_target[0:3, 3]
            Z_tool = T_target[0:3, 2]
            
            d6 = self.d[5]
            P_wc = P_tcp - d6 * Z_tool

            x_wc, y_wc = P_wc[0], P_wc[1]
            theta1 = atan2(y_wc, x_wc)
            
            d1 = self.d[0]
            d5 = self.d[4]
            r_wc = sqrt(x_wc**2 + y_wc**2) 
            d4 = self.d[3]
            if abs(r_wc) < d4: 
                return []
                
            r_arm = sqrt(r_wc**2 - d4**2) 
            h_arm = P_wc[2] - d1
            dist_s_w = sqrt(r_arm**2 + h_arm**2)
            
            a2, a3 = abs(self.a[1]), abs(self.a[2])
            
            max_dist = (a2 + a3) - 0.001 
            if dist_s_w > max_dist:
                dist_s_w = max_dist

            cos_theta3 = (dist_s_w**2 - a2**2 - a3**2) / (2 * a2 * a3)
            cos_theta3 = max(-1.0, min(1.0, cos_theta3))
            
            elbow_candidates = [-acos(cos_theta3), acos(cos_theta3)]

            for theta3 in elbow_candidates:
                alpha = atan2(h_arm, r_arm)
                beta = atan2(a3 * sin(theta3), a2 + a3 * cos(theta3))
                theta2 = -(alpha + beta)

                yaw = atan2(T_target[1, 0], T_target[0, 0])
                theta4 = -(pi/2) - theta2 - theta3
                theta5 = -pi/2 
                theta6_raw = yaw - theta1
                theta6 = ((theta6_raw + pi) % (2 * pi)) - pi

                solutions.append([theta1, theta2, theta3, theta4, theta5, theta6])

            return solutions
        except Exception:
            return []

    def move_linear_path(self, start_pose: List[float], end_pose: List[float], 
                     steps: int = 50, step_duration: float = 0.12) -> bool:
        """Executes a linear movement path enforcing shoulder backward bias."""
        p_start = np.array(start_pose[:3])
        p_end   = np.array(end_pose[:3])
        current_joints = np.array(self.joints_state)
        
        R_down = np.array([[ 0, -1,  0], 
                           [-1,  0,  0], 
                           [ 0,  0, -1]])
        
        yaw = end_pose[3] if len(end_pose) > 3 else 0.0
        cy, sy = cos(yaw), sin(yaw)
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_target = R_yaw @ R_down

        for i in range(1, steps + 1):
            t = i / float(steps)
            p_current = p_start + (p_end - p_start) * t
            
            T = np.eye(4)
            T[0:3, 3] = p_current
            T[0:3, 0:3] = R_target
            
            solutions = self._solve_ik_analytical(T)
            
            if not solutions: return False

            best_sol = None
            min_score = float('inf')
            
            for sol in solutions:
                sol_arr = np.array(sol)
                dist = np.linalg.norm(sol_arr - current_joints)
    
                shoulder_val = sol[1]
                penalty = 1000.0 if shoulder_val > -0.5 else 0.0
    
                wrist3_delta = abs(sol[5] - current_joints[5])
                if wrist3_delta > math.pi:
                    wrist3_delta = 2 * math.pi - wrist3_delta
                wrist3_penalty = 50.0 * wrist3_delta
    
                score = dist + penalty + wrist3_penalty
                if score < min_score:
                    min_score = score
                    best_sol = sol
            
            if best_sol:
                current_joints = np.array(best_sol)
                self.move_to_joint_positions(best_sol, duration=step_duration, wait=True)
            else:
                return False
                
        return True

    def move_to_joint_positions(self, target_joints, duration=3.0, wait=True):
        if not self.motors: return False
    
        normalized = list(target_joints)
        for i in range(len(normalized)):
            if i < len(self.joints_state):
                delta = normalized[i] - self.joints_state[i]
                while delta > math.pi:
                    delta -= 2 * math.pi
                    normalized[i] -= 2 * math.pi
                while delta < -math.pi:
                    delta += 2 * math.pi
                    normalized[i] += 2 * math.pi

        safe_duration = max(0.001, float(duration))
        
        for i, motor in enumerate(self.motors):
            if i < len(normalized):
                distance = abs(normalized[i] - self.joints_state[i])
                speed = distance / safe_duration
                speed = max(0.01, min(speed, 3.14))
                
                motor.setVelocity(speed)
                motor.setPosition(normalized[i])
                
        self.joints_state = normalized
        if wait: 
            self._wait_step(duration)
        return True

    def move_to_pose(self, target_pose: List[float], duration: float = 3.0, wait: bool = True) -> bool:
        """Move the robot to a target Cartesian pose."""
        ik_x, ik_y, ik_z = target_pose[0], target_pose[1], target_pose[2]
        yaw = target_pose[3] if len(target_pose) > 3 else 0.0

        R_down = np.array([[ 0, -1,  0], 
                           [-1,  0,  0], 
                           [ 0,  0, -1]])
        cy, sy = math.cos(yaw), math.sin(yaw)
        R_yaw = np.array([[cy, -sy, 0], 
                          [sy,  cy, 0], 
                          [ 0,   0, 1]])
        R_target = R_yaw @ R_down

        T = np.eye(4)
        T[0:3, 3] = [ik_x, ik_y, ik_z]
        T[0:3, 0:3] = R_target

        solutions = self._solve_ik_analytical(T)
    
        if solutions:
            current_joints = np.array(self.joints_state)
            best_sol = None
            min_score = float('inf')
        
            for sol in solutions:
                sol_arr = np.array(sol)
                joint_dist = np.linalg.norm(sol_arr - current_joints)
                
                shoulder_val = sol[1]
                penalty = 1000.0 if shoulder_val > -0.5 else 0.0
            
                wrist3_delta = abs(sol[5] - self.joints_state[5])
                if wrist3_delta > math.pi:
                    wrist3_delta = 2 * math.pi - wrist3_delta
                wrist3_penalty = 50.0 * wrist3_delta
            
                score = joint_dist + penalty + wrist3_penalty
                if score < min_score:
                    min_score = score
                    best_sol = sol
        
            return self.move_to_joint_positions(best_sol, duration, wait)
    
        return False

    def move_joints_linear(self, target_joints: List[float], 
                        steps: int = 30, 
                        step_duration: float = 0.15) -> bool:
        """Interpolates directly in joint space to prevent mid-motion IK snapping."""
        if not self.motors:
            return False

        start_joints = np.array(self.joints_state)
        end_joints   = np.array(target_joints)

        for i in range(len(end_joints)):
            delta = end_joints[i] - start_joints[i]
            while delta > math.pi:
                delta -= 2 * math.pi
                end_joints[i] -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi
                end_joints[i] += 2 * math.pi

        for step in range(1, steps + 1):
            t = step / float(steps)
            t_smooth = (1 - math.cos(t * math.pi)) / 2.0
            interp = start_joints + t_smooth * (end_joints - start_joints)
            self.move_to_joint_positions(interp.tolist(), duration=step_duration, wait=True)

        return True

    def home_position(self):
        """Moves the arm to the standard elbow-up home position."""
        return self.move_to_joint_positions(
            self.get_home_joints(simulation=self.is_sim), duration=2.0)

    def execute_grasp(self, pose: List[float]) -> bool:
        """Executes a full grasp sequence (hover -> descend -> close -> lift)."""
        w_x, w_y, w_z = pose[0], pose[1], pose[2]
        yaw = pose[5] if len(pose) > 5 else (pose[3] if len(pose) > 3 else 0.0)
        ik_x, ik_y, ik_z = self.transform_webots_to_ur3(w_x, w_y, w_z)

        r = math.sqrt(ik_x**2 + ik_y**2)
        MAX_SAFE_RADIUS = 0.43
        if r > MAX_SAFE_RADIUS:
            scale = MAX_SAFE_RADIUS / r
            ik_x *= scale
            ik_y *= scale

        target_z = ik_z
        PLATFORM_Z = 0.068
        FLOOR_MARGIN = 0.035
        target_z = max(target_z, PLATFORM_Z + FLOOR_MARGIN)

        grasp_z  = target_z + 0.129
        hover_z  = grasp_z  + 0.11
        safe_z   = 0.40

        base_yaw_offset = 0.0
        try:
            sup = self.webots_bridge.supervisor
            if hasattr(sup, 'supervisor'):
                sup = sup.supervisor
            robot_node = sup.getFromDef("UR3")
            if robot_node:
                rot_field = robot_node.getField("rotation").getSFRotation()
                axis_y = rot_field[1] if len(rot_field) == 4 else 1.0
                base_yaw_offset = rot_field[3] * np.sign(axis_y)
        except Exception as e:
            self.logger.warning(f"Could not read dynamic base rotation: {e}")

        GRIPPER_MOUNT_OFFSET = math.pi / 4 - (math.pi / 16) + math.radians(11.2)
        ik_yaw = -(yaw - base_yaw_offset + GRIPPER_MOUNT_OFFSET)

        if self.gripper:
            self.gripper.open_gripper()
            self._wait_step(0.8)

        R_down = np.array([[ 0, -1,  0],
                       [-1,  0,  0],
                       [ 0,  0, -1]])
        cy, sy = math.cos(ik_yaw), math.sin(ik_yaw)
        R_yaw  = np.array([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]])
        R_target = R_yaw @ R_down

        def solve_waypoint(x, y, z):
            T = np.eye(4)
            T[0:3, 3]   = [x, y, z]
            T[0:3, 0:3] = R_target
            solutions = self._solve_ik_analytical(T)
            if not solutions:
                return None
            current = np.array(self.joints_state)
            best, best_score = None, float('inf')
            for sol in solutions:
                sol_arr = np.array(sol)
                dist = np.linalg.norm(sol_arr - current)
                shoulder_penalty = 1000.0 if sol[1] > -0.5 else 0.0
                wrist_delta = abs(sol[5] - current[5])
                if wrist_delta > math.pi:
                    wrist_delta = 2 * math.pi - wrist_delta
                wrist_penalty = 50.0 * wrist_delta
                score = dist + shoulder_penalty + wrist_penalty
                if score < best_score:
                    best_score = score
                    best = sol
            return best

        joints_safe  = solve_waypoint(ik_x, ik_y, safe_z)
        joints_hover = solve_waypoint(ik_x, ik_y, hover_z)
        joints_grasp = solve_waypoint(ik_x, ik_y, grasp_z)

        if not all([joints_safe, joints_hover, joints_grasp]):
            self.logger.warning("Could not solve IK for one or more waypoints.")
            return False

        self.move_to_joint_positions(joints_hover, duration=1.0)

        HOME_BASE_ANGLE  = 0.0
        HOME_WRIST_ANGLE = 3.0
        base_delta = joints_hover[0] - HOME_BASE_ANGLE
        wrist_compensated = HOME_WRIST_ANGLE + base_delta

        joints_hover_compensated = list(joints_hover)
        joints_hover_compensated[5] = wrist_compensated
        self.move_to_joint_positions(joints_hover_compensated, duration=0.8)

        N_STEPS = 15
        for i in range(1, N_STEPS + 1):
            t = i / float(N_STEPS)
            interp_z = hover_z + t * (grasp_z - hover_z)
            j = solve_waypoint(ik_x, ik_y, interp_z)
            if j is None:
                self.logger.warning("IK failed during descent.")
                return False
            j = list(j)
            j[5] = wrist_compensated
            self.move_to_joint_positions(j, duration=0.05, wait=True)
        self._wait_step(0.2)

        try:
            closest = 9999.0
            if self.webots_bridge and hasattr(self.webots_bridge, 'supervisor'):
                sup = self.webots_bridge.supervisor
                if hasattr(sup, 'supervisor'):
                    sup = sup.supervisor
                o_node = sup.getFromDef("TARGET_OBJECT")
                g_node = sup.getFromDef("GRIPPER_MAIN")
                if g_node is None: g_node = sup.getFromDef("UR3e")
                if g_node is None: g_node = sup.getFromDef("UR3")
                if o_node and g_node:
                    import numpy as _np
                    o_pos = _np.array(o_node.getPosition())
                    g_pos = _np.array(g_node.getPosition())
                    closest = float(_np.linalg.norm(o_pos - g_pos))
            self._closest_approach_dist = closest
        except Exception:
            self._closest_approach_dist = 9999.0

        if self.gripper:
            self.gripper.close_gripper()
            self._wait_step(1.0)

        self.move_to_joint_positions(joints_hover, duration=1.0)

        return True


class GripperController:
    """
    Controller for the robotic gripper, managing both simulated and physical states.
    """
    
    def __init__(self, robot_instance=None):
        self.finger1 = None
        self.finger2 = None
        self.is_closed = False
        
        if robot_instance:
            self.finger1 = robot_instance.getDevice("finger1")
            self.finger2 = robot_instance.getDevice("finger2")
            
            if self.finger1 and self.finger2:
                self.finger1.setVelocity(0.15) 
                self.finger2.setVelocity(0.15)
                self.finger1.setAvailableForce(10.0)
                self.finger2.setAvailableForce(10.0)
            else:
                logging.warning("Gripper fingers not found. Check device definitions.")

        if ROS_AVAILABLE:
            self.gripper_pub = rospy.Publisher('/gripper/command', Bool, queue_size=1)

    def open_gripper(self, force: float = 50.0):
        """Opens the gripper fingers."""
        self.is_closed = False
        if self.finger1: self.finger1.setPosition(0.0)
        if self.finger2: self.finger2.setPosition(0.0)
        if ROS_AVAILABLE:
            self.gripper_pub.publish(Bool(data=False))

    def close_gripper(self):
        """Closes the gripper fingers."""
        self.is_closed = True
        if self.finger1: self.finger1.setPosition(0.024)
        if self.finger2: self.finger2.setPosition(0.024)
        if ROS_AVAILABLE:
            self.gripper_pub.publish(Bool(data=True))


class MotionPlanner:
    """High-level motion planning and coordination abstraction."""
    
    def __init__(self, robot_controller: UR3KinematicsController):
        self.robot = robot_controller
        self.logger = logging.getLogger('MotionPlanner')

    def plan_and_execute_grasp(self, pose: List[float]) -> bool:
        """Executes the grasp sequence via the main robot controller."""
        return self.robot.execute_grasp(pose)


def create_robot_system(config_path: str = "config.yaml", 
                        simulation: bool = True, 
                        webots_bridge: Any = None) -> Tuple[UR3KinematicsController, GripperController, MotionPlanner]:
    """Factory function to initialize and link the controller, gripper, and planner."""
    robot_instance = None
    if webots_bridge:
        if hasattr(webots_bridge, 'supervisor'):
            if hasattr(webots_bridge.supervisor, 'supervisor'):
                robot_instance = webots_bridge.supervisor.supervisor
            else:
                robot_instance = webots_bridge.supervisor

    robot_controller = UR3KinematicsController(
        config_path=config_path, 
        simulation=simulation, 
        robot_instance=robot_instance,
        webots_bridge=webots_bridge
    )
    
    gripper_controller = GripperController(robot_instance=robot_instance)
    robot_controller.gripper = gripper_controller
    motion_planner = MotionPlanner(robot_controller)
    
    return robot_controller, gripper_controller, motion_planner


if __name__ == "__main__":
    try:
        if ROS_AVAILABLE:
            rospy.init_node('ur3_controller_node', anonymous=True)
        
        from webots_bridge import WebotsBridge
        bridge = WebotsBridge(simulation=True)
        
        ur3, gripper, planner = create_robot_system("config.yaml", True, bridge)
        print("Controller ready. Press Ctrl+C to exit.")
        
        rate = rospy.Rate(30) if ROS_AVAILABLE else None
        
        while not (rospy.is_shutdown() if ROS_AVAILABLE else False):
            bridge.step()
            if rate: rate.sleep()
            else: time.sleep(0.033)

    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Critical Error: {e}")