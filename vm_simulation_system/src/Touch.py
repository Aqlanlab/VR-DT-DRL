#!/usr/bin/env python3
"""
ROS Robot Control Station Launcher
Fits a 1024x600 7-inch touchscreen display.
Launches each ROS command in its own terminal window,
monitors processes for crashes, and alerts when restart is needed.
"""

import sys
import subprocess
import shutil
import os
import time
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QDialog, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QTextCursor

# ─── Configuration ────────────────────────────────────────────────────────────
CONTAINER_NAME       = "frosty_goldberg"
HOST_SERVER_PATH     = "/home/seth/Downloads/UR3e_Hybrid_System/host_gpu_system"
SIM_CLIENT_PATH      = "vm_simulation_system/src"
WORKSPACE_SETUP_PI   = "/root/catkin_ws/install_isolated/setup.bash"
WORKSPACE_SETUP_DESK = "~/catkin_ws/install_isolated/setup.bash"

# ─── Terminal emulator detection ─────────────────────────────────────────────
def find_terminal():
    for t in ["lxterminal", "xterm", "gnome-terminal", "xfce4-terminal", "konsole"]:
        if shutil.which(t):
            return t
    return None

TERMINAL = find_terminal()

# ─── Process definitions ──────────────────────────────────────────────────────
# gpu_server is handled separately (Pi-only, shown above clients)
GPU_SERVER_PROC = {
    "id": "gpu_server",
    "label": "GPU Server",
    "terminal": "T1",
    "group": "server",
    "desktop_cmd": f"cd {HOST_SERVER_PATH} && source venv/bin/activate && python3 src/gpu_server.py",
    "pi_cmd":      f"cd {HOST_SERVER_PATH} && source venv/bin/activate && python3 src/gpu_server.py",
    "watchable": True,
    "note": "Runs host-side GPU server",
}
GPU_SERVER_PROC["cmd"] = GPU_SERVER_PROC["desktop_cmd"]

PROCESSES = [
    {
        "id": "roscore",
        "label": "roscore",
        "terminal": "T2",
        "group": "r1",
        "desktop_cmd": f"source {WORKSPACE_SETUP_DESK} && roscore",
        "pi_cmd":      f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "source {WORKSPACE_SETUP_PI} && roscore"',
        "watchable": False,
        "note": None,
    },
    {
        "id": "r1driver",
        "label": "UR3e Driver R1",
        "terminal": "T3",
        "group": "r1",
        "desktop_cmd": (
            f"source {WORKSPACE_SETUP_DESK} && "
            "roslaunch ur_robot_driver ur3e_bringup.launch "
            "robot_ip:=192.168.1.120 use_tool_communication:=true "
            "tool_voltage:=24 tool_device_name:=/tmp/ttyUR "
            "kinematics_config:=/home/seth/calibration.yaml"
        ),
        "pi_cmd": (
            f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "'
            f"source {WORKSPACE_SETUP_PI} && "
            "roslaunch ur_robot_driver ur3e_bringup.launch "
            "robot_ip:=192.168.1.120 use_tool_communication:=true "
            "tool_voltage:=24 tool_device_name:=/tmp/ttyUR "
            'kinematics_config:=/workspace/calibration.yaml"'
        ),
        "watchable": True,
        "note": "⚠  After launch: activate External Control on Robot 1 pendant",
    },
    {
        "id": "r1grip",
        "label": "Gripper R1",
        "terminal": "T4",
        "group": "r1",
        "desktop_cmd": (
            f"source {WORKSPACE_SETUP_DESK} && ROS_NAMESPACE=ur3e_robot1 "
            "rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py "
            "/tmp/ttyUR __name:=gripper_node"
        ),
        "pi_cmd": (
            f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "source {WORKSPACE_SETUP_PI} && '
            "ROS_NAMESPACE=ur3e_robot1 rosrun robotiq_2f_gripper_control "
            'Robotiq2FGripperRtuNode.py /tmp/ttyUR __name:=gripper_node"'
        ),
        "watchable": True,
        "note": None,
    },
    {
        "id": "r1cam",
        "label": "RealSense R1",
        "terminal": "T5",
        "group": "r1",
        "desktop_cmd": (
            f"source {WORKSPACE_SETUP_DESK} && roslaunch realsense2_camera rs_camera.launch "
            "align_depth:=true initial_reset:=true color_width:=640 color_height:=360 "
            "color_fps:=15 depth_width:=640 depth_height:=360 depth_fps:=15"
        ),
        "pi_cmd": (
            f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "source {WORKSPACE_SETUP_PI} && '
            "roslaunch realsense2_camera rs_camera.launch align_depth:=true "
            "initial_reset:=true color_width:=640 color_height:=360 color_fps:=15 "
            'depth_width:=640 depth_height:=360 depth_fps:=15"'
        ),
        "watchable": True,
        "note": None,
    },
    {
        "id": "r2driver",
        "label": "UR3e Driver R2",
        "terminal": "T6",
        "group": "r2",
        "desktop_cmd": f"source {WORKSPACE_SETUP_DESK} && roslaunch ~/robot2_bringup.launch",
        "pi_cmd":      f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "source {WORKSPACE_SETUP_PI} && roslaunch ~/robot2_bringup.launch"',
        "watchable": True,
        "note": "⚠  After launch: activate External Control on Robot 2 pendant",
    },
    {
        "id": "r2grip",
        "label": "Gripper R2",
        "terminal": "T7",
        "group": "r2",
        "desktop_cmd": (
            f"source {WORKSPACE_SETUP_DESK} && ROS_NAMESPACE=ur3e_robot2 "
            "rosrun robotiq_2f_gripper_control Robotiq2FGripperRtuNode.py "
            "/tmp/ttyUR2 __name:=gripper_node"
        ),
        "pi_cmd": (
            f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "source {WORKSPACE_SETUP_PI} && '
            "ROS_NAMESPACE=ur3e_robot2 rosrun robotiq_2f_gripper_control "
            'Robotiq2FGripperRtuNode.py /tmp/ttyUR2 __name:=gripper_node"'
        ),
        "watchable": True,
        "note": None,
    },
    {
        "id": "r2cam",
        "label": "RealSense R2",
        "terminal": "T8",
        "group": "r2",
        "desktop_cmd": (
            f"source {WORKSPACE_SETUP_DESK} && roslaunch realsense2_camera rs_camera.launch "
            "align_depth:=true initial_reset:=true color_width:=640 color_height:=360 "
            "color_fps:=15 depth_width:=640 depth_height:=360 depth_fps:=15 camera:=camera2"
        ),
        "pi_cmd": (
            f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "source {WORKSPACE_SETUP_PI} && '
            "roslaunch realsense2_camera rs_camera.launch align_depth:=true "
            "initial_reset:=true color_width:=640 color_height:=360 color_fps:=15 "
            'depth_width:=640 depth_height:=360 depth_fps:=15 camera:=camera2"'
        ),
        "watchable": True,
        "note": None,
    },
    {
        "id": "client1",
        "label": "Sim Client R1",
        "terminal": "T9",
        "group": "clients",
        "desktop_cmd": (
            f"source {WORKSPACE_SETUP_DESK} && cd ~/catkin_ws/src/{SIM_CLIENT_PATH} && "
            "python3 simulation_client.py --ros-camera --robot-id 1 --mode inference --free"
        ),
        "pi_cmd": (
            f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "source {WORKSPACE_SETUP_PI} && '
            f"cd {SIM_CLIENT_PATH} && python3 simulation_client.py "
            '--ros-camera --robot-id 1 --mode inference --free"'
        ),
        "watchable": True,
        "note": "Launch T9 and T10 roughly at the same time",
    },
    {
        "id": "client2",
        "label": "Sim Client R2",
        "terminal": "T10",
        "group": "clients",
        "desktop_cmd": (
            f"source {WORKSPACE_SETUP_DESK} && cd ~/catkin_ws/src/{SIM_CLIENT_PATH} && "
            "python3 simulation_client.py --ros-camera --robot-id 2 --mode inference --free"
        ),
        "pi_cmd": (
            f'docker start {CONTAINER_NAME} 2>/dev/null; docker exec -it {CONTAINER_NAME} bash -c "source {WORKSPACE_SETUP_PI} && '
            f"cd {SIM_CLIENT_PATH} && python3 simulation_client.py "
            '--ros-camera --robot-id 2 --mode inference --free"'
        ),
        "watchable": True,
        "note": "Launch T9 and T10 roughly at the same time",
    },
]

for p in PROCESSES:
    p["cmd"] = p["desktop_cmd"]

# ─── Status constants ─────────────────────────────────────────────────────────
STATUS_IDLE    = "idle"
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_ERROR   = "error"

STATUS_COLORS = {
    STATUS_IDLE:    "#555566",
    STATUS_RUNNING: "#3fb950",
    STATUS_STOPPED: "#f85149",
    STATUS_ERROR:   "#f85149",
}

# ─── Styles ───────────────────────────────────────────────────────────────────
BTN_BLUE_STYLE = (
    "QPushButton { background: #1f6feb33; border: 1px solid #1f6feb88; color: #58a6ff; "
    "border-radius: 3px; font-family: monospace; font-size: 10px; font-weight: 700; padding: 1px 6px; }"
    "QPushButton:hover { background: #1f6feb55; }"
    "QPushButton:pressed { background: #1f6feb77; }"
)
BTN_YELLOW_STYLE = (
    "QPushButton { background: #d2992233; border: 1px solid #d2992288; color: #d29922; "
    "border-radius: 3px; font-family: monospace; font-size: 12px; font-weight: 700; padding: 1px 3px; }"
    "QPushButton:hover { background: #d2992255; }"
    "QPushButton:pressed { background: #d2992277; }"
)
BTN_RED_STYLE = (
    "QPushButton { background: #f8514933; border: 1px solid #f8514988; color: #f85149; "
    "border-radius: 3px; font-family: monospace; font-size: 10px; font-weight: 700; padding: 1px 6px; }"
    "QPushButton:hover { background: #f8514955; }"
    "QPushButton:pressed { background: #f8514977; }"
)
BTN_GREEN_STYLE = (
    "QPushButton { background: #3fb95033; border: 1px solid #3fb95088; color: #3fb950; "
    "border-radius: 3px; font-family: monospace; font-size: 10px; font-weight: 700; padding: 3px 10px; }"
    "QPushButton:hover { background: #3fb95055; }"
    "QPushButton:pressed { background: #3fb95077; }"
)
BTN_CANCEL_STYLE = (
    "QPushButton { background: transparent; border: 1px solid #30363d; color: #8b949e; "
    "border-radius: 3px; font-family: monospace; font-size: 10px; padding: 3px 10px; }"
    "QPushButton:hover { background: #1c2128; }"
)
DIALOG_STYLE = (
    "QDialog { background: #161b22; color: #e6edf3; font-family: monospace; }"
    "QLabel { color: #e6edf3; background: transparent; }"
)

# ─── Terminal launch helper ───────────────────────────────────────────────────
def build_terminal_cmd(script_path, title):
    """
    Build a terminal command that keeps the window open after the command
    exits so the sentinel exit-file is written before the window closes.
    Uses hold/exec-bash tricks per terminal type.
    """
    t = TERMINAL
    if t == "lxterminal":
        # lxterminal: drop into bash after the script so window stays open
        hold_cmd = f'bash -c "{script_path}; echo; echo ---DONE---; exec bash"'
        return ["lxterminal", "--title", title, "-e", hold_cmd]
    elif t == "xterm":
        return ["xterm", "-hold", "-title", title, "-e", f"bash {script_path}"]
    elif t == "xfce4-terminal":
        return ["xfce4-terminal", "--hold", "--title", title, "--command",
                f"bash {script_path}"]
    elif t == "gnome-terminal":
        # gnome-terminal has no --hold; keep bash open after script
        return ["gnome-terminal", "--title", title, "--",
                "bash", "-c", f"{script_path}; echo; echo '[DONE]'; exec bash"]
    elif t == "konsole":
        return ["konsole", "--hold", "--title", title, "-e",
                f"bash {script_path}"]
    return None

# ─── Process watcher thread ───────────────────────────────────────────────────
class ProcessWatcher(QThread):
    status_changed = pyqtSignal(str, str)   # id, status
    crashed        = pyqtSignal(str, str)   # id, label

    def __init__(self, proc_def):
        super().__init__()
        self.proc_def = proc_def
        self._stop    = False

    def run(self):
        pid       = self.proc_def["id"]
        label     = self.proc_def["label"]
        cmd       = self.proc_def["cmd"]
        watchable = self.proc_def["watchable"]

        sentinel_script = f"/tmp/ros_launcher_sentinel_{pid}.sh"
        exit_file       = f"/tmp/ros_launcher_exit_{pid}.txt"

        with open(sentinel_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"{cmd}\n")
            f.write(f"echo $? > {exit_file}\n")
        os.chmod(sentinel_script, 0o755)

        if os.path.exists(exit_file):
            os.remove(exit_file)

        terminal_cmd = build_terminal_cmd(sentinel_script, label)
        if terminal_cmd is None:
            self.status_changed.emit(pid, STATUS_ERROR)
            return

        try:
            subprocess.Popen(terminal_cmd)
        except Exception:
            self.status_changed.emit(pid, STATUS_ERROR)
            return

        self.status_changed.emit(pid, STATUS_RUNNING)

        if not watchable:
            return

        while not self._stop:
            time.sleep(2)
            if os.path.exists(exit_file):
                self.status_changed.emit(pid, STATUS_STOPPED)
                self.crashed.emit(pid, label)
                break

    def stop(self):
        self._stop = True

# ─── Pendant confirm dialog ───────────────────────────────────────────────────
class PendantDialog(QDialog):
    """Shown after driver launches — asks user to activate External Control
    on the robot pendant before gripper/camera processes are started."""
    def __init__(self, robot_label, parent=None):
        super().__init__(parent, Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle("Pendant Action Required")
        self.setMinimumWidth(380)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 12)

        icon_lbl = QLabel("⚠")
        icon_lbl.setStyleSheet("font-size: 28px; color: #d29922; background: transparent;")
        icon_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_lbl)

        msg = QLabel(
            f"<b>{robot_label} driver is launching.</b><br><br>"
            "On the teach pendant, go to:<br>"
            "<tt>Program Robot → URCaps → External Control</tt><br><br>"
            "Activate <b>External Control</b>, then tap <b>Confirm</b>."
        )
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet(
            "font-size: 11px; color: #e6edf3; background: transparent; line-height: 1.5;"
        )
        layout.addWidget(msg)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(BTN_CANCEL_STYLE)
        cancel_btn.setMinimumHeight(30)
        cancel_btn.clicked.connect(self.reject)

        confirm_btn = QPushButton("✓  Confirmed — Continue")
        confirm_btn.setStyleSheet(BTN_GREEN_STYLE)
        confirm_btn.setMinimumHeight(30)
        confirm_btn.clicked.connect(self.accept)

        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(confirm_btn)
        layout.addLayout(btn_row)

# ─── Launch confirm dialog ────────────────────────────────────────────────────
class ConfirmDialog(QDialog):
    def __init__(self, proc_def, parent=None):
        super().__init__(parent, Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setWindowTitle(f"Launch: {proc_def['label']}")
        self.setMinimumWidth(440)
        self.setStyleSheet(DIALOG_STYLE)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(14, 14, 14, 10)

        title = QLabel(f"{proc_def['terminal']} — {proc_def['label']}")
        title.setStyleSheet("font-size: 12px; font-weight: bold; color: #e6edf3;")
        layout.addWidget(title)

        cmd_box = QTextEdit()
        cmd_box.setPlainText(proc_def["cmd"])
        cmd_box.setReadOnly(True)
        cmd_box.setStyleSheet(
            "background: #0d1117; color: #3fb950; font-family: monospace; "
            "font-size: 10px; border: 1px solid #30363d; border-radius: 4px; padding: 6px;"
        )
        cmd_box.setFixedHeight(68)
        layout.addWidget(cmd_box)

        if proc_def.get("note"):
            note = QLabel(proc_def["note"])
            note.setWordWrap(True)
            note.setStyleSheet(
                "background: #d2992215; border: 1px solid #d2992255; border-radius: 4px; "
                "color: #d29922; font-size: 10px; padding: 5px 8px;"
            )
            layout.addWidget(note)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel = QPushButton("Cancel")
        cancel.setStyleSheet(BTN_CANCEL_STYLE)
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)

        confirm = QPushButton("Open Terminal ↗")
        confirm.setStyleSheet(BTN_GREEN_STYLE)
        confirm.clicked.connect(self.accept)
        btn_row.addWidget(confirm)
        layout.addLayout(btn_row)

# ─── Process row widget ───────────────────────────────────────────────────────
class ProcessRow(QFrame):
    launch_requested  = pyqtSignal(dict)
    restart_requested = pyqtSignal(dict)

    def __init__(self, proc_def, parent=None):
        super().__init__(parent)
        self.proc_def = proc_def
        self.status   = STATUS_IDLE
        self.setFixedHeight(46)
        self._idle_style    = ("QFrame { background: #0d1117; border: 1px solid #30363d; border-radius: 4px; }"
                               "QFrame:hover { border-color: #484f58; }")
        self._running_style = "QFrame { background: #0d1117; border: 1px solid #3fb95044; border-radius: 4px; }"
        self._error_style   = "QFrame { background: #0d1117; border: 1px solid #f8514966; border-radius: 4px; }"
        self.setStyleSheet(self._idle_style)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 0, 8, 0)
        row.setSpacing(6)

        t_lbl = QLabel(proc_def["terminal"])
        t_lbl.setFixedWidth(26)
        t_lbl.setStyleSheet(
            "font-size: 10px; color: #8b949e; font-weight: bold; background: transparent; border: none;"
        )
        row.addWidget(t_lbl)

        name_lbl = QLabel(proc_def["label"])
        name_lbl.setStyleSheet(
            "font-size: 12px; color: #e6edf3; font-weight: 600; background: transparent; border: none;"
        )
        row.addWidget(name_lbl, stretch=1)

        self.dot_lbl = QLabel("●")
        self.dot_lbl.setFixedWidth(14)
        self.dot_lbl.setStyleSheet(
            f"font-size: 10px; color: {STATUS_COLORS[STATUS_IDLE]}; background: transparent; border: none;"
        )
        self.status_lbl = QLabel("idle")
        self.status_lbl.setFixedWidth(48)
        self.status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_lbl.setStyleSheet(
            "font-size: 10px; color: #555566; background: transparent; border: none;"
        )
        row.addWidget(self.dot_lbl)
        row.addWidget(self.status_lbl)

        self.launch_btn = QPushButton("Launch")
        self.launch_btn.setFixedSize(70, 32)
        self.launch_btn.setStyleSheet(BTN_BLUE_STYLE)
        self.launch_btn.clicked.connect(lambda: self.launch_requested.emit(self.proc_def))

        self.restart_btn = QPushButton("↺")
        self.restart_btn.setFixedSize(36, 32)
        self.restart_btn.setStyleSheet(BTN_YELLOW_STYLE)
        self.restart_btn.setToolTip("Restart")
        self.restart_btn.clicked.connect(lambda: self.restart_requested.emit(self.proc_def))

        row.addWidget(self.launch_btn)
        row.addWidget(self.restart_btn)

    def update_cmd(self, new_cmd):
        self.proc_def["cmd"] = new_cmd

    def set_status(self, status):
        self.status = status
        color = STATUS_COLORS.get(status, "#555566")
        self.dot_lbl.setStyleSheet(
            f"font-size: 10px; color: {color}; background: transparent; border: none;"
        )
        self.status_lbl.setStyleSheet(
            f"font-size: 10px; color: {color}; background: transparent; border: none;"
        )
        self.status_lbl.setText(status)
        if status in (STATUS_STOPPED, STATUS_ERROR):
            self.setStyleSheet(self._error_style)
        elif status == STATUS_RUNNING:
            self.setStyleSheet(self._running_style)
        else:
            self.setStyleSheet(self._idle_style)

# ─── Group panel ──────────────────────────────────────────────────────────────
class GroupPanel(QFrame):
    def __init__(self, group_id, group_label, accent_color, parent=None):
        super().__init__(parent)
        self.group_id = group_id
        self.setObjectName(f"group_{group_id}")
        self.setStyleSheet(
            f"QFrame#group_{group_id} {{"
            f"  background: #161b22;"
            f"  border: 1px solid #30363d;"
            f"  border-top: 2px solid {accent_color};"
            f"  border-radius: 5px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(32)
        header.setStyleSheet(
            "QFrame { background: #1c2128; border: none; "
            "border-bottom: 1px solid #30363d; border-radius: 0; }"
        )
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(8, 0, 6, 0)
        hrow.setSpacing(4)

        title = QLabel(group_label)
        title.setStyleSheet(
            "font-size: 10px; font-weight: 700; color: #8b949e; "
            "letter-spacing: 1px; background: transparent;"
        )
        hrow.addWidget(title)
        hrow.addStretch()

        self.launch_all_btn = QPushButton("All ▶")
        self.launch_all_btn.setFixedHeight(24)
        self.launch_all_btn.setStyleSheet(BTN_BLUE_STYLE)

        self.stop_all_btn = QPushButton("Stop")
        self.stop_all_btn.setFixedHeight(24)
        self.stop_all_btn.setStyleSheet(BTN_RED_STYLE)

        hrow.addWidget(self.launch_all_btn)
        hrow.addWidget(self.stop_all_btn)
        outer.addWidget(header)

        self.body = QWidget()
        self.body.setStyleSheet("background: transparent;")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(4, 4, 4, 4)
        self.body_layout.setSpacing(2)
        outer.addWidget(self.body)

    def add_row(self, row_widget):
        self.body_layout.addWidget(row_widget)

    def add_note(self, text):
        note = QLabel(text)
        note.setWordWrap(True)
        note.setStyleSheet(
            "background: #d2992215; border: 1px solid #d2992255; border-radius: 3px; "
            "color: #d29922; font-size: 9px; padding: 3px 6px; margin: 0;"
        )
        self.body_layout.addWidget(note)

# ─── Log panel ────────────────────────────────────────────────────────────────
class LogPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background: #161b22; border: 1px solid #30363d; "
            "border-top: 2px solid #484f58; border-radius: 5px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QFrame()
        header.setFixedHeight(24)
        header.setStyleSheet(
            "background: #1c2128; border: none; border-bottom: 1px solid #30363d;"
        )
        hrow = QHBoxLayout(header)
        hrow.setContentsMargins(8, 0, 6, 0)
        lbl = QLabel("EVENT LOG")
        lbl.setStyleSheet(
            "font-size: 9px; font-weight: 700; color: #8b949e; "
            "letter-spacing: 1px; background: transparent;"
        )
        hrow.addWidget(lbl)
        hrow.addStretch()
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(18)
        clear_btn.setStyleSheet(BTN_CANCEL_STYLE + "font-size: 9px; padding: 0 6px;")
        clear_btn.clicked.connect(self.clear)
        hrow.addWidget(clear_btn)
        layout.addWidget(header)

        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFixedHeight(52)
        self.text.setStyleSheet(
            "QTextEdit { background: #0d1117; color: #8b949e; font-family: monospace; "
            "font-size: 9px; border: none; padding: 3px 6px; }"
        )
        layout.addWidget(self.text)

    def add(self, msg, color="#8b949e"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.text.append(
            f'<span style="color:#484f58">{ts}</span> '
            f'<span style="color:{color}">{msg}</span>'
        )
        self.text.moveCursor(QTextCursor.End)

    def ok(self, msg):   self.add(msg, "#3fb950")
    def warn(self, msg): self.add(msg, "#d29922")
    def err(self, msg):  self.add(msg, "#f85149")
    def info(self, msg): self.add(msg, "#58a6ff")
    def clear(self):     self.text.clear()

# ─── Main window ─────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROS Robot Control Station")
        self.setFixedSize(1024, 600)
        self.setStyleSheet("QMainWindow { background: #0d1117; }")

        self.watchers    = {}
        self.rows        = {}
        self.groups      = {}
        self._is_pi_mode = False

        central = QWidget()
        central.setStyleSheet("background: #0d1117;")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = QFrame()
        topbar.setFixedHeight(30)
        topbar.setStyleSheet("background: #161b22; border-bottom: 1px solid #30363d;")
        trow = QHBoxLayout(topbar)
        trow.setContentsMargins(10, 0, 10, 0)
        trow.setSpacing(6)

        title_lbl = QLabel("⬡  ROS ROBOT CONTROL STATION")
        title_lbl.setStyleSheet(
            "font-size: 11px; font-weight: 700; color: #58a6ff; "
            "letter-spacing: 1px; background: transparent;"
        )
        trow.addWidget(title_lbl)
        trow.addStretch()

        # Pi / Desktop toggle pill
        toggle_frame = QFrame()
        toggle_frame.setFixedHeight(22)
        toggle_frame.setStyleSheet(
            "QFrame { background: #0d1117; border: 1px solid #30363d; border-radius: 11px; }"
        )
        tgl = QHBoxLayout(toggle_frame)
        tgl.setContentsMargins(2, 2, 2, 2)
        tgl.setSpacing(0)

        self.desktop_btn = QPushButton("🖥️  Desktop")
        self.desktop_btn.setFixedHeight(18)
        self.desktop_btn.clicked.connect(lambda: self._set_pi_mode(False))

        self.pi_btn = QPushButton("🍓  Pi")
        self.pi_btn.setFixedHeight(18)
        self.pi_btn.clicked.connect(lambda: self._set_pi_mode(True))

        tgl.addWidget(self.desktop_btn)
        tgl.addWidget(self.pi_btn)
        trow.addWidget(toggle_frame)
        self._apply_toggle_style()

        sep = QLabel("|")
        sep.setStyleSheet("color: #30363d; background: transparent; padding: 0 6px;")
        trow.addWidget(sep)

        self.clock_lbl = QLabel()
        self.clock_lbl.setStyleSheet(
            "font-size: 11px; font-weight: 600; color: #e6edf3; "
            "background: transparent; font-family: monospace;"
        )
        trow.addWidget(self.clock_lbl)
        root.addWidget(topbar)

        # ── Alert banner ──────────────────────────────────────────────────────
        self.alert_banner = QLabel()
        self.alert_banner.setFixedHeight(22)
        self.alert_banner.setAlignment(Qt.AlignCenter)
        self.alert_banner.setStyleSheet(
            "background: #f8514922; color: #f85149; font-size: 10px; font-weight: 700; "
            "border-bottom: 1px solid #f8514955; padding: 0 10px;"
        )
        self.alert_banner.hide()
        root.addWidget(self.alert_banner)

        # ── Main grid ─────────────────────────────────────────────────────────
        grid_container = QWidget()
        grid_container.setStyleSheet("background: #0d1117;")
        grid = QHBoxLayout(grid_container)
        grid.setContentsMargins(4, 4, 4, 3)
        grid.setSpacing(4)

        r1_panel = GroupPanel("r1", "ROBOT 1 — 192.168.1.120", "#1d4ed8")
        self.groups["r1"] = r1_panel
        grid.addWidget(r1_panel)

        r2_panel = GroupPanel("r2", "ROBOT 2 — 192.168.1.112", "#7c3aed")
        self.groups["r2"] = r2_panel
        grid.addWidget(r2_panel)

        # Right column: GPU Server (Pi-only) above Simulation Clients
        right_col = QVBoxLayout()
        right_col.setSpacing(4)

        self.gpu_server_panel = GroupPanel("server", "HOST GPU SERVER", "#e34c26")
        self.groups["server"] = self.gpu_server_panel
        right_col.addWidget(self.gpu_server_panel)

        clients_panel = GroupPanel("clients", "SIMULATION CLIENTS", "#f0883e")
        self.groups["clients"] = clients_panel
        right_col.addWidget(clients_panel, stretch=1)

        grid.addLayout(right_col)
        root.addWidget(grid_container, stretch=1)

        # ── Populate rows ─────────────────────────────────────────────────────
        gpu_row = ProcessRow(GPU_SERVER_PROC)
        gpu_row.launch_requested.connect(self.on_launch)
        gpu_row.restart_requested.connect(self.on_restart)
        self.rows["gpu_server"] = gpu_row
        self.gpu_server_panel.add_row(gpu_row)

        for p in PROCESSES:
            row = ProcessRow(p)
            row.launch_requested.connect(self.on_launch)
            row.restart_requested.connect(self.on_restart)
            self.rows[p["id"]] = row
            self.groups[p["group"]].add_row(row)

        # Sequence hint in clients panel
        clients_panel.body_layout.addStretch()
        seq_note = QLabel("① roscore → ② Drivers → ③ Pendant → ④ Grippers → ⑤ Cams → ⑥ Clients")
        seq_note.setWordWrap(True)
        seq_note.setStyleSheet(
            "font-size: 8px; color: #555566; background: #0d111788; "
            "border: 1px solid #30363d; border-radius: 3px; padding: 3px 5px;"
        )
        clients_panel.body_layout.addWidget(seq_note)

        # Wire launch-all / stop-all
        for gid, panel in self.groups.items():
            panel.launch_all_btn.clicked.connect(lambda checked, g=gid: self.launch_all(g))
            panel.stop_all_btn.clicked.connect(lambda checked, g=gid: self.stop_all(g))

        # ── Log panel ─────────────────────────────────────────────────────────
        self.log = LogPanel()
        log_container = QWidget()
        lc = QVBoxLayout(log_container)
        lc.setContentsMargins(4, 0, 4, 4)
        lc.addWidget(self.log)
        root.addWidget(log_container)

        # ── Timers ────────────────────────────────────────────────────────────
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.update_clock()

        self.alert_timer = QTimer()
        self.alert_timer.setSingleShot(True)
        self.alert_timer.timeout.connect(self.alert_banner.hide)

        self.update_commands_from_env()
        self.log.info("Control station ready — all processes idle")
        if not TERMINAL:
            self.log.err("No terminal emulator found! Install lxterminal or xterm.")

    # ── Toggle ────────────────────────────────────────────────────────────────
    def _set_pi_mode(self, is_pi):
        if is_pi == self._is_pi_mode:
            return
        self._is_pi_mode = is_pi
        self._apply_toggle_style()
        self.update_commands_from_env()

    def _apply_toggle_style(self):
        active = (
            "QPushButton { background: #1f6feb; color: #ffffff; border: none; "
            "border-radius: 9px; font-size: 10px; font-weight: 700; padding: 0 10px; }"
        )
        inactive = (
            "QPushButton { background: transparent; color: #8b949e; border: none; "
            "border-radius: 9px; font-size: 10px; font-weight: 600; padding: 0 10px; }"
            "QPushButton:hover { color: #e6edf3; }"
        )
        self.desktop_btn.setStyleSheet(active if not self._is_pi_mode else inactive)
        self.pi_btn.setStyleSheet(active if self._is_pi_mode else inactive)

    # ── Env switch ────────────────────────────────────────────────────────────
    def update_commands_from_env(self):
        is_pi = self._is_pi_mode
        self.gpu_server_panel.setVisible(is_pi)
        for pid, row in self.rows.items():
            new_cmd = row.proc_def["pi_cmd"] if is_pi else row.proc_def["desktop_cmd"]
            row.update_cmd(new_cmd)

    # ── Individual launch / restart ───────────────────────────────────────────
    def on_launch(self, proc_def):
        dlg = ConfirmDialog(proc_def, self)
        if dlg.exec_() == QDialog.Accepted:
            self._do_launch(proc_def)

    def on_restart(self, proc_def):
        pid = proc_def["id"]
        if pid in self.watchers:
            self.watchers[pid].stop()
            self.watchers[pid].quit()
            del self.watchers[pid]
        self.log.warn(f"Restarting {proc_def['label']}…")
        self._do_launch(proc_def)

    def _do_launch(self, proc_def):
        if not TERMINAL:
            QMessageBox.critical(self, "No Terminal",
                                 "No terminal emulator found.\nInstall lxterminal or xterm.")
            return
        pid = proc_def["id"]
        self.rows[pid].set_status(STATUS_RUNNING)
        w = ProcessWatcher(proc_def)
        w.status_changed.connect(self.on_status_changed)
        w.crashed.connect(self.on_crashed)
        self.watchers[pid] = w
        w.start()
        self.log.ok(f"Launched {proc_def['label']}")

    def on_status_changed(self, pid, status):
        if pid in self.rows:
            self.rows[pid].set_status(status)

    def on_crashed(self, pid, label):
        self.log.err(f"{label} stopped unexpectedly — restart required")
        self.show_alert(f"⚡  {label} has stopped — tap ↺ to restart", error=True)

    # ── Launch All ────────────────────────────────────────────────────────────
    def launch_all(self, group_id):
        """
        r1 / r2: launch roscore + driver first, pause for pendant confirmation,
        then continue with gripper + camera.
        clients / server: sequential with small delays.
        """
        if group_id == "server":
            self._do_launch(GPU_SERVER_PROC)
            self.log.info("Launching GPU Server…")
            return

        procs = [p for p in PROCESSES if p["group"] == group_id]

        if group_id in ("r1", "r2"):
            driver_id  = "r1driver" if group_id == "r1" else "r2driver"
            pre_procs  = [p for p in procs if p["id"] in ("roscore", driver_id)]
            post_procs = [p for p in procs if p["id"] not in ("roscore", driver_id)]
            robot_lbl  = "Robot 1" if group_id == "r1" else "Robot 2"

            for i, p in enumerate(pre_procs):
                QTimer.singleShot(i * 800, lambda p=p: self._do_launch(p))

            # Show pendant dialog after pre-procs have been triggered
            delay_ms = len(pre_procs) * 800 + 1200
            QTimer.singleShot(
                delay_ms,
                lambda rl=robot_lbl, pp=post_procs: self._show_pendant_then_continue(rl, pp)
            )
        else:
            for i, p in enumerate(procs):
                QTimer.singleShot(i * 700, lambda p=p: self._do_launch(p))

        self.log.info(f"Launching all {group_id} processes…")

    def _show_pendant_then_continue(self, robot_label, post_procs):
        dlg = PendantDialog(robot_label, self)
        if dlg.exec_() == QDialog.Accepted:
            for i, p in enumerate(post_procs):
                QTimer.singleShot(i * 700, lambda p=p: self._do_launch(p))
            self.log.ok(
                f"Pendant confirmed for {robot_label} — launching gripper & camera"
            )
        else:
            self.log.warn(
                f"Pendant step cancelled for {robot_label} — gripper/camera NOT launched"
            )
            self.show_alert(f"⚠  {robot_label} pendant step cancelled", error=False)

    def stop_all(self, group_id):
        targets = [GPU_SERVER_PROC] if group_id == "server" else PROCESSES
        for p in targets:
            if p["group"] == group_id:
                pid = p["id"]
                if pid in self.watchers:
                    self.watchers[pid].stop()
                    self.watchers[pid].quit()
                    del self.watchers[pid]
                if pid in self.rows:
                    self.rows[pid].set_status(STATUS_IDLE)
        self.log.warn(f"Stopped all {group_id} processes")

    def show_alert(self, msg, error=False):
        style = (
            "background: #f8514922; color: #f85149; font-size: 10px; font-weight: 700; "
            "border-bottom: 1px solid #f8514955; padding: 0 10px;"
        ) if error else (
            "background: #d2992222; color: #d29922; font-size: 10px; font-weight: 700; "
            "border-bottom: 1px solid #d2992255; padding: 0 10px;"
        )
        self.alert_banner.setStyleSheet(style)
        self.alert_banner.setText(msg)
        self.alert_banner.show()
        self.alert_timer.start(10000)

    def update_clock(self):
        self.clock_lbl.setText(datetime.now().strftime("%H:%M:%S"))

    def closeEvent(self, event):
        for w in self.watchers.values():
            w.stop()
            w.quit()
        event.accept()

# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.wayland=false")
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,     QColor("#0d1117"))
    pal.setColor(QPalette.WindowText, QColor("#e6edf3"))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())