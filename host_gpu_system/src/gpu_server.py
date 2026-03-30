#!/usr/bin/env python3
"""
GPU Inference Server for UR3 Grasping System.

Operates exclusively in Behavior Cloning mode. The system serves inferences 
and collects teacher demonstrations for continuous supervised training.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import socket
import json
import threading
import time
import yaml
import logging
import base64
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

from enhanced_neural_network import UR3GraspCNN_Enhanced, BehaviorCloningModule, create_model, ImageProcessor
from utils.logger import setup_logger
from utils.metrics import PerformanceMonitor


class GPUInferenceServer:
    def __init__(self, config_path: str = "config/network_config.yaml", model_path: str = None):
        self.config = self._load_config(config_path)
        self.model_path = model_path
        self.logger = setup_logger("gpu_server", "data/logs/gpu_server.log")

        # Device and Model Initialization
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_config = self._load_model_config()
        self.model, self.bc_module, self.image_processor = create_model(model_config)
        self.model = self.model.to(self.device)
        self.bc_module = self.bc_module.to(self.device)
        self.image_processor.device = self.device

        # Data Buffer Initialization (Every teacher sample is equally weighted)
        self.batch_size = 16
        self.data_buffer = deque(maxlen=10000)
        self.training_step_count = 0

        self._load_model_weights()

        # Learning Rate Scheduler (Reduces LR on pose loss plateau)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.bc_module.optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            min_lr=1e-5
        )

        # Utilities Setup
        self.image_processor = ImageProcessor(self.device)
        self.performance_monitor = PerformanceMonitor()

        # Networking and Threading Control
        self.is_running = False
        self.server_socket = None
        self.client_connections = []
        self.train_lock = threading.Lock()

    def _load_config(self, config_path: str) -> Dict:
        """Loads server configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception:
            return {'network': {'host_ip': '0.0.0.0', 'port': 8888}}

    def _load_model_config(self) -> Dict:
        """Provides default network configuration parameters."""
        return {
            'input_channels': 4,
            'input_size': [224, 224],
            'num_grasp_classes': 4,
            'output_6dof': True,
            'use_attention': True,
            'learning_rate': 5e-4,
            'weight_decay': 8e-4
        }

    def _load_model_weights(self):
        """Locates and loads the most recent model checkpoint."""
        script_dir = Path(__file__).resolve().parent

        if self.model_path:
            path = Path(self.model_path)
            if not path.is_absolute():
                path = script_dir / path
        else:
            path = script_dir.parent / "models" / "ur3_live_model.pth"
            if not path.exists():
                path = script_dir.parent / "models" / "ur3_model.pth"

        print(f"[INFO] Looking for weights at: {path.resolve()}")

        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=self.device)
                model_state = checkpoint.get('model_state_dict', checkpoint)

                missing, unexpected = self.model.load_state_dict(model_state, strict=False)
                if missing:
                    print(f"       New keys initialized with random weights: {missing}")
                if unexpected:
                    print(f"       Ignored deprecated keys: {unexpected}")

                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.bc_module.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except Exception as e:
                        print(f"       Optimizer state could not be restored: {e}")

                self.training_step_count = checkpoint.get('training_step', 0)
                print(f"[SUCCESS] Loaded weights: {path.resolve()}")
                print(f"          Resuming from step {self.training_step_count}")

            except Exception as e:
                print(f"[WARNING] Failed to load checkpoint ({e}). Starting with random weights.")
        else:
            print(f"[WARNING] No checkpoint found. Starting with random weights.")
            print(f"          Expected location: {path.resolve()}")

    def decode_b64_image(self, img_data: Dict) -> Dict[str, np.ndarray]:
        """Decodes base64 string payloads into RGB and depth numpy arrays."""
        rgb_bytes = base64.b64decode(img_data['rgb'])
        depth_bytes = base64.b64decode(img_data['depth'])

        rgb = cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Depth uses a custom encoding to avoid PNG data corruption:
        # Format: [H uint32 LE][W uint32 LE][H*W uint16 LE raw pixels]
        shape_header = np.frombuffer(depth_bytes[:8], dtype=np.uint32)
        h, w = int(shape_header[0]), int(shape_header[1])
        depth = np.frombuffer(depth_bytes[8:], dtype=np.uint16).reshape(h, w)

        return {'rgb': rgb, 'depth': depth}

    def format_batch_for_torch(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Converts raw experience dictionaries from the buffer into batched tensors."""
        states_list = []
        pose_labels_list = []
        grasp_labels_list = []
        rewards_list = []
        aux_pos_list = []

        for exp in batch:
            s_raw = exp['state']
            a_raw = exp['action']
            r_raw = exp['reward']
            obj_pos = exp.get('object_pos', [0.0, 0.0])
            is_sim = exp.get('source', 'real') == 'simulation'

            states_list.append(self.preprocess_rgbd_data(s_raw, is_simulation=is_sim))
            pose_labels_list.append(torch.tensor(a_raw, dtype=torch.float32))

            grasp_class = 1 if float(r_raw) >= 0.5 else 0
            grasp_labels_list.append(torch.tensor(grasp_class, dtype=torch.long))

            rewards_list.append(torch.tensor(r_raw, dtype=torch.float32).unsqueeze(0))
            aux_pos_list.append(torch.tensor(obj_pos, dtype=torch.float32))

        return {
            'states': torch.cat(states_list).to(self.device),
            'pose_labels': torch.stack(pose_labels_list).to(self.device),
            'grasp_labels': torch.stack(grasp_labels_list).to(self.device),
            'rewards': torch.stack(rewards_list).to(self.device),
            'aux_position_labels': torch.stack(aux_pos_list).to(self.device),
        }

    def preprocess_rgbd_data(self, rgbd_data: Dict, is_simulation: bool = False) -> torch.Tensor:
        """Decodes, scales, and crops the incoming sensory data."""
        img_data = self.decode_b64_image(rgbd_data)
        depth = img_data['depth']
        rgb = img_data['rgb']

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0

        if not is_simulation:
            # Shift hardware depth measurements to align with simulation expectations
            SIM_MEAN = 0.700
            REAL_MEAN = 0.743
            depth_shift = REAL_MEAN - SIM_MEAN 
            
            # np.where ensures 0 (no-data pixels) are not shifted into negative values
            depth = np.where(depth > 0, depth - depth_shift, 0)

        self.image_processor.is_simulation = is_simulation
        processed = self.image_processor.process_rgbd_image(rgb, depth)

        # Output raw crops for debugging orientation or normalization issues
        try:
            suffix = "_sim" if is_simulation else ""
            rgb_debug = rgb.copy()
            depth_debug = depth.copy()
            h2, w2 = rgb_debug.shape[:2]
            y0, y1 = int(0.215 * h2), int(0.584 * h2)
            x0, x1 = int(0.39 * w2), int(0.5565 * w2)
            
            cv2.imwrite(f"ai_vision_debug_rgb{suffix}.jpg", rgb_debug[y0:y1, x0:x1].copy())
            depth_crop = depth_debug[y0:y1, x0:x1].copy()
            depth_vis = cv2.normalize(depth_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(f"ai_vision_debug_depth{suffix}.png", depth_vis)
        except Exception:
            pass

        return processed

    def _handle_camera_data(self, full_message: Dict) -> Dict:
        """
        Processes incoming camera feeds.
        In training mode, yields control to the teacher.
        In inference mode, generates grasp predictions using the current model.
        """
        try:
            client_mode = full_message.get('mode', 'inference')

            if client_mode == 'training':
                return {
                    'type': 'grasp_prediction',
                    'mode': 'explore',
                    'pose': [0.0] * 6,
                    'timestamp': time.time()
                }

            is_sim = full_message.get('source', 'real') == 'simulation'
            camera_data = full_message['data']
            rgbd_tensor = self.preprocess_rgbd_data(camera_data, is_simulation=is_sim)

            self.model.eval()
            with torch.no_grad():
                prediction = self.model(rgbd_tensor)
                grasp_pose = prediction['pose_6dof'].cpu().numpy()[0]

            # Enforce tool-down orientation; AI currently only controls yaw (index 5)
            grasp_pose[3] = 3.14
            grasp_pose[4] = 0.0

            return {
                'type': 'grasp_prediction',
                'pose': grasp_pose.tolist(),
                'mode': 'exploit',
                'confidence': 1.0,
                'timestamp': time.time()
            }
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    def _handle_training_data(self, training_data: Dict, source: str = 'real') -> Dict:
        """Stores a teacher demonstration into the buffer and initiates training if full."""
        try:
            if source == 'simulation':
                try:
                    img_data = self.decode_b64_image(training_data['state'])
                    rgb = img_data['rgb']   
                    depth = img_data['depth'].astype(np.float32) / 1000.0

                    rgb_debug = rgb.copy()
                    depth_debug = depth.copy()

                    h, w = rgb_debug.shape[:2]
                    y0, y1 = int(0.215 * h), int(0.584 * h)
                    x0, x1 = int(0.39 * w), int(0.5565 * w)
                    rgb_crop = rgb_debug[y0:y1, x0:x1].copy()

                    cv2.imwrite("ai_vision_debug_rgb_sim.jpg", rgb_crop)
                    depth_crop = depth_debug[y0:y1, x0:x1].copy()
                    depth_vis = cv2.normalize(depth_crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite("ai_vision_debug_depth_sim.png", depth_vis)
                except Exception as e:
                    print(f"[DEBUG] Simulation image debug save failed: {e}")

            self.data_buffer.append({
                'state': training_data['state'],
                'action': training_data['action'],
                'reward': training_data['reward'],
                'object_pos': training_data.get('object_pos', [0.0, 0.0]),
                'source': source 
            })

            if len(self.data_buffer) >= self.batch_size:
                threading.Thread(target=self._run_training_step, daemon=True).start()

            return {'type': 'training_ack', 'buffer_len': len(self.data_buffer)}
        except Exception as e:
            return {'type': 'error', 'message': str(e)}

    def _run_training_step(self):
        """Samples a mini-batch from the demonstration buffer and runs one optimization step."""
        if self.train_lock.locked():
            return

        with self.train_lock:
            try:
                import random
                batch_raw = random.sample(list(self.data_buffer), self.batch_size)
                torch_batch = self.format_batch_for_torch(batch_raw)

                losses = self.bc_module.update_networks(torch_batch)
                self.lr_scheduler.step(losses['pose'])
                self.training_step_count += 1

                if self.training_step_count % 5 == 0:
                    print(
                        f"[TRAIN] Step {self.training_step_count:4d} | "
                        f"Total Loss: {losses['total']:.4f} "
                        f"(Pose: {losses['pose']:.4f}, Aux: {losses['aux']:.4f}) | "
                        f"GradNorm: {losses['grad_norm']:.3f}"
                    )

                if self.training_step_count % 100 == 0:
                    import os
                    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    save_dir = os.path.join(base_dir, "models")
                    os.makedirs(save_dir, exist_ok=True)
                    full_path = os.path.join(save_dir, "ur3_live_model.pth")
                    
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.bc_module.optimizer.state_dict(),
                        'training_step': self.training_step_count
                    }, full_path)
                    print(f"[SAVE] Model checkpoint saved to: {full_path}")

            except Exception as e:
                print(f"[ERROR] Critical failure during training step: {e}")
                import traceback
                traceback.print_exc()

    def handle_client_request(self, client_socket, address):
        """Handles incoming socket requests for data processing or inference."""
        self.is_running = True
        try:
            while self.is_running:
                size_data = client_socket.recv(4)
                if not size_data:
                    break
                message_size = int.from_bytes(size_data, byteorder='big')

                message_data = b''
                while len(message_data) < message_size:
                    chunk = client_socket.recv(min(message_size - len(message_data), 4096))
                    if not chunk:
                        break
                    message_data += chunk

                message = json.loads(message_data.decode('utf-8'))

                if message['type'] == 'camera_data':
                    response = self._handle_camera_data(message)
                elif message['type'] == 'training_data':
                    response = self._handle_training_data(
                        message['data'],
                        source=message.get('source', 'real')
                    )
                else:
                    response = {'type': 'ack'}

                response_data = json.dumps(response).encode('utf-8')
                client_socket.send(len(response_data).to_bytes(4, byteorder='big'))
                client_socket.send(response_data)
        except Exception as e:
            print(f"[ERROR] Socket error with client {address}: {e}")
        finally:
            client_socket.close()

    def start_server(self):
        """Initializes the socket server and listens for incoming hardware connections."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        host = self.config['network']['host_ip']
        port = self.config['network']['port']
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        
        print(f"[START] Behavior Cloning Server listening on {host}:{port}")
        
        while True:
            conn, addr = self.server_socket.accept()
            threading.Thread(
                target=self.handle_client_request, args=(conn, addr), daemon=True
            ).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help="Path to initial model weights")
    args = parser.parse_args()
    
    server = GPUInferenceServer(model_path=args.model)
    server.start_server()