import cv2
import numpy as np
import json
from pathlib import Path
import base64
import math
import re

class Visualizer:
    """Tools for visualizing robot data and network predictions"""
    
    @staticmethod
    def decode_image(b64_string):
        """Decode base64 string to numpy image"""
        try:
            img_bytes = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None

    @staticmethod
    def draw_grasp(image, pose, color=(0, 255, 0), thickness=2):
        """Draws the grasp pose on the image."""
        if image is None: return None
        
        # 1. ROTATE IMAGE (Fixed: Clockwise)
        # If floor was on Left, Clockwise brings it to Bottom.
        img_vis = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w = img_vis.shape[:2]
        
        # 2. COORDINATE MAPPING (Webots Y-Up)
        x_forward = pose[0]
        y_height  = pose[1] 
        z_side    = pose[2]
        yaw       = pose[3]

        # Camera Calibration 
        center_x_m = 0.40
        pixels_per_meter = 600 
        
        # Map World -> Image (Rotated Clockwise)
        # In Clockwise rotation:
        # Image X (U) = Original Y (Up/Down) -> Robot X (Forward)
        # Image Y (V) = Original X (Left/Right) -> Robot Z (Side)
        
        # WARNING: Coordinate transforms are tricky. 
        # If the dot moves opposite to the duck, flip the signs here.
        # Standard assumption for 90deg rotation:
        u = int(w/2 - (z_side * pixels_per_meter)) 
        v = int(h/2 - ((x_forward - center_x_m) * pixels_per_meter))
        
        # Draw Center Point
        cv2.circle(img_vis, (u, v), 6, color, -1)
        
        # Draw Orientation Line
        # Rotate yaw to match image rotation
        visual_yaw = yaw + (math.pi / 2) 
        
        gripper_len_px = int(0.08 * pixels_per_meter)
        dx = int((gripper_len_px/2) * math.sin(visual_yaw))
        dy = int((gripper_len_px/2) * math.cos(visual_yaw))
        
        p1 = (u - dx, v - dy)
        p2 = (u + dx, v + dy)
        cv2.line(img_vis, p1, p2, color, thickness)
        
        # Add Text
        text = f"H: {y_height:.3f}m"
        cv2.putText(img_vis, text, (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_vis

# ==========================================
#  STANDALONE INSPECTOR
# ==========================================
def inspect_data(data_dir="data/episodes"):
    path = Path(data_dir)
    if not path.exists():
        print(f"Directory {data_dir} not found!")
        return

    # --- FIX ORDERING ---
    # Sorts by the integer number in the filename (e.g. episode_2.json)
    # instead of alphabetical (where 10 comes before 2)
    def get_episode_num(filepath):
        # Extract all numbers, take the last one found
        nums = re.findall(r'\d+', filepath.name)
        return int(nums[-1]) if nums else 0

    files = sorted(list(path.glob("*.json")), key=get_episode_num)
    print(f"Found {len(files)} episodes.")

    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            episode_num = data.get('episode', '?')
            if 'data' in data and len(data['data']) > 0:
                last_step = data['data'][-1]
                
                if 'command' in last_step:
                    cmd = last_step['command']
                elif 'prediction' in last_step:
                    cmd = last_step['prediction'].get('pose', [0,0,0,0])
                else: cmd = [0,0,0,0]

                if 'outcome' in last_step:
                    reward = last_step['outcome'].get('reward', 0)
                elif 'execution' in last_step:
                    reward = last_step['execution'].get('reward', 0)
                else: reward = 0

                b64_img = None
                if 'visual_input' in last_step:
                    b64_img = last_step['visual_input']['rgb']
                elif 'prediction' in last_step and 'data' in last_step['prediction']:
                    b64_img = last_step['prediction']['data']['rgb']

                if b64_img:
                    img = Visualizer.decode_image(b64_img)
                    if img is not None:
                        vis_img = Visualizer.draw_grasp(img, cmd)
                        
                        status = "SUCCESS" if reward > 0 else "FAIL"
                        col = (0, 255, 0) if reward > 0 else (0, 0, 255)
                        cv2.putText(vis_img, f"Ep {episode_num}: {status} ({reward:.2f})", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

                        cv2.imshow("Data Inspector", vis_img)
                        print(f"Ep {episode_num}: Pose={cmd} | Reward={reward}")
                        
                        key = cv2.waitKey(0)
                        if key == ord('q'): break
        except Exception as e:
            print(f"Error reading {filepath.name}: {e}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    inspect_data("data/grasp_dataset")