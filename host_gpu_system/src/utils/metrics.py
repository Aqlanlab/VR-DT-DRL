#!/usr/bin/env python3
"""
Performance monitoring utilities
"""

import time
import psutil
import threading
from collections import deque
from typing import List, Dict

class PerformanceMonitor:
    """Monitor system and inference performance"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.inference_times = deque(maxlen=max_history)
        self.start_time = time.time()
        self.total_inferences = 0
        
    def log_inference_time(self, inference_time: float):
        """Log an inference time"""
        self.inference_times.append(inference_time)
        self.total_inferences += 1
    
    def get_average_inference_time(self) -> float:
        """Get average inference time"""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def get_inference_fps(self) -> float:
        """Get inference FPS"""
        avg_time = self.get_average_inference_time()
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'uptime_hours': (time.time() - self.start_time) / 3600,
            'total_inferences': self.total_inferences,
            'avg_inference_time': self.get_average_inference_time(),
            'inference_fps': self.get_inference_fps()
        }
    
    def log_summary(self):
        """Log performance summary"""
        stats = self.get_system_stats()
        print(f"📊 Performance Summary:")
        print(f"  Total inferences: {stats['total_inferences']}")
        print(f"  Average inference time: {stats['avg_inference_time']:.3f}s")
        print(f"  Inference FPS: {stats['inference_fps']:.1f}")
        print(f"  Uptime: {stats['uptime_hours']:.1f} hours")
