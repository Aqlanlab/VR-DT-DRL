import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, object_pos=None):
        """
        Store a transition.

        object_pos: (object_X, object_Z) in robot world frame - used to supervise
        the auxiliary position head so the backbone learns spatial encoding.
        Defaults to [0.0, 0.0] for backward compatibility with old call sites
        that don't supply it; those entries are masked out during training.
        """
        if object_pos is None:
            object_pos = [0.0, 0.0]
        self.buffer.append((state, action, reward, next_state, done, object_pos))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)