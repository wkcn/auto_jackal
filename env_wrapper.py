import cv2
import numpy as np
import retro


class RetroWrapper:
    """Wrapper for Retro environment with preprocessing"""
    
    def __init__(self, game='Jackal-Nes', state=retro.State.DEFAULT, 
                 frame_skip=4, frame_stack=4, resize_shape=(84, 84),
                 no_reward_penalty=-0.01, no_reward_timeout_steps=450):
        self.env = retro.make(game=game, state=state)
        self.frame_skip = frame_skip
        self.frame_stack = frame_stack
        self.resize_shape = resize_shape
        
        # Frame buffer for stacking
        self.frames = []
        
        # Get action space info
        # Retro uses MultiBinary action space (button array)
        self.button_count = self.env.action_space.n
        self.n_actions = self.button_count  # Direct button control
        
        # Reward tracking for penalty mechanism
        self.no_reward_penalty = no_reward_penalty
        self.no_reward_timeout_steps = no_reward_timeout_steps  # ~30 seconds at 60fps with frame_skip=4
        self.steps_without_reward = 0
        self.last_total_reward = 0
    
    def reset(self):
        """Reset environment and return initial state"""
        obs = self.env.reset()
        obs = self._preprocess(obs)
        
        # Initialize frame stack
        self.frames = [obs] * 1 # self.frame_stack
        
        # Reset reward tracking
        self.steps_without_reward = 0
        self.last_total_reward = 0
        
        return self._get_state()
    
    def step(self, action):
        """Execute action and return next state
        
        Args:
            action: numpy array or tensor of shape (button_count,) with 0s and 1s
        """
        total_reward = 0
        done = False
        
        # Convert action to numpy array if it's a tensor
        if hasattr(action, 'cpu'):
            action = action.cpu().numpy()
        action = np.array(action, dtype=np.int8)
        
        # Frame skipping
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        obs = self._preprocess(obs)
        self.frames.append(obs)
        self.frames.pop(0)
        
        # Apply reward penalty mechanism
        modified_reward = total_reward
        
        # Check if reward increased
        if total_reward > 0:
            # Got positive reward, reset counter
            self.steps_without_reward = 0
            self.last_total_reward += total_reward
        else:
            # No reward this step
            self.steps_without_reward += 1
            
            # Apply penalty for not getting reward
            modified_reward += self.no_reward_penalty
            
            # Check if timeout reached (e.g., 30 seconds without reward)
            if self.steps_without_reward >= self.no_reward_timeout_steps:
                done = True
                modified_reward -= 1.0  # Additional penalty for timeout
                info['timeout'] = True
                info['reason'] = 'no_reward_timeout'
        
        return self._get_state(), modified_reward, done, info
    
    def render(self):
        """Render environment"""
        return self.env.render()
    
    def close(self):
        """Close environment"""
        self.env.close()
    
    def _preprocess(self, frame):
        """Preprocess frame: grayscale and resize"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize
        resized = cv2.resize(gray, self.resize_shape, interpolation=cv2.INTER_AREA)
        return resized
    
    def _get_state(self):
        """Get stacked frames as state"""
        return np.array(self.frames, dtype=np.float32)
