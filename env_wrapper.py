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
        
        # Create a simplified discrete action space
        # Define common NES actions (combinations of buttons)
        self.action_map = self._create_action_map()
        self.n_actions = len(self.action_map)
        
        # Reward tracking for penalty mechanism
        self.no_reward_penalty = no_reward_penalty
        self.no_reward_timeout_steps = no_reward_timeout_steps  # ~30 seconds at 60fps with frame_skip=4
        self.steps_without_reward = 0
        self.last_total_reward = 0
    
    def _create_action_map(self):
        """Create mapping from discrete actions to button arrays"""
        # NES controller buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
        # We'll create common action combinations
        actions = []
        
        # No action
        actions.append([0] * self.button_count)
        
        # Single button presses
        for i in range(self.button_count):
            action = [0] * self.button_count
            action[i] = 1
            actions.append(action)
        
        # Common combinations for NES games
        # UP + A (jump forward)
        if self.button_count >= 9:
            actions.append([0, 0, 0, 0, 1, 0, 0, 0, 1])  # UP + A
            actions.append([0, 0, 0, 0, 0, 0, 1, 0, 1])  # LEFT + A
            actions.append([0, 0, 0, 0, 0, 0, 0, 1, 1])  # RIGHT + A
            actions.append([0, 0, 0, 0, 1, 0, 0, 0, 0])  # UP
            actions.append([0, 0, 0, 0, 0, 1, 0, 0, 0])  # DOWN
            actions.append([0, 0, 0, 0, 0, 0, 1, 0, 0])  # LEFT
            actions.append([0, 0, 0, 0, 0, 0, 0, 1, 0])  # RIGHT
            actions.append([1, 0, 0, 0, 0, 0, 0, 0, 0])  # B
            actions.append([0, 0, 0, 0, 0, 0, 0, 0, 1])  # A
        
        return actions
    
    def _discrete_to_multi_binary(self, action):
        """Convert discrete action to MultiBinary format"""
        return np.array(self.action_map[action], dtype=np.int8)
        
    def reset(self):
        """Reset environment and return initial state"""
        obs = self.env.reset()
        obs = self._preprocess(obs)
        
        # Initialize frame stack
        self.frames = [obs] * self.frame_stack
        
        # Reset reward tracking
        self.steps_without_reward = 0
        self.last_total_reward = 0
        
        return self._get_state()
    
    def step(self, action):
        """Execute action and return next state"""
        total_reward = 0
        done = False
        
        # Convert discrete action to MultiBinary format
        multi_binary_action = self._discrete_to_multi_binary(action)
        
        # Frame skipping
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(multi_binary_action)
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
