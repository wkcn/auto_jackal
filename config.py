"""
Training Configuration Loader
Loads configuration from YAML files for flexible training setups
"""

import yaml
import os
import argparse
from pathlib import Path


class Config:
    """Configuration class that loads settings from YAML files"""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration from YAML file
        
        Args:
            config_path: Path to YAML config file. If None, uses default.yaml
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml')
        
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file with inheritance support"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Handle inheritance
        if 'inherit' in config_dict:
            inherit_path = config_dict.pop('inherit')
            
            # Resolve relative path
            if not os.path.isabs(inherit_path):
                config_dir = os.path.dirname(self.config_path)
                inherit_path = os.path.join(config_dir, inherit_path)
            
            # Load parent config
            if not os.path.exists(inherit_path):
                raise FileNotFoundError(f"Parent config file not found: {inherit_path}")
            
            with open(inherit_path, 'r') as f:
                parent_dict = yaml.safe_load(f)
            
            # Merge configs: child overrides parent
            config_dict = self._deep_merge(parent_dict, config_dict)
        
        self._parse_config(config_dict)
    
    def _deep_merge(self, base, override):
        """
        Deep merge two dictionaries
        
        Args:
            base: Base dictionary (parent config)
            override: Override dictionary (child config)
        
        Returns:
            Merged dictionary where override values take precedence
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = value
        
        return result
    
    def _parse_config(self, config_dict):
        """Parse configuration dictionary and set attributes"""
        
        # Environment settings
        env = config_dict.get('environment', {})
        self.GAME = env.get('game', 'Jackal-Nes')
        self.FRAME_SKIP = env.get('frame_skip', 4)
        
        # Training hyperparameters
        training = config_dict.get('training', {})
        self.MAX_EPISODES = training.get('max_episodes', 100000)
        self.MAX_STEPS = training.get('max_steps', 10000)
        self.UPDATE_INTERVAL = training.get('update_interval', 2048)
        
        # Reward shaping
        rewards = config_dict.get('rewards', {})
        self.LIFE_LOSS_PENALTY = rewards.get('life_loss_penalty', -5.0)
        self.LIFE_GAIN_BONUS = rewards.get('life_gain_bonus', 10.0)
        self.UPWARD_SCORE_BONUS = rewards.get('upward_score_bonus', 0.5)
        
        # Checkpoint management
        checkpoint = config_dict.get('checkpoint', {})
        self.SAVE_INTERVAL = checkpoint.get('save_interval', 50)
        self.MAX_CHECKPOINTS = checkpoint.get('max_checkpoints', 100)
        self.CHECKPOINT_DIR = checkpoint.get('checkpoint_dir', 'checkpoints')
        
        # Visualization settings
        viz = config_dict.get('visualization', {})
        self.HEADLESS = viz.get('headless', True)
        self.RENDER = viz.get('render', False)
        self.RENDER_INTERVAL = viz.get('render_interval', 5)
        self.PLOT_UPDATE_INTERVAL = viz.get('plot_update_interval', 5)
        
        # Parallel training settings
        parallel = config_dict.get('parallel', {})
        self.N_WORKERS = parallel.get('n_workers', 32)
        self.SHOW_GAME_SCREENS = parallel.get('show_game_screens', True)
        self.VISUALIZATION_UPDATE_INTERVAL = parallel.get('visualization_update_interval', 0.5)
        
        # Model architecture
        model = config_dict.get('model', {})
        self.INPUT_SHAPE = tuple(model.get('input_shape', [4, 84, 84]))
        
        # PPO algorithm parameters
        ppo = config_dict.get('ppo', {})
        self.LEARNING_RATE = ppo.get('learning_rate', 3e-4)
        self.GAMMA = ppo.get('gamma', 0.99)
        self.GAE_LAMBDA = ppo.get('gae_lambda', 0.95)
        self.CLIP_EPSILON = ppo.get('clip_epsilon', 0.2)
        self.VALUE_COEF = ppo.get('value_coef', 0.5)
        self.ENTROPY_COEF = ppo.get('entropy_coef', 0.01)
        self.MAX_GRAD_NORM = ppo.get('max_grad_norm', 0.5)
        self.PPO_EPOCHS = ppo.get('ppo_epochs', 4)
        self.BATCH_SIZE = ppo.get('batch_size', 64)
        
        # Logging settings
        logging = config_dict.get('logging', {})
        self.LOGGING_ENABLED = logging.get('enabled', False)
        self.LOGGING_BACKEND = logging.get('backend', 'wandb')
        self.LOGGING_PROJECT = logging.get('project', 'auto-jackal')
        self.LOGGING_ENTITY = logging.get('entity', None)
        self.LOGGING_NAME = logging.get('name', None)
        self.LOGGING_TAGS = logging.get('tags', [])
        self.LOGGING_NOTES = logging.get('notes', '')
        self.LOG_INTERVAL = logging.get('log_interval', 1)
        self.LOG_GRADIENTS = logging.get('log_gradients', False)
        self.LOG_MODEL = logging.get('log_model', False)
    
    def __repr__(self):
        """String representation of configuration"""
        return f"Config(config_path='{self.config_path}')"
    
    def print_config(self):
        """Print all configuration values"""
        print(f"\n{'='*60}")
        print(f"Configuration loaded from: {self.config_path}")
        print(f"{'='*60}")
        
        print("\n[Environment]")
        print(f"  Game: {self.GAME}")
        print(f"  Frame Skip: {self.FRAME_SKIP}")
        
        print("\n[Training]")
        print(f"  Max Episodes: {self.MAX_EPISODES}")
        print(f"  Max Steps: {self.MAX_STEPS}")
        print(f"  Update Interval: {self.UPDATE_INTERVAL}")
        
        print("\n[Rewards]")
        print(f"  Life Loss Penalty: {self.LIFE_LOSS_PENALTY}")
        print(f"  Life Gain Bonus: {self.LIFE_GAIN_BONUS}")
        print(f"  Upward Score Bonus: {self.UPWARD_SCORE_BONUS}")
        
        print("\n[Checkpoint]")
        print(f"  Save Interval: {self.SAVE_INTERVAL}")
        print(f"  Max Checkpoints: {self.MAX_CHECKPOINTS}")
        print(f"  Checkpoint Dir: {self.CHECKPOINT_DIR}")
        
        print("\n[Visualization]")
        print(f"  Headless: {self.HEADLESS}")
        print(f"  Render: {self.RENDER}")
        print(f"  Render Interval: {self.RENDER_INTERVAL}")
        print(f"  Plot Update Interval: {self.PLOT_UPDATE_INTERVAL}")
        
        print("\n[Parallel Training]")
        print(f"  N Workers: {self.N_WORKERS}")
        print(f"  Show Game Screens: {self.SHOW_GAME_SCREENS}")
        print(f"  Visualization Update Interval: {self.VISUALIZATION_UPDATE_INTERVAL}")
        
        print("\n[Model]")
        print(f"  Input Shape: {self.INPUT_SHAPE}")
        
        print("\n[PPO]")
        print(f"  Learning Rate: {self.LEARNING_RATE}")
        print(f"  Gamma: {self.GAMMA}")
        print(f"  GAE Lambda: {self.GAE_LAMBDA}")
        print(f"  Clip Epsilon: {self.CLIP_EPSILON}")
        print(f"  Value Coef: {self.VALUE_COEF}")
        print(f"  Entropy Coef: {self.ENTROPY_COEF}")
        print(f"  Max Grad Norm: {self.MAX_GRAD_NORM}")
        print(f"  PPO Epochs: {self.PPO_EPOCHS}")
        print(f"  Batch Size: {self.BATCH_SIZE}")
        
        print("\n[Logging]")
        print(f"  Enabled: {self.LOGGING_ENABLED}")
        print(f"  Backend: {self.LOGGING_BACKEND}")
        print(f"  Project: {self.LOGGING_PROJECT}")
        print(f"  Entity: {self.LOGGING_ENTITY}")
        print(f"  Name: {self.LOGGING_NAME}")
        print(f"  Tags: {self.LOGGING_TAGS}")
        print(f"  Log Interval: {self.LOG_INTERVAL}")
        print(f"  Log Gradients: {self.LOG_GRADIENTS}")
        print(f"  Log Model: {self.LOG_MODEL}")
        print(f"{'='*60}\n")


def get_config_from_args():
    """
    Parse command line arguments and return Config object
    
    Usage:
        python train.py --config configs/default.yaml
        python train.py --config configs/high_performance.yaml
        python train.py --config configs/debug.yaml
    """
    parser = argparse.ArgumentParser(description='Train PPO agent with custom config')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file (default: configs/default.yaml)')
    
    args, unknown = parser.parse_known_args()
    
    config = Config(args.config)
    return config


# For backward compatibility, create a default config instance
# This allows existing code to import config values directly
_default_config = Config()

# Export all config values as module-level variables
GAME = _default_config.GAME
FRAME_SKIP = _default_config.FRAME_SKIP
MAX_EPISODES = _default_config.MAX_EPISODES
MAX_STEPS = _default_config.MAX_STEPS
UPDATE_INTERVAL = _default_config.UPDATE_INTERVAL
LIFE_LOSS_PENALTY = _default_config.LIFE_LOSS_PENALTY
LIFE_GAIN_BONUS = _default_config.LIFE_GAIN_BONUS
UPWARD_SCORE_BONUS = _default_config.UPWARD_SCORE_BONUS
SAVE_INTERVAL = _default_config.SAVE_INTERVAL
MAX_CHECKPOINTS = _default_config.MAX_CHECKPOINTS
CHECKPOINT_DIR = _default_config.CHECKPOINT_DIR
HEADLESS = _default_config.HEADLESS
RENDER = _default_config.RENDER
RENDER_INTERVAL = _default_config.RENDER_INTERVAL
PLOT_UPDATE_INTERVAL = _default_config.PLOT_UPDATE_INTERVAL
N_WORKERS = _default_config.N_WORKERS
SHOW_GAME_SCREENS = _default_config.SHOW_GAME_SCREENS
VISUALIZATION_UPDATE_INTERVAL = _default_config.VISUALIZATION_UPDATE_INTERVAL
INPUT_SHAPE = _default_config.INPUT_SHAPE
LEARNING_RATE = _default_config.LEARNING_RATE
GAMMA = _default_config.GAMMA
GAE_LAMBDA = _default_config.GAE_LAMBDA
CLIP_EPSILON = _default_config.CLIP_EPSILON
VALUE_COEF = _default_config.VALUE_COEF
ENTROPY_COEF = _default_config.ENTROPY_COEF
MAX_GRAD_NORM = _default_config.MAX_GRAD_NORM
PPO_EPOCHS = _default_config.PPO_EPOCHS
BATCH_SIZE = _default_config.BATCH_SIZE
LOGGING_ENABLED = _default_config.LOGGING_ENABLED
LOGGING_BACKEND = _default_config.LOGGING_BACKEND
LOGGING_PROJECT = _default_config.LOGGING_PROJECT
LOGGING_ENTITY = _default_config.LOGGING_ENTITY
LOGGING_NAME = _default_config.LOGGING_NAME
LOGGING_TAGS = _default_config.LOGGING_TAGS
LOGGING_NOTES = _default_config.LOGGING_NOTES
LOG_INTERVAL = _default_config.LOG_INTERVAL
LOG_GRADIENTS = _default_config.LOG_GRADIENTS
LOG_MODEL = _default_config.LOG_MODEL


if __name__ == '__main__':
    # Test config loading
    config = get_config_from_args()
    config.print_config()
