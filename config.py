"""
Training Configuration
All adjustable parameters for the training process
"""

# ============================================================================
# Environment Settings
# ============================================================================
GAME = 'Jackal-Nes'  # Game to train on
FRAME_SKIP = 4  # Number of frames to skip (repeat action), higher = faster but less control

# ============================================================================
# Training Hyperparameters
# ============================================================================
MAX_EPISODES = 100000  # Maximum number of episodes to train
MAX_STEPS = 10000  # Maximum steps per episode
UPDATE_INTERVAL = 2048  # Number of steps between policy updates

# ============================================================================
# Reward Shaping Parameters
# ============================================================================
# Life management rewards
LIFE_LOSS_PENALTY = -5.0  # Penalty when lives decrease (negative value)
LIFE_GAIN_BONUS = 10.0  # Bonus when lives increase (extra life gained)

# Movement rewards
UPWARD_SCORE_BONUS = 0.5  # Bonus when moving up AND score increases

# ============================================================================
# Checkpoint Management
# ============================================================================
SAVE_INTERVAL = 50  # Save checkpoint every N episodes
MAX_CHECKPOINTS = 100  # Maximum number of checkpoints to keep
CHECKPOINT_DIR = 'checkpoints'  # Directory to save checkpoints

# ============================================================================
# Visualization Settings
# ============================================================================
HEADLESS = True  # Run in headless mode (no GUI, for background/server training)
RENDER = False  # Enable visualization during training (ignored if HEADLESS=True)
RENDER_INTERVAL = 5  # Render every N episodes (to reduce overhead)
PLOT_UPDATE_INTERVAL = 5  # Update plots every N episodes

# ============================================================================
# Parallel Training Settings
# ============================================================================
N_WORKERS = 32  # Number of parallel environments
SHOW_GAME_SCREENS = True  # Display real-time game screens from all workers
VISUALIZATION_UPDATE_INTERVAL = 0.5  # Update visualization every N seconds (to avoid slowdown)

# ============================================================================
# Model Architecture (for reference, actual implementation in model.py)
# ============================================================================
INPUT_SHAPE = (4, 84, 84)  # (frame_stack, height, width)
# Frame stack: number of consecutive frames to stack as input
# Height/Width: resized game screen dimensions

# ============================================================================
# PPO Algorithm Parameters (for reference, actual implementation in ppo_agent.py)
# ============================================================================
# These are typically set in PPOAgent class, but listed here for reference
# LEARNING_RATE = 3e-4
# GAMMA = 0.99  # Discount factor
# GAE_LAMBDA = 0.95  # GAE parameter
# CLIP_EPSILON = 0.2  # PPO clip parameter
# VALUE_COEF = 0.5  # Value loss coefficient
# ENTROPY_COEF = 0.01  # Entropy bonus coefficient
# MAX_GRAD_NORM = 0.5  # Gradient clipping
# PPO_EPOCHS = 4  # Number of epochs per update
# BATCH_SIZE = 64  # Mini-batch size for PPO update
