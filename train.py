import torch
import torch.multiprocessing as mp
import numpy as np
import os
import sys

# Import config first to check HEADLESS mode
from config import get_config_from_args

# Load configuration from command line arguments
config = get_config_from_args()
config.print_config()

# Set matplotlib backend before importing pyplot
if config.HEADLESS:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for headless mode
    print("Running in HEADLESS mode (no GUI)")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import glob
import re
from collections import deque
from env_wrapper import RetroWrapper
from ppo_agent import PPOAgent

# Initialize logging backend (wandb or swanlab)
logger = None
if config.LOGGING_ENABLED:
    try:
        if config.LOGGING_BACKEND == 'wandb':
            import wandb
            logger = wandb
            print(f"✓ Loaded wandb for experiment tracking")
        elif config.LOGGING_BACKEND == 'swanlab':
            import swanlab
            logger = swanlab
            print(f"✓ Loaded swanlab for experiment tracking")
        else:
            print(f"⚠️  Unknown logging backend: {config.LOGGING_BACKEND}")
            print(f"   Supported backends: 'wandb', 'swanlab'")
            config.LOGGING_ENABLED = False
    except ImportError as e:
        print(f"⚠️  Failed to import {config.LOGGING_BACKEND}: {e}")
        print(f"   Install with: pip install {config.LOGGING_BACKEND}")
        config.LOGGING_ENABLED = False


def worker_process(worker_id, game, frame_skip, task_queue, result_queue, max_steps, 
                   life_loss_penalty, life_gain_bonus, upward_score_bonus):
    """Worker process that runs episodes in parallel (without rendering)"""
    # Set different random seeds for each worker
    np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
    torch.manual_seed(worker_id * 1000 + int(time.time()) % 1000)
    
    # Create environment for this worker
    env = RetroWrapper(game=game)
    
    # Create local policy for action selection
    input_shape = config.INPUT_SHAPE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Workers use CPU to avoid GPU memory issues
    from model import ActorCritic
    policy = ActorCritic(input_shape, env.n_actions).to(device)
    
    print(f"Worker {worker_id} started")
    
    while True:
        # Get task
        task = task_queue.get()
        if task is None:
            break
        
        episode_num, policy_state = task
        
        # Update local policy
        policy.load_state_dict(policy_state)
        policy.eval()
        
        # Run episode
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        transitions = []
        
        # Initialize info tracking
        prev_lives = None
        prev_score = 0
        prev_upward_score = 0
        
        for step in range(max_steps):
            # Select action using local policy
            state_tensor = torch.tensor(state, device=device)
            with torch.no_grad():
                action, log_prob, value = policy.act(state_tensor)
            
            # Execute the same action for frame_skip frames
            frame_reward = 0
            life_penalty = 0
            life_bonus = 0
            upward_bonus = 0
            score_reward = 0
            
            # Check if UP button is pressed
            is_moving_up = False
            if hasattr(action, '__len__') and len(action) > 4:
                is_moving_up = (action[4] > 0.5)
            
            for _ in range(frame_skip):
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Process info: check for life changes
                if info and 'lives' in info:
                    current_lives = info['lives']
                    if prev_lives is not None:
                        if current_lives < prev_lives:
                            life_penalty += life_loss_penalty
                        elif current_lives > prev_lives:
                            life_bonus += life_gain_bonus
                    prev_lives = current_lives
                
                # Process info: check for score increase with upward movement
                if info and 'score' in info:
                    current_score = info['score']
                    if is_moving_up and current_score > prev_upward_score:
                        upward_bonus += upward_score_bonus
                        prev_upward_score = current_score
                    if current_score > prev_score:
                        score_reward += current_score - prev_score
                        prev_score = current_score
                
                # Accumulate reward
                episode_length += 1
                
                if done:
                    break
            
            # Apply penalties/bonuses
            frame_reward += life_penalty + life_bonus + upward_bonus + score_reward
            
            # Store transition
            transitions.append((
                state, action, frame_reward,
                value.item(), log_prob.item(), done
            ))
            
            state = next_state
            episode_reward += frame_reward
            
            if done:
                break
        
        # Send result back
        result_queue.put((episode_num, episode_reward, episode_length, transitions, prev_score, prev_lives))
    
    env.close()
    print(f"Worker {worker_id} stopped")


class Trainer:
    """Training manager with visualization and parallel workers"""
    
    def __init__(self, game=None, render=None, save_interval=None, max_checkpoints=None, frame_skip=None, n_workers=None):
        # Use config values as defaults
        self.game = game or config.GAME
        # Force render=False in headless mode
        if config.HEADLESS:
            self.render = False
        else:
            self.render = render if render is not None else config.RENDER
        self.save_interval = save_interval or config.SAVE_INTERVAL
        self.max_checkpoints = max_checkpoints or config.MAX_CHECKPOINTS
        self.frame_skip = frame_skip or config.FRAME_SKIP  # Number of frames to skip (repeat action)
        self.n_workers = n_workers or config.N_WORKERS  # Number of parallel workers
        self.checkpoint_dir = config.CHECKPOINT_DIR
        
        # Create main environment (for rendering)
        self.env = RetroWrapper(game=self.game)
        
        # Initialize agent
        input_shape = config.INPUT_SHAPE  # (frame_stack, height, width)
        self.agent = PPOAgent(input_shape, self.env.n_actions)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.start_episode = 0
        self.global_step = 0
        
        # Logging
        self.logger = logger
        self.logging_enabled = config.LOGGING_ENABLED
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load latest checkpoint if exists
        self.load_latest_checkpoint()
        
        # Initialize logging
        if self.logging_enabled:
            self.init_logging()
        
        # Visualization setup (only if not in headless mode)
        if self.render and not config.HEADLESS:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('Training Progress')
            
    def load_latest_checkpoint(self):
        """Load the latest checkpoint if exists"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.pth'))
        
        if not checkpoint_files:
            print("No checkpoint found. Starting from scratch.")
            return
        
        # Extract episode numbers and find the latest
        checkpoint_episodes = []
        for f in checkpoint_files:
            match = re.search(r'checkpoint_(\d+)\.pth', f)
            if match:
                checkpoint_episodes.append((int(match.group(1)), f))
        
        if not checkpoint_episodes:
            print("No valid checkpoint found. Starting from scratch.")
            return
        
        # Sort by episode number and get the latest
        checkpoint_episodes.sort(key=lambda x: x[0], reverse=True)
        latest_episode, latest_checkpoint = checkpoint_episodes[0]
        
        print(f"Loading checkpoint: {latest_checkpoint}")
        try:
            checkpoint = self.agent.load(latest_checkpoint)
            self.start_episode = checkpoint.get('episode', 0)
            self.global_step = checkpoint.get('global_step', 0)
            
            # Restore training statistics if available
            if 'training_stats' in checkpoint:
                stats = checkpoint['training_stats']
                self.episode_rewards = stats.get('episode_rewards', [])
                self.episode_lengths = stats.get('episode_lengths', [])
                self.losses = stats.get('losses', [])
            
            print(f"Resumed from episode {self.start_episode}, global step {self.global_step}")
            print(f"Loaded {len(self.episode_rewards)} episode records")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch.")
            self.start_episode = 0
            self.global_step = 0
    
    def save_checkpoint(self, episode):
        """Save checkpoint and manage checkpoint files"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{episode}.pth')
        
        # Prepare training statistics
        training_stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
        }
        
        # Save checkpoint
        self.agent.save(
            checkpoint_path,
            episode=episode,
            global_step=self.global_step,
            training_stats=training_stats
        )
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Manage checkpoint files (keep only max_checkpoints)
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the latest max_checkpoints"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.pth'))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Extract episode numbers
        checkpoint_episodes = []
        for f in checkpoint_files:
            match = re.search(r'checkpoint_(\d+)\.pth', f)
            if match:
                checkpoint_episodes.append((int(match.group(1)), f))
        
        # Sort by episode number (descending)
        checkpoint_episodes.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old checkpoints
        for _, checkpoint_path in checkpoint_episodes[self.max_checkpoints:]:
            try:
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Failed to remove {checkpoint_path}: {e}")
    
    def init_logging(self):
        """Initialize wandb or swanlab logging"""
        if not self.logging_enabled or self.logger is None:
            return
        
        # Prepare config dict for logging
        log_config = {
            'game': self.game,
            'frame_skip': self.frame_skip,
            'n_workers': self.n_workers,
            'max_episodes': config.MAX_EPISODES,
            'max_steps': config.MAX_STEPS,
            'update_interval': config.UPDATE_INTERVAL,
            'learning_rate': config.LEARNING_RATE,
            'gamma': config.GAMMA,
            'gae_lambda': config.GAE_LAMBDA,
            'clip_epsilon': config.CLIP_EPSILON,
            'value_coef': config.VALUE_COEF,
            'entropy_coef': config.ENTROPY_COEF,
            'max_grad_norm': config.MAX_GRAD_NORM,
            'ppo_epochs': config.PPO_EPOCHS,
            'batch_size': config.BATCH_SIZE,
            'life_loss_penalty': config.LIFE_LOSS_PENALTY,
            'life_gain_bonus': config.LIFE_GAIN_BONUS,
            'upward_score_bonus': config.UPWARD_SCORE_BONUS,
        }
        
        try:
            self.logger.init(
                project=config.LOGGING_PROJECT,
                entity=config.LOGGING_ENTITY,
                name=config.LOGGING_NAME,
                config=log_config,
                tags=config.LOGGING_TAGS,
                notes=config.LOGGING_NOTES,
                resume='allow',  # Allow resuming runs
            )
            print(f"✓ Initialized {config.LOGGING_BACKEND} logging")
            print(f"  Project: {config.LOGGING_PROJECT}")
            if config.LOGGING_NAME:
                print(f"  Run name: {config.LOGGING_NAME}")
        except Exception as e:
            print(f"⚠️  Failed to initialize logging: {e}")
            self.logging_enabled = False
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to wandb or swanlab"""
        if not self.logging_enabled or self.logger is None:
            return
        
        try:
            if step is not None:
                self.logger.log(metrics, step=step)
            else:
                self.logger.log(metrics)
        except Exception as e:
            print(f"⚠️  Failed to log metrics: {e}")
    
    def train(self, max_episodes=None, max_steps=None, update_interval=None, 
                life_loss_penalty=None, life_gain_bonus=None, upward_score_bonus=None):
        """Train the agent with parallel workers and one rendering environment
        
        Args:
            max_episodes: Maximum number of episodes to train
            max_steps: Maximum steps per episode
            update_interval: Number of steps between policy updates
            life_loss_penalty: Penalty to apply when lives decrease (negative value)
            life_gain_bonus: Bonus reward when lives increase (positive value)
            upward_score_bonus: Bonus reward when moving up AND score increases (positive value)
        """
        # Use config values as defaults
        max_episodes = max_episodes or config.MAX_EPISODES
        max_steps = max_steps or config.MAX_STEPS
        update_interval = update_interval or config.UPDATE_INTERVAL
        life_loss_penalty = life_loss_penalty if life_loss_penalty is not None else config.LIFE_LOSS_PENALTY
        life_gain_bonus = life_gain_bonus if life_gain_bonus is not None else config.LIFE_GAIN_BONUS
        upward_score_bonus = upward_score_bonus if upward_score_bonus is not None else config.UPWARD_SCORE_BONUS
        
        print(f"Training on device: {self.agent.device}")
        print(f"Action space: {self.env.n_actions}")
        print(f"Frame skip: {self.frame_skip} (agent decides every {self.frame_skip} frames)")
        print(f"Number of workers: {self.n_workers}")
        print(f"Life loss penalty: {life_loss_penalty}")
        print(f"Life gain bonus: {life_gain_bonus}")
        print(f"Upward score bonus: {upward_score_bonus}")
        print(f"Starting from episode: {self.start_episode}")
        print(f"Global step: {self.global_step}")
        
        # Create task and result queues
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Start worker processes
        workers = []
        for i in range(self.n_workers):
            worker = mp.Process(
                target=worker_process,
                args=(i, self.game, self.frame_skip, task_queue, result_queue, max_steps,
                      life_loss_penalty, life_gain_bonus, upward_score_bonus)
            )
            worker.start()
            workers.append(worker)
        
        # Training loop
        episode = self.start_episode
        pending_episodes = {}  # Track episodes sent to workers
        next_episode_to_process = self.start_episode
        
        # Also run one episode in main process for rendering
        render_episode_interval = config.RENDER_INTERVAL if self.render else float('inf')
        
        while episode < max_episodes:
            # Distribute tasks to workers
            while len(pending_episodes) < self.n_workers and episode < max_episodes:
                # Get current policy state
                policy_state = self.agent.policy.state_dict()
                
                # Send task to worker
                task_queue.put((episode, policy_state))
                pending_episodes[episode] = True
                episode += 1
            
            # Collect results from workers
            if not result_queue.empty():
                episode_num, episode_reward, episode_length, transitions, final_score, final_lives = result_queue.get()
                
                # Store transitions in agent's memory
                for state, action, reward, value, log_prob, done in transitions:
                    self.agent.store_transition(state, action, reward, value, log_prob, done)
                    self.global_step += 1
                    
                    # Update policy
                    if self.global_step % update_interval == 0:
                        # Get the last state for value estimation
                        last_state = transitions[-1][0] if transitions else state
                        metrics = self.agent.update(last_state)
                        
                        # Store total loss for backward compatibility
                        total_loss = metrics['loss/total']
                        self.losses.append(total_loss)
                        print(f"Step {self.global_step}: Loss = {total_loss:.4f}")
                        
                        # Log training metrics
                        if self.logging_enabled:
                            log_data = {
                                'global_step': self.global_step,
                                **metrics,
                            }
                            self.log_metrics(log_data, step=episode_num + 1)

                # Record episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Remove from pending
                del pending_episodes[episode_num]
                
                # Print progress
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                avg_length = np.mean(self.episode_lengths[-100:]) if len(self.episode_lengths) >= 100 else np.mean(self.episode_lengths)
                print(f"Episode {episode_num + 1}/{max_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Length: {episode_length} | "
                      f"Score: {final_score} | "
                      f"Lives: {final_lives if final_lives is not None else 'N/A'} | "
                      f"Avg Reward (100): {avg_reward:.2f}")
                
                # Log episode metrics
                if self.logging_enabled and (episode_num + 1) % config.LOG_INTERVAL == 0:
                    log_data = {
                        'episode': episode_num + 1,
                        'episode/reward': episode_reward,
                        'episode/length': episode_length,
                        'episode/score': final_score,
                        'episode/avg_reward_100': avg_reward,
                        'episode/avg_length_100': avg_length,
                    }
                    if final_lives is not None:
                        log_data['episode/lives'] = final_lives
                    self.log_metrics(log_data, step=episode_num + 1)
                
                # Update visualization
                if self.render and (episode_num + 1) % config.PLOT_UPDATE_INTERVAL == 0:
                    self.update_plots()
                
                # Save checkpoint
                if (episode_num + 1) % self.save_interval == 0:
                    self.save_checkpoint(episode_num + 1)
            
            # Run rendering episode in main process
            if self.render and (episode % render_episode_interval == 0):
                self.run_render_episode(max_steps, life_loss_penalty, life_gain_bonus, upward_score_bonus)
            
            # Small sleep to avoid busy waiting
            time.sleep(0.001)
        
        # Wait for all pending episodes to complete
        print("Waiting for workers to finish...")
        while pending_episodes:
            episode_num, episode_reward, episode_length, transitions, final_score, final_lives = result_queue.get()
            
            # Store transitions
            for state, action, reward, value, log_prob, done in transitions:
                self.agent.store_transition(state, action, reward, value, log_prob, done)
                self.global_step += 1
            
            # Record statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            del pending_episodes[episode_num]
            
            print(f"Episode {episode_num + 1} completed (cleanup)")
        
        # Stop workers
        print("Stopping workers...")
        for _ in range(self.n_workers):
            task_queue.put(None)
        
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
    
        self.env.close()
        print("Training completed!")
        
        # Finish logging
        if self.logging_enabled and self.logger is not None:
            try:
                self.logger.finish()
                print(f"✓ Logging finished")
            except Exception as e:
                print(f"⚠️  Failed to finish logging: {e}")
    
    def run_render_episode(self, max_steps, life_loss_penalty, life_gain_bonus, upward_score_bonus):
        """Run one episode in main process with rendering"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Initialize info tracking
        prev_lives = None
        prev_score = 0
        
        for step in range(max_steps):
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            
            # Execute action with frame skip
            frame_reward = 0
            life_penalty = 0
            life_bonus = 0
            upward_bonus = 0
            
            # Check if UP button is pressed
            is_moving_up = False
            if hasattr(action, '__len__') and len(action) > 4:
                is_moving_up = (action[4] > 0.5)
            
            for _ in range(self.frame_skip):
                # Take action
                next_state, reward, done, info = self.env.step(action)
                
                # Render
                self.env.render()
                
                # Process info
                if info and 'lives' in info:
                    current_lives = info['lives']
                    if prev_lives is not None:
                        if current_lives < prev_lives:
                            life_penalty += life_loss_penalty
                        elif current_lives > prev_lives:
                            life_bonus += life_gain_bonus
                    prev_lives = current_lives
                
                if info and 'score' in info:
                    current_score = info['score']
                    if is_moving_up and current_score > prev_score:
                        upward_bonus += upward_score_bonus
                    prev_score = current_score
                
                frame_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Apply penalties/bonuses
            frame_reward += life_penalty + life_bonus + upward_bonus
            
            state = next_state
            episode_reward += frame_reward
            
            if done:
                break
        
        print(f"[RENDER] Episode reward: {episode_reward:.2f}, Length: {episode_length}, Score: {prev_score}")
        
    def update_plots(self):
        """Update training visualization"""
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Episode Rewards
        if len(self.episode_rewards) > 0:
            self.axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
            if len(self.episode_rewards) >= 10:
                moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                self.axes[0, 0].plot(range(9, len(self.episode_rewards)), moving_avg, 
                                     'r-', linewidth=2, label='Moving Avg (10)')
            self.axes[0, 0].set_xlabel('Episode')
            self.axes[0, 0].set_ylabel('Reward')
            self.axes[0, 0].set_title('Episode Rewards')
            self.axes[0, 0].legend()
            self.axes[0, 0].grid(True)
        
        # Plot 2: Episode Lengths
        if len(self.episode_lengths) > 0:
            self.axes[0, 1].plot(self.episode_lengths, alpha=0.6)
            self.axes[0, 1].set_xlabel('Episode')
            self.axes[0, 1].set_ylabel('Length')
            self.axes[0, 1].set_title('Episode Lengths')
            self.axes[0, 1].grid(True)
        
        # Plot 3: Training Loss
        if len(self.losses) > 0:
            self.axes[1, 0].plot(self.losses, alpha=0.6)
            self.axes[1, 0].set_xlabel('Update Step')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].set_title('Training Loss')
            self.axes[1, 0].grid(True)
        
        # Plot 4: Average Reward Trend
        if len(self.episode_rewards) >= 50:
            window_sizes = [10, 50, 100]
            for window in window_sizes:
                if len(self.episode_rewards) >= window:
                    moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                    self.axes[1, 1].plot(range(window-1, len(self.episode_rewards)), 
                                        moving_avg, label=f'MA-{window}')
            self.axes[1, 1].set_xlabel('Episode')
            self.axes[1, 1].set_ylabel('Average Reward')
            self.axes[1, 1].set_title('Reward Trends')
            self.axes[1, 1].legend()
            self.axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.pause(0.001)


def main():
    """Main training function with parallel workers"""
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Initialize trainer with config defaults
    # All parameters can be overridden by passing arguments
    # If not specified, values from config.py will be used
    trainer = Trainer()
    
    # Start training (will auto-resume from latest checkpoint if exists)
    # All parameters use config.py defaults unless overridden
    trainer.train()
    
    # Keep plot window open (only if not in headless mode)
    if trainer.render and not config.HEADLESS:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
