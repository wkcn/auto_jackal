import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import os
import glob
import re
from env_wrapper import RetroWrapper
from ppo_agent import PPOAgent
import config


class Trainer:
    """Training manager with visualization"""
    
    def __init__(self, game=None, render=None, save_interval=None, max_checkpoints=None, frame_skip=None):
        # Use config values as defaults
        self.env = RetroWrapper(game=game or config.GAME)
        self.render = render if render is not None else config.RENDER
        self.save_interval = save_interval or config.SAVE_INTERVAL
        self.max_checkpoints = max_checkpoints or config.MAX_CHECKPOINTS
        self.frame_skip = frame_skip or config.FRAME_SKIP  # Number of frames to skip (repeat action)
        self.checkpoint_dir = config.CHECKPOINT_DIR
        
        # Initialize agent
        input_shape = config.INPUT_SHAPE  # (frame_stack, height, width)
        self.agent = PPOAgent(input_shape, self.env.n_actions)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.start_episode = 0
        self.global_step = 0
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load latest checkpoint if exists
        self.load_latest_checkpoint()
        
        # Visualization setup
        if self.render:
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
    
    def train(self, max_episodes=None, max_steps=None, update_interval=None, 
                life_loss_penalty=None, life_gain_bonus=None, upward_score_bonus=None):
        # Use config values as defaults
        max_episodes = max_episodes or config.MAX_EPISODES
        max_steps = max_steps or config.MAX_STEPS
        update_interval = update_interval or config.UPDATE_INTERVAL
        life_loss_penalty = life_loss_penalty if life_loss_penalty is not None else config.LIFE_LOSS_PENALTY
        life_gain_bonus = life_gain_bonus if life_gain_bonus is not None else config.LIFE_GAIN_BONUS
        upward_score_bonus = upward_score_bonus if upward_score_bonus is not None else config.UPWARD_SCORE_BONUS
        """Train the agent
        
        Args:
            max_episodes: Maximum number of episodes to train
            max_steps: Maximum steps per episode
            update_interval: Number of steps between policy updates
            life_loss_penalty: Penalty to apply when lives decrease (negative value)
            life_gain_bonus: Bonus reward when lives increase (positive value)
            upward_score_bonus: Bonus reward when moving up AND score increases (positive value)
        """
        print(f"Training on device: {self.agent.device}")
        print(f"Action space: {self.env.n_actions}")
        print(f"Frame skip: {self.frame_skip} (agent decides every {self.frame_skip} frames)")
        print(f"Life loss penalty: {life_loss_penalty}")
        print(f"Life gain bonus: {life_gain_bonus}")
        print(f"Upward score bonus: {upward_score_bonus}")
        print(f"Starting from episode: {self.start_episode}")
        print(f"Global step: {self.global_step}")
        
        for episode in range(self.start_episode, max_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Initialize info tracking
            prev_lives = None
            prev_score = 0  # Track score for upward movement bonus
            
            for step in range(max_steps):
                # Select action (only once per frame_skip frames)
                action, log_prob, value = self.agent.select_action(state)
                
                # Execute the same action for frame_skip frames
                frame_reward = 0
                life_penalty = 0
                life_bonus = 0
                upward_bonus = 0
                
                # Check if UP button is pressed (index 4 in NES controller)
                # Action is a numpy array of shape (9,) with 0s and 1s
                is_moving_up = False
                if hasattr(action, '__len__') and len(action) > 4:
                    is_moving_up = (action[4] > 0.5)  # UP button pressed
                
                for _ in range(self.frame_skip):
                    # Take action
                    next_state, reward, done, info = self.env.step(action)
                    
                    # Process info: check for life changes
                    if info and 'lives' in info:
                        current_lives = info['lives']
                        if prev_lives is not None:
                            if current_lives < prev_lives:
                                # Lives decreased, apply penalty
                                life_penalty += life_loss_penalty
                                print(f"  ⚠️  Life lost! Lives: {prev_lives} -> {current_lives}, Penalty: {life_loss_penalty}")
                            elif current_lives > prev_lives:
                                # Lives increased, apply bonus!
                                life_bonus += life_gain_bonus
                                print(f"  ❤️  Extra life gained! Lives: {prev_lives} -> {current_lives}, Bonus: +{life_gain_bonus}")
                        prev_lives = current_lives
                    
                    # Process info: check for score increase with upward movement
                    if info and 'score' in info:
                        current_score = info['score']
                        if is_moving_up and current_score > prev_score:
                            # Moving up AND score increased, apply bonus
                            score_increase = current_score - prev_score
                            upward_bonus += upward_score_bonus
                            print(f"  ⬆️  Upward progress! Score: {prev_score} -> {current_score} (+{score_increase}), Bonus: +{upward_score_bonus}")
                        prev_score = current_score
                    
                    # Render game
                    if self.render and episode % config.RENDER_INTERVAL == 0:
                        self.env.render()
                    
                    # Accumulate reward from skipped frames
                    frame_reward += reward
                    episode_length += 1
                    
                    if done:
                        break
                
                # Apply life penalty/bonus and upward bonus to the frame reward
                frame_reward += life_penalty + life_bonus + upward_bonus
                
                # Store transition (with accumulated reward from skipped frames)
                self.agent.store_transition(state, action, frame_reward, value, log_prob, done)
                
                state = next_state
                episode_reward += frame_reward
                self.global_step += 1
                
                # Update policy
                if self.global_step % update_interval == 0:
                    loss = self.agent.update(next_state)
                    self.losses.append(loss)
                    print(f"Step {self.global_step}: Loss = {loss:.4f}")
                
                if done:
                    break
            
            # Record episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Print progress
            avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
            print(f"Episode {episode + 1}/{max_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Length: {episode_length} | "
                  f"Score: {prev_score} | "
                  f"Lives: {prev_lives if prev_lives is not None else 'N/A'} | "
                  f"Avg Reward (100): {avg_reward:.2f}")
            
            # Update visualization
            if self.render and episode % config.PLOT_UPDATE_INTERVAL == 0:
                self.update_plots()
            
            # Save checkpoint every save_interval episodes
            if (episode + 1) % self.save_interval == 0:
                self.save_checkpoint(episode + 1)
        
        self.env.close()
        print("Training completed!")
        
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
    """Main training function"""
    # Initialize trainer with config defaults
    # All parameters can be overridden by passing arguments
    # If not specified, values from config.py will be used
    trainer = Trainer()
    
    # Start training (will auto-resume from latest checkpoint if exists)
    # All parameters use config.py defaults unless overridden
    trainer.train()
    
    # Keep plot window open
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
