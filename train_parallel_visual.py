import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import os
import glob
import re
from collections import deque
from env_wrapper import RetroWrapper
from ppo_agent import PPOAgent
import config


class ParallelTrainerWithVisual:    """Parallel training manager with multiple environments and visual display"""
    
    def __init__(self, game=None, n_workers=None, render=None, 
                 save_interval=None, max_checkpoints=None, show_game_screens=None):
        # Use config values as defaults
        self.game = game or config.GAME
        self.n_workers = n_workers or config.N_WORKERS
        self.render = render if render is not None else config.RENDER
        self.show_game_screens = show_game_screens if show_game_screens is not None else config.SHOW_GAME_SCREENS
        self.save_interval = save_interval or config.SAVE_INTERVAL
        self.max_checkpoints = max_checkpoints or config.MAX_CHECKPOINTS
        self.checkpoint_dir = config.CHECKPOINT_DIR
        
        # Create a single environment to get dimensions
        temp_env = RetroWrapper(game=self.game)
        input_shape = config.INPUT_SHAPE
        self.n_actions = temp_env.n_actions
        temp_env.close()
        
        # Initialize shared agent
        self.agent = PPOAgent(input_shape, self.n_actions)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.start_episode = 0
        self.global_step = 0
        
        # Worker screens (shared memory for displaying game frames)
        if self.show_game_screens:
            self.manager = mp.Manager()
            self.worker_screens = self.manager.list([None] * n_workers)
            self.worker_stats = self.manager.list([{'reward': 0, 'steps': 0}] * n_workers)
            
            # Create a separate window for real-time simulation
            self.sim_fig = None
            self.sim_ax = None
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Load latest checkpoint if exists
        self.load_latest_checkpoint()
        
        # Visualization setup
        if self.render:
            plt.ion()
            if self.show_game_screens:
                # Create figure with game screens and training plots
                self.fig = plt.figure(figsize=(20, 12))
                gs = GridSpec(4, 6, figure=self.fig, hspace=0.3, wspace=0.3)
                
                # Top 2 rows: 8 game screens (4x2 layout) - COLOR
                self.screen_axes = []
                self.screen_images = []  # Store image objects for faster update
                for i in range(2):
                    for j in range(4):
                        ax = self.fig.add_subplot(gs[i, j:j+1])
                        ax.set_title(f'Worker {i*4+j}', fontsize=10)
                        ax.axis('off')
                        self.screen_axes.append(ax)
                        # Pre-create image object
                        img = ax.imshow(np.zeros((84, 84, 3), dtype=np.uint8))
                        self.screen_images.append(img)
                
                # Bottom 2 rows: 4 training plots (2x2 layout)
                self.plot_axes = []
                self.plot_axes.append(self.fig.add_subplot(gs[2, 0:3]))  # Episode Rewards
                self.plot_axes.append(self.fig.add_subplot(gs[2, 3:6]))  # Episode Lengths
                self.plot_axes.append(self.fig.add_subplot(gs[3, 0:3]))  # Training Loss
                self.plot_axes.append(self.fig.add_subplot(gs[3, 3:6]))  # Reward Trends
                
                self.fig.suptitle(f'Parallel Training with {n_workers} Workers (Color Display)', fontsize=14, fontweight='bold')
                
                # Create separate window for real-time simulation (larger view)
                self.sim_fig, self.sim_ax = plt.subplots(figsize=(8, 8))
                self.sim_fig.canvas.manager.set_window_title('Real-time Simulation - Worker 0')
                self.sim_ax.axis('off')
                self.sim_image = self.sim_ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
                self.sim_ax.set_title('Worker 0 - Real-time View', fontsize=14, fontweight='bold')
            else:
                # Original layout without game screens
                self.fig, self.plot_axes = plt.subplots(2, 2, figsize=(12, 8))
                self.plot_axes = self.plot_axes.flatten()
                self.fig.suptitle(f'Parallel Training Progress ({n_workers} workers)')
    
    def load_latest_checkpoint(self):
        """Load the latest checkpoint if exists"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.pth'))
        
        if not checkpoint_files:
            print("No checkpoint found. Starting from scratch.")
            return
        
        checkpoint_episodes = []
        for f in checkpoint_files:
            match = re.search(r'checkpoint_(\d+)\.pth', f)
            if match:
                checkpoint_episodes.append((int(match.group(1)), f))
        
        if not checkpoint_episodes:
            print("No valid checkpoint found. Starting from scratch.")
            return
        
        checkpoint_episodes.sort(key=lambda x: x[0], reverse=True)
        latest_episode, latest_checkpoint = checkpoint_episodes[0]
        
        print(f"Loading checkpoint: {latest_checkpoint}")
        try:
            checkpoint = self.agent.load(latest_checkpoint)
            self.start_episode = checkpoint.get('episode', 0)
            self.global_step = checkpoint.get('global_step', 0)
            
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
        
        training_stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
        }
        
        self.agent.save(
            checkpoint_path,
            episode=episode,
            global_step=self.global_step,
            training_stats=training_stats
        )
        print(f"Checkpoint saved: {checkpoint_path}")
        
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the latest max_checkpoints"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, 'checkpoint_*.pth'))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        checkpoint_episodes = []
        for f in checkpoint_files:
            match = re.search(r'checkpoint_(\d+)\.pth', f)
            if match:
                checkpoint_episodes.append((int(match.group(1)), f))
        
        checkpoint_episodes.sort(key=lambda x: x[0], reverse=True)
        
        for _, checkpoint_path in checkpoint_episodes[self.max_checkpoints:]:
            try:
                os.remove(checkpoint_path)
                print(f"Removed old checkpoint: {checkpoint_path}")
            except Exception as e:
                print(f"Failed to remove {checkpoint_path}: {e}")
    
    def train(self, max_episodes=None, max_steps=None, update_interval=None):
        """Train the agent with parallel environments"""
        # Use config values as defaults
        max_episodes = max_episodes or config.MAX_EPISODES
        max_steps = max_steps or config.MAX_STEPS
        update_interval = update_interval or config.UPDATE_INTERVAL
        print(f"Training on device: {self.agent.device}")
        print(f"Action space: {self.n_actions}")
        print(f"Number of parallel workers: {self.n_workers}")
        print(f"Starting from episode: {self.start_episode}")
        print(f"Global step: {self.global_step}")
        print(f"Show game screens: {self.show_game_screens}")
        
        # Create multiprocessing queues for communication
        task_queue = mp.Queue()
        result_queue = mp.Queue()
        
        # Start worker processes
        workers = []
        for worker_id in range(self.n_workers):
            if self.show_game_screens:
                p = mp.Process(
                    target=self.worker_process_with_visual,
                    args=(worker_id, task_queue, result_queue, max_steps, 
                          self.worker_screens, self.worker_stats)
                )
            else:
                p = mp.Process(
                    target=self.worker_process,
                    args=(worker_id, task_queue, result_queue, max_steps)
                )
            p.start()
            workers.append(p)
        
        episode = self.start_episode
        episodes_completed = 0
        last_update_time = time.time()
        
        try:
            while episode < max_episodes:
                # Send tasks to workers (one episode per worker)
                for worker_id in range(self.n_workers):
                    if episode + worker_id < max_episodes:
                        # Get current policy state
                        policy_state = self.agent.policy.state_dict()
                        task_queue.put((episode + worker_id, policy_state))
                
                # Collect results from workers
                batch_results = []
                for _ in range(min(self.n_workers, max_episodes - episode)):
                    result = result_queue.get()
                    batch_results.append(result)
                    episodes_completed += 1
                
                # Process results
                for result in batch_results:
                    ep_num, ep_reward, ep_length, transitions = result
                    
                    # Store episode statistics
                    self.episode_rewards.append(ep_reward)
                    self.episode_lengths.append(ep_length)
                    
                    # Store transitions in agent
                    for state, action, reward, value, log_prob, done in transitions:
                        self.agent.store_transition(state, action, reward, value, log_prob, done)
                        self.global_step += 1
                    
                    # Print progress
                    avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                    print(f"Episode {ep_num + 1}/{max_episodes} | "
                          f"Reward: {ep_reward:.2f} | "
                          f"Length: {ep_length} | "
                          f"Avg Reward (100): {avg_reward:.2f}")
                
                # Update policy after collecting enough data
                if len(self.agent.states) >= update_interval:
                    # Use the last state from the last episode for next_value estimation
                    last_state = batch_results[-1][3][-1][0]  # Last transition's state
                    loss = self.agent.update(last_state)
                    self.losses.append(loss)
                    print(f"Policy updated | Loss = {loss:.4f} | Buffer size: {len(self.agent.states)}")
                
                # Update episode counter
                episode += self.n_workers
                
                # Update visualization (use config interval to avoid slowdown)
                current_time = time.time()
                if self.render and (current_time - last_update_time) > config.VISUALIZATION_UPDATE_INTERVAL:
                    self.update_visualization()
                    last_update_time = current_time
                
                # Save checkpoint
                if episodes_completed % self.save_interval == 0:
                    self.save_checkpoint(episode)
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Stop workers
            for _ in range(self.n_workers):
                task_queue.put(None)
            
            for p in workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
            
            print("All workers stopped")
        
        print("Training completed!")
    
    @staticmethod
    def worker_process(worker_id, task_queue, result_queue, max_steps):
        """Worker process that runs episodes (without visual)"""
        # Set different random seeds for each worker
        np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
        torch.manual_seed(worker_id * 1000 + int(time.time()) % 1000)
        
        # Create environment for this worker
        env = RetroWrapper(game=config.GAME)
        
        # Create local policy for action selection
        input_shape = config.INPUT_SHAPE
        device = 'cpu'  # Workers use CPU to avoid GPU memory issues
        from model import ActorCritic
        policy = ActorCritic(input_shape, env.n_actions).to(device)
        
        print(f"Worker {worker_id} started with seed {worker_id * 1000}")
        
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
            
            for step in range(max_steps):
                # Select action using local policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, log_prob, value = policy.act(state_tensor)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                transitions.append((
                    state, action, reward, 
                    value.item(), log_prob.item(), done
                ))
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Send result back
            result_queue.put((episode_num, episode_reward, episode_length, transitions))
        
        env.close()
        print(f"Worker {worker_id} stopped")
    
    @staticmethod
    def worker_process_with_visual(worker_id, task_queue, result_queue, max_steps, 
                                   worker_screens, worker_stats):
        """Worker process that runs episodes and shares screen data (COLOR)"""
        # Set different random seeds for each worker
        np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
        torch.manual_seed(worker_id * 1000 + int(time.time()) % 1000)
        
        # Create environment for this worker
        env = RetroWrapper(game=config.GAME)
        
        # Create local policy for action selection
        input_shape = config.INPUT_SHAPE
        device = 'cpu'  # Workers use CPU to avoid GPU memory issues
        from model import ActorCritic
        policy = ActorCritic(input_shape, env.n_actions).to(device)
        
        print(f"Worker {worker_id} started with seed {worker_id * 1000} (with color visual)")
        
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
            
            for step in range(max_steps):
                # Select action using local policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    action, log_prob, value = policy.act(state_tensor)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Update shared screen data (every 3 steps to reduce overhead)
                if step % 3 == 0:
                    # Get RGB frame from environment (before preprocessing)
                    # We need to access the raw observation from retro env
                    try:
                        # Get the current RGB frame from the environment
                        rgb_frame = env.env.render(mode='rgb_array')
                        if rgb_frame is not None:
                            # Resize to 84x84 for display consistency
                            import cv2
                            rgb_frame_resized = cv2.resize(rgb_frame, (84, 84), interpolation=cv2.INTER_AREA)
                            worker_screens[worker_id] = rgb_frame_resized.copy()
                    except:
                        # Fallback: convert grayscale to RGB
                        frame = state[-1]  # Shape: (84, 84)
                        rgb_frame = np.stack([frame, frame, frame], axis=-1).astype(np.uint8)
                        worker_screens[worker_id] = rgb_frame.copy()
                    
                    worker_stats[worker_id] = {
                        'reward': episode_reward,
                        'steps': episode_length
                    }
                
                # Store transition
                transitions.append((
                    state, action, reward, 
                    value.item(), log_prob.item(), done
                ))
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            # Send result back
            result_queue.put((episode_num, episode_reward, episode_length, transitions))
        
        env.close()
        print(f"Worker {worker_id} stopped")
    
    def update_visualization(self):
        """Update all visualizations including game screens and training plots"""
        # Update game screens (COLOR)
        if self.show_game_screens:
            for i, (ax, img_obj) in enumerate(zip(self.screen_axes, self.screen_images)):
                screen = self.worker_screens[i]
                stats = self.worker_stats[i]
                
                if screen is not None:
                    # Update image data (faster than ax.clear() + ax.imshow())
                    if len(screen.shape) == 3:  # RGB
                        img_obj.set_data(screen)
                    else:  # Grayscale fallback
                        rgb_screen = np.stack([screen, screen, screen], axis=-1)
                        img_obj.set_data(rgb_screen)
                    ax.set_title(f'Worker {i} | R:{stats["reward"]:.1f} S:{stats["steps"]}', 
                               fontsize=9)
                else:
                    # Show black screen with text
                    black_screen = np.zeros((84, 84, 3), dtype=np.uint8)
                    img_obj.set_data(black_screen)
                    ax.set_title(f'Worker {i} - Waiting...', fontsize=9)
            
            # Update real-time simulation window (Worker 0, larger view)
            if self.sim_fig is not None and self.sim_ax is not None:
                screen = self.worker_screens[0]  # Show Worker 0
                stats = self.worker_stats[0]
                
                if screen is not None:
                    # Resize to larger view (224x224)
                    import cv2
                    if len(screen.shape) == 3:  # RGB
                        large_screen = cv2.resize(screen, (224, 224), interpolation=cv2.INTER_NEAREST)
                    else:  # Grayscale fallback
                        rgb_screen = np.stack([screen, screen, screen], axis=-1)
                        large_screen = cv2.resize(rgb_screen, (224, 224), interpolation=cv2.INTER_NEAREST)
                    
                    self.sim_image.set_data(large_screen)
                    self.sim_ax.set_title(
                        f'Worker 0 - Real-time View | Reward: {stats["reward"]:.1f} | Steps: {stats["steps"]}',
                        fontsize=14, fontweight='bold'
                    )
                else:
                    black_screen = np.zeros((224, 224, 3), dtype=np.uint8)
                    self.sim_image.set_data(black_screen)
                    self.sim_ax.set_title('Worker 0 - Waiting...', fontsize=14)
        
        # Update training plots
        for ax in self.plot_axes:
            ax.clear()
        
        # Plot 1: Episode Rewards
        if len(self.episode_rewards) > 0:
            self.plot_axes[0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward', linewidth=1)
            if len(self.episode_rewards) >= 10:
                moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
                self.plot_axes[0].plot(range(9, len(self.episode_rewards)), moving_avg, 
                                     'r-', linewidth=2, label='Moving Avg (10)')
            self.plot_axes[0].set_xlabel('Episode', fontsize=9)
            self.plot_axes[0].set_ylabel('Reward', fontsize=9)
            self.plot_axes[0].set_title('Episode Rewards', fontsize=10)
            self.plot_axes[0].legend(fontsize=8)
            self.plot_axes[0].grid(True, alpha=0.3)
            self.plot_axes[0].tick_params(labelsize=8)
        
        # Plot 2: Episode Lengths
        if len(self.episode_lengths) > 0:
            self.plot_axes[1].plot(self.episode_lengths, alpha=0.6, linewidth=1)
            self.plot_axes[1].set_xlabel('Episode', fontsize=9)
            self.plot_axes[1].set_ylabel('Length', fontsize=9)
            self.plot_axes[1].set_title('Episode Lengths', fontsize=10)
            self.plot_axes[1].grid(True, alpha=0.3)
            self.plot_axes[1].tick_params(labelsize=8)
        
        # Plot 3: Training Loss
        if len(self.losses) > 0:
            self.plot_axes[2].plot(self.losses, alpha=0.6, linewidth=1)
            self.plot_axes[2].set_xlabel('Update Step', fontsize=9)
            self.plot_axes[2].set_ylabel('Loss', fontsize=9)
            self.plot_axes[2].set_title('Training Loss', fontsize=10)
            self.plot_axes[2].grid(True, alpha=0.3)
            self.plot_axes[2].tick_params(labelsize=8)
        
        # Plot 4: Average Reward Trend
        if len(self.episode_rewards) >= 50:
            window_sizes = [10, 50, 100]
            for window in window_sizes:
                if len(self.episode_rewards) >= window:
                    moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                    self.plot_axes[3].plot(range(window-1, len(self.episode_rewards)), 
                                        moving_avg, label=f'MA-{window}', linewidth=1.5)
            self.plot_axes[3].set_xlabel('Episode', fontsize=9)
            self.plot_axes[3].set_ylabel('Average Reward', fontsize=9)
            self.plot_axes[3].set_title('Reward Trends', fontsize=10)
            self.plot_axes[3].legend(fontsize=8)
            self.plot_axes[3].grid(True, alpha=0.3)
            self.plot_axes[3].tick_params(labelsize=8)
        
        plt.pause(0.001)


def main():
    """Main training function with parallel environments and visual display"""
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Initialize parallel trainer with config defaults
    # All parameters can be overridden by passing arguments
    # If not specified, values from config.py will be used
    trainer = ParallelTrainerWithVisual()
    
    # Start training (will auto-resume from latest checkpoint if exists)
    # All parameters use config.py defaults unless overridden
    trainer.train()
    
    # Keep plot window open
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
