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


class ParallelTrainerWithVisual:
    """Parallel training manager with multiple environments and visual display"""
    
    def __init__(self, game='Jackal-Nes', n_workers=8, render=True, 
                 save_interval=10, max_checkpoints=100, show_game_screens=True):
        self.game = game
        self.n_workers = n_workers
        self.render = render
        self.show_game_screens = show_game_screens
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = 'checkpoints'
        
        # Create a single environment to get dimensions
        temp_env = RetroWrapper(game=game)
        input_shape = (4, 84, 84)
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
                
                # Top 2 rows: 8 game screens (4x2 layout)
                self.screen_axes = []
                for i in range(2):
                    for j in range(4):
                        ax = self.fig.add_subplot(gs[i, j:j+1])
                        ax.set_title(f'Worker {i*4+j}', fontsize=10)
                        ax.axis('off')
                        self.screen_axes.append(ax)
                
                # Bottom 2 rows: 4 training plots (2x2 layout)
                self.plot_axes = []
                self.plot_axes.append(self.fig.add_subplot(gs[2, 0:3]))  # Episode Rewards
                self.plot_axes.append(self.fig.add_subplot(gs[2, 3:6]))  # Episode Lengths
                self.plot_axes.append(self.fig.add_subplot(gs[3, 0:3]))  # Training Loss
                self.plot_axes.append(self.fig.add_subplot(gs[3, 3:6]))  # Reward Trends
                
                self.fig.suptitle(f'Parallel Training with {n_workers} Workers', fontsize=14, fontweight='bold')
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
    
    def train(self, max_episodes=1000, max_steps=10000, update_interval=2048):
        """Train the agent with parallel environments"""
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
                
                # Update visualization (every 0.5 seconds to avoid slowdown)
                current_time = time.time()
                if self.render and (current_time - last_update_time) > 0.5:
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
        env = RetroWrapper(game='Jackal-Nes')
        
        # Create local policy for action selection
        input_shape = (4, 84, 84)
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
        """Worker process that runs episodes and shares screen data"""
        # Set different random seeds for each worker
        np.random.seed(worker_id * 1000 + int(time.time()) % 1000)
        torch.manual_seed(worker_id * 1000 + int(time.time()) % 1000)
        
        # Create environment for this worker
        env = RetroWrapper(game='Jackal-Nes')
        
        # Create local policy for action selection
        input_shape = (4, 84, 84)
        device = 'cpu'  # Workers use CPU to avoid GPU memory issues
        from model import ActorCritic
        policy = ActorCritic(input_shape, env.n_actions).to(device)
        
        print(f"Worker {worker_id} started with seed {worker_id * 1000} (with visual)")
        
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
                
                # Update shared screen data (every 5 steps to reduce overhead)
                if step % 5 == 0:
                    # Get the latest frame (last channel of state)
                    frame = state[-1]  # Shape: (84, 84)
                    worker_screens[worker_id] = frame.copy()
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
        # Update game screens
        if self.show_game_screens:
            for i, ax in enumerate(self.screen_axes):
                ax.clear()
                ax.axis('off')
                
                screen = self.worker_screens[i]
                stats = self.worker_stats[i]
                
                if screen is not None:
                    ax.imshow(screen, cmap='gray')
                    ax.set_title(f'Worker {i} | R:{stats["reward"]:.1f} S:{stats["steps"]}', 
                               fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'Waiting...', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'Worker {i}', fontsize=9)
        
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
    
    # Initialize parallel trainer with visual display
    # show_game_screens=True: display 8 game screens in real-time
    # n_workers=8: run 8 parallel environments
    # save_interval=10: save every 10 episodes
    # max_checkpoints=100: keep only the latest 100 checkpoints
    trainer = ParallelTrainerWithVisual(
        game='Jackal-Nes',
        n_workers=8,
        render=True,
        show_game_screens=True,  # Set to False to disable game screens
        save_interval=10,
        max_checkpoints=100
    )
    
    # Start training (will auto-resume from latest checkpoint if exists)
    # With 8 workers, effective speed is ~8x faster
    trainer.train(max_episodes=1000, max_steps=10000, update_interval=2048)
    
    # Keep plot window open
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
