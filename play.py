import torch
import time
from env_wrapper import RetroWrapper
from ppo_agent import PPOAgent


def play(model_path, episodes=5, render=True):
    """Play game using trained model"""
    # Initialize environment
    env = RetroWrapper(game='Jackal-Nes')
    
    # Initialize agent
    input_shape = (1, 84, 84)
    agent = PPOAgent(input_shape, env.n_actions)
    
    # Load trained model
    try:
        agent.load(model_path)
        print(f"Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return
    
    agent.policy.eval()
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"\n=== Episode {episode + 1}/{episodes} ===")
        
        while True:
            # Select action (greedy)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action, log_prob, value = agent.policy.act(state_tensor)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            if render:
                env.render()
                time.sleep(0.01)  # Slow down for visualization
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        print(f"Episode {episode + 1} finished!")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Episode Length: {episode_length}")
    
    env.close()
    print("\nPlayback completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'checkpoints/model_episode_100.pth'
    
    play(model_path, episodes=5, render=True)
