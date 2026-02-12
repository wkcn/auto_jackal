import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO algorithm"""
    
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        # Convolutional layers for processing game frames
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
    def _get_conv_out(self, shape):
        """Calculate the output size of convolutional layers"""
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))
    
    def forward(self, x):
        """Forward pass through the network"""
        x = x / 255.0 - 0.5  # Normalize pixel values
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.actor(conv_out), self.critic(conv_out)
    
    def act(self, state):
        """Select action based on current policy (multi-label with sigmoid)"""
        logits, value = self.forward(state)
        probs = torch.sigmoid(logits)  # Use sigmoid for multi-label
        dist = Bernoulli(probs)
        action = dist.sample()  # Sample each button independently
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum log probs for all buttons
        return action.squeeze(0), log_prob, value
    
    def evaluate(self, states, actions):
        """Evaluate actions for PPO update (multi-label with sigmoid)"""
        logits, values = self.forward(states)
        probs = torch.sigmoid(logits)  # Use sigmoid for multi-label
        dist = Bernoulli(probs)
        log_probs = dist.log_prob(actions).sum(dim=-1)  # Sum log probs for all buttons
        entropy = dist.entropy().sum(dim=-1)  # Sum entropy for all buttons
        return log_probs, values, entropy
