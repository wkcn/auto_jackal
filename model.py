import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO algorithm
    
    Action space design for Jackal-Nes (6 buttons):
    - Direction: Categorical(9) - None, Up, Down, Left, Right, UpLeft, UpRight, DownLeft, DownRight
    - Weapons: Independent Bernoulli(2) - Bullet(0), Grenade(8)
    
    Button indices in Retro environment:
    [0:Bullet, 1:NULL, 2:NULL, 3:NULL, 4:UP, 5:DOWN, 6:LEFT, 7:RIGHT, 8:Grenade]
    
    Total action space: 9 directions × 2^2 weapons = 36 valid combinations
    (vs 2^6 = 64 combinations with all independent, many invalid)
    """
    
    # Direction mapping: Categorical index -> (UP, DOWN, LEFT, RIGHT) button states
    DIRECTION_MAP = [
        [0, 0, 0, 0],  # 0: None
        [1, 0, 0, 0],  # 1: Up
        [0, 1, 0, 0],  # 2: Down
        [0, 0, 1, 0],  # 3: Left
        [0, 0, 0, 1],  # 4: Right
        [1, 0, 1, 0],  # 5: UpLeft
        [1, 0, 0, 1],  # 6: UpRight
        [0, 1, 1, 0],  # 7: DownLeft
        [0, 1, 0, 1],  # 8: DownRight
    ]
    
    # Button indices in the Retro action array
    BUTTON_BULLET = 0
    BUTTON_NULL_1 = 1
    BUTTON_NULL_2 = 2
    BUTTON_NULL_3 = 3
    BUTTON_UP = 4
    BUTTON_DOWN = 5
    BUTTON_LEFT = 6
    BUTTON_RIGHT = 7
    BUTTON_GRENADE = 8
    
    # Number of outputs
    N_DIRECTIONS = 9  # Categorical for direction
    N_WEAPONS = 2     # Bernoulli for Bullet and Grenade
    
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        self.n_actions = n_actions  # Should be 9 (total buttons in Retro)
        
        # Convolutional layers for processing game frames
        # input: 256x256
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=16, stride=16),   # (256,256) -> (16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),   # (16,16) -> (16,16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),   # (16,16) -> (8,8)
            nn.ReLU(),
        )
        
        # Calculate conv output size
        conv_out_size = self._get_conv_out(input_shape)
        
        # Shared feature layer
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )
        
        # Actor head - direction (Categorical)
        self.actor_direction = nn.Linear(512, self.N_DIRECTIONS)
        
        # Actor head - weapons (Bernoulli)
        self.actor_weapons = nn.Linear(512, self.N_WEAPONS)

        for fc in [self.actor_direction, self.actor_weapons]:
            nn.init.orthogonal_(fc.weight, gain=0.01)
            nn.init.constant_(fc.bias, 0.0)
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def _get_conv_out(self, shape):
        """Calculate the output size of convolutional layers"""
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))
    
    def forward(self, x):
        """Forward pass through the network
        
        Returns:
            direction_logits: (batch, 9) logits for direction Categorical
            weapon_logits: (batch, 2) logits for weapons Bernoulli
            value: (batch, 1) state value
        """
        conv_out = self.conv(x).view(x.size(0), -1)
        features = self.shared_fc(conv_out)
        
        direction_logits = self.actor_direction(features)
        weapon_logits = self.actor_weapons(features)
        value = self.critic(features)
        
        return direction_logits, weapon_logits, value
    
    def _direction_to_buttons(self, direction_idx):
        """Convert direction index to button states
        
        Args:
            direction_idx: int or tensor of direction indices
            
        Returns:
            button_states: (4,) or (batch, 4) tensor of [UP, DOWN, LEFT, RIGHT]
        """
        if isinstance(direction_idx, int):
            return torch.tensor(self.DIRECTION_MAP[direction_idx], dtype=torch.float32)
        else:
            # Batch processing
            device = direction_idx.device
            direction_map_tensor = torch.tensor(self.DIRECTION_MAP, dtype=torch.float32, device=device)
            return direction_map_tensor[direction_idx]
    
    def _components_to_action(self, direction_idx, weapons):
        """Convert direction and weapons to full action array
        
        Args:
            direction_idx: int or (batch,) tensor of direction indices
            weapons: (2,) or (batch, 2) tensor of [Bullet, Grenade]
            
        Returns:
            action: (9,) or (batch, 9) tensor of full button array
        """
        is_batch = not isinstance(direction_idx, int)
        
        if is_batch:
            batch_size = direction_idx.shape[0]
            device = direction_idx.device
            action = torch.zeros(batch_size, self.n_actions, device=device)
            
            # Set directional buttons
            dir_buttons = self._direction_to_buttons(direction_idx)  # (batch, 4)
            action[:, self.BUTTON_UP] = dir_buttons[:, 0]
            action[:, self.BUTTON_DOWN] = dir_buttons[:, 1]
            action[:, self.BUTTON_LEFT] = dir_buttons[:, 2]
            action[:, self.BUTTON_RIGHT] = dir_buttons[:, 3]
            
            # Set weapon buttons
            action[:, self.BUTTON_BULLET] = weapons[:, 0]
            action[:, self.BUTTON_GRENADE] = weapons[:, 1]
            
            # NULL buttons are always 0
            action[:, self.BUTTON_NULL_1] = 0
            action[:, self.BUTTON_NULL_2] = 0
            action[:, self.BUTTON_NULL_3] = 0
        else:
            device = weapons.device if torch.is_tensor(weapons) else 'cpu'
            action = torch.zeros(self.n_actions, device=device)
            
            # Set directional buttons
            dir_buttons = self._direction_to_buttons(direction_idx)
            action[self.BUTTON_UP] = dir_buttons[0]
            action[self.BUTTON_DOWN] = dir_buttons[1]
            action[self.BUTTON_LEFT] = dir_buttons[2]
            action[self.BUTTON_RIGHT] = dir_buttons[3]
            
            # Set weapon buttons
            if torch.is_tensor(weapons):
                action[self.BUTTON_BULLET] = weapons[0]
                action[self.BUTTON_GRENADE] = weapons[1]
            else:
                action[self.BUTTON_BULLET] = weapons[0]
                action[self.BUTTON_GRENADE] = weapons[1]
            
            # NULL buttons are always 0
            action[self.BUTTON_NULL_1] = 0
            action[self.BUTTON_NULL_2] = 0
            action[self.BUTTON_NULL_3] = 0
            
        return action
    
    def _action_to_components(self, action):
        """Convert full action array to direction index and weapons
        
        Args:
            action: (9,) or (batch, 9) tensor of full button array
            
        Returns:
            direction_idx: int or (batch,) tensor of direction indices
            weapons: (2,) or (batch, 2) tensor of [Bullet, Grenade]
        """
        is_batch = action.dim() == 2
        if not is_batch:
            direction_idx, weapons = self._action_to_components(action[None])
            direction_idx = direction_idx[0]
            weapons = weapons[0]
        else:
            # Extract directional buttons
            dir_buttons = action[:, [self.BUTTON_UP, self.BUTTON_DOWN, 
                                    self.BUTTON_LEFT, self.BUTTON_RIGHT]]  # (batch, 4)
            
            # Find matching direction index
            device = action.device
            direction_map_tensor = torch.tensor(self.DIRECTION_MAP, dtype=torch.float32, device=device)
            
            # Compare with all direction patterns
            # dir_buttons: (batch, 4), direction_map_tensor: (9, 4)
            # Expand for broadcasting: (batch, 1, 4) vs (1, 9, 4)
            matches = (dir_buttons.unsqueeze(1) == direction_map_tensor.unsqueeze(0)).all(dim=2)  # (batch, 9)
            direction_idx = matches.float().argmax(dim=1)  # (batch,)
            
            # Extract weapon buttons
            weapons = action[:, [self.BUTTON_BULLET, self.BUTTON_GRENADE]]  # (batch, 2)
          
        return direction_idx, weapons
    
    def act(self, state):
        """Select action based on current policy
        
        Returns:
            action: (9,) tensor of full button array
            log_prob: scalar log probability
            value: scalar state value
        """
        direction_logits, weapon_logits, value = self.forward(state)
        
        # Sample direction (Categorical)
        direction_dist = Categorical(logits=direction_logits)
        direction_idx = direction_dist.sample()
        direction_log_prob = direction_dist.log_prob(direction_idx)
        
        # Sample weapons (Bernoulli)
        weapon_dist = Bernoulli(logits=weapon_logits)
        weapons = weapon_dist.sample()
        weapon_log_prob = weapon_dist.log_prob(weapons).sum(dim=-1)
        
        # Combine log probabilities
        total_log_prob = direction_log_prob + weapon_log_prob
        
        # Convert to full action array
        action = self._components_to_action(direction_idx, weapons)
        
        return action.squeeze(0), total_log_prob, value
    
    def evaluate(self, states, actions):
        """Evaluate actions for PPO update
        
        Args:
            states: (batch, C, H, W) tensor of states
            actions: (batch, 9) tensor of full button arrays
            
        Returns:
            log_probs: (batch,) tensor of log probabilities
            values: (batch, 1) tensor of state values
            entropy: (batch,) tensor of entropy
        """
        direction_logits, weapon_logits, values = self.forward(states)
        
        # Decompose actions into components
        direction_idx, weapons = self._action_to_components(actions)
        
        # Evaluate direction (Categorical)
        direction_dist = Categorical(logits=direction_logits)
        direction_log_prob = direction_dist.log_prob(direction_idx)
        direction_entropy = direction_dist.entropy()
        
        # Evaluate weapons (Bernoulli)
        weapon_dist = Bernoulli(logits=weapon_logits)
        weapon_log_prob = weapon_dist.log_prob(weapons).sum(dim=-1)
        weapon_entropy = weapon_dist.entropy().sum(dim=-1)
        
        # Combine
        log_probs = direction_log_prob + weapon_log_prob
        entropy = direction_entropy + weapon_entropy
        
        return log_probs, values, entropy
