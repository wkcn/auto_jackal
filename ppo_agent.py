import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import ActorCritic


class PPOAgent:
    """PPO Agent for training"""
    
    def __init__(self, input_shape, n_actions, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, c1=0.5, c2=0.01,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.c1 = c1  # Value loss coefficient
        self.c2 = c2  # Entropy coefficient
        
        # Initialize network
        self.policy = ActorCritic(input_shape, n_actions).to(device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr)
        
        # Storage for rollout data
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def select_action(self, state):
        """Select action using current policy (returns button array)"""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state)
        return action.cpu().numpy(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns
    
    def update(self, next_state, epochs=4, batch_size=64):
        """Update policy using PPO"""
        # Get next value for GAE computation
        next_state = torch.FloatTensor(next_state).to(self.device)
        with torch.no_grad():
            _, _, next_value = self.policy(next_state)
            next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.concatenate(self.states)).to(self.device)
        actions = torch.stack(self.actions).to(self.device)  # Changed to FloatTensor for multi-label
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        dataset_size = len(self.states)
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        total_clipfrac = 0
        n_updates = 0
        
        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Policy loss (clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # Additional metrics for monitoring
                with torch.no_grad():
                    # Approximate KL divergence
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    total_approx_kl += approx_kl
                    
                    # Fraction of samples where clipping occurred
                    clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    total_clipfrac += clipfrac
                
                n_updates += 1
        
        # Clear memory
        self.clear_memory()
        
        # Return detailed metrics
        metrics = {
            'loss/total': total_loss / n_updates,
            'loss/policy': total_policy_loss / n_updates,
            'loss/value': total_value_loss / n_updates,
            'loss/entropy': -total_entropy / n_updates,  # Negative because we want to maximize entropy
            'train/approx_kl': total_approx_kl / n_updates,
            'train/clipfrac': total_clipfrac / n_updates,
            'train/entropy': total_entropy / n_updates,
        }
        
        return metrics
    
    def clear_memory(self):
        """Clear stored transitions"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def save(self, path, episode=None, global_step=None, training_stats=None):
        """Save model and training state"""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        if episode is not None:
            checkpoint['episode'] = episode
        if global_step is not None:
            checkpoint['global_step'] = global_step
        if training_stats is not None:
            checkpoint['training_stats'] = training_stats
        
        torch.save(checkpoint, path)
    
    def load(self, path, strict=False):
        """Load model and training state
        
        Args:
            path: Path to checkpoint file
            strict: If False, allows loading checkpoints with shape mismatches (for backward compatibility)
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        skipped_param_ids = set()
        
        if strict:
            # Strict mode: load all weights, fail if shape mismatch
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
        else:
            # Compatible mode: skip layers with shape mismatch
            model_state_dict = self.policy.state_dict()
            checkpoint_state_dict = checkpoint['policy_state_dict']
            
            # Filter out layers with shape mismatch
            compatible_state_dict = {}
            skipped_layers = []
            
            # Build parameter name to id mapping for current model
            param_name_to_id = {}
            for i, (name, param) in enumerate(self.policy.named_parameters()):
                param_name_to_id[name] = i
            
            for name, param in checkpoint_state_dict.items():
                if name in model_state_dict:
                    if param.shape == model_state_dict[name].shape:
                        compatible_state_dict[name] = param
                    else:
                        skipped_layers.append(name)
                        # Mark this parameter's optimizer state for reinitialization
                        if name in param_name_to_id:
                            skipped_param_ids.add(param_name_to_id[name])
                        print(f"⚠️  Skipping layer '{name}': shape mismatch "
                              f"(checkpoint: {param.shape}, model: {model_state_dict[name].shape})")
                else:
                    skipped_layers.append(name)
                    print(f"⚠️  Skipping layer '{name}': not found in current model")
            
            # Load compatible weights
            self.policy.load_state_dict(compatible_state_dict, strict=False)
            
            if skipped_layers:
                print(f"\n✓ Loaded checkpoint with {len(skipped_layers)} layer(s) skipped")
                print(f"  Skipped layers: {', '.join(skipped_layers)}")
                print(f"  These layers will be randomly initialized")
            else:
                print(f"✓ Loaded checkpoint successfully (all layers compatible)")
        
        # Load optimizer state with compatibility check
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer_state_dict = checkpoint['optimizer_state_dict']
                
                if skipped_param_ids:
                    # Remove optimizer states for skipped parameters
                    if 'state' in optimizer_state_dict:
                        filtered_state = {}
                        for param_id, state in optimizer_state_dict['state'].items():
                            if param_id not in skipped_param_ids:
                                filtered_state[param_id] = state
                        optimizer_state_dict['state'] = filtered_state
                    
                    print(f"⚠️  Reinitializing optimizer state for {len(skipped_param_ids)} parameter group(s)")
                
                # Load filtered optimizer state
                self.optimizer.load_state_dict(optimizer_state_dict)
                print(f"✓ Loaded optimizer state successfully")
                
            except Exception as e:
                print(f"⚠️  Could not load optimizer state: {e}")
                print(f"  Optimizer will be fully reinitialized")
        
        return checkpoint
