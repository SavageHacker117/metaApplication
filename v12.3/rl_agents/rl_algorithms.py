"""
Advanced RL Algorithms for RL-LLM System

This module implements state-of-the-art reinforcement learning algorithms including
PPO, SAC, Rainbow DQN, and other modern techniques with optimized implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass, field
from collections import deque, namedtuple
import random
import math
from abc import ABC, abstractmethod
import copy

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'log_prob', 'value'])


@dataclass
class AlgorithmConfig:
    """Configuration for RL algorithms."""
    algorithm_name: str
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100000
    update_frequency: int = 4
    target_update_frequency: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    tau: float = 0.005  # Soft update parameter
    clip_epsilon: float = 0.2  # PPO clipping parameter
    entropy_coefficient: float = 0.01
    value_loss_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """Experience replay buffer with prioritized sampling support."""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer capacity
            alpha: Prioritization exponent
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0
    
    def add(self, experience: Experience, priority: Optional[float] = None):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        # Set priority
        if priority is None:
            priority = self.max_priority
        
        self.priorities[self.pos] = priority
        self.max_priority = max(self.max_priority, priority)
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: Optional[float] = None) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Sample batch with prioritized sampling."""
        if beta is None:
            beta = self.beta
        
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for policy gradient methods."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 continuous: bool = False):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            continuous: Whether action space is continuous
        """
        super().__init__()
        
        self.continuous = continuous
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value."""
        features = self.feature_extractor(state)
        
        if self.continuous:
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_logstd.expand_as(mean))
            action_dist = Normal(mean, std)
        else:
            logits = self.actor(features)
            action_dist = Categorical(logits=logits)
        
        value = self.critic(features)
        
        return action_dist, value
    
    def act(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action and return action, log probability, and value."""
        action_dist, value = self.forward(state)
        
        if deterministic and self.continuous:
            action = action_dist.mean
        else:
            action = action_dist.sample()
        
        log_prob = action_dist.log_prob(action)
        
        if not self.continuous:
            log_prob = log_prob.unsqueeze(-1)
        else:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, value


class PPOAlgorithm:
    """Proximal Policy Optimization (PPO) algorithm."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int, 
                 continuous: bool = False):
        """
        Initialize PPO algorithm.
        
        Args:
            config: Algorithm configuration
            state_dim: State space dimension
            action_dim: Action space dimension
            continuous: Whether action space is continuous
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.policy = ActorCriticNetwork(state_dim, action_dim, continuous=continuous).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        
        # Training data storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Training statistics
        self.episode_rewards = deque(maxlen=100)
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.entropy_losses = deque(maxlen=100)
        
        logger.info("Initialized PPO algorithm")
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor, deterministic)
        
        action_np = action.cpu().numpy().squeeze()
        
        info = {
            'log_prob': log_prob.cpu().numpy().squeeze(),
            'value': value.cpu().numpy().squeeze()
        }
        
        return action_np, info
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, info: Dict[str, Any]):
        """Store transition for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(info['log_prob'])
        self.values.append(info['value'])
        self.dones.append(done)
    
    def update(self) -> Dict[str, float]:
        """Update policy using collected experiences."""
        if len(self.states) < self.config.batch_size:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        
        # Calculate advantages and returns
        advantages, returns = self._compute_gae(rewards, values.cpu().numpy(), dones)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # Multiple epochs of updates
        for _ in range(4):  # PPO typically uses 4 epochs
            # Get current policy outputs
            action_dist, current_values = self.policy(states)
            
            if hasattr(action_dist, 'log_prob'):
                current_log_probs = action_dist.log_prob(actions)
                if len(current_log_probs.shape) > 1:
                    current_log_probs = current_log_probs.sum(dim=-1, keepdim=True)
            else:
                current_log_probs = action_dist.log_prob(actions.long().squeeze())
                current_log_probs = current_log_probs.unsqueeze(-1)
            
            # Calculate ratio
            ratio = torch.exp(current_log_probs - old_log_probs)
            
            # Calculate policy loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(current_values.squeeze(), returns)
            
            # Calculate entropy loss
            entropy_loss = -action_dist.entropy().mean()
            
            # Total loss
            total_loss = (policy_loss + 
                         self.config.value_loss_coefficient * value_loss + 
                         self.config.entropy_coefficient * entropy_loss)
            
            # Update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        # Store losses
        self.policy_losses.append(total_policy_loss / 4)
        self.value_losses.append(total_value_loss / 4)
        self.entropy_losses.append(total_entropy_loss / 4)
        
        # Clear storage
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
        
        return {
            'policy_loss': total_policy_loss / 4,
            'value_loss': total_value_loss / 4,
            'entropy_loss': total_entropy_loss / 4
        }
    
    def _compute_gae(self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
                    gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def save(self, path: Path):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Saved PPO model to {path}")
    
    def load(self, path: Path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded PPO model from {path}")


class SACAlgorithm:
    """Soft Actor-Critic (SAC) algorithm for continuous control."""
    
    def __init__(self, config: AlgorithmConfig, state_dim: int, action_dim: int):
        """
        Initialize SAC algorithm.
        
        Args:
            config: Algorithm configuration
            state_dim: State space dimension
            action_dim: Action space dimension
        """
        self.config = config
        self.device = torch.device(config.device)
        self.action_dim = action_dim
        
        # Networks
        self.actor = self._build_actor(state_dim, action_dim).to(self.device)
        self.critic1 = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic2 = self._build_critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=config.learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=config.learning_rate)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training statistics
        self.actor_losses = deque(maxlen=100)
        self.critic_losses = deque(maxlen=100)
        self.alpha_losses = deque(maxlen=100)
        
        logger.info("Initialized SAC algorithm")
    
    def _build_actor(self, state_dim: int, action_dim: int) -> nn.Module:
        """Build actor network."""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)  # mean and log_std
        )
    
    def _build_critic(self, state_dim: int, action_dim: int) -> nn.Module:
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action using current policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self._sample_action(state_tensor, deterministic)
        
        return action.cpu().numpy().squeeze(), {'log_prob': log_prob.cpu().numpy().squeeze()}
    
    def _sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        actor_output = self.actor(state)
        mean, log_std = torch.chunk(actor_output, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action)
        else:
            normal = Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def store_transition(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, info: Dict[str, Any]):
        """Store transition in replay buffer."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=info.get('log_prob', 0),
            value=0  # Not used in SAC
        )
        self.replay_buffer.add(experience)
    
    def update(self) -> Dict[str, float]:
        """Update SAC networks."""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        experiences, indices, weights = self.replay_buffer.sample(self.config.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self._sample_action(next_states)
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self._sample_action(states)
        q1_new = self.critic1(torch.cat([states, new_actions], dim=1))
        q2_new = self.critic2(torch.cat([states, new_actions], dim=1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append((critic1_loss.item() + critic2_loss.item()) / 2)
        self.alpha_losses.append(alpha_loss.item())
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
            'alpha_loss': alpha_loss.item(),
            'alpha': self.log_alpha.exp().item()
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau)
    
    def save(self, path: Path):
        """Save model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'config': self.config
        }, path)
        logger.info(f"Saved SAC model to {path}")
    
    def load(self, path: Path):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        logger.info(f"Loaded SAC model from {path}")


class AlgorithmFactory:
    """Factory for creating RL algorithms."""
    
    @staticmethod
    def create_algorithm(algorithm_name: str, config: AlgorithmConfig, 
                        state_dim: int, action_dim: int, **kwargs):
        """
        Create RL algorithm instance.
        
        Args:
            algorithm_name: Name of algorithm
            config: Algorithm configuration
            state_dim: State space dimension
            action_dim: Action space dimension
            **kwargs: Additional arguments
            
        Returns:
            Algorithm instance
        """
        if algorithm_name.lower() == 'ppo':
            continuous = kwargs.get('continuous', False)
            return PPOAlgorithm(config, state_dim, action_dim, continuous)
        
        elif algorithm_name.lower() == 'sac':
            return SACAlgorithm(config, state_dim, action_dim)
        
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")


def create_algorithm(algorithm_name: str, config: Dict[str, Any], 
                    state_dim: int, action_dim: int, **kwargs):
    """
    Factory function to create RL algorithm.
    
    Args:
        algorithm_name: Name of algorithm
        config: Configuration dictionary
        state_dim: State space dimension
        action_dim: Action space dimension
        **kwargs: Additional arguments
        
    Returns:
        Algorithm instance
    """
    # Convert config dict to AlgorithmConfig
    algo_config = AlgorithmConfig(
        algorithm_name=algorithm_name,
        **{k: v for k, v in config.items() if hasattr(AlgorithmConfig, k)}
    )
    
    return AlgorithmFactory.create_algorithm(
        algorithm_name, algo_config, state_dim, action_dim, **kwargs
    )

