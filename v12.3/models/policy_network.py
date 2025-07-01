"""
Policy Network Implementation for RL-LLM Tower Defense

This module implements neural network architectures for policy learning
in the tower defense environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TowerDefensePolicyNetwork(nn.Module):
    """
    Policy network for tower defense RL agent.
    
    Features:
    - Convolutional layers for spatial map processing
    - Dense layers for game state processing
    - Multi-head output for different action types
    - Attention mechanism for strategic decision making
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the policy network.
        
        Args:
            config: Network configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Extract configuration
        self.map_size = config.get('map_size', (12, 12))
        self.game_info_size = config.get('game_info_size', 10)
        self.hidden_size = config.get('hidden_size', 256)
        self.num_tower_types = config.get('num_tower_types', 3)
        
        # Calculate input sizes
        self.map_input_size = self.map_size[0] * self.map_size[1]
        self.total_input_size = self.map_input_size + self.game_info_size
        
        # Spatial processing layers (for map state)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Reduce to 4x4
        )
        
        # Calculate conv output size
        self.conv_output_size = 32 * 4 * 4  # 512
        
        # Game state processing
        self.game_state_fc = nn.Sequential(
            nn.Linear(self.game_info_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Combined feature processing
        combined_size = self.conv_output_size + 64
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )
        
        # Action heads
        self.action_type_head = nn.Linear(self.hidden_size, 4)  # 4 action types
        self.position_head = nn.Linear(self.hidden_size, self.map_size[0] * self.map_size[1])
        self.tower_type_head = nn.Linear(self.hidden_size, self.num_tower_types)
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Attention mechanism for strategic focus
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        logger.info(f"Initialized TowerDefensePolicyNetwork with config: {config}")
    
    def forward(self, observation: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            observation: Input observation tensor
            
        Returns:
            Dictionary containing action logits and value estimate
        """
        batch_size = observation.shape[0]
        
        # Split observation into map and game info
        map_obs = observation[:, :self.map_input_size]
        game_info = observation[:, self.map_input_size:]
        
        # Process spatial information
        map_reshaped = map_obs.view(batch_size, 1, self.map_size[0], self.map_size[1])
        spatial_features = self.spatial_conv(map_reshaped)
        spatial_features = spatial_features.view(batch_size, -1)
        
        # Process game state information
        game_features = self.game_state_fc(game_info)
        
        # Combine features
        combined_features = torch.cat([spatial_features, game_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # Apply attention for strategic focus
        attended_features, attention_weights = self.attention(
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1),
            fused_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Generate action logits
        action_type_logits = self.action_type_head(attended_features)
        position_logits = self.position_head(attended_features)
        tower_type_logits = self.tower_type_head(attended_features)
        
        # Reshape position logits to match map dimensions
        position_logits = position_logits.view(batch_size, self.map_size[0], self.map_size[1])
        
        # Generate value estimate
        value = self.value_head(attended_features)
        
        return {
            'action_type_logits': action_type_logits,
            'position_logits': position_logits,
            'tower_type_logits': tower_type_logits,
            'value': value,
            'attention_weights': attention_weights,
            'features': attended_features
        }
    
    def get_action(self, observation: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get action from the policy.
        
        Args:
            observation: Input observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, additional_info)
        """
        with torch.no_grad():
            output = self.forward(observation)
            
            if deterministic:
                # Use argmax for deterministic action
                action_type = torch.argmax(output['action_type_logits'], dim=-1)
                position_flat = torch.argmax(output['position_logits'].view(observation.shape[0], -1), dim=-1)
                tower_type = torch.argmax(output['tower_type_logits'], dim=-1)
            else:
                # Sample from distributions
                action_type_dist = torch.distributions.Categorical(
                    logits=output['action_type_logits']
                )
                action_type = action_type_dist.sample()
                
                position_dist = torch.distributions.Categorical(
                    logits=output['position_logits'].view(observation.shape[0], -1)
                )
                position_flat = position_dist.sample()
                
                tower_type_dist = torch.distributions.Categorical(
                    logits=output['tower_type_logits']
                )
                tower_type = tower_type_dist.sample()
            
            # Convert flat position to x, y coordinates
            position_x = position_flat // self.map_size[1]
            position_y = position_flat % self.map_size[1]
            
            action = torch.stack([action_type, position_x, position_y, tower_type], dim=-1)
            
            info = {
                'value': output['value'],
                'attention_weights': output['attention_weights'],
                'action_type_probs': F.softmax(output['action_type_logits'], dim=-1),
                'position_probs': F.softmax(output['position_logits'].view(observation.shape[0], -1), dim=-1),
                'tower_type_probs': F.softmax(output['tower_type_logits'], dim=-1)
            }
            
            return action, info
    
    def evaluate_actions(self, observation: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            observation: Input observations
            actions: Actions to evaluate
            
        Returns:
            Dictionary containing log probabilities and values
        """
        output = self.forward(observation)
        
        # Extract action components
        action_type = actions[:, 0].long()
        position_x = actions[:, 1].long()
        position_y = actions[:, 2].long()
        tower_type = actions[:, 3].long()
        
        # Calculate log probabilities
        action_type_dist = torch.distributions.Categorical(logits=output['action_type_logits'])
        action_type_log_prob = action_type_dist.log_prob(action_type)
        
        position_flat = position_x * self.map_size[1] + position_y
        position_dist = torch.distributions.Categorical(
            logits=output['position_logits'].view(observation.shape[0], -1)
        )
        position_log_prob = position_dist.log_prob(position_flat)
        
        tower_type_dist = torch.distributions.Categorical(logits=output['tower_type_logits'])
        tower_type_log_prob = tower_type_dist.log_prob(tower_type)
        
        # Combined log probability
        total_log_prob = action_type_log_prob + position_log_prob + tower_type_log_prob
        
        # Calculate entropy for regularization
        entropy = (action_type_dist.entropy() + 
                  position_dist.entropy() + 
                  tower_type_dist.entropy())
        
        return {
            'log_prob': total_log_prob,
            'value': output['value'].squeeze(-1),
            'entropy': entropy,
            'action_type_log_prob': action_type_log_prob,
            'position_log_prob': position_log_prob,
            'tower_type_log_prob': tower_type_log_prob
        }


class SimplePolicyNetwork(nn.Module):
    """
    Simplified policy network for quick testing and prototyping.
    """
    
    def __init__(self, input_size: int, action_size: int, hidden_size: int = 128):
        """Initialize simple policy network."""
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        action_logits = self.network(x)
        value = self.value_head(x)
        
        return {
            'action_logits': action_logits,
            'value': value
        }


def create_policy_network(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create policy networks.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Policy network instance
    """
    network_type = config.get('network_type', 'tower_defense')
    
    if network_type == 'tower_defense':
        return TowerDefensePolicyNetwork(config)
    elif network_type == 'simple':
        return SimplePolicyNetwork(
            input_size=config['input_size'],
            action_size=config['action_size'],
            hidden_size=config.get('hidden_size', 128)
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")

