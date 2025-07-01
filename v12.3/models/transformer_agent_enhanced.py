"""
Enhanced Transformer-Based RL Agent v3

Improvements based on feedback:
- Gradient checkpointing for memory-efficient attention
- Reward history and context concatenation
- Cross-modal embeddings (code tokens + image)
- Decision transformer style recurrence
- Enhanced memory mechanisms
- Context-aware architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
from collections import deque
import warnings

@dataclass
class EnhancedAgentConfig:
    """Enhanced configuration for transformer RL agents."""
    # Model architecture
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    # Memory efficiency
    use_gradient_checkpointing: bool = True
    max_sequence_length: int = 256
    memory_efficient_attention: bool = True
    
    # Context and history
    reward_history_length: int = 50
    action_history_length: int = 100
    enable_reward_context: bool = True
    enable_action_context: bool = True
    
    # Cross-modal features
    enable_cross_modal: bool = True
    code_vocab_size: int = 10000
    code_embedding_dim: int = 256
    image_feature_dim: int = 512
    
    # Decision transformer features
    use_decision_transformer: bool = True
    return_to_go_conditioning: bool = True
    timestep_embedding: bool = True
    
    # GNN configuration
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 4
    gnn_heads: int = 4
    
    # Training configuration
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MemoryEfficientPositionalEncoding(nn.Module):
    """Memory-efficient positional encoding with gradient checkpointing."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Forward with optional gradient checkpointing."""
        return x + self.pe[:x.size(0), :]

class CrossModalEmbedding(nn.Module):
    """Cross-modal embedding for code tokens and image features."""
    
    def __init__(self, config: EnhancedAgentConfig):
        super().__init__()
        self.config = config
        
        # Code token embedding
        self.code_embedding = nn.Embedding(
            config.code_vocab_size, 
            config.code_embedding_dim
        )
        
        # Image feature projection
        self.image_projection = nn.Linear(
            config.image_feature_dim, 
            config.d_model
        )
        
        # Code feature projection
        self.code_projection = nn.Linear(
            config.code_embedding_dim, 
            config.d_model
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, code_tokens: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse code tokens and image features.
        
        Args:
            code_tokens: Tokenized code [batch_size, seq_len]
            image_features: Image features [batch_size, feature_dim]
            
        Returns:
            Fused cross-modal representation [batch_size, d_model]
        """
        batch_size = code_tokens.shape[0]
        
        # Embed code tokens
        code_embedded = self.code_embedding(code_tokens)  # [batch, seq_len, embed_dim]
        code_features = self.code_projection(code_embedded.mean(dim=1))  # [batch, d_model]
        
        # Project image features
        image_projected = self.image_projection(image_features)  # [batch, d_model]
        
        # Cross-modal attention
        code_features_expanded = code_features.unsqueeze(1)  # [batch, 1, d_model]
        image_features_expanded = image_projected.unsqueeze(1)  # [batch, 1, d_model]
        
        # Attend code to image
        code_attended, _ = self.cross_attention(
            code_features_expanded, 
            image_features_expanded, 
            image_features_expanded
        )
        
        # Attend image to code
        image_attended, _ = self.cross_attention(
            image_features_expanded,
            code_features_expanded,
            code_features_expanded
        )
        
        # Fuse attended features
        fused_features = torch.cat([
            code_attended.squeeze(1), 
            image_attended.squeeze(1)
        ], dim=-1)
        
        fused_output = self.fusion_layer(fused_features)
        
        return fused_output

class RewardHistoryEncoder(nn.Module):
    """Encoder for reward history and context."""
    
    def __init__(self, config: EnhancedAgentConfig):
        super().__init__()
        self.config = config
        
        # Reward embedding
        self.reward_embedding = nn.Linear(1, config.d_model // 4)
        
        # Action embedding for history
        self.action_embedding = nn.Linear(
            config.d_model, 
            config.d_model // 4
        )
        
        # Timestep embedding
        if config.timestep_embedding:
            self.timestep_embedding = nn.Embedding(
                config.reward_history_length + config.action_history_length,
                config.d_model // 4
            )
        
        # History encoder
        self.history_encoder = nn.LSTM(
            input_size=config.d_model // 2 if not config.timestep_embedding else config.d_model // 4 * 3,
            hidden_size=config.d_model // 2,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.d_model // 2, config.d_model)
        
    def forward(self, reward_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
        """
        Encode reward and action history.
        
        Args:
            reward_history: Historical rewards [batch_size, history_len]
            action_history: Historical actions [batch_size, history_len, action_dim]
            
        Returns:
            Encoded history context [batch_size, d_model]
        """
        batch_size, history_len = reward_history.shape
        
        # Embed rewards and actions
        reward_embedded = self.reward_embedding(reward_history.unsqueeze(-1))  # [batch, len, d_model//4]
        action_embedded = self.action_embedding(action_history)  # [batch, len, d_model//4]
        
        # Combine embeddings
        if self.config.timestep_embedding:
            timesteps = torch.arange(history_len, device=reward_history.device).unsqueeze(0).expand(batch_size, -1)
            timestep_embedded = self.timestep_embedding(timesteps)  # [batch, len, d_model//4]
            
            combined = torch.cat([reward_embedded, action_embedded, timestep_embedded], dim=-1)
        else:
            combined = torch.cat([reward_embedded, action_embedded], dim=-1)
        
        # Encode with LSTM
        encoded, (hidden, _) = self.history_encoder(combined)
        
        # Use final hidden state as context
        context = self.output_projection(hidden[-1])  # [batch, d_model]
        
        return context

class DecisionTransformerLayer(nn.Module):
    """Decision transformer layer with return-to-go conditioning."""
    
    def __init__(self, config: EnhancedAgentConfig):
        super().__init__()
        self.config = config
        
        # Return-to-go embedding
        if config.return_to_go_conditioning:
            self.rtg_embedding = nn.Linear(1, config.d_model)
        
        # Standard transformer layer
        self.transformer_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
        
    def forward(self, x: torch.Tensor, return_to_go: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional return-to-go conditioning.
        
        Args:
            x: Input features [batch_size, seq_len, d_model]
            return_to_go: Return-to-go values [batch_size, seq_len]
            
        Returns:
            Transformed features [batch_size, seq_len, d_model]
        """
        # Add return-to-go conditioning if enabled
        if self.config.return_to_go_conditioning and return_to_go is not None:
            rtg_embedded = self.rtg_embedding(return_to_go.unsqueeze(-1))
            x = x + rtg_embedded
        
        # Apply transformer layer
        if self.config.use_gradient_checkpointing and self.training:
            x = checkpoint(self.transformer_layer, x)
        else:
            x = self.transformer_layer(x)
        
        x = self.layer_norm(x)
        
        return x

class EnhancedSpatialGraphEncoder(nn.Module):
    """Enhanced spatial graph encoder with memory efficiency."""
    
    def __init__(self, config: EnhancedAgentConfig):
        super().__init__()
        self.config = config
        
        # Graph convolution layers with gradient checkpointing
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            GATConv(
                in_channels=config.d_model,
                out_channels=config.gnn_hidden_dim,
                heads=config.gnn_heads,
                dropout=config.dropout
            )
        )
        
        for _ in range(config.gnn_num_layers - 1):
            self.conv_layers.append(
                GATConv(
                    in_channels=config.gnn_hidden_dim * config.gnn_heads,
                    out_channels=config.gnn_hidden_dim,
                    heads=config.gnn_heads,
                    dropout=config.dropout
                )
            )
        
        # Output projection
        self.output_proj = nn.Linear(
            config.gnn_hidden_dim * config.gnn_heads,
            config.d_model
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass with optional gradient checkpointing."""
        # Apply graph convolutions
        for i, conv in enumerate(self.conv_layers):
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint(conv, x, edge_index)
            else:
                x = conv(x, edge_index)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Output projection and normalization
        x = self.output_proj(x)
        x = self.layer_norm(x)
        
        return x

class EnhancedTransformerRLAgent(nn.Module):
    """
    Enhanced Transformer-based RL Agent with all improvements.
    
    Features:
    - Gradient checkpointing for memory efficiency
    - Reward history and context concatenation
    - Cross-modal embeddings (code + image)
    - Decision transformer style recurrence
    - Memory-efficient attention
    - Enhanced context awareness
    """
    
    def __init__(self, config: EnhancedAgentConfig, action_space_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.action_space_config = action_space_config
        self.logger = logging.getLogger(__name__)
        
        # Input embeddings
        self.state_embedding = nn.Linear(action_space_config['state_dim'], config.d_model)
        
        # Cross-modal embedding
        if config.enable_cross_modal:
            self.cross_modal_embedding = CrossModalEmbedding(config)
        
        # Reward and action history encoder
        if config.enable_reward_context or config.enable_action_context:
            self.history_encoder = RewardHistoryEncoder(config)
        
        # Positional encoding
        self.pos_encoder = MemoryEfficientPositionalEncoding(
            config.d_model, 
            config.max_sequence_length
        )
        
        # Spatial graph encoder
        self.spatial_encoder = EnhancedSpatialGraphEncoder(config)
        
        # Decision transformer layers
        self.decision_transformer_layers = nn.ModuleList([
            DecisionTransformerLayer(config) 
            for _ in range(config.num_encoder_layers)
        ])
        
        # Multi-modal fusion
        fusion_input_dim = config.d_model
        if config.enable_cross_modal:
            fusion_input_dim += config.d_model
        if config.enable_reward_context or config.enable_action_context:
            fusion_input_dim += config.d_model
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, action_space_config['action_dim'])
        )
        
        # Value function
        self.value_function = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward // 2, 1)
        )
        
        # Memory for context
        self.reward_memory = deque(maxlen=config.reward_history_length)
        self.action_memory = deque(maxlen=config.action_history_length)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info("Enhanced Transformer RL Agent initialized")
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def update_memory(self, reward: float, action: torch.Tensor):
        """Update reward and action memory."""
        self.reward_memory.append(reward)
        self.action_memory.append(action.detach().cpu())
    
    def get_history_tensors(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get reward and action history as tensors."""
        device = next(self.parameters()).device
        
        # Reward history
        if len(self.reward_memory) > 0:
            reward_history = torch.tensor(
                list(self.reward_memory), 
                dtype=torch.float32, 
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
            
            # Pad if necessary
            if reward_history.shape[1] < self.config.reward_history_length:
                padding = torch.zeros(
                    batch_size, 
                    self.config.reward_history_length - reward_history.shape[1],
                    device=device
                )
                reward_history = torch.cat([padding, reward_history], dim=1)
        else:
            reward_history = torch.zeros(
                batch_size, 
                self.config.reward_history_length, 
                device=device
            )
        
        # Action history
        if len(self.action_memory) > 0:
            action_list = list(self.action_memory)
            action_history = torch.stack(action_list).to(device)
            action_history = action_history.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Pad if necessary
            if action_history.shape[1] < self.config.action_history_length:
                padding = torch.zeros(
                    batch_size,
                    self.config.action_history_length - action_history.shape[1],
                    action_history.shape[2],
                    device=device
                )
                action_history = torch.cat([padding, action_history], dim=1)
        else:
            action_history = torch.zeros(
                batch_size,
                self.config.action_history_length,
                self.action_space_config['action_dim'],
                device=device
            )
        
        return reward_history, action_history
    
    def forward(self, 
                state: torch.Tensor,
                code_tokens: Optional[torch.Tensor] = None,
                image_features: Optional[torch.Tensor] = None,
                spatial_graph: Optional[Data] = None,
                return_to_go: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with all improvements.
        
        Args:
            state: State features [batch_size, state_dim]
            code_tokens: Code tokens [batch_size, seq_len] (optional)
            image_features: Image features [batch_size, image_dim] (optional)
            spatial_graph: Spatial graph data (optional)
            return_to_go: Return-to-go values [batch_size, seq_len] (optional)
            
        Returns:
            Dictionary containing action logits and value
        """
        batch_size = state.shape[0]
        features_list = []
        
        # State embedding
        state_features = self.state_embedding(state)
        features_list.append(state_features)
        
        # Cross-modal embedding
        if (self.config.enable_cross_modal and 
            code_tokens is not None and 
            image_features is not None):
            
            cross_modal_features = self.cross_modal_embedding(code_tokens, image_features)
            features_list.append(cross_modal_features)
        
        # History context
        if self.config.enable_reward_context or self.config.enable_action_context:
            reward_history, action_history = self.get_history_tensors(batch_size)
            history_context = self.history_encoder(reward_history, action_history)
            features_list.append(history_context)
        
        # Spatial graph encoding
        if spatial_graph is not None:
            spatial_features = self.spatial_encoder(
                spatial_graph.x, 
                spatial_graph.edge_index, 
                spatial_graph.batch
            )
            # Expand to match batch size if needed
            if spatial_features.shape[0] != batch_size:
                spatial_features = spatial_features.expand(batch_size, -1)
            features_list.append(spatial_features)
        
        # Fuse all features
        if len(features_list) > 1:
            fused_features = torch.cat(features_list, dim=-1)
            fused_features = self.fusion_layer(fused_features)
        else:
            fused_features = features_list[0]
        
        # Add sequence dimension for transformer
        fused_features = fused_features.unsqueeze(1)  # [batch, 1, d_model]
        
        # Add positional encoding
        fused_features = self.pos_encoder(fused_features.transpose(0, 1)).transpose(0, 1)
        
        # Apply decision transformer layers
        for layer in self.decision_transformer_layers:
            fused_features = layer(fused_features, return_to_go)
        
        # Remove sequence dimension
        final_features = fused_features.squeeze(1)  # [batch, d_model]
        
        # Decode actions and value
        action_logits = self.action_decoder(final_features)
        value = self.value_function(final_features)
        
        return {
            'action_logits': action_logits,
            'value': value,
            'features': final_features
        }
    
    def get_action(self, 
                   state: torch.Tensor,
                   code_tokens: Optional[torch.Tensor] = None,
                   image_features: Optional[torch.Tensor] = None,
                   spatial_graph: Optional[Data] = None,
                   deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get action from the agent.
        
        Args:
            state: Current state
            code_tokens: Code tokens (optional)
            image_features: Image features (optional)
            spatial_graph: Spatial graph (optional)
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action and additional info
        """
        with torch.no_grad():
            output = self.forward(state, code_tokens, image_features, spatial_graph)
            
            action_logits = output['action_logits']
            value = output['value']
            
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action_probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).squeeze(-1)
            
            info = {
                'value': value,
                'action_probs': F.softmax(action_logits, dim=-1),
                'features': output['features']
            }
            
            # Update memory
            self.update_memory(0.0, action)  # Reward will be updated later
            
            return action, info
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'reward_memory_size': len(self.reward_memory),
            'action_memory_size': len(self.action_memory),
            'reward_memory_capacity': self.reward_memory.maxlen,
            'action_memory_capacity': self.action_memory.maxlen
        }

# Factory function for easy creation
def create_enhanced_transformer_agent(
    state_dim: int,
    action_dim: int,
    code_vocab_size: int = 10000,
    image_feature_dim: int = 512,
    config: Optional[EnhancedAgentConfig] = None
) -> EnhancedTransformerRLAgent:
    """
    Factory function to create enhanced transformer agent.
    
    Args:
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        code_vocab_size: Size of code vocabulary
        image_feature_dim: Dimension of image features
        config: Agent configuration
        
    Returns:
        Configured EnhancedTransformerRLAgent
    """
    if config is None:
        config = EnhancedAgentConfig()
    
    # Update config with provided dimensions
    config.code_vocab_size = code_vocab_size
    config.image_feature_dim = image_feature_dim
    
    action_space_config = {
        'state_dim': state_dim,
        'action_dim': action_dim
    }
    
    return EnhancedTransformerRLAgent(config, action_space_config)

