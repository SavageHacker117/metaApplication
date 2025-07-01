"""
Next-Generation Transformer-Based RL Agent

Revolutionary features:
- Transformer architecture for sequential decision making
- Graph Neural Networks for spatial reasoning
- Mixed-precision training for massive scale
- Multi-head attention for complex state understanding
- Hierarchical action spaces
- Advanced memory mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import autocast, GradScaler
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

@dataclass
class AgentConfig:
    """Configuration for next-generation RL agents."""
    # Model architecture
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    # GNN configuration
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 4
    gnn_heads: int = 4
    
    # Training configuration
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    
    # Memory configuration
    memory_size: int = 10000
    sequence_length: int = 128
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architecture."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SpatialGraphEncoder(nn.Module):
    """Graph Neural Network for spatial reasoning."""
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # Graph convolution layers
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
        """
        Forward pass through spatial graph encoder.
        
        Args:
            x: Node features [num_nodes, d_model]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment for nodes
            
        Returns:
            Graph-encoded features [batch_size, d_model]
        """
        # Apply graph convolutions
        for conv in self.conv_layers:
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

class MultiModalAttention(nn.Module):
    """Multi-modal attention for combining different input modalities."""
    
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        
        # Attention mechanisms for different modalities
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Cross-modal fusion
        self.fusion_layer = nn.Linear(config.d_model * 3, config.d_model)
        self.layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, visual_features, spatial_features, temporal_features):
        """
        Fuse multi-modal features using attention.
        
        Args:
            visual_features: Visual input features
            spatial_features: Spatial graph features
            temporal_features: Temporal sequence features
            
        Returns:
            Fused multi-modal representation
        """
        # Self-attention for each modality
        visual_attended, _ = self.visual_attention(
            visual_features, visual_features, visual_features
        )
        
        spatial_attended, _ = self.spatial_attention(
            spatial_features, spatial_features, spatial_features
        )
        
        temporal_attended, _ = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )
        
        # Concatenate and fuse
        fused = torch.cat([visual_attended, spatial_attended, temporal_attended], dim=-1)
        fused = self.fusion_layer(fused)
        fused = self.layer_norm(fused)
        
        return fused

class HierarchicalActionDecoder(nn.Module):
    """Hierarchical action decoder for complex action spaces."""
    
    def __init__(self, config: AgentConfig, action_space_config: Dict[str, int]):
        super().__init__()
        self.config = config
        self.action_space_config = action_space_config
        
        # High-level action decoder
        self.high_level_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, action_space_config['high_level'])
        )
        
        # Low-level action decoders for each high-level action
        self.low_level_decoders = nn.ModuleDict()
        for action_type, action_dim in action_space_config['low_level'].items():
            self.low_level_decoders[action_type] = nn.Sequential(
                nn.Linear(config.d_model, config.dim_feedforward // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim_feedforward // 2, action_dim)
            )
        
        # Value function decoder
        self.value_decoder = nn.Sequential(
            nn.Linear(config.d_model, config.dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward // 2, 1)
        )
    
    def forward(self, features):
        """
        Decode hierarchical actions and value.
        
        Args:
            features: Encoded state features
            
        Returns:
            Dictionary containing action logits and value
        """
        # High-level action logits
        high_level_logits = self.high_level_decoder(features)
        
        # Low-level action logits
        low_level_logits = {}
        for action_type, decoder in self.low_level_decoders.items():
            low_level_logits[action_type] = decoder(features)
        
        # Value function
        value = self.value_decoder(features)
        
        return {
            'high_level_logits': high_level_logits,
            'low_level_logits': low_level_logits,
            'value': value
        }

class TransformerRLAgent(nn.Module):
    """
    Next-generation Transformer-based RL Agent.
    
    Features:
    - Transformer encoder for sequential reasoning
    - Graph Neural Networks for spatial understanding
    - Multi-modal attention for complex state fusion
    - Hierarchical action spaces
    - Mixed-precision training support
    - Advanced memory mechanisms
    """
    
    def __init__(self, config: AgentConfig, action_space_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.action_space_config = action_space_config
        
        # Input embeddings
        self.visual_embedding = nn.Linear(action_space_config['visual_dim'], config.d_model)
        self.state_embedding = nn.Linear(action_space_config['state_dim'], config.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.sequence_length)
        
        # Spatial graph encoder
        self.spatial_encoder = SpatialGraphEncoder(config)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, 
            num_layers=config.num_encoder_layers
        )
        
        # Multi-modal attention
        self.multimodal_attention = MultiModalAttention(config)
        
        # Hierarchical action decoder
        self.action_decoder = HierarchicalActionDecoder(config, action_space_config)
        
        # Memory mechanism
        self.memory = nn.LSTM(
            input_size=config.d_model,
            hidden_size=config.d_model,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def create_spatial_graph(self, grid_state, tower_positions):
        """
        Create spatial graph from game state.
        
        Args:
            grid_state: Grid representation of the game state
            tower_positions: Positions of towers
            
        Returns:
            Graph data structure for GNN processing
        """
        # Create nodes for each grid cell and tower
        nodes = []
        node_features = []
        edges = []
        
        # Grid nodes
        for i in range(grid_state.shape[0]):
            for j in range(grid_state.shape[1]):
                nodes.append((i, j))
                # Node features: position + grid state
                features = torch.cat([
                    torch.tensor([i / grid_state.shape[0], j / grid_state.shape[1]]),
                    grid_state[i, j].flatten()
                ])
                node_features.append(features)
        
        # Tower nodes
        for tower_pos in tower_positions:
            nodes.append(tower_pos)
            # Tower features
            features = torch.cat([
                torch.tensor([tower_pos[0] / grid_state.shape[0], tower_pos[1] / grid_state.shape[1]]),
                torch.ones(grid_state.shape[2])  # Tower indicator
            ])
            node_features.append(features)
        
        # Create edges (spatial connectivity)
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j:
                    # Connect nodes within certain distance
                    dist = np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
                    if dist <= 2.0:  # Adjustable connectivity radius
                        edges.append([i, j])
        
        # Convert to tensors
        node_features = torch.stack(node_features).to(self.config.device)
        edge_index = torch.tensor(edges).t().contiguous().to(self.config.device)
        
        return Data(x=node_features, edge_index=edge_index)
    
    def forward(self, observations, hidden_state=None, return_attention=False):
        """
        Forward pass through the transformer RL agent.
        
        Args:
            observations: Dictionary containing different observation modalities
            hidden_state: Previous hidden state for memory
            return_attention: Whether to return attention weights
            
        Returns:
            Action logits, value, and updated hidden state
        """
        batch_size = observations['visual'].shape[0]
        seq_len = observations['visual'].shape[1]
        
        # Process visual features
        visual_features = self.visual_embedding(observations['visual'])
        visual_features = self.pos_encoder(visual_features.transpose(0, 1)).transpose(0, 1)
        
        # Process state features
        state_features = self.state_embedding(observations['state'])
        state_features = self.pos_encoder(state_features.transpose(0, 1)).transpose(0, 1)
        
        # Process spatial features using GNN
        spatial_features_list = []
        for b in range(batch_size):
            for s in range(seq_len):
                # Create spatial graph for this timestep
                graph_data = self.create_spatial_graph(
                    observations['grid'][b, s],
                    observations['tower_positions'][b, s]
                )
                
                # Encode spatial features
                spatial_encoded = self.spatial_encoder(
                    graph_data.x,
                    graph_data.edge_index
                )
                spatial_features_list.append(spatial_encoded)
        
        # Reshape spatial features
        spatial_features = torch.stack(spatial_features_list).view(batch_size, seq_len, -1)
        
        # Multi-modal attention fusion
        fused_features = self.multimodal_attention(
            visual_features,
            spatial_features,
            state_features
        )
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(fused_features)
        
        # Memory processing
        if hidden_state is None:
            hidden_state = (
                torch.zeros(2, batch_size, self.config.d_model).to(self.config.device),
                torch.zeros(2, batch_size, self.config.d_model).to(self.config.device)
            )
        
        memory_output, new_hidden_state = self.memory(transformer_output, hidden_state)
        
        # Take the last timestep for action prediction
        final_features = memory_output[:, -1, :]
        
        # Decode actions and value
        outputs = self.action_decoder(final_features)
        
        if return_attention:
            # Extract attention weights (simplified)
            attention_weights = {
                'transformer': None,  # Would need to modify transformer to return attention
                'multimodal': None    # Would need to modify multimodal attention
            }
            outputs['attention_weights'] = attention_weights
        
        outputs['hidden_state'] = new_hidden_state
        
        return outputs
    
    def get_action(self, observations, hidden_state=None, deterministic=False):
        """
        Get action from observations.
        
        Args:
            observations: Current observations
            hidden_state: Previous hidden state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Selected action and updated hidden state
        """
        with torch.no_grad():
            outputs = self.forward(observations, hidden_state)
            
            # Sample high-level action
            high_level_probs = F.softmax(outputs['high_level_logits'], dim=-1)
            if deterministic:
                high_level_action = torch.argmax(high_level_probs, dim=-1)
            else:
                high_level_action = torch.multinomial(high_level_probs, 1).squeeze(-1)
            
            # Sample low-level actions
            low_level_actions = {}
            for action_type, logits in outputs['low_level_logits'].items():
                probs = F.softmax(logits, dim=-1)
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = torch.multinomial(probs, 1).squeeze(-1)
                low_level_actions[action_type] = action
            
            return {
                'high_level_action': high_level_action,
                'low_level_actions': low_level_actions,
                'value': outputs['value'],
                'hidden_state': outputs['hidden_state']
            }

class MixedPrecisionTrainer:
    """Mixed-precision trainer for transformer RL agent."""
    
    def __init__(self, agent: TransformerRLAgent, config: AgentConfig):
        self.agent = agent
        self.config = config
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            agent.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,  # Adjust based on training length
            eta_min=config.learning_rate * 0.01
        )
        
        self.logger = logging.getLogger(__name__)
    
    def train_step(self, batch):
        """
        Perform one training step with mixed precision.
        
        Args:
            batch: Training batch containing observations, actions, rewards, etc.
            
        Returns:
            Training metrics
        """
        self.optimizer.zero_grad()
        
        if self.config.mixed_precision:
            with autocast():
                outputs = self.agent(batch['observations'])
                loss = self._compute_loss(outputs, batch)
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.gradient_clipping
            )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.agent(batch['observations'])
            loss = self._compute_loss(outputs, batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.gradient_clipping
            )
            
            self.optimizer.step()
        
        self.scheduler.step()
        
        return {
            'loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _compute_loss(self, outputs, batch):
        """Compute training loss."""
        # Policy loss (cross-entropy for discrete actions)
        high_level_loss = F.cross_entropy(
            outputs['high_level_logits'],
            batch['high_level_actions']
        )
        
        low_level_loss = 0
        for action_type, logits in outputs['low_level_logits'].items():
            low_level_loss += F.cross_entropy(
                logits,
                batch['low_level_actions'][action_type]
            )
        
        # Value loss (MSE)
        value_loss = F.mse_loss(
            outputs['value'].squeeze(-1),
            batch['returns']
        )
        
        # Total loss
        total_loss = high_level_loss + low_level_loss + 0.5 * value_loss
        
        return total_loss

# Factory function for creating optimized agents
def create_transformer_agent(
    visual_dim: int,
    state_dim: int,
    action_space_config: Dict[str, Any],
    device: Optional[str] = None,
    mixed_precision: bool = True
) -> Tuple[TransformerRLAgent, MixedPrecisionTrainer]:
    """
    Factory function to create optimized transformer RL agent.
    
    Args:
        visual_dim: Dimension of visual observations
        state_dim: Dimension of state observations
        action_space_config: Configuration for action spaces
        device: Target device
        mixed_precision: Enable mixed precision training
        
    Returns:
        Configured agent and trainer
    """
    config = AgentConfig(
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        mixed_precision=mixed_precision
    )
    
    # Add dimensions to action space config
    action_space_config.update({
        'visual_dim': visual_dim,
        'state_dim': state_dim
    })
    
    agent = TransformerRLAgent(config, action_space_config).to(config.device)
    trainer = MixedPrecisionTrainer(agent, config)
    
    return agent, trainer

