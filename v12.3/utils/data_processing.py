"""
Data Processing Utilities for RL-LLM System

This module provides utilities for processing training data, experience replay,
and data transformations used throughout the RL training pipeline.
"""

import numpy as np
import pandas as pd
import pickle
import json
import h5py
import zarr
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import logging
from collections import deque, defaultdict
import torch

logger = logging.getLogger(__name__)


class ExperienceBuffer:
    """
    Experience replay buffer for RL training.
    
    Supports multiple data formats and efficient sampling.
    """
    
    def __init__(self, capacity: int = 100000, batch_size: int = 32):
        """
        Initialize experience buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            batch_size: Default batch size for sampling
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
        logger.info(f"Initialized ExperienceBuffer with capacity {capacity}")
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool, info: Optional[Dict] = None):
        """Add experience to buffer."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {}
        }
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Sample batch of experiences."""
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Stack experiences
        states = np.stack([exp['state'] for exp in batch])
        actions = np.stack([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.stack([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'indices': indices
        }
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def save(self, filepath: Union[str, Path]):
        """Save buffer to file."""
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)
        logger.info(f"Saved experience buffer to {filepath}")
    
    def load(self, filepath: Union[str, Path]):
        """Load buffer from file."""
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            experiences = pickle.load(f)
        
        self.buffer = deque(experiences, maxlen=self.capacity)
        logger.info(f"Loaded {len(self.buffer)} experiences from {filepath}")


class DataNormalizer:
    """
    Data normalization utilities for RL observations and rewards.
    """
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
        """
        self.method = method
        self.fitted = False
        self.stats = {}
        
    def fit(self, data: np.ndarray):
        """Fit normalizer to data."""
        if self.method == 'standard':
            self.stats['mean'] = np.mean(data, axis=0)
            self.stats['std'] = np.std(data, axis=0) + 1e-8
        elif self.method == 'minmax':
            self.stats['min'] = np.min(data, axis=0)
            self.stats['max'] = np.max(data, axis=0)
            self.stats['range'] = self.stats['max'] - self.stats['min'] + 1e-8
        elif self.method == 'robust':
            self.stats['median'] = np.median(data, axis=0)
            self.stats['mad'] = np.median(np.abs(data - self.stats['median']), axis=0) + 1e-8
        
        self.fitted = True
        logger.info(f"Fitted {self.method} normalizer to data shape {data.shape}")
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted normalizer."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        if self.method == 'standard':
            return (data - self.stats['mean']) / self.stats['std']
        elif self.method == 'minmax':
            return (data - self.stats['min']) / self.stats['range']
        elif self.method == 'robust':
            return (data - self.stats['median']) / self.stats['mad']
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data."""
        if not self.fitted:
            raise ValueError("Normalizer must be fitted before inverse transform")
        
        if self.method == 'standard':
            return data * self.stats['std'] + self.stats['mean']
        elif self.method == 'minmax':
            return data * self.stats['range'] + self.stats['min']
        elif self.method == 'robust':
            return data * self.stats['mad'] + self.stats['median']
    
    def save(self, filepath: Union[str, Path]):
        """Save normalizer state."""
        filepath = Path(filepath)
        with open(filepath, 'json') as f:
            save_data = {
                'method': self.method,
                'fitted': self.fitted,
                'stats': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                         for k, v in self.stats.items()}
            }
            json.dump(save_data, f)
    
    def load(self, filepath: Union[str, Path]):
        """Load normalizer state."""
        filepath = Path(filepath)
        with open(filepath, 'r') as f:
            save_data = json.load(f)
        
        self.method = save_data['method']
        self.fitted = save_data['fitted']
        self.stats = {k: np.array(v) if isinstance(v, list) else v 
                     for k, v in save_data['stats'].items()}


class TrajectoryProcessor:
    """
    Process and analyze RL training trajectories.
    """
    
    def __init__(self):
        """Initialize trajectory processor."""
        self.trajectories = []
        self.metrics = defaultdict(list)
    
    def add_trajectory(self, trajectory: List[Dict[str, Any]]):
        """Add a complete trajectory."""
        self.trajectories.append(trajectory)
        
        # Calculate trajectory metrics
        total_reward = sum(step['reward'] for step in trajectory)
        trajectory_length = len(trajectory)
        
        self.metrics['total_rewards'].append(total_reward)
        self.metrics['trajectory_lengths'].append(trajectory_length)
        self.metrics['average_rewards'].append(total_reward / trajectory_length)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get trajectory statistics."""
        if not self.trajectories:
            return {}
        
        stats = {}
        for metric_name, values in self.metrics.items():
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        
        return stats
    
    def compute_returns(self, gamma: float = 0.99) -> List[List[float]]:
        """Compute discounted returns for all trajectories."""
        all_returns = []
        
        for trajectory in self.trajectories:
            returns = []
            G = 0
            
            # Compute returns backwards
            for step in reversed(trajectory):
                G = step['reward'] + gamma * G
                returns.append(G)
            
            returns.reverse()
            all_returns.append(returns)
        
        return all_returns
    
    def compute_advantages(self, values: List[List[float]], gamma: float = 0.99, 
                          lam: float = 0.95) -> List[List[float]]:
        """Compute GAE advantages."""
        all_advantages = []
        
        for traj_idx, trajectory in enumerate(self.trajectories):
            traj_values = values[traj_idx]
            advantages = []
            gae = 0
            
            for t in reversed(range(len(trajectory))):
                if t == len(trajectory) - 1:
                    next_value = 0
                else:
                    next_value = traj_values[t + 1]
                
                delta = trajectory[t]['reward'] + gamma * next_value - traj_values[t]
                gae = delta + gamma * lam * gae
                advantages.append(gae)
            
            advantages.reverse()
            all_advantages.append(advantages)
        
        return all_advantages


def save_training_data(data: Dict[str, Any], filepath: Union[str, Path], 
                      format: str = 'pickle'):
    """
    Save training data in various formats.
    
    Args:
        data: Data to save
        filepath: Output file path
        format: Save format ('pickle', 'json', 'hdf5', 'zarr')
    """
    filepath = Path(filepath)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    elif format == 'json':
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    elif format == 'hdf5':
        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                else:
                    f.attrs[key] = value
    
    elif format == 'zarr':
        zarr.save(filepath, **data)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved training data to {filepath} in {format} format")


def load_training_data(filepath: Union[str, Path], format: str = 'pickle') -> Dict[str, Any]:
    """
    Load training data from various formats.
    
    Args:
        filepath: Input file path
        format: File format ('pickle', 'json', 'hdf5', 'zarr')
        
    Returns:
        Loaded data dictionary
    """
    filepath = Path(filepath)
    
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    elif format == 'json':
        with open(filepath, 'r') as f:
            return json.load(f)
    
    elif format == 'hdf5':
        data = {}
        with h5py.File(filepath, 'r') as f:
            # Load datasets
            for key in f.keys():
                data[key] = f[key][:]
            
            # Load attributes
            for key, value in f.attrs.items():
                data[key] = value
        
        return data
    
    elif format == 'zarr':
        return dict(zarr.load(filepath))
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def preprocess_observations(observations: np.ndarray, 
                          normalizer: Optional[DataNormalizer] = None) -> np.ndarray:
    """
    Preprocess RL observations.
    
    Args:
        observations: Raw observations
        normalizer: Optional normalizer to apply
        
    Returns:
        Preprocessed observations
    """
    # Handle different observation types
    if observations.dtype == np.uint8:
        # Image observations - normalize to [0, 1]
        observations = observations.astype(np.float32) / 255.0
    
    # Apply normalizer if provided
    if normalizer is not None:
        observations = normalizer.transform(observations)
    
    return observations


def compute_episode_metrics(episode_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute metrics for a single episode.
    
    Args:
        episode_data: List of step data dictionaries
        
    Returns:
        Episode metrics dictionary
    """
    if not episode_data:
        return {}
    
    rewards = [step['reward'] for step in episode_data]
    
    metrics = {
        'total_reward': sum(rewards),
        'episode_length': len(episode_data),
        'average_reward': np.mean(rewards),
        'reward_std': np.std(rewards),
        'min_reward': min(rewards),
        'max_reward': max(rewards)
    }
    
    # Add custom metrics if available
    if 'info' in episode_data[0]:
        info_keys = set()
        for step in episode_data:
            if 'info' in step and step['info']:
                info_keys.update(step['info'].keys())
        
        for key in info_keys:
            values = []
            for step in episode_data:
                if 'info' in step and step['info'] and key in step['info']:
                    if isinstance(step['info'][key], (int, float)):
                        values.append(step['info'][key])
            
            if values:
                metrics[f'info_{key}_mean'] = np.mean(values)
                metrics[f'info_{key}_final'] = values[-1]
    
    return metrics

