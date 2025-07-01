"""
Episodic Replay Buffer for Version 5 BETA 1

This module implements an advanced episodic replay buffer that stores complete
episodes and provides sophisticated sampling strategies for improved learning.

Key features:
- Episodic storage with complete trajectory information
- Priority-based sampling with multiple strategies
- NeRF asset correlation tracking
- Curriculum-aware sampling
- Memory-efficient storage with compression
- Advanced analytics and visualization
"""

import numpy as np
import torch
import pickle
import gzip
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import time
import random
import heapq
from pathlib import Path
import threading
import json

logger = logging.getLogger(__name__)

class SamplingStrategy(Enum):
    """Different sampling strategies for replay buffer."""
    UNIFORM = "uniform"                  # Uniform random sampling
    PRIORITY = "priority"                # Priority-based sampling
    CURRICULUM = "curriculum"            # Curriculum-aware sampling
    DIVERSITY = "diversity"              # Diversity-based sampling
    RECENT = "recent"                    # Recent episode bias
    PERFORMANCE = "performance"          # Performance-based sampling
    NERF_CORRELATION = "nerf_correlation" # NeRF asset correlation sampling

class CompressionType(Enum):
    """Compression types for episode storage."""
    NONE = "none"
    GZIP = "gzip"
    PICKLE = "pickle"
    TORCH = "torch"

@dataclass
class EpisodeMetadata:
    """Metadata for stored episodes."""
    episode_id: int
    timestamp: float
    total_reward: float
    episode_length: int
    difficulty: float
    success: bool
    nerf_assets_used: set
    performance_metrics: Dict[str, float]
    priority_score: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0
    compression_type: CompressionType = CompressionType.NONE

@dataclass
class ReplayBufferConfig:
    """Configuration for episodic replay buffer."""
    
    # Capacity settings
    max_episodes: int = 10000
    max_memory_mb: int = 1000
    auto_cleanup: bool = True
    cleanup_threshold: float = 0.9  # Cleanup when 90% full
    
    # Sampling settings
    default_sampling_strategy: SamplingStrategy = SamplingStrategy.PRIORITY
    priority_alpha: float = 0.6  # Priority exponent
    priority_beta: float = 0.4   # Importance sampling exponent
    curriculum_weight: float = 0.3
    diversity_weight: float = 0.2
    
    # Storage settings
    compression_enabled: bool = True
    compression_type: CompressionType = CompressionType.GZIP
    compression_threshold: int = 1000  # Compress episodes longer than this
    
    # Performance settings
    batch_size: int = 32
    prefetch_size: int = 64
    async_loading: bool = True
    
    # Analytics settings
    track_correlations: bool = True
    correlation_window: int = 100
    performance_tracking: bool = True

class EpisodicReplayBuffer:
    """
    Advanced episodic replay buffer with sophisticated sampling and storage.
    
    Features:
    - Multiple sampling strategies
    - Priority-based experience replay
    - Curriculum-aware sampling
    - NeRF asset correlation tracking
    - Memory-efficient compression
    - Advanced analytics
    """
    
    def __init__(self, config: Optional[ReplayBufferConfig] = None, device: str = "cpu"):
        self.config = config or ReplayBufferConfig()
        self.device = device
        
        # Storage
        self.episodes: Dict[int, Any] = {}  # Actual episode data
        self.metadata: Dict[int, EpisodeMetadata] = {}  # Episode metadata
        self.episode_counter = 0
        
        # Priority management
        self.priority_heap: List[Tuple[float, int]] = []  # (priority, episode_id)
        self.priority_sum = 0.0
        self.max_priority = 1.0
        
        # Sampling strategies
        self.sampling_strategies = {
            SamplingStrategy.UNIFORM: self._uniform_sampling,
            SamplingStrategy.PRIORITY: self._priority_sampling,
            SamplingStrategy.CURRICULUM: self._curriculum_sampling,
            SamplingStrategy.DIVERSITY: self._diversity_sampling,
            SamplingStrategy.RECENT: self._recent_sampling,
            SamplingStrategy.PERFORMANCE: self._performance_sampling,
            SamplingStrategy.NERF_CORRELATION: self._nerf_correlation_sampling
        }
        
        # Analytics and tracking
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.performance_history: deque = deque(maxlen=1000)
        self.sampling_statistics: Dict[str, int] = defaultdict(int)
        
        # Memory management
        self.current_memory_mb = 0.0
        self.compression_stats = defaultdict(int)
        
        # Threading for async operations
        self.lock = threading.Lock()
        self.prefetch_queue: deque = deque(maxlen=self.config.prefetch_size)
        
        logger.info(f"Episodic Replay Buffer initialized with capacity {self.config.max_episodes}")
    
    def add_episode(self, episode_data: Dict[str, Any]) -> int:
        """
        Add a complete episode to the replay buffer.
        
        Args:
            episode_data: Complete episode data including observations, actions, rewards
            
        Returns:
            Episode ID assigned to the stored episode
        """
        
        with self.lock:
            episode_id = self.episode_counter
            self.episode_counter += 1
            
            # Create metadata
            metadata = self._create_episode_metadata(episode_id, episode_data)
            
            # Compress episode if needed
            compressed_data = self._compress_episode(episode_data)
            
            # Store episode and metadata
            self.episodes[episode_id] = compressed_data
            self.metadata[episode_id] = metadata
            
            # Update priority heap
            self._update_priority(episode_id, metadata.priority_score)
            
            # Update correlations
            if self.config.track_correlations:
                self._update_correlations(episode_data)
            
            # Update memory tracking
            self._update_memory_usage(episode_id, compressed_data)
            
            # Cleanup if necessary
            if self.config.auto_cleanup and self._should_cleanup():
                self._cleanup_old_episodes()
            
            logger.debug(f"Added episode {episode_id} with priority {metadata.priority_score:.4f}")
            
            return episode_id
    
    def sample_batch(self, 
                    batch_size: Optional[int] = None,
                    strategy: Optional[SamplingStrategy] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Sample a batch of episodes using the specified strategy.
        
        Args:
            batch_size: Number of episodes to sample
            strategy: Sampling strategy to use
            **kwargs: Additional arguments for sampling strategy
            
        Returns:
            Dictionary containing sampled episodes and metadata
        """
        
        batch_size = batch_size or self.config.batch_size
        strategy = strategy or self.config.default_sampling_strategy
        
        if len(self.episodes) == 0:
            return {'episodes': [], 'metadata': [], 'indices': [], 'weights': []}
        
        # Get sampling function
        sampling_func = self.sampling_strategies.get(strategy, self._uniform_sampling)
        
        # Sample episode indices
        indices, weights = sampling_func(batch_size, **kwargs)
        
        # Load episodes
        episodes = []
        metadata_list = []
        
        for idx in indices:
            episode = self._load_episode(idx)
            metadata = self.metadata[idx]
            
            episodes.append(episode)
            metadata_list.append(metadata)
            
            # Update access tracking
            metadata.access_count += 1
            metadata.last_accessed = time.time()
        
        # Update sampling statistics
        self.sampling_statistics[strategy.value] += len(indices)
        
        return {
            'episodes': episodes,
            'metadata': metadata_list,
            'indices': indices,
            'weights': weights,
            'strategy': strategy.value
        }
    
    def _create_episode_metadata(self, episode_id: int, episode_data: Dict[str, Any]) -> EpisodeMetadata:
        """Create metadata for an episode."""
        
        # Extract basic information
        total_reward = episode_data.get('total_reward', 0.0)
        episode_length = episode_data.get('steps', 0)
        difficulty = episode_data.get('difficulty', 0.5)
        success = episode_data.get('success', False)
        nerf_assets = set(episode_data.get('nerf_assets_used', []))
        performance_metrics = episode_data.get('performance_metrics', {})
        
        # Calculate priority score
        priority_score = self._calculate_priority_score(episode_data)
        
        return EpisodeMetadata(
            episode_id=episode_id,
            timestamp=time.time(),
            total_reward=total_reward,
            episode_length=episode_length,
            difficulty=difficulty,
            success=success,
            nerf_assets_used=nerf_assets,
            performance_metrics=performance_metrics,
            priority_score=priority_score
        )
    
    def _calculate_priority_score(self, episode_data: Dict[str, Any]) -> float:
        """Calculate priority score for an episode."""
        
        # Base priority from reward
        reward_score = episode_data.get('total_reward', 0.0)
        normalized_reward = max(0.0, min(1.0, reward_score / 50.0))  # Normalize to [0,1]
        
        # Success bonus
        success_bonus = 0.2 if episode_data.get('success', False) else 0.0
        
        # Difficulty bonus
        difficulty = episode_data.get('difficulty', 0.5)
        difficulty_bonus = difficulty * 0.1
        
        # NeRF diversity bonus
        nerf_assets = episode_data.get('nerf_assets_used', set())
        diversity_bonus = min(len(nerf_assets) * 0.05, 0.2)
        
        # Performance metrics bonus
        performance_metrics = episode_data.get('performance_metrics', {})
        performance_bonus = 0.0
        if performance_metrics:
            avg_metrics = np.mean(list(performance_metrics.values()))
            performance_bonus = min(avg_metrics * 0.1, 0.1)
        
        # Combine scores
        priority_score = (
            normalized_reward * 0.5 +
            success_bonus +
            difficulty_bonus +
            diversity_bonus +
            performance_bonus
        )
        
        return max(0.01, priority_score)  # Minimum priority
    
    def _compress_episode(self, episode_data: Dict[str, Any]) -> Any:
        """Compress episode data if needed."""
        
        if not self.config.compression_enabled:
            return episode_data
        
        episode_length = episode_data.get('steps', 0)
        
        if episode_length < self.config.compression_threshold:
            return episode_data
        
        try:
            if self.config.compression_type == CompressionType.GZIP:
                serialized = pickle.dumps(episode_data)
                compressed = gzip.compress(serialized)
                self.compression_stats['gzip_compressed'] += 1
                return {'compressed': True, 'type': 'gzip', 'data': compressed}
            
            elif self.config.compression_type == CompressionType.TORCH:
                # Convert numpy arrays to torch tensors for better compression
                torch_data = self._convert_to_torch(episode_data)
                compressed = torch.save(torch_data, compression='gzip')
                self.compression_stats['torch_compressed'] += 1
                return {'compressed': True, 'type': 'torch', 'data': compressed}
            
            else:
                return episode_data
                
        except Exception as e:
            logger.warning(f"Compression failed: {e}, storing uncompressed")
            return episode_data
    
    def _convert_to_torch(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert episode data to torch tensors."""
        torch_data = {}
        
        for key, value in episode_data.items():
            if isinstance(value, np.ndarray):
                torch_data[key] = torch.from_numpy(value)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                torch_data[key] = [torch.from_numpy(v) for v in value]
            else:
                torch_data[key] = value
        
        return torch_data
    
    def _load_episode(self, episode_id: int) -> Dict[str, Any]:
        """Load and decompress episode data."""
        
        episode_data = self.episodes[episode_id]
        
        # Check if compressed
        if isinstance(episode_data, dict) and episode_data.get('compressed', False):
            compression_type = episode_data.get('type', 'gzip')
            compressed_data = episode_data['data']
            
            try:
                if compression_type == 'gzip':
                    decompressed = gzip.decompress(compressed_data)
                    return pickle.loads(decompressed)
                
                elif compression_type == 'torch':
                    return torch.load(compressed_data)
                
                else:
                    logger.warning(f"Unknown compression type: {compression_type}")
                    return episode_data
                    
            except Exception as e:
                logger.error(f"Decompression failed for episode {episode_id}: {e}")
                return {}
        
        return episode_data
    
    def _update_priority(self, episode_id: int, priority: float):
        """Update priority for an episode."""
        
        # Update max priority
        self.max_priority = max(self.max_priority, priority)
        
        # Add to priority heap
        heapq.heappush(self.priority_heap, (-priority, episode_id))  # Negative for max heap
        
        # Update priority sum
        self.priority_sum += priority
    
    def _update_correlations(self, episode_data: Dict[str, Any]):
        """Update NeRF asset correlations."""
        
        nerf_assets = list(episode_data.get('nerf_assets_used', []))
        
        if len(nerf_assets) < 2:
            return
        
        # Update pairwise correlations
        for i, asset1 in enumerate(nerf_assets):
            for asset2 in nerf_assets[i+1:]:
                correlation_key = tuple(sorted([asset1, asset2]))
                
                if correlation_key not in self.correlation_matrix:
                    self.correlation_matrix[correlation_key] = 0.0
                
                # Simple correlation update (could be more sophisticated)
                self.correlation_matrix[correlation_key] += 1.0
    
    def _update_memory_usage(self, episode_id: int, episode_data: Any):
        """Update memory usage tracking."""
        
        # Estimate memory usage
        if isinstance(episode_data, dict) and episode_data.get('compressed', False):
            memory_size = len(episode_data['data']) / (1024 * 1024)  # MB
        else:
            # Rough estimate for uncompressed data
            memory_size = len(str(episode_data)) / (1024 * 1024)  # MB
        
        self.current_memory_mb += memory_size
    
    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        
        # Check episode count
        if len(self.episodes) >= self.config.max_episodes * self.config.cleanup_threshold:
            return True
        
        # Check memory usage
        if self.current_memory_mb >= self.config.max_memory_mb * self.config.cleanup_threshold:
            return True
        
        return False
    
    def _cleanup_old_episodes(self):
        """Remove old episodes to free up space."""
        
        logger.info("Starting replay buffer cleanup...")
        
        # Calculate how many episodes to remove
        target_episodes = int(self.config.max_episodes * 0.8)  # Remove to 80% capacity
        episodes_to_remove = len(self.episodes) - target_episodes
        
        if episodes_to_remove <= 0:
            return
        
        # Sort episodes by priority and age (remove low priority, old episodes first)
        episode_scores = []
        current_time = time.time()
        
        for episode_id, metadata in self.metadata.items():
            # Combine priority and recency
            age_penalty = (current_time - metadata.timestamp) / 86400.0  # Days
            score = metadata.priority_score - age_penalty * 0.1
            episode_scores.append((score, episode_id))
        
        # Sort by score (ascending, so lowest scores are removed first)
        episode_scores.sort()
        
        # Remove episodes
        removed_count = 0
        for score, episode_id in episode_scores[:episodes_to_remove]:
            if episode_id in self.episodes:
                del self.episodes[episode_id]
                del self.metadata[episode_id]
                removed_count += 1
        
        # Rebuild priority heap
        self.priority_heap = []
        self.priority_sum = 0.0
        
        for episode_id, metadata in self.metadata.items():
            self._update_priority(episode_id, metadata.priority_score)
        
        # Update memory tracking
        self.current_memory_mb *= (len(self.episodes) / (len(self.episodes) + removed_count))
        
        logger.info(f"Cleanup completed: removed {removed_count} episodes")
    
    # Sampling strategies
    def _uniform_sampling(self, batch_size: int, **kwargs) -> Tuple[List[int], List[float]]:
        """Uniform random sampling."""
        
        episode_ids = list(self.episodes.keys())
        
        if len(episode_ids) <= batch_size:
            indices = episode_ids
        else:
            indices = random.sample(episode_ids, batch_size)
        
        weights = [1.0] * len(indices)  # Uniform weights
        
        return indices, weights
    
    def _priority_sampling(self, batch_size: int, **kwargs) -> Tuple[List[int], List[float]]:
        """Priority-based sampling using prioritized experience replay."""
        
        if self.priority_sum == 0:
            return self._uniform_sampling(batch_size, **kwargs)
        
        indices = []
        weights = []
        
        # Sample based on priorities
        for _ in range(min(batch_size, len(self.episodes))):
            rand_val = random.random() * self.priority_sum
            cumulative_priority = 0.0
            
            for episode_id, metadata in self.metadata.items():
                cumulative_priority += metadata.priority_score
                if cumulative_priority >= rand_val:
                    indices.append(episode_id)
                    
                    # Calculate importance sampling weight
                    prob = metadata.priority_score / self.priority_sum
                    weight = (1.0 / (len(self.episodes) * prob)) ** self.config.priority_beta
                    weights.append(weight)
                    break
        
        # Normalize weights
        if weights:
            max_weight = max(weights)
            weights = [w / max_weight for w in weights]
        
        return indices, weights
    
    def _curriculum_sampling(self, batch_size: int, current_difficulty: float = 0.5, **kwargs) -> Tuple[List[int], List[float]]:
        """Curriculum-aware sampling that prefers episodes near current difficulty."""
        
        episode_scores = []
        
        for episode_id, metadata in self.metadata.items():
            # Score based on difficulty proximity
            difficulty_diff = abs(metadata.difficulty - current_difficulty)
            difficulty_score = 1.0 / (1.0 + difficulty_diff)
            
            # Combine with priority
            combined_score = (
                difficulty_score * self.config.curriculum_weight +
                metadata.priority_score * (1.0 - self.config.curriculum_weight)
            )
            
            episode_scores.append((combined_score, episode_id))
        
        # Sort by score and sample top episodes
        episode_scores.sort(reverse=True)
        
        # Sample with some randomness
        top_candidates = episode_scores[:batch_size * 2]  # Get more candidates
        
        if len(top_candidates) <= batch_size:
            indices = [episode_id for _, episode_id in top_candidates]
        else:
            # Weighted sampling from top candidates
            scores = [score for score, _ in top_candidates]
            total_score = sum(scores)
            
            indices = []
            for _ in range(batch_size):
                rand_val = random.random() * total_score
                cumulative_score = 0.0
                
                for score, episode_id in top_candidates:
                    cumulative_score += score
                    if cumulative_score >= rand_val and episode_id not in indices:
                        indices.append(episode_id)
                        break
        
        weights = [1.0] * len(indices)  # Equal weights for curriculum sampling
        
        return indices, weights
    
    def _diversity_sampling(self, batch_size: int, **kwargs) -> Tuple[List[int], List[float]]:
        """Diversity-based sampling that prefers episodes with diverse NeRF assets."""
        
        # Track assets already selected
        selected_assets = set()
        episode_scores = []
        
        for episode_id, metadata in self.metadata.items():
            # Calculate diversity score
            episode_assets = metadata.nerf_assets_used
            new_assets = episode_assets - selected_assets
            diversity_score = len(new_assets) / max(len(episode_assets), 1)
            
            # Combine with priority
            combined_score = (
                diversity_score * self.config.diversity_weight +
                metadata.priority_score * (1.0 - self.config.diversity_weight)
            )
            
            episode_scores.append((combined_score, episode_id, episode_assets))
        
        # Greedy selection for diversity
        episode_scores.sort(reverse=True)
        
        indices = []
        for score, episode_id, episode_assets in episode_scores:
            if len(indices) >= batch_size:
                break
            
            indices.append(episode_id)
            selected_assets.update(episode_assets)
        
        weights = [1.0] * len(indices)
        
        return indices, weights
    
    def _recent_sampling(self, batch_size: int, recency_weight: float = 0.7, **kwargs) -> Tuple[List[int], List[float]]:
        """Recent episode biased sampling."""
        
        current_time = time.time()
        episode_scores = []
        
        for episode_id, metadata in self.metadata.items():
            # Calculate recency score
            age_hours = (current_time - metadata.timestamp) / 3600.0
            recency_score = 1.0 / (1.0 + age_hours * 0.1)  # Decay over time
            
            # Combine with priority
            combined_score = (
                recency_score * recency_weight +
                metadata.priority_score * (1.0 - recency_weight)
            )
            
            episode_scores.append((combined_score, episode_id))
        
        # Sort and sample top episodes
        episode_scores.sort(reverse=True)
        
        indices = [episode_id for _, episode_id in episode_scores[:batch_size]]
        weights = [1.0] * len(indices)
        
        return indices, weights
    
    def _performance_sampling(self, batch_size: int, **kwargs) -> Tuple[List[int], List[float]]:
        """Performance-based sampling that prefers high-performing episodes."""
        
        episode_scores = []
        
        for episode_id, metadata in self.metadata.items():
            # Performance score based on success and reward
            performance_score = metadata.total_reward
            if metadata.success:
                performance_score *= 1.5  # Success bonus
            
            episode_scores.append((performance_score, episode_id))
        
        # Sort by performance
        episode_scores.sort(reverse=True)
        
        # Sample top performers with some randomness
        top_count = min(batch_size * 3, len(episode_scores))
        top_episodes = episode_scores[:top_count]
        
        if len(top_episodes) <= batch_size:
            indices = [episode_id for _, episode_id in top_episodes]
        else:
            indices = random.sample([episode_id for _, episode_id in top_episodes], batch_size)
        
        weights = [1.0] * len(indices)
        
        return indices, weights
    
    def _nerf_correlation_sampling(self, batch_size: int, target_assets: Optional[List[str]] = None, **kwargs) -> Tuple[List[int], List[float]]:
        """NeRF correlation-based sampling."""
        
        if not target_assets:
            # If no target assets specified, use most common assets
            asset_counts = defaultdict(int)
            for metadata in self.metadata.values():
                for asset in metadata.nerf_assets_used:
                    asset_counts[asset] += 1
            
            target_assets = [asset for asset, _ in sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        episode_scores = []
        
        for episode_id, metadata in self.metadata.items():
            # Calculate correlation score
            episode_assets = metadata.nerf_assets_used
            correlation_score = 0.0
            
            for target_asset in target_assets:
                if target_asset in episode_assets:
                    correlation_score += 1.0
                    
                    # Add correlation bonuses
                    for other_asset in episode_assets:
                        if other_asset != target_asset:
                            correlation_key = tuple(sorted([target_asset, other_asset]))
                            correlation_score += self.correlation_matrix.get(correlation_key, 0.0) * 0.1
            
            episode_scores.append((correlation_score, episode_id))
        
        # Sort by correlation score
        episode_scores.sort(reverse=True)
        
        indices = [episode_id for _, episode_id in episode_scores[:batch_size]]
        weights = [1.0] * len(indices)
        
        return indices, weights
    
    def update_priorities(self, episode_ids: List[int], td_errors: List[float]):
        """Update priorities based on TD errors."""
        
        with self.lock:
            for episode_id, td_error in zip(episode_ids, td_errors):
                if episode_id in self.metadata:
                    # Update priority based on TD error
                    new_priority = abs(td_error) + 1e-6  # Small epsilon to avoid zero priority
                    self.metadata[episode_id].priority_score = new_priority
                    
                    # Update max priority
                    self.max_priority = max(self.max_priority, new_priority)
            
            # Rebuild priority sum
            self.priority_sum = sum(metadata.priority_score for metadata in self.metadata.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive replay buffer statistics."""
        
        stats = {
            'total_episodes': len(self.episodes),
            'memory_usage_mb': self.current_memory_mb,
            'memory_limit_mb': self.config.max_memory_mb,
            'compression_stats': dict(self.compression_stats),
            'sampling_statistics': dict(self.sampling_statistics),
            'priority_stats': {
                'max_priority': self.max_priority,
                'avg_priority': self.priority_sum / len(self.metadata) if self.metadata else 0.0,
                'priority_sum': self.priority_sum
            }
        }
        
        # Episode statistics
        if self.metadata:
            rewards = [m.total_reward for m in self.metadata.values()]
            lengths = [m.episode_length for m in self.metadata.values()]
            difficulties = [m.difficulty for m in self.metadata.values()]
            
            stats['episode_stats'] = {
                'avg_reward': np.mean(rewards),
                'reward_std': np.std(rewards),
                'avg_length': np.mean(lengths),
                'avg_difficulty': np.mean(difficulties),
                'success_rate': np.mean([m.success for m in self.metadata.values()])
            }
        
        # Asset correlation statistics
        if self.correlation_matrix:
            stats['correlation_stats'] = {
                'total_correlations': len(self.correlation_matrix),
                'top_correlations': sorted(
                    [(assets, count) for assets, count in self.correlation_matrix.items()],
                    key=lambda x: x[1], reverse=True
                )[:10]
            }
        
        return stats
    
    def save_buffer(self, filepath: str):
        """Save replay buffer to disk."""
        
        buffer_data = {
            'config': self.config.__dict__,
            'episodes': self.episodes,
            'metadata': {k: v.__dict__ for k, v in self.metadata.items()},
            'correlation_matrix': dict(self.correlation_matrix),
            'statistics': self.get_statistics(),
            'episode_counter': self.episode_counter
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(buffer_data, f)
        
        logger.info(f"Replay buffer saved to {filepath}")
    
    def load_buffer(self, filepath: str):
        """Load replay buffer from disk."""
        
        try:
            with open(filepath, 'rb') as f:
                buffer_data = pickle.load(f)
            
            # Restore data
            self.episodes = buffer_data['episodes']
            self.correlation_matrix = buffer_data.get('correlation_matrix', {})
            self.episode_counter = buffer_data.get('episode_counter', 0)
            
            # Restore metadata
            self.metadata = {}
            for episode_id, metadata_dict in buffer_data['metadata'].items():
                metadata = EpisodeMetadata(**metadata_dict)
                self.metadata[int(episode_id)] = metadata
            
            # Rebuild priority structures
            self.priority_heap = []
            self.priority_sum = 0.0
            self.max_priority = 1.0
            
            for episode_id, metadata in self.metadata.items():
                self._update_priority(episode_id, metadata.priority_score)
            
            logger.info(f"Replay buffer loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load replay buffer: {e}")
    
    def clear(self):
        """Clear all stored episodes."""
        
        with self.lock:
            self.episodes.clear()
            self.metadata.clear()
            self.priority_heap.clear()
            self.correlation_matrix.clear()
            self.priority_sum = 0.0
            self.max_priority = 1.0
            self.current_memory_mb = 0.0
            self.episode_counter = 0
        
        logger.info("Replay buffer cleared")

# Example usage
if __name__ == "__main__":
    config = ReplayBufferConfig(
        max_episodes=1000,
        default_sampling_strategy=SamplingStrategy.PRIORITY,
        compression_enabled=True
    )
    
    buffer = EpisodicReplayBuffer(config)
    
    # Add some example episodes
    for i in range(100):
        episode_data = {
            'total_reward': np.random.normal(10, 3),
            'steps': np.random.randint(50, 200),
            'success': np.random.random() < 0.6,
            'difficulty': np.random.uniform(0.1, 0.9),
            'nerf_assets_used': set(np.random.choice(['asset1', 'asset2', 'asset3'], 
                                                   size=np.random.randint(1, 4), replace=False)),
            'observations': np.random.randn(100, 10),
            'actions': np.random.randn(100, 5),
            'rewards': np.random.randn(100),
            'performance_metrics': {
                'avg_reward_per_step': np.random.uniform(0.05, 0.15),
                'nerf_reward_ratio': np.random.uniform(0.1, 0.3)
            }
        }
        
        buffer.add_episode(episode_data)
    
    # Sample batches with different strategies
    for strategy in SamplingStrategy:
        try:
            batch = buffer.sample_batch(batch_size=8, strategy=strategy)
            print(f"{strategy.value}: sampled {len(batch['episodes'])} episodes")
        except Exception as e:
            print(f"{strategy.value}: failed - {e}")
    
    # Get statistics
    stats = buffer.get_statistics()
    print(f"Buffer statistics: {stats}")
    
    # Save buffer
    buffer.save_buffer("replay_buffer_v5.pkl")

