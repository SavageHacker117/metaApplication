"""
Advanced Training Loop with Curriculum Learning

This module implements a sophisticated training loop for RL agents with curriculum learning,
adaptive difficulty adjustment, and comprehensive monitoring capabilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json
import pickle
from collections import deque, defaultdict
import time
import random

logger = logging.getLogger(__name__)


class CurriculumManager:
    """
    Manages curriculum learning progression for RL training.
    
    Features:
    - Automatic difficulty adjustment based on performance
    - Multiple curriculum strategies
    - Performance-based progression criteria
    - Curriculum state persistence
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize curriculum manager.
        
        Args:
            config: Curriculum configuration
        """
        self.config = config
        self.strategy = config.get('strategy', 'performance_based')
        self.difficulty_levels = config.get('difficulty_levels', 5)
        self.current_level = config.get('initial_level', 0)
        
        # Performance tracking
        self.performance_window = config.get('performance_window', 100)
        self.success_threshold = config.get('success_threshold', 0.8)
        self.failure_threshold = config.get('failure_threshold', 0.3)
        
        # Performance history
        self.performance_history = deque(maxlen=self.performance_window)
        self.level_history = []
        
        # Progression criteria
        self.min_episodes_per_level = config.get('min_episodes_per_level', 50)
        self.episodes_at_current_level = 0
        
        logger.info(f"Initialized CurriculumManager with strategy: {self.strategy}")
    
    def get_current_difficulty(self) -> Dict[str, Any]:
        """Get current difficulty parameters."""
        base_config = self.config.get('base_difficulty', {})
        
        if self.strategy == 'linear':
            # Linear progression
            progress = self.current_level / max(1, self.difficulty_levels - 1)
            return self._interpolate_difficulty(base_config, progress)
        
        elif self.strategy == 'exponential':
            # Exponential progression
            progress = (2 ** self.current_level - 1) / (2 ** self.difficulty_levels - 1)
            return self._interpolate_difficulty(base_config, progress)
        
        elif self.strategy == 'performance_based':
            # Performance-based difficulty
            return self._get_performance_based_difficulty()
        
        else:
            return base_config
    
    def _interpolate_difficulty(self, base_config: Dict[str, Any], progress: float) -> Dict[str, Any]:
        """Interpolate difficulty parameters based on progress."""
        difficulty = {}
        max_config = self.config.get('max_difficulty', base_config)
        
        for key, base_value in base_config.items():
            if key in max_config:
                max_value = max_config[key]
                if isinstance(base_value, (int, float)):
                    difficulty[key] = base_value + (max_value - base_value) * progress
                else:
                    difficulty[key] = base_value
            else:
                difficulty[key] = base_value
        
        return difficulty
    
    def _get_performance_based_difficulty(self) -> Dict[str, Any]:
        """Get difficulty based on current performance level."""
        level_configs = self.config.get('level_configs', [])
        
        if level_configs and self.current_level < len(level_configs):
            return level_configs[self.current_level]
        else:
            # Fallback to interpolation
            progress = self.current_level / max(1, self.difficulty_levels - 1)
            return self._interpolate_difficulty(
                self.config.get('base_difficulty', {}), 
                progress
            )
    
    def update_performance(self, episode_result: Dict[str, Any]):
        """
        Update performance tracking and potentially adjust difficulty.
        
        Args:
            episode_result: Dictionary containing episode performance metrics
        """
        # Extract performance metric (success rate, reward, etc.)
        performance = episode_result.get('success', 0)
        if 'normalized_reward' in episode_result:
            performance = episode_result['normalized_reward']
        elif 'total_reward' in episode_result:
            # Normalize reward to [0, 1] range
            reward = episode_result['total_reward']
            max_reward = self.config.get('max_expected_reward', 100)
            performance = max(0, min(1, reward / max_reward))
        
        self.performance_history.append(performance)
        self.episodes_at_current_level += 1
        
        # Check for level progression
        if self.episodes_at_current_level >= self.min_episodes_per_level:
            self._check_progression()
    
    def _check_progression(self):
        """Check if we should progress to next difficulty level."""
        if len(self.performance_history) < self.performance_window // 2:
            return
        
        recent_performance = np.mean(list(self.performance_history)[-50:])
        
        # Progress to next level if performing well
        if (recent_performance >= self.success_threshold and 
            self.current_level < self.difficulty_levels - 1):
            
            self.current_level += 1
            self.episodes_at_current_level = 0
            self.level_history.append({
                'level': self.current_level,
                'timestamp': datetime.now(),
                'performance': recent_performance
            })
            
            logger.info(f"Progressed to difficulty level {self.current_level} "
                       f"(performance: {recent_performance:.3f})")
        
        # Regress to previous level if performing poorly
        elif (recent_performance <= self.failure_threshold and 
              self.current_level > 0):
            
            self.current_level -= 1
            self.episodes_at_current_level = 0
            
            logger.info(f"Regressed to difficulty level {self.current_level} "
                       f"(performance: {recent_performance:.3f})")
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get curriculum progress information."""
        recent_performance = 0.0
        if self.performance_history:
            recent_performance = np.mean(list(self.performance_history)[-20:])
        
        return {
            'current_level': self.current_level,
            'max_level': self.difficulty_levels - 1,
            'episodes_at_level': self.episodes_at_current_level,
            'recent_performance': recent_performance,
            'total_episodes': len(self.performance_history),
            'level_history': self.level_history
        }
    
    def save_state(self, filepath: Path):
        """Save curriculum state."""
        state = {
            'current_level': self.current_level,
            'episodes_at_current_level': self.episodes_at_current_level,
            'performance_history': list(self.performance_history),
            'level_history': self.level_history,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"Saved curriculum state to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load curriculum state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_level = state['current_level']
        self.episodes_at_current_level = state['episodes_at_current_level']
        self.performance_history = deque(state['performance_history'], 
                                       maxlen=self.performance_window)
        self.level_history = state['level_history']
        
        logger.info(f"Loaded curriculum state from {filepath}")


class AdvancedTrainingLoop:
    """
    Advanced training loop with curriculum learning, adaptive scheduling,
    and comprehensive monitoring.
    """
    
    def __init__(self, agent, environment, config: Dict[str, Any]):
        """
        Initialize advanced training loop.
        
        Args:
            agent: RL agent to train
            environment: Training environment
            config: Training configuration
        """
        self.agent = agent
        self.environment = environment
        self.config = config
        
        # Curriculum learning
        if config.get('use_curriculum', False):
            self.curriculum = CurriculumManager(config.get('curriculum', {}))
        else:
            self.curriculum = None
        
        # Training parameters
        self.max_episodes = config.get('max_episodes', 10000)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 1000)
        self.eval_frequency = config.get('eval_frequency', 100)
        self.save_frequency = config.get('save_frequency', 500)
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.evaluation_results = []
        self.training_metrics = defaultdict(list)
        
        # Adaptive learning rate
        self.use_adaptive_lr = config.get('use_adaptive_lr', False)
        self.lr_scheduler = None
        if self.use_adaptive_lr:
            self.lr_scheduler = self._create_lr_scheduler()
        
        # Early stopping
        self.early_stopping = config.get('early_stopping', {})
        self.best_performance = float('-inf')
        self.patience_counter = 0
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized AdvancedTrainingLoop")
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler."""
        if hasattr(self.agent, 'optimizer'):
            scheduler_type = self.config.get('lr_scheduler_type', 'plateau')
            
            if scheduler_type == 'plateau':
                return optim.lr_scheduler.ReduceLROnPlateau(
                    self.agent.optimizer,
                    mode='max',
                    factor=0.5,
                    patience=50,
                    verbose=True
                )
            elif scheduler_type == 'cosine':
                return optim.lr_scheduler.CosineAnnealingLR(
                    self.agent.optimizer,
                    T_max=self.max_episodes
                )
        
        return None
    
    def train(self, callbacks: Optional[List[Callable]] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            callbacks: Optional list of callback functions
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting advanced training loop")
        start_time = time.time()
        
        callbacks = callbacks or []
        
        try:
            for episode in range(self.max_episodes):
                # Update curriculum difficulty
                if self.curriculum:
                    difficulty = self.curriculum.get_current_difficulty()
                    self.environment.update_difficulty(difficulty)
                
                # Run episode
                episode_result = self._run_episode(episode)
                
                # Update curriculum
                if self.curriculum:
                    self.curriculum.update_performance(episode_result)
                
                # Update learning rate
                if self.lr_scheduler:
                    if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step(episode_result['total_reward'])
                    else:
                        self.lr_scheduler.step()
                
                # Track metrics
                self._update_metrics(episode, episode_result)
                
                # Execute callbacks
                for callback in callbacks:
                    callback(episode, episode_result, self)
                
                # Evaluation
                if episode % self.eval_frequency == 0:
                    eval_result = self._evaluate()
                    self.evaluation_results.append(eval_result)
                    
                    # Early stopping check
                    if self._check_early_stopping(eval_result):
                        logger.info(f"Early stopping triggered at episode {episode}")
                        break
                
                # Checkpointing
                if episode % self.save_frequency == 0:
                    self._save_checkpoint(episode)
                
                # Progress logging
                if episode % 100 == 0:
                    self._log_progress(episode)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final checkpoint
            self._save_checkpoint('final')
            
            # Training summary
            training_time = time.time() - start_time
            results = self._generate_training_summary(training_time)
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            return results
    
    def _run_episode(self, episode: int) -> Dict[str, Any]:
        """Run a single training episode."""
        obs = self.environment.reset()
        total_reward = 0.0
        episode_length = 0
        episode_losses = []
        
        for step in range(self.max_steps_per_episode):
            # Agent action
            action = self.agent.act(obs)
            
            # Environment step
            next_obs, reward, done, info = self.environment.step(action)
            
            # Agent learning
            if hasattr(self.agent, 'learn'):
                loss = self.agent.learn(obs, action, reward, next_obs, done)
                if loss is not None:
                    episode_losses.append(loss)
            
            total_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
        
        # Episode result
        result = {
            'episode': episode,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'average_loss': np.mean(episode_losses) if episode_losses else 0.0,
            'success': info.get('success', total_reward > 0),
            'info': info
        }
        
        return result
    
    def _evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate agent performance."""
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        
        # Temporarily disable learning
        if hasattr(self.agent, 'training'):
            was_training = self.agent.training
            self.agent.eval()
        
        for _ in range(num_episodes):
            obs = self.environment.reset()
            total_reward = 0.0
            episode_length = 0
            
            for step in range(self.max_steps_per_episode):
                action = self.agent.act(obs, deterministic=True)
                obs, reward, done, info = self.environment.step(action)
                
                total_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(episode_length)
            eval_successes.append(info.get('success', total_reward > 0))
        
        # Restore training mode
        if hasattr(self.agent, 'training'):
            if was_training:
                self.agent.train()
        
        result = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean(eval_successes),
            'num_episodes': num_episodes
        }
        
        return result
    
    def _update_metrics(self, episode: int, episode_result: Dict[str, Any]):
        """Update training metrics."""
        self.episode_rewards.append(episode_result['total_reward'])
        self.episode_lengths.append(episode_result['episode_length'])
        
        # Store additional metrics
        for key, value in episode_result.items():
            if isinstance(value, (int, float)):
                self.training_metrics[key].append(value)
    
    def _check_early_stopping(self, eval_result: Dict[str, Any]) -> bool:
        """Check early stopping criteria."""
        if not self.early_stopping:
            return False
        
        current_performance = eval_result['mean_reward']
        patience = self.early_stopping.get('patience', 100)
        min_delta = self.early_stopping.get('min_delta', 0.01)
        
        if current_performance > self.best_performance + min_delta:
            self.best_performance = current_performance
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        return self.patience_counter >= patience
    
    def _save_checkpoint(self, episode):
        """Save training checkpoint."""
        checkpoint = {
            'episode': episode,
            'agent_state': self.agent.state_dict() if hasattr(self.agent, 'state_dict') else None,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'evaluation_results': self.evaluation_results,
            'training_metrics': dict(self.training_metrics),
            'best_performance': self.best_performance,
            'config': self.config
        }
        
        if self.curriculum:
            checkpoint['curriculum_state'] = {
                'current_level': self.curriculum.current_level,
                'episodes_at_level': self.curriculum.episodes_at_current_level,
                'performance_history': list(self.curriculum.performance_history)
            }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_{episode}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _log_progress(self, episode: int):
        """Log training progress."""
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        mean_reward = np.mean(recent_rewards)
        
        log_msg = f"Episode {episode}: Mean Reward (last 100): {mean_reward:.2f}"
        
        if self.curriculum:
            progress = self.curriculum.get_progress_info()
            log_msg += f", Curriculum Level: {progress['current_level']}"
        
        if self.lr_scheduler and hasattr(self.agent, 'optimizer'):
            current_lr = self.agent.optimizer.param_groups[0]['lr']
            log_msg += f", LR: {current_lr:.6f}"
        
        logger.info(log_msg)
    
    def _generate_training_summary(self, training_time: float) -> Dict[str, Any]:
        """Generate comprehensive training summary."""
        summary = {
            'total_episodes': len(self.episode_rewards),
            'training_time': training_time,
            'final_performance': {
                'mean_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                'best_reward': max(self.episode_rewards) if self.episode_rewards else 0,
                'final_reward': self.episode_rewards[-1] if self.episode_rewards else 0
            },
            'evaluation_results': self.evaluation_results,
            'training_metrics': dict(self.training_metrics)
        }
        
        if self.curriculum:
            summary['curriculum_progress'] = self.curriculum.get_progress_info()
        
        return summary


def create_training_callbacks(config: Dict[str, Any]) -> List[Callable]:
    """Create training callbacks based on configuration."""
    callbacks = []
    
    # Logging callback
    if config.get('enable_logging_callback', True):
        def logging_callback(episode, episode_result, trainer):
            if episode % config.get('log_frequency', 50) == 0:
                logger.info(f"Episode {episode}: Reward={episode_result['total_reward']:.2f}, "
                           f"Length={episode_result['episode_length']}")
        callbacks.append(logging_callback)
    
    # Visualization callback
    if config.get('enable_viz_callback', False):
        def viz_callback(episode, episode_result, trainer):
            if episode % config.get('viz_frequency', 100) == 0:
                # Placeholder for visualization updates
                pass
        callbacks.append(viz_callback)
    
    return callbacks

