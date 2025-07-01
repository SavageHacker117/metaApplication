"""
Training Loop Infrastructure for RL Training

This module provides the core training loop infrastructure for reinforcement
learning agents in the tower defense game development environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Callable, Tuple
import time
import logging
from collections import deque, defaultdict
import json
import pickle
from pathlib import Path

from ..environment.base_environment import TowerDefenseRLEnvironment
from ..rewards.reward_system import CompositeRewardSystem
from ..actions.action_space import ActionSpace, ActionInterpreter
from ..observations.observation_space import CompositeObservationExtractor, ObservationNormalizer
from ...utils.logging import setup_logger
from ...utils.math import calculate_moving_average


class TrainingConfig:
    """Configuration for training loop."""
    
    def __init__(self, **kwargs):
        # Training parameters
        self.max_episodes = kwargs.get('max_episodes', 10000)
        self.max_steps_per_episode = kwargs.get('max_steps_per_episode', 1000)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 0.005)
        
        # Exploration parameters
        self.epsilon_start = kwargs.get('epsilon_start', 1.0)
        self.epsilon_end = kwargs.get('epsilon_end', 0.01)
        self.epsilon_decay = kwargs.get('epsilon_decay', 0.995)
        
        # Memory parameters
        self.memory_size = kwargs.get('memory_size', 100000)
        self.min_memory_size = kwargs.get('min_memory_size', 1000)
        
        # Training frequency
        self.train_frequency = kwargs.get('train_frequency', 4)
        self.target_update_frequency = kwargs.get('target_update_frequency', 100)
        
        # Evaluation parameters
        self.eval_frequency = kwargs.get('eval_frequency', 100)
        self.eval_episodes = kwargs.get('eval_episodes', 10)
        
        # Saving parameters
        self.save_frequency = kwargs.get('save_frequency', 1000)
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        
        # Logging parameters
        self.log_frequency = kwargs.get('log_frequency', 10)
        self.tensorboard_log = kwargs.get('tensorboard_log', True)
        
        # Multi-objective training
        self.use_multi_objective = kwargs.get('use_multi_objective', True)
        self.reward_weights = kwargs.get('reward_weights', {
            'gameplay': 0.4,
            'visual_quality': 0.2,
            'code_quality': 0.2,
            'performance': 0.2
        })


class ExperienceBuffer:
    """Experience replay buffer for storing and sampling experiences."""
    
    def __init__(self, capacity: int, observation_space, action_space):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.observation_space = observation_space
        self.action_space = action_space
        
    def push(self, state: Dict[str, Any], action: Dict[str, Any], 
             reward: float, next_state: Dict[str, Any], done: bool):
        """Add experience to buffer."""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, Any]:
        """Sample batch of experiences."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        # Organize batch data
        states = [exp['state'] for exp in batch]
        actions = [exp['action'] for exp in batch]
        rewards = np.array([exp['reward'] for exp in batch], dtype=np.float32)
        next_states = [exp['next_state'] for exp in batch]
        dones = np.array([exp['done'] for exp in batch], dtype=bool)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """Prioritized experience replay buffer."""
    
    def __init__(self, capacity: int, observation_space, action_space, alpha: float = 0.6):
        super().__init__(capacity, observation_space, action_space)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state: Dict[str, Any], action: Dict[str, Any], 
             reward: float, next_state: Dict[str, Any], done: bool):
        """Add experience with maximum priority."""
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Dict[str, Any]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, 
                                 replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        batch = [self.buffer[i] for i in indices]
        
        # Organize batch data
        states = [exp['state'] for exp in batch]
        actions = [exp['action'] for exp in batch]
        rewards = np.array([exp['reward'] for exp in batch], dtype=np.float32)
        next_states = [exp['next_state'] for exp in batch]
        dones = np.array([exp['done'] for exp in batch], dtype=bool)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'indices': indices,
            'weights': weights
        }
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)


class TrainingMetrics:
    """Tracks and manages training metrics."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        self.q_values = []
        self.exploration_rates = []
        self.evaluation_scores = []
        
        # Multi-objective metrics
        self.reward_components = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
        # Training statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.training_time = 0.0
        
    def update_episode_metrics(self, episode_reward: float, episode_length: int, 
                             reward_breakdown: Dict[str, float]):
        """Update metrics after episode completion."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.total_episodes += 1
        self.total_steps += episode_length
        
        # Update reward component tracking
        for component, value in reward_breakdown.items():
            self.reward_components[component].append(value)
    
    def update_training_metrics(self, loss: float, q_value: float, epsilon: float):
        """Update training-specific metrics."""
        self.loss_history.append(loss)
        self.q_values.append(q_value)
        self.exploration_rates.append(epsilon)
    
    def update_evaluation_metrics(self, eval_score: float):
        """Update evaluation metrics."""
        self.evaluation_scores.append(eval_score)
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """Get recent performance statistics."""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        
        return {
            'mean_reward': np.mean(recent_rewards),
            'std_reward': np.std(recent_rewards),
            'mean_length': np.mean(recent_lengths),
            'success_rate': len([r for r in recent_rewards if r > 0]) / len(recent_rewards),
            'episodes_completed': len(recent_rewards)
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'training_time': self.training_time,
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'best_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'recent_performance': self.get_recent_performance(),
            'reward_components': {k: np.mean(v) for k, v in self.reward_components.items()},
            'convergence_indicator': self._calculate_convergence_indicator()
        }
    
    def _calculate_convergence_indicator(self) -> float:
        """Calculate convergence indicator based on recent performance stability."""
        if len(self.episode_rewards) < 100:
            return 0.0
        
        recent_rewards = self.episode_rewards[-100:]
        first_half = recent_rewards[:50]
        second_half = recent_rewards[50:]
        
        # Calculate stability as inverse of relative difference
        mean_first = np.mean(first_half)
        mean_second = np.mean(second_half)
        
        if mean_first == 0:
            return 0.0
        
        relative_change = abs(mean_second - mean_first) / abs(mean_first)
        stability = 1.0 / (1.0 + relative_change)
        
        return stability


class TrainingLoop:
    """
    Main training loop for RL agents.
    """
    
    def __init__(self, config: TrainingConfig, environment: TowerDefenseRLEnvironment,
                 agent, reward_system: CompositeRewardSystem):
        self.config = config
        self.env = environment
        self.agent = agent
        self.reward_system = reward_system
        
        # Initialize components
        self.action_interpreter = ActionInterpreter({})
        self.observation_extractor = CompositeObservationExtractor(environment.config)
        self.observation_normalizer = ObservationNormalizer(environment.config)
        
        # Initialize experience buffer
        if hasattr(config, 'use_prioritized_replay') and config.use_prioritized_replay:
            self.experience_buffer = PrioritizedExperienceBuffer(
                config.memory_size, 
                environment.observation_space,
                environment.action_space
            )
        else:
            self.experience_buffer = ExperienceBuffer(
                config.memory_size,
                environment.observation_space,
                environment.action_space
            )
        
        # Initialize metrics tracking
        self.metrics = TrainingMetrics()
        self.logger = setup_logger(__name__)
        
        # Training state
        self.current_episode = 0
        self.current_step = 0
        self.epsilon = config.epsilon_start
        self.best_eval_score = float('-inf')
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Training summary and final metrics
        """
        self.logger.info("Starting training loop")
        start_time = time.time()
        
        try:
            for episode in range(self.config.max_episodes):
                self.current_episode = episode
                
                # Run episode
                episode_metrics = self._run_episode()
                
                # Update metrics
                self.metrics.update_episode_metrics(
                    episode_metrics['total_reward'],
                    episode_metrics['episode_length'],
                    episode_metrics['reward_breakdown']
                )
                
                # Train agent
                if len(self.experience_buffer) >= self.config.min_memory_size:
                    if episode % self.config.train_frequency == 0:
                        training_metrics = self._train_agent()
                        if training_metrics:
                            self.metrics.update_training_metrics(
                                training_metrics['loss'],
                                training_metrics['q_value'],
                                self.epsilon
                            )
                
                # Update exploration rate
                self.epsilon = max(
                    self.config.epsilon_end,
                    self.epsilon * self.config.epsilon_decay
                )
                
                # Evaluation
                if episode % self.config.eval_frequency == 0:
                    eval_score = self._evaluate_agent()
                    self.metrics.update_evaluation_metrics(eval_score)
                    
                    # Save best model
                    if eval_score > self.best_eval_score:
                        self.best_eval_score = eval_score
                        self._save_checkpoint('best_model')
                
                # Periodic saving
                if episode % self.config.save_frequency == 0:
                    self._save_checkpoint(f'episode_{episode}')
                
                # Logging
                if episode % self.config.log_frequency == 0:
                    self._log_progress(episode, episode_metrics)
                
                # Early stopping check
                if self._should_stop_early():
                    self.logger.info(f"Early stopping at episode {episode}")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Final cleanup and summary
            self.metrics.training_time = time.time() - start_time
            self._save_checkpoint('final_model')
            
            training_summary = self.metrics.get_training_summary()
            self.logger.info(f"Training completed. Summary: {training_summary}")
            
            return training_summary
    
    def _run_episode(self) -> Dict[str, Any]:
        """Run a single training episode."""
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        reward_breakdown = defaultdict(float)
        
        for step in range(self.config.max_steps_per_episode):
            # Extract and normalize observations
            observations = self.observation_extractor.extract_all(state)
            normalized_obs = self.observation_normalizer.normalize(observations)
            
            # Select action
            action = self.agent.select_action(normalized_obs, self.epsilon)
            
            # Interpret action
            interpreted_action = self.action_interpreter.interpret_action(action, state)
            
            # Execute action in environment
            next_state, env_reward, done, info = self.env.step(interpreted_action)
            
            # Calculate comprehensive reward
            reward_components = self.reward_system.calculate_total_reward(
                state, interpreted_action, info
            )
            total_reward = reward_components['total']
            
            # Store experience
            next_observations = self.observation_extractor.extract_all(next_state)
            next_normalized_obs = self.observation_normalizer.normalize(next_observations)
            
            self.experience_buffer.push(
                normalized_obs, action, total_reward, next_normalized_obs, done
            )
            
            # Update episode metrics
            episode_reward += total_reward
            episode_length += 1
            
            for component, value in reward_components.items():
                reward_breakdown[component] += value
            
            # Update state
            state = next_state
            self.current_step += 1
            
            if done:
                break
        
        return {
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'reward_breakdown': dict(reward_breakdown)
        }
    
    def _train_agent(self) -> Optional[Dict[str, float]]:
        """Train the agent using experience replay."""
        if len(self.experience_buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        batch = self.experience_buffer.sample(self.config.batch_size)
        
        # Train agent
        training_metrics = self.agent.train_step(batch)
        
        # Update priorities if using prioritized replay
        if isinstance(self.experience_buffer, PrioritizedExperienceBuffer):
            if 'td_errors' in training_metrics:
                self.experience_buffer.update_priorities(
                    batch['indices'], 
                    np.abs(training_metrics['td_errors']) + 1e-6
                )
        
        return training_metrics
    
    def _evaluate_agent(self) -> float:
        """Evaluate agent performance."""
        total_reward = 0.0
        
        for _ in range(self.config.eval_episodes):
            state = self.env.reset()
            episode_reward = 0.0
            
            for _ in range(self.config.max_steps_per_episode):
                observations = self.observation_extractor.extract_all(state)
                normalized_obs = self.observation_normalizer.normalize(observations)
                
                # Use greedy action selection for evaluation
                action = self.agent.select_action(normalized_obs, epsilon=0.0)
                interpreted_action = self.action_interpreter.interpret_action(action, state)
                
                next_state, reward, done, info = self.env.step(interpreted_action)
                
                reward_components = self.reward_system.calculate_total_reward(
                    state, interpreted_action, info
                )
                episode_reward += reward_components['total']
                
                state = next_state
                
                if done:
                    break
            
            total_reward += episode_reward
        
        return total_reward / self.config.eval_episodes
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early."""
        # Check convergence
        if len(self.metrics.episode_rewards) >= 1000:
            convergence = self.metrics._calculate_convergence_indicator()
            if convergence > 0.95:  # High stability
                return True
        
        # Check if performance has plateaued
        if len(self.metrics.evaluation_scores) >= 10:
            recent_scores = self.metrics.evaluation_scores[-10:]
            if max(recent_scores) - min(recent_scores) < 0.01:  # Very small improvement
                return True
        
        return False
    
    def _save_checkpoint(self, name: str):
        """Save training checkpoint."""
        checkpoint = {
            'episode': self.current_episode,
            'step': self.current_step,
            'epsilon': self.epsilon,
            'best_eval_score': self.best_eval_score,
            'agent_state': self.agent.get_state(),
            'metrics': self.metrics,
            'config': self.config.__dict__
        }
        
        checkpoint_path = Path(self.config.checkpoint_dir) / f'{name}.pkl'
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        self.current_episode = checkpoint['episode']
        self.current_step = checkpoint['step']
        self.epsilon = checkpoint['epsilon']
        self.best_eval_score = checkpoint['best_eval_score']
        self.agent.load_state(checkpoint['agent_state'])
        self.metrics = checkpoint['metrics']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def _log_progress(self, episode: int, episode_metrics: Dict[str, Any]):
        """Log training progress."""
        recent_perf = self.metrics.get_recent_performance()
        
        self.logger.info(
            f"Episode {episode}: "
            f"Reward={episode_metrics['total_reward']:.2f}, "
            f"Length={episode_metrics['episode_length']}, "
            f"Epsilon={self.epsilon:.3f}, "
            f"Recent Mean={recent_perf.get('mean_reward', 0):.2f}, "
            f"Success Rate={recent_perf.get('success_rate', 0):.2f}"
        )
        
        # Log reward breakdown
        reward_breakdown = episode_metrics['reward_breakdown']
        breakdown_str = ", ".join([f"{k}={v:.2f}" for k, v in reward_breakdown.items()])
        self.logger.debug(f"Reward breakdown: {breakdown_str}")


class MultiObjectiveTrainingLoop(TrainingLoop):
    """
    Training loop specialized for multi-objective optimization.
    """
    
    def __init__(self, config: TrainingConfig, environment: TowerDefenseRLEnvironment,
                 agents: Dict[str, Any], reward_system: CompositeRewardSystem):
        # Initialize with primary agent
        super().__init__(config, environment, agents['primary'], reward_system)
        self.agents = agents
        self.objective_weights = config.reward_weights
        
    def _train_agent(self) -> Optional[Dict[str, float]]:
        """Train multiple agents for different objectives."""
        if len(self.experience_buffer) < self.config.batch_size:
            return None
        
        batch = self.experience_buffer.sample(self.config.batch_size)
        combined_metrics = {}
        
        # Train each agent with different reward emphasis
        for agent_name, agent in self.agents.items():
            if agent_name == 'primary':
                continue
            
            # Modify rewards based on objective
            modified_batch = self._modify_batch_for_objective(batch, agent_name)
            agent_metrics = agent.train_step(modified_batch)
            
            combined_metrics[agent_name] = agent_metrics
        
        # Train primary agent with combined experience
        primary_metrics = self.agents['primary'].train_step(batch)
        combined_metrics['primary'] = primary_metrics
        
        return combined_metrics
    
    def _modify_batch_for_objective(self, batch: Dict[str, Any], 
                                  objective: str) -> Dict[str, Any]:
        """Modify batch rewards to emphasize specific objective."""
        modified_batch = batch.copy()
        
        # This would require storing detailed reward breakdowns in experience buffer
        # For now, return original batch
        return modified_batch

