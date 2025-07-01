"""
Base Environment Class for RL-LLM Tower Defense Game

This module provides the foundational environment interface for reinforcement learning
training in the procedural tower defense game context.
"""

import gym
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseTowerDefenseEnv(gym.Env, ABC):
    """
    Abstract base class for tower defense environments.
    
    This class defines the common interface and functionality that all
    tower defense environments should implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base environment.
        
        Args:
            config: Configuration dictionary containing environment parameters
        """
        super().__init__()
        self.config = config
        self.episode_step = 0
        self.max_episode_steps = config.get('max_episode_steps', 1000)
        self.reward_scale = config.get('reward_scale', 1.0)
        
        # Game state
        self.game_state = {}
        self.towers = []
        self.enemies = []
        self.resources = config.get('initial_resources', 100)
        self.lives = config.get('initial_lives', 20)
        
        # Metrics tracking
        self.episode_reward = 0.0
        self.episode_metrics = {
            'towers_built': 0,
            'enemies_defeated': 0,
            'resources_spent': 0,
            'damage_dealt': 0,
            'damage_taken': 0
        }
        
        self._setup_spaces()
        logger.info(f"Initialized {self.__class__.__name__} with config: {config}")
    
    @abstractmethod
    def _setup_spaces(self):
        """Setup observation and action spaces. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current observation. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _calculate_reward(self, action: Any, prev_state: Dict) -> float:
        """Calculate reward for the current step. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _is_terminal(self) -> bool:
        """Check if episode should terminate. Must be implemented by subclasses."""
        pass
    
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self._is_terminal():
            logger.warning("Step called on terminated environment")
            return self._get_observation(), 0.0, True, {}
        
        prev_state = self.game_state.copy()
        
        # Execute action
        self._execute_action(action)
        
        # Update game state
        self._update_game_state()
        
        # Calculate reward
        reward = self._calculate_reward(action, prev_state)
        self.episode_reward += reward
        
        # Check termination
        done = self._is_terminal()
        
        # Update step counter
        self.episode_step += 1
        
        # Prepare info dict
        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'lives': self.lives,
            'resources': self.resources,
            'metrics': self.episode_metrics.copy()
        }
        
        if done:
            info['episode_length'] = self.episode_step
            info['final_reward'] = self.episode_reward
            logger.info(f"Episode completed: {info}")
        
        return self._get_observation(), reward * self.reward_scale, done, info
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        self.episode_step = 0
        self.episode_reward = 0.0
        self.towers = []
        self.enemies = []
        self.resources = self.config.get('initial_resources', 100)
        self.lives = self.config.get('initial_lives', 20)
        
        # Reset metrics
        self.episode_metrics = {
            'towers_built': 0,
            'enemies_defeated': 0,
            'resources_spent': 0,
            'damage_dealt': 0,
            'damage_taken': 0
        }
        
        # Reset game state
        self.game_state = self._initialize_game_state()
        
        logger.debug("Environment reset completed")
        return self._get_observation()
    
    def _initialize_game_state(self) -> Dict[str, Any]:
        """Initialize the game state. Can be overridden by subclasses."""
        return {
            'wave_number': 0,
            'time_step': 0,
            'map_state': np.zeros((10, 10)),  # Default 10x10 grid
            'tower_positions': [],
            'enemy_positions': []
        }
    
    def _execute_action(self, action: Any):
        """Execute the given action. Can be overridden by subclasses."""
        # Default implementation - subclasses should override
        pass
    
    def _update_game_state(self):
        """Update the game state after action execution. Can be overridden by subclasses."""
        # Default implementation - subclasses should override
        self.game_state['time_step'] += 1
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
            
        Returns:
            Rendered image if mode is 'rgb_array', None otherwise
        """
        if mode == 'human':
            print(f"Episode Step: {self.episode_step}")
            print(f"Lives: {self.lives}, Resources: {self.resources}")
            print(f"Towers: {len(self.towers)}, Enemies: {len(self.enemies)}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            print("-" * 40)
        elif mode == 'rgb_array':
            # Return a simple visualization as RGB array
            # Subclasses should implement proper rendering
            return np.zeros((64, 64, 3), dtype=np.uint8)
        
        return None
    
    def close(self):
        """Clean up environment resources."""
        logger.info("Environment closed")
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current episode metrics."""
        return {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'lives': self.lives,
            'resources': self.resources,
            **self.episode_metrics
        }