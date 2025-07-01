"""
Two-State World Environment for RL Testing

This module implements a simple two-state environment for testing
RL algorithms and debugging training pipelines.
"""

import gym
import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TwoStateWorld(gym.Env):
    """
    Simple two-state environment for RL testing.
    
    States:
    - State 0: Starting state
    - State 1: Terminal state (goal)
    
    Actions:
    - Action 0: Stay in current state
    - Action 1: Move to next state (if possible)
    
    Rewards:
    - +1 for reaching terminal state
    - -0.1 for each step (encourages efficiency)
    - 0 for invalid actions
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the two-state world."""
        super().__init__()
        
        if config is None:
            config = {}
        
        self.config = config
        self.max_episode_steps = config.get('max_episode_steps', 100)
        self.step_penalty = config.get('step_penalty', -0.1)
        self.goal_reward = config.get('goal_reward', 1.0)
        self.invalid_action_penalty = config.get('invalid_action_penalty', -0.01)
        
        # Environment setup
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        
        # State variables
        self.current_state = 0
        self.episode_step = 0
        self.episode_reward = 0.0
        
        logger.info("Initialized TwoStateWorld environment")
    
    def reset(self) -> int:
        """Reset environment to initial state."""
        self.current_state = 0
        self.episode_step = 0
        self.episode_reward = 0.0
        
        logger.debug("Environment reset to initial state")
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0 or 1)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if not self.action_space.contains(action):
            logger.warning(f"Invalid action: {action}")
            return self.current_state, self.invalid_action_penalty, False, {'invalid_action': True}
        
        reward = 0.0
        done = False
        info = {'episode_step': self.episode_step}
        
        if action == 0:  # Stay in current state
            reward = self.step_penalty
            info['action_type'] = 'stay'
        
        elif action == 1:  # Try to move to next state
            if self.current_state == 0:
                # Move from state 0 to state 1 (goal)
                self.current_state = 1
                reward = self.goal_reward
                done = True
                info['action_type'] = 'move_to_goal'
                info['reached_goal'] = True
            else:
                # Already in terminal state, invalid action
                reward = self.invalid_action_penalty
                info['action_type'] = 'invalid_move'
                info['invalid_action'] = True
        
        # Update episode tracking
        self.episode_step += 1
        self.episode_reward += reward
        
        # Check for episode timeout
        if self.episode_step >= self.max_episode_steps:
            done = True
            info['timeout'] = True
        
        info.update({
            'episode_reward': self.episode_reward,
            'current_state': self.current_state,
            'episode_length': self.episode_step
        })
        
        if done:
            logger.debug(f"Episode completed: {info}")
        
        return self.current_state, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render the environment state."""
        state_repr = f"State: {self.current_state}, Step: {self.episode_step}, Reward: {self.episode_reward:.2f}"
        
        if mode == 'human':
            print(state_repr)
            if self.current_state == 0:
                print("Current: START")
            else:
                print("Current: GOAL")
            print("-" * 30)
        
        return state_repr
    
    def close(self):
        """Clean up environment resources."""
        logger.info("TwoStateWorld environment closed")


class MultiStateBandits(gym.Env):
    """
    Multi-armed bandit variant with multiple states.
    
    Each state has different reward distributions for actions.
    Useful for testing exploration vs exploitation strategies.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize multi-state bandits."""
        super().__init__()
        
        if config is None:
            config = {}
        
        self.config = config
        self.num_states = config.get('num_states', 3)
        self.num_actions = config.get('num_actions', 4)
        self.max_episode_steps = config.get('max_episode_steps', 50)
        self.state_transition_prob = config.get('state_transition_prob', 0.1)
        
        # Environment setup
        self.observation_space = gym.spaces.Discrete(self.num_states)
        self.action_space = gym.spaces.Discrete(self.num_actions)
        
        # Create reward distributions for each state-action pair
        np.random.seed(config.get('seed', 42))
        self.reward_means = np.random.uniform(-1, 1, (self.num_states, self.num_actions))
        self.reward_stds = np.random.uniform(0.1, 0.5, (self.num_states, self.num_actions))
        
        # State variables
        self.current_state = 0
        self.episode_step = 0
        self.episode_reward = 0.0
        
        logger.info(f"Initialized MultiStateBandits with {self.num_states} states and {self.num_actions} actions")
    
    def reset(self) -> int:
        """Reset environment to random initial state."""
        self.current_state = np.random.randint(0, self.num_states)
        self.episode_step = 0
        self.episode_reward = 0.0
        
        return self.current_state
    
    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """Execute one step in the bandit environment."""
        if not self.action_space.contains(action):
            return self.current_state, -1.0, False, {'invalid_action': True}
        
        # Get reward from current state-action pair
        mean_reward = self.reward_means[self.current_state, action]
        std_reward = self.reward_stds[self.current_state, action]
        reward = np.random.normal(mean_reward, std_reward)
        
        # Possibly transition to new state
        if np.random.random() < self.state_transition_prob:
            self.current_state = np.random.randint(0, self.num_states)
        
        # Update episode tracking
        self.episode_step += 1
        self.episode_reward += reward
        
        # Check for episode termination
        done = self.episode_step >= self.max_episode_steps
        
        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'state_changed': False  # Could track this if needed
        }
        
        return self.current_state, reward, done, info
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """Render the bandit state."""
        state_repr = f"State: {self.current_state}, Step: {self.episode_step}, Total Reward: {self.episode_reward:.2f}"
        
        if mode == 'human':
            print(state_repr)
            print(f"Action rewards (mean±std): {self.reward_means[self.current_state]:.2f}±{self.reward_stds[self.current_state]:.2f}")
            print("-" * 40)
        
        return state_repr
    
    def get_optimal_action(self, state: Optional[int] = None) -> int:
        """Get the optimal action for a given state (for analysis)."""
        if state is None:
            state = self.current_state
        return np.argmax(self.reward_means[state])
    
    def get_state_info(self) -> Dict[str, Any]:
        """Get information about all states (for analysis)."""
        return {
            'reward_means': self.reward_means.tolist(),
            'reward_stds': self.reward_stds.tolist(),
            'optimal_actions': [self.get_optimal_action(s) for s in range(self.num_states)]
        }


def create_simple_env(env_type: str = 'two_state', config: Optional[Dict[str, Any]] = None) -> gym.Env:
    """
    Factory function to create simple test environments.
    
    Args:
        env_type: Type of environment ('two_state' or 'bandits')
        config: Environment configuration
        
    Returns:
        Environment instance
    """
    if env_type == 'two_state':
        return TwoStateWorld(config)
    elif env_type == 'bandits':
        return MultiStateBandits(config)
    else:
        raise ValueError(f"Unknown environment type: {env_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test TwoStateWorld
    env = TwoStateWorld()
    obs = env.reset()
    
    print("Testing TwoStateWorld:")
    for step in range(5):
        action = np.random.randint(0, 2)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    
    print("\nTesting MultiStateBandits:")
    # Test MultiStateBandits
    bandit_env = MultiStateBandits({'num_states': 2, 'num_actions': 3})
    obs = bandit_env.reset()
    
    for step in range(5):
        action = np.random.randint(0, 3)
        obs, reward, done, info = bandit_env.step(action)
        bandit_env.render()
        if done:
            break