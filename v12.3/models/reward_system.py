"""
Reward System for RL Training

This module provides various reward calculation mechanisms for different
aspects of tower defense game development and optimization.
"""

import numpy as np
from typing import Dict, List, Any, Callable
from abc import ABC, abstractmethod
import math

from ..utils.math import normalize_value, calculate_weighted_average


class BaseRewardCalculator(ABC):
    """Abstract base class for reward calculators."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weight = config.get('weight', 1.0)
        self.enabled = config.get('enabled', True)
    
    @abstractmethod
    def calculate(self, state: Dict[str, Any], action: Dict[str, Any], 
                 result: Dict[str, Any]) -> float:
        """Calculate reward for given state, action, and result."""
        pass
    
    def normalize_reward(self, reward: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
        """Normalize reward to specified range."""
        return normalize_value(reward, min_val, max_val)


class GameplayRewardCalculator(BaseRewardCalculator):
    """Calculates rewards based on gameplay performance."""
    
    def calculate(self, state: Dict[str, Any], action: Dict[str, Any], 
                 result: Dict[str, Any]) -> float:
        """Calculate gameplay-based reward."""
        if not self.enabled:
            return 0.0
        
        reward = 0.0
        game_stats = state.get('game_stats', {})
        
        # Reward for successful actions
        if result.get('success', False):
            reward += 10.0
        else:
            reward -= 5.0
        
        # Reward for enemy elimination
        enemies_killed = game_stats.get('enemies_killed_this_step', 0)
        reward += enemies_killed * 20.0
        
        # Penalty for losing lives
        lives_lost = game_stats.get('lives_lost_this_step', 0)
        reward -= lives_lost * 100.0
        
        # Reward for wave completion
        if game_stats.get('wave_completed_this_step', False):
            reward += 200.0
        
        # Efficiency bonus
        efficiency = game_stats.get('efficiency', 0.0)
        reward += efficiency * 10.0
        
        # Resource management reward
        gold_ratio = state.get('player_state', {}).get('gold', 0) / max(state.get('starting_gold', 100), 1)
        if gold_ratio > 0.5:  # Maintaining good resource levels
            reward += 5.0
        elif gold_ratio < 0.1:  # Running low on resources
            reward -= 10.0
        
        return self.normalize_reward(reward * self.weight, -200.0, 300.0)


class VisualQualityRewardCalculator(BaseRewardCalculator):
    """Calculates rewards based on visual quality metrics."""
    
    def calculate(self, state: Dict[str, Any], action: Dict[str, Any], 
                 result: Dict[str, Any]) -> float:
        """Calculate visual quality-based reward."""
        if not self.enabled:
            return 0.0
        
        reward = 0.0
        visual_metrics = result.get('visual_metrics', {})
        
        # LPIPS similarity score (lower is better)
        lpips_score = visual_metrics.get('lpips_score', 1.0)
        reward += (1.0 - lpips_score) * 100.0
        
        # Frame rate consistency
        fps = visual_metrics.get('fps', 60.0)
        if fps >= 60.0:
            reward += 20.0
        elif fps >= 30.0:
            reward += 10.0
        else:
            reward -= 20.0
        
        # Rendering quality score
        render_quality = visual_metrics.get('render_quality', 0.5)
        reward += render_quality * 50.0
        
        # Particle effect quality
        particle_quality = visual_metrics.get('particle_quality', 0.5)
        reward += particle_quality * 30.0
        
        # Visual coherence score
        coherence = visual_metrics.get('visual_coherence', 0.5)
        reward += coherence * 40.0
        
        return self.normalize_reward(reward * self.weight, -50.0, 250.0)


class CodeQualityRewardCalculator(BaseRewardCalculator):
    """Calculates rewards based on code quality metrics."""
    
    def calculate(self, state: Dict[str, Any], action: Dict[str, Any], 
                 result: Dict[str, Any]) -> float:
        """Calculate code quality-based reward."""
        if not self.enabled:
            return 0.0
        
        reward = 0.0
        code_metrics = result.get('code_metrics', {})
        
        # Syntax correctness
        if code_metrics.get('syntax_valid', True):
            reward += 50.0
        else:
            reward -= 100.0
        
        # Code complexity (lower is better)
        complexity = code_metrics.get('cyclomatic_complexity', 10)
        if complexity <= 5:
            reward += 20.0
        elif complexity <= 10:
            reward += 10.0
        else:
            reward -= complexity * 2.0
        
        # Test coverage
        coverage = code_metrics.get('test_coverage', 0.0)
        reward += coverage * 30.0
        
        # Code style compliance
        style_score = code_metrics.get('style_score', 0.5)
        reward += style_score * 20.0
        
        # Performance metrics
        execution_time = code_metrics.get('execution_time', 1.0)
        if execution_time < 0.1:
            reward += 15.0
        elif execution_time < 0.5:
            reward += 5.0
        else:
            reward -= execution_time * 10.0
        
        # Memory usage efficiency
        memory_usage = code_metrics.get('memory_usage', 100.0)  # MB
        if memory_usage < 50.0:
            reward += 10.0
        elif memory_usage < 100.0:
            reward += 5.0
        else:
            reward -= memory_usage * 0.1
        
        return self.normalize_reward(reward * self.weight, -200.0, 150.0)


class PerformanceRewardCalculator(BaseRewardCalculator):
    """Calculates rewards based on system performance metrics."""
    
    def calculate(self, state: Dict[str, Any], action: Dict[str, Any], 
                 result: Dict[str, Any]) -> float:
        """Calculate performance-based reward."""
        if not self.enabled:
            return 0.0
        
        reward = 0.0
        perf_metrics = result.get('performance_metrics', {})
        
        # CPU usage efficiency
        cpu_usage = perf_metrics.get('cpu_usage', 50.0)  # Percentage
        if cpu_usage < 30.0:
            reward += 20.0
        elif cpu_usage < 60.0:
            reward += 10.0
        else:
            reward -= cpu_usage * 0.5
        
        # GPU usage efficiency
        gpu_usage = perf_metrics.get('gpu_usage', 50.0)  # Percentage
        if 50.0 <= gpu_usage <= 80.0:  # Optimal range
            reward += 15.0
        elif gpu_usage < 20.0:  # Underutilized
            reward -= 10.0
        elif gpu_usage > 90.0:  # Overloaded
            reward -= 20.0
        
        # Memory efficiency
        memory_efficiency = perf_metrics.get('memory_efficiency', 0.5)
        reward += memory_efficiency * 25.0
        
        # Network latency (for multiplayer features)
        latency = perf_metrics.get('network_latency', 100.0)  # ms
        if latency < 50.0:
            reward += 15.0
        elif latency < 100.0:
            reward += 5.0
        else:
            reward -= latency * 0.1
        
        # Throughput (operations per second)
        throughput = perf_metrics.get('throughput', 100.0)
        reward += min(throughput / 10.0, 50.0)  # Cap at 50 points
        
        return self.normalize_reward(reward * self.weight, -100.0, 150.0)


class LearningProgressRewardCalculator(BaseRewardCalculator):
    """Calculates rewards based on learning progress and improvement."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.baseline_metrics = {}
        self.improvement_history = []
    
    def calculate(self, state: Dict[str, Any], action: Dict[str, Any], 
                 result: Dict[str, Any]) -> float:
        """Calculate learning progress-based reward."""
        if not self.enabled:
            return 0.0
        
        reward = 0.0
        current_metrics = result.get('learning_metrics', {})
        
        # Calculate improvement over baseline
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric_name]
                improvement = (current_value - baseline_value) / max(abs(baseline_value), 1e-6)
                reward += improvement * 20.0
            else:
                # First measurement becomes baseline
                self.baseline_metrics[metric_name] = current_value
        
        # Reward for consistent improvement
        if len(self.improvement_history) >= 5:
            recent_improvements = self.improvement_history[-5:]
            if all(imp > 0 for imp in recent_improvements):
                reward += 30.0  # Consistency bonus
        
        # Update improvement history
        total_improvement = sum(current_metrics.values()) - sum(self.baseline_metrics.values())
        self.improvement_history.append(total_improvement)
        
        # Keep history manageable
        if len(self.improvement_history) > 100:
            self.improvement_history = self.improvement_history[-50:]
        
        # Exploration bonus
        exploration_score = result.get('exploration_score', 0.0)
        reward += exploration_score * 10.0
        
        return self.normalize_reward(reward * self.weight, -50.0, 100.0)


class AdaptiveRewardCalculator(BaseRewardCalculator):
    """Adaptive reward calculator that adjusts based on training progress."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.adaptation_rate = config.get('adaptation_rate', 0.01)
        self.performance_history = []
        self.current_difficulty = 1.0
    
    def calculate(self, state: Dict[str, Any], action: Dict[str, Any], 
                 result: Dict[str, Any]) -> float:
        """Calculate adaptive reward based on current performance."""
        if not self.enabled:
            return 0.0
        
        base_reward = result.get('base_reward', 0.0)
        
        # Adjust reward based on current difficulty
        adjusted_reward = base_reward * self.current_difficulty
        
        # Track performance
        performance = result.get('performance_score', 0.0)
        self.performance_history.append(performance)
        
        # Adapt difficulty based on recent performance
        if len(self.performance_history) >= 10:
            recent_performance = np.mean(self.performance_history[-10:])
            
            if recent_performance > 0.8:  # Too easy
                self.current_difficulty *= (1.0 + self.adaptation_rate)
            elif recent_performance < 0.3:  # Too hard
                self.current_difficulty *= (1.0 - self.adaptation_rate)
            
            # Keep difficulty in reasonable bounds
            self.current_difficulty = np.clip(self.current_difficulty, 0.1, 3.0)
        
        # Keep history manageable
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]
        
        return self.normalize_reward(adjusted_reward * self.weight, -100.0, 100.0)


class CompositeRewardSystem:
    """
    Composite reward system that combines multiple reward calculators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.calculators = {}
        self.global_weight = config.get('global_weight', 1.0)
        
        # Initialize reward calculators
        self._initialize_calculators()
    
    def _initialize_calculators(self):
        """Initialize all reward calculators."""
        calculator_configs = self.config.get('calculators', {})
        
        # Gameplay rewards
        if 'gameplay' in calculator_configs:
            self.calculators['gameplay'] = GameplayRewardCalculator(
                calculator_configs['gameplay']
            )
        
        # Visual quality rewards
        if 'visual_quality' in calculator_configs:
            self.calculators['visual_quality'] = VisualQualityRewardCalculator(
                calculator_configs['visual_quality']
            )
        
        # Code quality rewards
        if 'code_quality' in calculator_configs:
            self.calculators['code_quality'] = CodeQualityRewardCalculator(
                calculator_configs['code_quality']
            )
        
        # Performance rewards
        if 'performance' in calculator_configs:
            self.calculators['performance'] = PerformanceRewardCalculator(
                calculator_configs['performance']
            )
        
        # Learning progress rewards
        if 'learning_progress' in calculator_configs:
            self.calculators['learning_progress'] = LearningProgressRewardCalculator(
                calculator_configs['learning_progress']
            )
        
        # Adaptive rewards
        if 'adaptive' in calculator_configs:
            self.calculators['adaptive'] = AdaptiveRewardCalculator(
                calculator_configs['adaptive']
            )
    
    def calculate_total_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                              result: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate total reward from all enabled calculators.
        
        Returns:
            Dictionary with individual and total rewards
        """
        rewards = {}
        total_reward = 0.0
        
        for name, calculator in self.calculators.items():
            if calculator.enabled:
                reward = calculator.calculate(state, action, result)
                rewards[name] = reward
                total_reward += reward
        
        rewards['total'] = total_reward * self.global_weight
        return rewards
    
    def update_calculator_weights(self, weight_updates: Dict[str, float]):
        """Update weights for specific calculators."""
        for name, weight in weight_updates.items():
            if name in self.calculators:
                self.calculators[name].weight = weight
    
    def enable_calculator(self, name: str, enabled: bool = True):
        """Enable or disable a specific calculator."""
        if name in self.calculators:
            self.calculators[name].enabled = enabled
    
    def get_reward_breakdown(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed breakdown of all calculators."""
        breakdown = {}
        for name, calculator in self.calculators.items():
            breakdown[name] = {
                'enabled': calculator.enabled,
                'weight': calculator.weight,
                'type': type(calculator).__name__
            }
        return breakdown


class RewardShaping:
    """
    Reward shaping utilities for improving learning efficiency.
    """
    
    @staticmethod
    def potential_based_shaping(state: Dict[str, Any], next_state: Dict[str, Any], 
                               gamma: float = 0.99) -> float:
        """
        Apply potential-based reward shaping.
        
        Args:
            state: Current state
            next_state: Next state
            gamma: Discount factor
            
        Returns:
            Shaping reward
        """
        current_potential = RewardShaping._calculate_potential(state)
        next_potential = RewardShaping._calculate_potential(next_state)
        
        return gamma * next_potential - current_potential
    
    @staticmethod
    def _calculate_potential(state: Dict[str, Any]) -> float:
        """Calculate potential function value for a state."""
        # Example potential based on game progress
        wave = state.get('current_wave', 0)
        lives = state.get('player_lives', 20)
        score = state.get('score', 0)
        
        # Potential increases with progress and decreases with life loss
        potential = wave * 10.0 + score * 0.1 - (20 - lives) * 50.0
        
        return potential
    
    @staticmethod
    def curiosity_driven_reward(state: Dict[str, Any], action: Dict[str, Any], 
                               prediction_error: float) -> float:
        """
        Calculate curiosity-driven intrinsic reward.
        
        Args:
            state: Current state
            action: Taken action
            prediction_error: Error in predicting next state
            
        Returns:
            Curiosity reward
        """
        # Higher prediction error indicates novel situations
        curiosity_reward = min(prediction_error * 10.0, 50.0)  # Cap the reward
        
        # Bonus for exploring new areas of the state space
        novelty_bonus = RewardShaping._calculate_novelty_bonus(state)
        
        return curiosity_reward + novelty_bonus
    
    @staticmethod
    def _calculate_novelty_bonus(state: Dict[str, Any]) -> float:
        """Calculate bonus for novel states."""
        # Simplified novelty calculation
        # In practice, this would use more sophisticated methods
        state_hash = hash(str(sorted(state.items())))
        
        # This would be replaced with actual novelty detection
        # For now, return a small random bonus
        return np.random.random() * 5.0

