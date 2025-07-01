"""
Enhanced Reward System for Version 5 BETA 1

Major improvements for v5:
- Deep integration with NeRF asset rewards
- Automated reward shaping with diversity bonuses
- Anti-reward-hacking mechanisms
- Multi-objective reward optimization
- Comprehensive reward logging and analysis
- Dynamic reward weight adjustment
- Performance-aware reward calculation
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import json
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)

class RewardComponent(Enum):
    """Different components of the reward system."""
    BASE_GAME = "base_game"
    NERF_QUALITY = "nerf_quality"
    NERF_DIVERSITY = "nerf_diversity"
    PERFORMANCE = "performance"
    NOVELTY = "novelty"
    AESTHETIC = "aesthetic"
    CURRICULUM = "curriculum"
    HITL_FEEDBACK = "hitl_feedback"

@dataclass
class RewardWeights:
    """Dynamic weights for different reward components."""
    base_game: float = 0.4
    nerf_quality: float = 0.15
    nerf_diversity: float = 0.1
    performance: float = 0.1
    novelty: float = 0.1
    aesthetic: float = 0.1
    curriculum: float = 0.05
    hitl_feedback: float = 0.0  # Activated when HITL feedback is available

@dataclass
class RewardConfig:
    """Configuration for the enhanced reward system."""
    
    # Anti-reward-hacking settings
    enable_anti_hacking: bool = True
    trivial_solution_penalty: float = -0.5
    repetition_penalty_threshold: int = 5
    repetition_penalty_factor: float = 0.8
    
    # Diversity settings
    diversity_bonus_threshold: int = 3
    diversity_bonus_multiplier: float = 1.2
    diversity_window_size: int = 100
    
    # Performance settings
    performance_target_fps: float = 60.0
    performance_penalty_threshold: float = 30.0
    performance_bonus_threshold: float = 90.0
    
    # Novelty settings
    novelty_decay_rate: float = 0.95
    novelty_bonus_cap: float = 0.3
    
    # Curriculum settings
    curriculum_difficulty_bonus: float = 0.1
    curriculum_progression_reward: float = 0.2
    
    # Logging settings
    log_detailed_rewards: bool = True
    reward_history_size: int = 10000

class EnhancedRewardSystemV5:
    """
    Enhanced Reward System for Version 5 with comprehensive improvements.
    
    Key features:
    - Deep NeRF integration with quality and diversity rewards
    - Anti-reward-hacking mechanisms
    - Multi-objective optimization
    - Dynamic weight adjustment
    - Comprehensive logging and analysis
    """
    
    def __init__(self, 
                 nerf_integration=None,
                 config: Optional[RewardConfig] = None,
                 weights: Optional[RewardWeights] = None):
        
        self.nerf_integration = nerf_integration
        self.config = config or RewardConfig()
        self.weights = weights or RewardWeights()
        
        # Tracking and history
        self.reward_history: deque = deque(maxlen=self.config.reward_history_size)
        self.component_history: Dict[str, deque] = {
            component.value: deque(maxlen=1000) 
            for component in RewardComponent
        }
        self.action_history: deque = deque(maxlen=self.config.diversity_window_size)
        self.performance_history: deque = deque(maxlen=100)
        
        # Anti-hacking detection
        self.trivial_patterns: Dict[str, int] = defaultdict(int)
        self.repetition_tracker: Dict[str, int] = defaultdict(int)
        
        # Dynamic adjustment
        self.weight_adjustment_history: List[Dict[str, float]] = []
        self.performance_baseline: float = 0.0
        self.adaptation_rate: float = 0.01
        
        # Statistics
        self.total_rewards_calculated: int = 0
        self.reward_statistics: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        logger.info("Enhanced Reward System V5 initialized")
    
    def calculate_reward(self,
                        state: Any,
                        action: Any,
                        next_state: Any,
                        base_reward: float,
                        info: Dict[str, Any]) -> float:
        """
        Calculate comprehensive reward with all Version 5 enhancements.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            base_reward: Base game reward
            info: Additional information dictionary
            
        Returns:
            Enhanced reward value
        """
        
        start_time = time.time()
        reward_components = {}
        
        # Base game reward
        reward_components[RewardComponent.BASE_GAME.value] = base_reward
        
        # NeRF-based rewards
        if self.nerf_integration and 'nerf_asset' in info:
            nerf_rewards = self._calculate_nerf_rewards(info)
            reward_components.update(nerf_rewards)
        
        # Performance reward
        performance_reward = self._calculate_performance_reward(info)
        reward_components[RewardComponent.PERFORMANCE.value] = performance_reward
        
        # Novelty reward
        novelty_reward = self._calculate_novelty_reward(action, state)
        reward_components[RewardComponent.NOVELTY.value] = novelty_reward
        
        # Aesthetic reward
        aesthetic_reward = self._calculate_aesthetic_reward(info)
        reward_components[RewardComponent.AESTHETIC.value] = aesthetic_reward
        
        # Curriculum reward
        curriculum_reward = self._calculate_curriculum_reward(info)
        reward_components[RewardComponent.CURRICULUM.value] = curriculum_reward
        
        # HITL feedback reward
        hitl_reward = self._calculate_hitl_reward(info)
        reward_components[RewardComponent.HITL_FEEDBACK.value] = hitl_reward
        
        # Apply anti-reward-hacking measures
        if self.config.enable_anti_hacking:
            hacking_penalty = self._detect_and_penalize_hacking(action, reward_components)
            if hacking_penalty < 0:
                reward_components['anti_hacking_penalty'] = hacking_penalty
        
        # Combine rewards with dynamic weights
        total_reward = self._combine_rewards(reward_components)
        
        # Update tracking and statistics
        self._update_tracking(action, reward_components, total_reward, time.time() - start_time)
        
        # Dynamic weight adjustment
        self._adjust_weights_if_needed(reward_components, total_reward)
        
        self.total_rewards_calculated += 1
        
        return total_reward
    
    def _calculate_nerf_rewards(self, info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate NeRF-specific rewards."""
        nerf_rewards = {}
        
        if not self.nerf_integration:
            return nerf_rewards
        
        asset_id = info.get('nerf_asset')
        rendering_result = info.get('rendering_result', {})
        
        if asset_id:
            # Get NeRF reward from integration
            nerf_reward = self.nerf_integration.calculate_nerf_reward(
                asset_id, rendering_result, info
            )
            
            # Break down into components
            nerf_rewards[RewardComponent.NERF_QUALITY.value] = nerf_reward * 0.7
            
            # Diversity bonus
            diversity_bonus = self._calculate_nerf_diversity_bonus(asset_id)
            nerf_rewards[RewardComponent.NERF_DIVERSITY.value] = diversity_bonus
        
        return nerf_rewards
    
    def _calculate_nerf_diversity_bonus(self, asset_id: str) -> float:
        """Calculate bonus for using diverse NeRF assets."""
        # Track recent asset usage
        recent_assets = set()
        for i, action_info in enumerate(reversed(self.action_history)):
            if i >= self.config.diversity_window_size:
                break
            if 'nerf_asset' in action_info:
                recent_assets.add(action_info['nerf_asset'])
        
        diversity_count = len(recent_assets)
        
        if diversity_count >= self.config.diversity_bonus_threshold:
            bonus = (diversity_count - self.config.diversity_bonus_threshold) * 0.1
            return min(bonus, 0.5)  # Cap the bonus
        
        return 0.0
    
    def _calculate_performance_reward(self, info: Dict[str, Any]) -> float:
        """Calculate reward based on performance metrics."""
        rendering_time = info.get('rendering_time', 0)
        fps = info.get('fps', self.config.performance_target_fps)
        memory_usage = info.get('memory_usage', 0)
        
        # FPS-based reward
        if fps >= self.config.performance_bonus_threshold:
            fps_reward = 0.2  # Bonus for high FPS
        elif fps <= self.config.performance_penalty_threshold:
            fps_reward = -0.1  # Penalty for low FPS
        else:
            # Linear interpolation
            ratio = (fps - self.config.performance_penalty_threshold) / \
                   (self.config.performance_bonus_threshold - self.config.performance_penalty_threshold)
            fps_reward = -0.1 + ratio * 0.3
        
        # Rendering time penalty
        time_penalty = max(0, (rendering_time - 50) * 0.001)  # Penalty for slow rendering
        
        # Memory usage consideration
        memory_penalty = max(0, (memory_usage - 1000) * 0.0001)  # Penalty for high memory usage
        
        total_performance_reward = fps_reward - time_penalty - memory_penalty
        
        # Update performance history
        self.performance_history.append(total_performance_reward)
        
        return total_performance_reward
    
    def _calculate_novelty_reward(self, action: Any, state: Any) -> float:
        """Calculate reward for novel actions or states."""
        # Create action signature
        action_signature = self._create_action_signature(action, state)
        
        # Check how recently this action was taken
        novelty_score = 1.0
        for i, past_action in enumerate(reversed(self.action_history)):
            if i >= 50:  # Only check last 50 actions
                break
            
            past_signature = past_action.get('signature', '')
            if past_signature == action_signature:
                # Reduce novelty based on recency
                novelty_score *= self.config.novelty_decay_rate ** (50 - i)
        
        # Convert to reward
        novelty_reward = (novelty_score - 0.5) * self.config.novelty_bonus_cap
        return max(-self.config.novelty_bonus_cap, min(self.config.novelty_bonus_cap, novelty_reward))
    
    def _create_action_signature(self, action: Any, state: Any) -> str:
        """Create a signature for action-state combination."""
        # Simplified signature creation
        action_str = str(action) if not isinstance(action, (list, np.ndarray)) else str(hash(tuple(action.flatten())))
        state_str = str(hash(str(state)[:100]))  # Use first 100 chars of state string
        return f"{action_str}_{state_str}"
    
    def _calculate_aesthetic_reward(self, info: Dict[str, Any]) -> float:
        """Calculate reward based on aesthetic quality."""
        # Use visual assessment if available
        if 'aesthetic_score' in info:
            return (info['aesthetic_score'] - 0.5) * 0.2  # Scale to [-0.1, 0.1]
        
        # Fallback: use basic visual metrics
        visual_metrics = info.get('visual_metrics', {})
        if visual_metrics:
            brightness = visual_metrics.get('brightness', 0.5)
            contrast = visual_metrics.get('contrast', 0.5)
            color_harmony = visual_metrics.get('color_harmony', 0.5)
            
            aesthetic_score = (brightness + contrast + color_harmony) / 3.0
            return (aesthetic_score - 0.5) * 0.1
        
        return 0.0
    
    def _calculate_curriculum_reward(self, info: Dict[str, Any]) -> float:
        """Calculate reward based on curriculum progression."""
        difficulty = info.get('difficulty', 0.5)
        success = info.get('success', False)
        
        if success:
            # Bonus for succeeding at higher difficulties
            difficulty_bonus = difficulty * self.config.curriculum_difficulty_bonus
            
            # Progression bonus if difficulty increased
            if hasattr(self, 'last_difficulty') and difficulty > self.last_difficulty:
                progression_bonus = self.config.curriculum_progression_reward
            else:
                progression_bonus = 0.0
            
            self.last_difficulty = difficulty
            return difficulty_bonus + progression_bonus
        
        return 0.0
    
    def _calculate_hitl_reward(self, info: Dict[str, Any]) -> float:
        """Calculate reward based on human feedback."""
        hitl_rating = info.get('hitl_rating')
        if hitl_rating is not None:
            # Convert rating (1-10) to reward (-0.2 to 0.2)
            normalized_rating = (hitl_rating - 5.5) / 4.5  # Center around 5.5, scale by 4.5
            return normalized_rating * 0.2
        
        return 0.0
    
    def _detect_and_penalize_hacking(self, action: Any, reward_components: Dict[str, float]) -> float:
        """Detect and penalize reward hacking attempts."""
        penalty = 0.0
        
        # Detect trivial solutions
        action_signature = str(action)
        self.trivial_patterns[action_signature] += 1
        
        # Penalize excessive repetition
        if self.trivial_patterns[action_signature] > self.config.repetition_penalty_threshold:
            repetition_ratio = self.trivial_patterns[action_signature] / len(self.action_history)
            if repetition_ratio > 0.5:  # More than 50% repetition
                penalty += self.config.trivial_solution_penalty
        
        # Detect reward component exploitation
        max_component_value = max(reward_components.values()) if reward_components else 0
        if max_component_value > 2.0:  # Suspiciously high individual component
            penalty += -0.2
        
        # Detect unrealistic reward combinations
        total_positive_reward = sum(max(0, r) for r in reward_components.values())
        if total_positive_reward > 3.0:  # Suspiciously high total positive reward
            penalty += -0.3
        
        return penalty
    
    def _combine_rewards(self, reward_components: Dict[str, float]) -> float:
        """Combine reward components using dynamic weights."""
        total_reward = 0.0
        
        # Apply weights to each component
        for component, value in reward_components.items():
            if hasattr(self.weights, component):
                weight = getattr(self.weights, component)
                total_reward += value * weight
            else:
                # Handle special components
                if component == 'anti_hacking_penalty':
                    total_reward += value  # Apply penalty directly
                else:
                    total_reward += value * 0.1  # Default small weight
        
        return total_reward
    
    def _adjust_weights_if_needed(self, reward_components: Dict[str, float], total_reward: float):
        """Dynamically adjust reward weights based on performance."""
        # Simple adaptive mechanism
        if len(self.reward_history) > 100:
            recent_rewards = list(self.reward_history)[-100:]
            reward_variance = np.var(recent_rewards)
            
            # If variance is too high, reduce weights of high-variance components
            if reward_variance > 1.0:
                for component, values in self.component_history.items():
                    if len(values) > 50:
                        component_variance = np.var(list(values)[-50:])
                        if component_variance > 0.5 and hasattr(self.weights, component):
                            current_weight = getattr(self.weights, component)
                            new_weight = current_weight * 0.95  # Slight reduction
                            setattr(self.weights, component, max(0.01, new_weight))
    
    def _update_tracking(self, action: Any, reward_components: Dict[str, float], 
                        total_reward: float, calculation_time: float):
        """Update tracking and statistics."""
        # Update histories
        self.reward_history.append(total_reward)
        
        action_info = {
            'signature': self._create_action_signature(action, {}),
            'timestamp': time.time(),
            'total_reward': total_reward,
            'calculation_time': calculation_time
        }
        self.action_history.append(action_info)
        
        # Update component histories
        for component, value in reward_components.items():
            if component in self.component_history:
                self.component_history[component].append(value)
        
        # Update statistics
        for component, value in reward_components.items():
            stats = self.reward_statistics[component]
            stats['count'] += 1
            stats['sum'] += value
            stats['sum_squared'] += value ** 2
            stats['min'] = min(stats.get('min', value), value)
            stats['max'] = max(stats.get('max', value), value)
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get comprehensive reward statistics."""
        stats = {
            'total_rewards_calculated': self.total_rewards_calculated,
            'current_weights': self.weights.__dict__,
            'component_statistics': {},
            'performance_metrics': {
                'avg_calculation_time': np.mean([a.get('calculation_time', 0) for a in self.action_history]),
                'reward_variance': np.var(list(self.reward_history)) if self.reward_history else 0,
                'avg_reward': np.mean(list(self.reward_history)) if self.reward_history else 0
            }
        }
        
        # Calculate component statistics
        for component, component_stats in self.reward_statistics.items():
            if component_stats['count'] > 0:
                mean = component_stats['sum'] / component_stats['count']
                variance = (component_stats['sum_squared'] / component_stats['count']) - mean ** 2
                
                stats['component_statistics'][component] = {
                    'count': component_stats['count'],
                    'mean': mean,
                    'variance': variance,
                    'std': np.sqrt(max(0, variance)),
                    'min': component_stats['min'],
                    'max': component_stats['max']
                }
        
        return stats
    
    def save_reward_analysis(self, filepath: str):
        """Save comprehensive reward analysis to file."""
        analysis = {
            'statistics': self.get_reward_statistics(),
            'reward_history': list(self.reward_history),
            'component_history': {k: list(v) for k, v in self.component_history.items()},
            'trivial_patterns': dict(self.trivial_patterns),
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Reward analysis saved to {filepath}")
    
    def reset_tracking(self):
        """Reset all tracking data."""
        self.reward_history.clear()
        for history in self.component_history.values():
            history.clear()
        self.action_history.clear()
        self.performance_history.clear()
        self.trivial_patterns.clear()
        self.repetition_tracker.clear()
        self.reward_statistics.clear()
        
        logger.info("Reward tracking data reset")

# Example usage
if __name__ == "__main__":
    from .nerf_integration_v5 import NeRFIntegrationV5
    
    # Initialize components
    nerf_integration = NeRFIntegrationV5()
    reward_system = EnhancedRewardSystemV5(nerf_integration=nerf_integration)
    
    # Example reward calculation
    state = np.array([1, 2, 3])
    action = np.array([0.5, 0.3])
    next_state = np.array([1.1, 2.1, 3.1])
    base_reward = 1.0
    
    info = {
        'nerf_asset': 'asset_123',
        'rendering_result': {'rendering_time': 45.0},
        'fps': 75.0,
        'difficulty': 0.7,
        'success': True
    }
    
    reward = reward_system.calculate_reward(state, action, next_state, base_reward, info)
    print(f"Enhanced reward: {reward:.4f}")
    
    # Get statistics
    stats = reward_system.get_reward_statistics()
    print(f"Reward statistics: {stats}")
    
    # Save analysis
    reward_system.save_reward_analysis("reward_analysis_v5.json")

