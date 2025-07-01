"""
Modular Reward Shaping Framework for RL-LLM Tree
Compositional reward components and dynamic adaptation system

This module implements a sophisticated reward shaping system that enables
modular reward design, dynamic adaptation, and multi-objective optimization.
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import json
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RewardType(Enum):
    """Types of reward components."""
    TASK_COMPLETION = "task_completion"
    EFFICIENCY = "efficiency"
    SAFETY = "safety"
    EXPLORATION = "exploration"
    SOCIAL_APPROPRIATENESS = "social_appropriateness"
    LEARNING_PROGRESS = "learning_progress"
    RESOURCE_CONSERVATION = "resource_conservation"
    COLLABORATION = "collaboration"


class AggregationStrategy(Enum):
    """Strategies for combining multiple reward components."""
    WEIGHTED_SUM = "weighted_sum"
    HIERARCHICAL = "hierarchical"
    PARETO_OPTIMAL = "pareto_optimal"
    DYNAMIC_BALANCING = "dynamic_balancing"


@dataclass
class RewardComponentConfig:
    """Configuration for a reward component."""
    component_id: str
    reward_type: RewardType
    weight: float
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    adaptation_rate: float = 0.1
    min_weight: float = 0.0
    max_weight: float = 1.0


class RewardComponent(ABC):
    """Abstract base class for reward components."""
    
    def __init__(self, config: RewardComponentConfig):
        self.config = config
        self.history: deque = deque(maxlen=1000)
        self.adaptation_history: List[float] = []
    
    @abstractmethod
    def calculate_reward(self, state: Dict[str, Any], action: Any, 
                        next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Calculate reward for the given transition."""
        pass
    
    @abstractmethod
    def get_reward_info(self) -> Dict[str, Any]:
        """Get information about the reward component's current state."""
        pass
    
    def update_weight(self, new_weight: float):
        """Update the component's weight with bounds checking."""
        self.config.weight = np.clip(
            new_weight, self.config.min_weight, self.config.max_weight
        )
        self.adaptation_history.append(self.config.weight)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical information about the component."""
        if not self.history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        rewards = list(self.history)
        return {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "min": np.min(rewards),
            "max": np.max(rewards),
            "count": len(rewards)
        }


class TaskCompletionReward(RewardComponent):
    """Reward component for task completion."""
    
    def calculate_reward(self, state: Dict[str, Any], action: Any,
                        next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Calculate task completion reward."""
        # Check if task was completed
        task_completed = info.get('task_completed', False)
        completion_quality = info.get('completion_quality', 1.0)
        
        if task_completed:
            base_reward = self.config.parameters.get('completion_reward', 10.0)
            quality_bonus = base_reward * (completion_quality - 1.0) * 0.5
            reward = base_reward + quality_bonus
        else:
            # Small progress reward for partial completion
            progress = info.get('task_progress', 0.0)
            progress_reward = self.config.parameters.get('progress_reward', 1.0)
            reward = progress * progress_reward
        
        self.history.append(reward)
        return reward * self.config.weight
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get task completion reward information."""
        return {
            "component_type": "task_completion",
            "weight": self.config.weight,
            "parameters": self.config.parameters,
            "statistics": self.get_statistics()
        }


class EfficiencyReward(RewardComponent):
    """Reward component for efficiency."""
    
    def calculate_reward(self, state: Dict[str, Any], action: Any,
                        next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Calculate efficiency reward."""
        # Time-based efficiency
        time_taken = info.get('time_taken', 1.0)
        optimal_time = self.config.parameters.get('optimal_time', 1.0)
        time_efficiency = max(0.0, 1.0 - (time_taken - optimal_time) / optimal_time)
        
        # Action efficiency (fewer actions = better)
        action_count = info.get('action_count', 1)
        optimal_actions = self.config.parameters.get('optimal_actions', 1)
        action_efficiency = max(0.0, 1.0 - (action_count - optimal_actions) / optimal_actions)
        
        # Resource efficiency
        resources_used = info.get('resources_used', 0.0)
        resource_budget = self.config.parameters.get('resource_budget', 1.0)
        resource_efficiency = max(0.0, 1.0 - resources_used / resource_budget)
        
        # Combine efficiency metrics
        efficiency_weights = self.config.parameters.get('efficiency_weights', {
            'time': 0.4, 'action': 0.3, 'resource': 0.3
        })
        
        total_efficiency = (
            efficiency_weights['time'] * time_efficiency +
            efficiency_weights['action'] * action_efficiency +
            efficiency_weights['resource'] * resource_efficiency
        )
        
        reward = total_efficiency * self.config.parameters.get('max_efficiency_reward', 5.0)
        self.history.append(reward)
        return reward * self.config.weight
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get efficiency reward information."""
        return {
            "component_type": "efficiency",
            "weight": self.config.weight,
            "parameters": self.config.parameters,
            "statistics": self.get_statistics()
        }


class SafetyReward(RewardComponent):
    """Reward component for safety considerations."""
    
    def calculate_reward(self, state: Dict[str, Any], action: Any,
                        next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Calculate safety reward."""
        # Safety violations
        safety_violations = info.get('safety_violations', [])
        violation_penalty = len(safety_violations) * self.config.parameters.get('violation_penalty', -5.0)
        
        # Risk assessment
        risk_level = info.get('risk_level', 0.0)  # 0.0 = safe, 1.0 = very risky
        risk_penalty = risk_level * self.config.parameters.get('risk_penalty', -2.0)
        
        # Safety margin bonus
        safety_margin = info.get('safety_margin', 0.0)  # Distance from danger
        margin_bonus = safety_margin * self.config.parameters.get('margin_bonus', 1.0)
        
        reward = violation_penalty + risk_penalty + margin_bonus
        self.history.append(reward)
        return reward * self.config.weight
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get safety reward information."""
        return {
            "component_type": "safety",
            "weight": self.config.weight,
            "parameters": self.config.parameters,
            "statistics": self.get_statistics()
        }


class ExplorationReward(RewardComponent):
    """Reward component for exploration behavior."""
    
    def __init__(self, config: RewardComponentConfig):
        super().__init__(config)
        self.visited_states = set()
        self.state_visit_counts = defaultdict(int)
    
    def calculate_reward(self, state: Dict[str, Any], action: Any,
                        next_state: Dict[str, Any], info: Dict[str, Any]) -> float:
        """Calculate exploration reward."""
        # State novelty reward
        state_key = self._get_state_key(next_state)
        is_novel = state_key not in self.visited_states
        
        if is_novel:
            self.visited_states.add(state_key)
            novelty_reward = self.config.parameters.get('novelty_reward', 2.0)
        else:
            # Diminishing returns for revisiting states
            visit_count = self.state_visit_counts[state_key]
            novelty_reward = self.config.parameters.get('novelty_reward', 2.0) / (visit_count + 1)
        
        self.state_visit_counts[state_key] += 1
        
        # Diversity bonus for exploring different types of states
        diversity_bonus = self._calculate_diversity_bonus()
        
        reward = novelty_reward + diversity_bonus
        self.history.append(reward)
        return reward * self.config.weight
    
    def _get_state_key(self, state: Dict[str, Any]) -> str:
        """Generate a key for state identification."""
        # Simplified state key generation - should be customized based on state representation
        position = state.get('position', (0, 0))
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            # Discretize position for exploration tracking
            discretization = self.config.parameters.get('position_discretization', 1.0)
            discrete_pos = (
                int(position[0] / discretization),
                int(position[1] / discretization)
            )
            return f"pos_{discrete_pos[0]}_{discrete_pos[1]}"
        return "unknown_state"
    
    def _calculate_diversity_bonus(self) -> float:
        """Calculate bonus for exploring diverse states."""
        unique_states = len(self.visited_states)
        total_visits = sum(self.state_visit_counts.values())
        
        if total_visits == 0:
            return 0.0
        
        diversity_ratio = unique_states / total_visits
        max_diversity_bonus = self.config.parameters.get('max_diversity_bonus', 1.0)
        
        return diversity_ratio * max_diversity_bonus
    
    def get_reward_info(self) -> Dict[str, Any]:
        """Get exploration reward information."""
        return {
            "component_type": "exploration",
            "weight": self.config.weight,
            "parameters": self.config.parameters,
            "statistics": self.get_statistics(),
            "unique_states_visited": len(self.visited_states),
            "total_state_visits": sum(self.state_visit_counts.values())
        }


class RewardShaper:
    """Main reward shaping system that combines multiple reward components."""
    
    def __init__(self, aggregation_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_SUM):
        self.components: Dict[str, RewardComponent] = {}
        self.aggregation_strategy = aggregation_strategy
        self.reward_history: List[Dict[str, float]] = []
        self.adaptation_enabled = True
        self.adaptation_frequency = 100  # episodes
        self.episode_count = 0
        
        # Multi-objective optimization state
        self.pareto_front: List[Dict[str, float]] = []
        self.objective_preferences: Dict[str, float] = {}
    
    def add_component(self, component: RewardComponent):
        """Add a reward component to the shaper."""
        self.components[component.config.component_id] = component
        logger.info(f"Added reward component: {component.config.component_id}")
    
    def remove_component(self, component_id: str):
        """Remove a reward component from the shaper."""
        if component_id in self.components:
            del self.components[component_id]
            logger.info(f"Removed reward component: {component_id}")
    
    def calculate_total_reward(self, state: Dict[str, Any], action: Any,
                             next_state: Dict[str, Any], info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculate total reward using all enabled components."""
        component_rewards = {}
        
        # Calculate individual component rewards
        for component_id, component in self.components.items():
            if component.config.enabled:
                reward = component.calculate_reward(state, action, next_state, info)
                component_rewards[component_id] = reward
            else:
                component_rewards[component_id] = 0.0
        
        # Aggregate rewards based on strategy
        total_reward = self._aggregate_rewards(component_rewards)
        
        # Store reward history
        reward_record = {
            "total": total_reward,
            "components": component_rewards.copy(),
            "episode": self.episode_count
        }
        self.reward_history.append(reward_record)
        
        # Perform adaptation if enabled
        if self.adaptation_enabled and self.episode_count % self.adaptation_frequency == 0:
            self._adapt_weights()
        
        self.episode_count += 1
        
        return total_reward, component_rewards
    
    def _aggregate_rewards(self, component_rewards: Dict[str, float]) -> float:
        """Aggregate component rewards based on the selected strategy."""
        if self.aggregation_strategy == AggregationStrategy.WEIGHTED_SUM:
            return sum(component_rewards.values())
        
        elif self.aggregation_strategy == AggregationStrategy.HIERARCHICAL:
            return self._hierarchical_aggregation(component_rewards)
        
        elif self.aggregation_strategy == AggregationStrategy.PARETO_OPTIMAL:
            return self._pareto_aggregation(component_rewards)
        
        elif self.aggregation_strategy == AggregationStrategy.DYNAMIC_BALANCING:
            return self._dynamic_balancing(component_rewards)
        
        else:
            # Default to weighted sum
            return sum(component_rewards.values())
    
    def _hierarchical_aggregation(self, component_rewards: Dict[str, float]) -> float:
        """Hierarchical aggregation with priority ordering."""
        # Define priority order (can be customized)
        priority_order = [
            RewardType.SAFETY,
            RewardType.TASK_COMPLETION,
            RewardType.EFFICIENCY,
            RewardType.EXPLORATION,
            RewardType.SOCIAL_APPROPRIATENESS
        ]
        
        total_reward = 0.0
        
        for reward_type in priority_order:
            # Find components of this type
            type_components = [
                (comp_id, comp) for comp_id, comp in self.components.items()
                if comp.config.reward_type == reward_type and comp.config.enabled
            ]
            
            if type_components:
                # Sum rewards for this type
                type_reward = sum(
                    component_rewards.get(comp_id, 0.0) for comp_id, _ in type_components
                )
                
                # Apply hierarchical weighting (higher priority = higher weight)
                priority_weight = (len(priority_order) - priority_order.index(reward_type)) / len(priority_order)
                total_reward += type_reward * priority_weight
        
        return total_reward
    
    def _pareto_aggregation(self, component_rewards: Dict[str, float]) -> float:
        """Pareto-optimal aggregation for multi-objective optimization."""
        # Update Pareto front
        self._update_pareto_front(component_rewards)
        
        # Use scalarization based on preferences
        if not self.objective_preferences:
            # Equal weighting if no preferences specified
            return sum(component_rewards.values()) / len(component_rewards)
        
        # Weighted scalarization
        total_reward = 0.0
        for comp_id, reward in component_rewards.items():
            preference = self.objective_preferences.get(comp_id, 1.0)
            total_reward += reward * preference
        
        return total_reward
    
    def _dynamic_balancing(self, component_rewards: Dict[str, float]) -> float:
        """Dynamic balancing based on recent performance."""
        if len(self.reward_history) < 10:
            # Not enough history, use weighted sum
            return sum(component_rewards.values())
        
        # Analyze recent performance trends
        recent_history = self.reward_history[-10:]
        component_trends = {}
        
        for comp_id in component_rewards.keys():
            recent_rewards = [h["components"].get(comp_id, 0.0) for h in recent_history]
            trend = np.mean(recent_rewards[-5:]) - np.mean(recent_rewards[:5])
            component_trends[comp_id] = trend
        
        # Adjust weights based on trends
        total_reward = 0.0
        for comp_id, reward in component_rewards.items():
            component = self.components[comp_id]
            
            # Increase weight for improving components, decrease for declining ones
            trend = component_trends.get(comp_id, 0.0)
            dynamic_weight = component.config.weight * (1.0 + 0.1 * np.tanh(trend))
            
            total_reward += reward * dynamic_weight
        
        return total_reward
    
    def _update_pareto_front(self, component_rewards: Dict[str, float]):
        """Update the Pareto front with new reward vector."""
        # Check if new point dominates any existing points
        new_point = component_rewards.copy()
        
        # Remove dominated points
        self.pareto_front = [
            point for point in self.pareto_front
            if not self._dominates(new_point, point)
        ]
        
        # Add new point if it's not dominated
        if not any(self._dominates(point, new_point) for point in self.pareto_front):
            self.pareto_front.append(new_point)
        
        # Limit Pareto front size
        if len(self.pareto_front) > 100:
            # Keep most diverse points
            self.pareto_front = self._select_diverse_points(self.pareto_front, 100)
    
    def _dominates(self, point1: Dict[str, float], point2: Dict[str, float]) -> bool:
        """Check if point1 dominates point2 in Pareto sense."""
        all_better_or_equal = True
        at_least_one_better = False
        
        for key in point1.keys():
            if key in point2:
                if point1[key] < point2[key]:
                    all_better_or_equal = False
                    break
                elif point1[key] > point2[key]:
                    at_least_one_better = True
        
        return all_better_or_equal and at_least_one_better
    
    def _select_diverse_points(self, points: List[Dict[str, float]], max_points: int) -> List[Dict[str, float]]:
        """Select diverse points from Pareto front."""
        if len(points) <= max_points:
            return points
        
        # Simple diversity selection based on distance
        selected = [points[0]]  # Start with first point
        
        for _ in range(max_points - 1):
            best_point = None
            best_min_distance = -1
            
            for candidate in points:
                if candidate in selected:
                    continue
                
                # Calculate minimum distance to selected points
                min_distance = min(
                    self._euclidean_distance(candidate, selected_point)
                    for selected_point in selected
                )
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_point = candidate
            
            if best_point:
                selected.append(best_point)
        
        return selected
    
    def _euclidean_distance(self, point1: Dict[str, float], point2: Dict[str, float]) -> float:
        """Calculate Euclidean distance between two reward vectors."""
        distance = 0.0
        for key in point1.keys():
            if key in point2:
                distance += (point1[key] - point2[key]) ** 2
        return np.sqrt(distance)
    
    def _adapt_weights(self):
        """Adapt component weights based on performance history."""
        if len(self.reward_history) < self.adaptation_frequency:
            return
        
        recent_history = self.reward_history[-self.adaptation_frequency:]
        
        # Analyze component performance
        for comp_id, component in self.components.items():
            component_rewards = [h["components"].get(comp_id, 0.0) for h in recent_history]
            
            # Calculate performance metrics
            mean_reward = np.mean(component_rewards)
            reward_variance = np.var(component_rewards)
            
            # Adapt weight based on performance
            if mean_reward > 0 and reward_variance < 1.0:
                # Good, stable performance - slightly increase weight
                new_weight = component.config.weight * 1.05
            elif mean_reward < 0 or reward_variance > 5.0:
                # Poor or unstable performance - slightly decrease weight
                new_weight = component.config.weight * 0.95
            else:
                # Maintain current weight
                new_weight = component.config.weight
            
            component.update_weight(new_weight)
        
        logger.info("Adapted reward component weights")
    
    def set_objective_preferences(self, preferences: Dict[str, float]):
        """Set preferences for multi-objective optimization."""
        self.objective_preferences = preferences.copy()
    
    def get_reward_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of reward system state."""
        component_info = {}
        for comp_id, component in self.components.items():
            component_info[comp_id] = component.get_reward_info()
        
        recent_performance = {}
        if len(self.reward_history) >= 10:
            recent_rewards = self.reward_history[-10:]
            recent_performance = {
                "mean_total_reward": np.mean([h["total"] for h in recent_rewards]),
                "reward_variance": np.var([h["total"] for h in recent_rewards]),
                "component_contributions": {}
            }
            
            for comp_id in self.components.keys():
                comp_rewards = [h["components"].get(comp_id, 0.0) for h in recent_rewards]
                recent_performance["component_contributions"][comp_id] = {
                    "mean": np.mean(comp_rewards),
                    "contribution_ratio": np.mean(comp_rewards) / recent_performance["mean_total_reward"] if recent_performance["mean_total_reward"] != 0 else 0
                }
        
        return {
            "aggregation_strategy": self.aggregation_strategy.value,
            "total_episodes": self.episode_count,
            "components": component_info,
            "recent_performance": recent_performance,
            "pareto_front_size": len(self.pareto_front),
            "adaptation_enabled": self.adaptation_enabled
        }
    
    def save_configuration(self, filepath: str):
        """Save reward shaper configuration to file."""
        config_data = {
            "aggregation_strategy": self.aggregation_strategy.value,
            "adaptation_enabled": self.adaptation_enabled,
            "adaptation_frequency": self.adaptation_frequency,
            "objective_preferences": self.objective_preferences,
            "components": {}
        }
        
        for comp_id, component in self.components.items():
            config_data["components"][comp_id] = {
                "component_id": component.config.component_id,
                "reward_type": component.config.reward_type.value,
                "weight": component.config.weight,
                "enabled": component.config.enabled,
                "parameters": component.config.parameters,
                "adaptation_rate": component.config.adaptation_rate,
                "min_weight": component.config.min_weight,
                "max_weight": component.config.max_weight
            }
        
        with open(filepath, 'w') as f:
            json.dump(config_data, f, indent=2)


# Example usage and testing
if __name__ == "__main__":
    # Create reward shaper
    shaper = RewardShaper(AggregationStrategy.WEIGHTED_SUM)
    
    # Add reward components
    task_config = RewardComponentConfig(
        component_id="task_completion",
        reward_type=RewardType.TASK_COMPLETION,
        weight=1.0,
        parameters={"completion_reward": 10.0, "progress_reward": 1.0}
    )
    shaper.add_component(TaskCompletionReward(task_config))
    
    efficiency_config = RewardComponentConfig(
        component_id="efficiency",
        reward_type=RewardType.EFFICIENCY,
        weight=0.5,
        parameters={"optimal_time": 10.0, "optimal_actions": 5, "max_efficiency_reward": 5.0}
    )
    shaper.add_component(EfficiencyReward(efficiency_config))
    
    safety_config = RewardComponentConfig(
        component_id="safety",
        reward_type=RewardType.SAFETY,
        weight=2.0,
        parameters={"violation_penalty": -10.0, "risk_penalty": -2.0, "margin_bonus": 1.0}
    )
    shaper.add_component(SafetyReward(safety_config))
    
    exploration_config = RewardComponentConfig(
        component_id="exploration",
        reward_type=RewardType.EXPLORATION,
        weight=0.3,
        parameters={"novelty_reward": 2.0, "max_diversity_bonus": 1.0}
    )
    shaper.add_component(ExplorationReward(exploration_config))
    
    # Test reward calculation
    state = {"position": (0, 0), "health": 100}
    action = "move_forward"
    next_state = {"position": (1, 0), "health": 100}
    info = {
        "task_completed": False,
        "task_progress": 0.2,
        "time_taken": 1.0,
        "action_count": 1,
        "safety_violations": [],
        "risk_level": 0.1
    }
    
    total_reward, component_rewards = shaper.calculate_total_reward(state, action, next_state, info)
    
    print(f"Total reward: {total_reward}")
    print(f"Component rewards: {component_rewards}")
    
    # Get summary
    summary = shaper.get_reward_summary()
    print(f"Reward system summary: {json.dumps(summary, indent=2)}")

