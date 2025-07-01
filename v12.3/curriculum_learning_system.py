"""
Curriculum Learning System for RL-LLM Tree
Progressive scene definitions and task logic implementation

This module implements the core curriculum learning framework that enables
progressive difficulty scaling and adaptive learning experiences.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DifficultyDimension(Enum):
    """Dimensions along which curriculum difficulty can be scaled."""
    ENVIRONMENTAL_COMPLEXITY = "env_complexity"
    TASK_OBJECTIVES = "task_objectives"
    TIME_CONSTRAINTS = "time_constraints"
    INTERACTION_REQUIREMENTS = "interaction_requirements"
    SPATIAL_REASONING = "spatial_reasoning"
    MULTI_AGENT_COORDINATION = "multi_agent_coordination"


@dataclass
class SceneConfiguration:
    """Configuration for a single curriculum scene."""
    scene_id: str
    name: str
    description: str
    difficulty_level: float  # 0.0 to 1.0
    difficulty_dimensions: Dict[DifficultyDimension, float]
    prerequisites: List[str]  # Scene IDs that must be mastered first
    learning_objectives: List[str]
    success_criteria: Dict[str, float]
    environment_config: Dict[str, Any]
    reward_config: Dict[str, Any]
    estimated_training_time: int  # in episodes
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert enum keys to strings
        result['difficulty_dimensions'] = {
            dim.value: val for dim, val in self.difficulty_dimensions.items()
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SceneConfiguration':
        """Create from dictionary."""
        # Convert string keys back to enums
        difficulty_dims = {
            DifficultyDimension(k): v 
            for k, v in data['difficulty_dimensions'].items()
        }
        data['difficulty_dimensions'] = difficulty_dims
        return cls(**data)


@dataclass
class LearningProgress:
    """Tracks agent's learning progress across curriculum."""
    agent_id: str
    completed_scenes: List[str]
    current_scene: Optional[str]
    scene_performance: Dict[str, float]  # scene_id -> performance score
    learning_rate: Dict[str, float]  # scene_id -> learning rate
    mastery_levels: Dict[str, float]  # scene_id -> mastery level (0-1)
    total_training_episodes: int
    last_updated: str


class CurriculumManager:
    """Manages curriculum progression and scene selection."""
    
    def __init__(self, curriculum_config_path: str):
        """Initialize curriculum manager with configuration."""
        self.config_path = Path(curriculum_config_path)
        self.scenes: Dict[str, SceneConfiguration] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        self.progression_history: Dict[str, LearningProgress] = {}
        
        # Performance thresholds
        self.mastery_threshold = 0.8
        self.progression_threshold = 0.7
        self.regression_threshold = 0.5
        
        # Adaptive parameters
        self.difficulty_adaptation_rate = 0.1
        self.performance_window_size = 100
        
        self._load_curriculum()
    
    def _load_curriculum(self):
        """Load curriculum configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Load scene configurations
            for scene_data in config_data.get('scenes', []):
                scene = SceneConfiguration.from_dict(scene_data)
                self.scenes[scene.scene_id] = scene
                
            # Build dependency graph
            self._build_dependency_graph()
            
            logger.info(f"Loaded {len(self.scenes)} scenes from curriculum")
            
        except Exception as e:
            logger.error(f"Failed to load curriculum: {e}")
            raise
    
    def _build_dependency_graph(self):
        """Build scene dependency graph for progression planning."""
        self.dependency_graph = {}
        
        for scene_id, scene in self.scenes.items():
            self.dependency_graph[scene_id] = scene.prerequisites.copy()
    
    def get_available_scenes(self, agent_id: str) -> List[str]:
        """Get list of scenes available for the agent based on progress."""
        if agent_id not in self.progression_history:
            # New agent - start with scenes that have no prerequisites
            return [
                scene_id for scene_id, prereqs in self.dependency_graph.items()
                if not prereqs
            ]
        
        progress = self.progression_history[agent_id]
        available = []
        
        for scene_id, prereqs in self.dependency_graph.items():
            # Skip if already mastered
            if scene_id in progress.completed_scenes:
                continue
                
            # Check if all prerequisites are met
            prereqs_met = all(
                prereq in progress.completed_scenes or
                progress.mastery_levels.get(prereq, 0) >= self.mastery_threshold
                for prereq in prereqs
            )
            
            if prereqs_met:
                available.append(scene_id)
        
        return available
    
    def select_next_scene(self, agent_id: str) -> Optional[str]:
        """Select the next scene for the agent based on curriculum strategy."""
        available_scenes = self.get_available_scenes(agent_id)
        
        if not available_scenes:
            return None
        
        # If only one scene available, select it
        if len(available_scenes) == 1:
            return available_scenes[0]
        
        # Multi-scene selection strategy
        return self._select_optimal_scene(agent_id, available_scenes)
    
    def _select_optimal_scene(self, agent_id: str, available_scenes: List[str]) -> str:
        """Select optimal scene from available options."""
        if agent_id not in self.progression_history:
            # For new agents, start with easiest available scene
            scene_difficulties = [
                (scene_id, self.scenes[scene_id].difficulty_level)
                for scene_id in available_scenes
            ]
            return min(scene_difficulties, key=lambda x: x[1])[0]
        
        progress = self.progression_history[agent_id]
        
        # Calculate scene scores based on multiple factors
        scene_scores = {}
        
        for scene_id in available_scenes:
            scene = self.scenes[scene_id]
            
            # Base score from difficulty appropriateness
            current_skill_level = self._estimate_skill_level(agent_id)
            difficulty_match = 1.0 - abs(scene.difficulty_level - current_skill_level)
            
            # Bonus for scenes that build on recent learning
            recency_bonus = self._calculate_recency_bonus(agent_id, scene_id)
            
            # Penalty for scenes attempted recently without success
            failure_penalty = self._calculate_failure_penalty(agent_id, scene_id)
            
            scene_scores[scene_id] = (
                difficulty_match + recency_bonus - failure_penalty
            )
        
        # Select scene with highest score
        return max(scene_scores.items(), key=lambda x: x[1])[0]
    
    def _estimate_skill_level(self, agent_id: str) -> float:
        """Estimate agent's current skill level."""
        if agent_id not in self.progression_history:
            return 0.0
        
        progress = self.progression_history[agent_id]
        
        if not progress.mastery_levels:
            return 0.0
        
        # Weighted average of mastery levels, with recent scenes weighted more
        completed_scenes = progress.completed_scenes[-10:]  # Last 10 scenes
        
        if not completed_scenes:
            return 0.0
        
        total_weight = 0
        weighted_sum = 0
        
        for i, scene_id in enumerate(completed_scenes):
            weight = (i + 1) / len(completed_scenes)  # More recent = higher weight
            mastery = progress.mastery_levels.get(scene_id, 0)
            scene_difficulty = self.scenes[scene_id].difficulty_level
            
            weighted_sum += weight * mastery * scene_difficulty
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_recency_bonus(self, agent_id: str, scene_id: str) -> float:
        """Calculate bonus for scenes that build on recent learning."""
        if agent_id not in self.progression_history:
            return 0.0
        
        progress = self.progression_history[agent_id]
        scene = self.scenes[scene_id]
        
        # Check if any prerequisites were recently completed
        recent_scenes = progress.completed_scenes[-5:]  # Last 5 scenes
        
        bonus = 0.0
        for prereq in scene.prerequisites:
            if prereq in recent_scenes:
                # Higher bonus for more recent prerequisites
                recency_index = len(recent_scenes) - recent_scenes[::-1].index(prereq)
                bonus += 0.1 * (recency_index / len(recent_scenes))
        
        return bonus
    
    def _calculate_failure_penalty(self, agent_id: str, scene_id: str) -> float:
        """Calculate penalty for recently failed scenes."""
        if agent_id not in self.progression_history:
            return 0.0
        
        progress = self.progression_history[agent_id]
        
        # Check recent performance on this scene
        recent_performance = progress.scene_performance.get(scene_id, 1.0)
        
        if recent_performance < self.regression_threshold:
            return 0.2 * (self.regression_threshold - recent_performance)
        
        return 0.0
    
    def update_progress(self, agent_id: str, scene_id: str, 
                       performance_score: float, episode_count: int):
        """Update agent's learning progress."""
        if agent_id not in self.progression_history:
            self.progression_history[agent_id] = LearningProgress(
                agent_id=agent_id,
                completed_scenes=[],
                current_scene=scene_id,
                scene_performance={},
                learning_rate={},
                mastery_levels={},
                total_training_episodes=0,
                last_updated=""
            )
        
        progress = self.progression_history[agent_id]
        progress.current_scene = scene_id
        progress.scene_performance[scene_id] = performance_score
        progress.total_training_episodes += episode_count
        
        # Update mastery level using exponential moving average
        current_mastery = progress.mastery_levels.get(scene_id, 0.0)
        alpha = 0.1  # Learning rate for mastery update
        new_mastery = alpha * performance_score + (1 - alpha) * current_mastery
        progress.mastery_levels[scene_id] = new_mastery
        
        # Check if scene is mastered
        if (new_mastery >= self.mastery_threshold and 
            scene_id not in progress.completed_scenes):
            progress.completed_scenes.append(scene_id)
            logger.info(f"Agent {agent_id} mastered scene {scene_id}")
        
        # Update learning rate estimate
        self._update_learning_rate(agent_id, scene_id, performance_score)
    
    def _update_learning_rate(self, agent_id: str, scene_id: str, 
                             performance_score: float):
        """Update learning rate estimate for the scene."""
        progress = self.progression_history[agent_id]
        
        # Simple learning rate estimation based on performance improvement
        previous_performance = progress.scene_performance.get(scene_id, 0.0)
        improvement = performance_score - previous_performance
        
        # Exponential moving average of improvement rate
        current_lr = progress.learning_rate.get(scene_id, 0.0)
        alpha = 0.2
        new_lr = alpha * improvement + (1 - alpha) * current_lr
        progress.learning_rate[scene_id] = new_lr
    
    def get_curriculum_status(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive curriculum status for the agent."""
        if agent_id not in self.progression_history:
            return {
                "agent_id": agent_id,
                "total_scenes": len(self.scenes),
                "completed_scenes": 0,
                "current_scene": None,
                "available_scenes": self.get_available_scenes(agent_id),
                "overall_progress": 0.0,
                "estimated_skill_level": 0.0
            }
        
        progress = self.progression_history[agent_id]
        
        return {
            "agent_id": agent_id,
            "total_scenes": len(self.scenes),
            "completed_scenes": len(progress.completed_scenes),
            "current_scene": progress.current_scene,
            "available_scenes": self.get_available_scenes(agent_id),
            "overall_progress": len(progress.completed_scenes) / len(self.scenes),
            "estimated_skill_level": self._estimate_skill_level(agent_id),
            "mastery_levels": progress.mastery_levels,
            "total_episodes": progress.total_training_episodes
        }
    
    def save_progress(self, filepath: str):
        """Save learning progress to file."""
        progress_data = {
            agent_id: asdict(progress) 
            for agent_id, progress in self.progression_history.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def load_progress(self, filepath: str):
        """Load learning progress from file."""
        try:
            with open(filepath, 'r') as f:
                progress_data = json.load(f)
            
            self.progression_history = {
                agent_id: LearningProgress(**data)
                for agent_id, data in progress_data.items()
            }
            
            logger.info(f"Loaded progress for {len(self.progression_history)} agents")
            
        except Exception as e:
            logger.warning(f"Failed to load progress: {e}")


class AdaptiveDifficultyScaler:
    """Handles dynamic difficulty adjustment within scenes."""
    
    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.performance_history: Dict[str, List[float]] = {}
        self.difficulty_adjustments: Dict[str, Dict[str, float]] = {}
    
    def adjust_scene_difficulty(self, scene_config: SceneConfiguration,
                              agent_id: str, recent_performance: List[float]) -> SceneConfiguration:
        """Adjust scene difficulty based on agent performance."""
        if len(recent_performance) < 5:
            return scene_config  # Need sufficient data for adjustment
        
        avg_performance = np.mean(recent_performance[-10:])
        performance_trend = np.mean(recent_performance[-5:]) - np.mean(recent_performance[-10:-5])
        
        # Calculate adjustment factor
        if avg_performance > 0.8 and performance_trend > 0:
            # Agent is doing well, increase difficulty
            adjustment_factor = 1.0 + self.adaptation_rate
        elif avg_performance < 0.4 or performance_trend < -0.1:
            # Agent is struggling, decrease difficulty
            adjustment_factor = 1.0 - self.adaptation_rate
        else:
            # Performance is appropriate, no adjustment
            adjustment_factor = 1.0
        
        # Apply adjustments to difficulty dimensions
        adjusted_config = SceneConfiguration(**asdict(scene_config))
        
        for dimension in adjusted_config.difficulty_dimensions:
            current_value = adjusted_config.difficulty_dimensions[dimension]
            new_value = np.clip(current_value * adjustment_factor, 0.0, 1.0)
            adjusted_config.difficulty_dimensions[dimension] = new_value
        
        # Update overall difficulty level
        adjusted_config.difficulty_level = np.mean(
            list(adjusted_config.difficulty_dimensions.values())
        )
        
        # Store adjustment for tracking
        if agent_id not in self.difficulty_adjustments:
            self.difficulty_adjustments[agent_id] = {}
        
        self.difficulty_adjustments[agent_id][scene_config.scene_id] = adjustment_factor
        
        return adjusted_config


# Example usage and testing
if __name__ == "__main__":
    # Example curriculum configuration
    example_scenes = [
        {
            "scene_id": "basic_navigation",
            "name": "Basic Navigation",
            "description": "Learn to navigate in simple environments",
            "difficulty_level": 0.2,
            "difficulty_dimensions": {
                "env_complexity": 0.1,
                "task_objectives": 0.2,
                "time_constraints": 0.3,
                "interaction_requirements": 0.1,
                "spatial_reasoning": 0.2,
                "multi_agent_coordination": 0.0
            },
            "prerequisites": [],
            "learning_objectives": ["basic_movement", "obstacle_avoidance"],
            "success_criteria": {"completion_rate": 0.8, "efficiency": 0.6},
            "environment_config": {"map_size": "small", "obstacles": "few"},
            "reward_config": {"completion_reward": 10, "efficiency_bonus": 5},
            "estimated_training_time": 100
        }
    ]
    
    # Create example curriculum file
    curriculum_config = {"scenes": example_scenes}
    
    with open("/tmp/example_curriculum.json", "w") as f:
        json.dump(curriculum_config, f, indent=2)
    
    # Test curriculum manager
    manager = CurriculumManager("/tmp/example_curriculum.json")
    
    # Test scene selection for new agent
    next_scene = manager.select_next_scene("agent_001")
    print(f"Next scene for new agent: {next_scene}")
    
    # Test progress update
    manager.update_progress("agent_001", "basic_navigation", 0.75, 50)
    
    # Get status
    status = manager.get_curriculum_status("agent_001")
    print(f"Curriculum status: {status}")

