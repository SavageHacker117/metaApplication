"""
Curriculum Learning Manager for Version 5 BETA 1

This module implements adaptive curriculum learning that progressively increases
difficulty based on agent performance, enabling more efficient and stable training.

Key features:
- Adaptive difficulty progression based on performance metrics
- Multiple curriculum strategies (linear, exponential, performance-based)
- Automatic difficulty adjustment with safety bounds
- Comprehensive progress tracking and analysis
- Integration with NeRF asset complexity
- Performance-based milestone system
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class CurriculumStrategy(Enum):
    """Different curriculum learning strategies."""
    LINEAR = "linear"                    # Linear progression
    EXPONENTIAL = "exponential"          # Exponential progression
    PERFORMANCE_BASED = "performance"    # Based on performance metrics
    ADAPTIVE = "adaptive"                # Adaptive based on multiple factors
    MILESTONE = "milestone"              # Milestone-based progression

class DifficultyMetric(Enum):
    """Metrics used to determine difficulty."""
    REWARD_THRESHOLD = "reward_threshold"
    SUCCESS_RATE = "success_rate"
    EPISODE_LENGTH = "episode_length"
    NERF_COMPLEXITY = "nerf_complexity"
    COMBINED_SCORE = "combined_score"

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    # Basic settings
    strategy: CurriculumStrategy = CurriculumStrategy.ADAPTIVE
    initial_difficulty: float = 0.1
    max_difficulty: float = 1.0
    min_difficulty: float = 0.05
    
    # Progression settings
    progression_rate: float = 0.01
    regression_rate: float = 0.005
    stability_threshold: int = 10  # Episodes of stable performance
    
    # Performance thresholds
    success_threshold: float = 0.7
    failure_threshold: float = 0.3
    reward_improvement_threshold: float = 0.1
    
    # Evaluation windows
    evaluation_window: int = 20
    trend_analysis_window: int = 50
    
    # NeRF integration
    nerf_complexity_weight: float = 0.2
    asset_diversity_bonus: float = 0.1
    
    # Safety settings
    max_progression_per_step: float = 0.05
    max_regression_per_step: float = 0.02
    emergency_regression_threshold: float = 0.1

@dataclass
class CurriculumMilestone:
    """Represents a curriculum milestone."""
    difficulty: float
    name: str
    description: str
    requirements: Dict[str, float]
    rewards_bonus: float = 0.0
    unlocked: bool = False
    unlock_episode: Optional[int] = None

class CurriculumLearningManager:
    """
    Advanced curriculum learning manager with adaptive difficulty progression.
    
    Features:
    - Multiple progression strategies
    - Performance-based adaptation
    - NeRF asset complexity integration
    - Milestone system with rewards
    - Comprehensive progress tracking
    """
    
    def __init__(self, config: Optional[CurriculumConfig] = None):
        self.config = config or CurriculumConfig()
        
        # Current state
        self.current_difficulty = self.config.initial_difficulty
        self.current_episode = 0
        self.stable_episodes = 0
        
        # Performance tracking
        self.performance_history: deque = deque(maxlen=self.config.evaluation_window)
        self.reward_history: deque = deque(maxlen=self.config.trend_analysis_window)
        self.success_history: deque = deque(maxlen=self.config.evaluation_window)
        self.difficulty_history: List[Tuple[int, float]] = []
        
        # Advanced metrics
        self.performance_metrics: Dict[str, deque] = {
            'episode_length': deque(maxlen=self.config.evaluation_window),
            'nerf_complexity': deque(maxlen=self.config.evaluation_window),
            'asset_diversity': deque(maxlen=self.config.evaluation_window),
            'learning_rate': deque(maxlen=self.config.evaluation_window)
        }
        
        # Milestone system
        self.milestones: List[CurriculumMilestone] = self._initialize_milestones()
        self.milestone_progress: Dict[str, float] = {}
        
        # Analysis and statistics
        self.progression_events: List[Dict[str, Any]] = []
        self.regression_events: List[Dict[str, Any]] = []
        self.statistics: Dict[str, Any] = defaultdict(float)
        
        logger.info(f"Curriculum Learning Manager initialized with {self.config.strategy.value} strategy")
    
    def _initialize_milestones(self) -> List[CurriculumMilestone]:
        """Initialize curriculum milestones."""
        milestones = [
            CurriculumMilestone(
                difficulty=0.2,
                name="Basic Competency",
                description="Demonstrates basic game understanding",
                requirements={'success_rate': 0.5, 'avg_reward': 5.0},
                rewards_bonus=0.1
            ),
            CurriculumMilestone(
                difficulty=0.4,
                name="Intermediate Skills",
                description="Shows strategic thinking and planning",
                requirements={'success_rate': 0.6, 'avg_reward': 10.0, 'asset_diversity': 3},
                rewards_bonus=0.15
            ),
            CurriculumMilestone(
                difficulty=0.6,
                name="Advanced Tactics",
                description="Demonstrates advanced tactical abilities",
                requirements={'success_rate': 0.7, 'avg_reward': 15.0, 'asset_diversity': 5},
                rewards_bonus=0.2
            ),
            CurriculumMilestone(
                difficulty=0.8,
                name="Expert Performance",
                description="Exhibits expert-level performance",
                requirements={'success_rate': 0.8, 'avg_reward': 20.0, 'asset_diversity': 7},
                rewards_bonus=0.25
            ),
            CurriculumMilestone(
                difficulty=1.0,
                name="Master Level",
                description="Achieves master-level performance",
                requirements={'success_rate': 0.9, 'avg_reward': 25.0, 'asset_diversity': 10},
                rewards_bonus=0.3
            )
        ]
        return milestones
    
    def update(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update curriculum based on episode performance.
        
        Args:
            episode_data: Complete episode data including metrics
            
        Returns:
            Dictionary with curriculum update information
        """
        
        self.current_episode += 1
        
        # Extract performance metrics
        performance_metrics = self._extract_performance_metrics(episode_data)
        
        # Update tracking
        self._update_tracking(performance_metrics)
        
        # Determine difficulty adjustment
        adjustment_info = self._calculate_difficulty_adjustment(performance_metrics)
        
        # Apply adjustment
        old_difficulty = self.current_difficulty
        self._apply_difficulty_adjustment(adjustment_info)
        
        # Check milestones
        milestone_updates = self._check_milestones(performance_metrics)
        
        # Update statistics
        self._update_statistics(adjustment_info, milestone_updates)
        
        # Prepare update summary
        update_summary = {
            'episode': self.current_episode,
            'old_difficulty': old_difficulty,
            'new_difficulty': self.current_difficulty,
            'adjustment': adjustment_info,
            'milestone_updates': milestone_updates,
            'performance_metrics': performance_metrics,
            'stable_episodes': self.stable_episodes
        }
        
        logger.debug(f"Curriculum update: Episode {self.current_episode}, "
                    f"Difficulty: {old_difficulty:.3f} -> {self.current_difficulty:.3f}")
        
        return update_summary
    
    def _extract_performance_metrics(self, episode_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant performance metrics from episode data."""
        metrics = {}
        
        # Basic metrics
        metrics['total_reward'] = episode_data.get('total_reward', 0.0)
        metrics['episode_length'] = episode_data.get('steps', 0)
        metrics['success'] = float(episode_data.get('success', False))
        
        # NeRF-related metrics
        nerf_assets = episode_data.get('nerf_assets_used', set())
        metrics['asset_diversity'] = len(nerf_assets)
        metrics['nerf_complexity'] = self._calculate_nerf_complexity(nerf_assets)
        
        # Performance metrics
        perf_metrics = episode_data.get('performance_metrics', {})
        metrics['avg_reward_per_step'] = perf_metrics.get('avg_reward_per_step', 0.0)
        metrics['nerf_reward_ratio'] = perf_metrics.get('nerf_reward_ratio', 0.0)
        
        # Learning indicators
        metrics['exploration_rate'] = perf_metrics.get('exploration_rate', 0.5)
        metrics['decision_confidence'] = perf_metrics.get('decision_confidence', 0.5)
        
        return metrics
    
    def _calculate_nerf_complexity(self, nerf_assets: set) -> float:
        """Calculate complexity score based on NeRF assets used."""
        if not nerf_assets:
            return 0.0
        
        # Simple complexity calculation based on asset count and diversity
        base_complexity = len(nerf_assets) * 0.1
        diversity_bonus = min(len(nerf_assets) / 10.0, 0.5)  # Max 0.5 bonus
        
        return min(base_complexity + diversity_bonus, 1.0)
    
    def _update_tracking(self, metrics: Dict[str, float]):
        """Update performance tracking with new metrics."""
        # Update main tracking
        self.performance_history.append(metrics)
        self.reward_history.append(metrics['total_reward'])
        self.success_history.append(metrics['success'])
        
        # Update detailed metrics
        for key, value in metrics.items():
            if key in self.performance_metrics:
                self.performance_metrics[key].append(value)
        
        # Update difficulty history
        self.difficulty_history.append((self.current_episode, self.current_difficulty))
    
    def _calculate_difficulty_adjustment(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate how difficulty should be adjusted."""
        
        if len(self.performance_history) < self.config.evaluation_window:
            return {'action': 'wait', 'reason': 'insufficient_data', 'magnitude': 0.0}
        
        # Calculate performance indicators
        recent_performance = self._analyze_recent_performance()
        trend_analysis = self._analyze_performance_trend()
        
        # Determine adjustment based on strategy
        if self.config.strategy == CurriculumStrategy.LINEAR:
            adjustment = self._linear_adjustment()
        elif self.config.strategy == CurriculumStrategy.EXPONENTIAL:
            adjustment = self._exponential_adjustment()
        elif self.config.strategy == CurriculumStrategy.PERFORMANCE_BASED:
            adjustment = self._performance_based_adjustment(recent_performance)
        elif self.config.strategy == CurriculumStrategy.ADAPTIVE:
            adjustment = self._adaptive_adjustment(recent_performance, trend_analysis)
        elif self.config.strategy == CurriculumStrategy.MILESTONE:
            adjustment = self._milestone_based_adjustment(recent_performance)
        else:
            adjustment = {'action': 'maintain', 'reason': 'unknown_strategy', 'magnitude': 0.0}
        
        # Apply safety constraints
        adjustment = self._apply_safety_constraints(adjustment, recent_performance)
        
        return adjustment
    
    def _analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent performance metrics."""
        if not self.performance_history:
            return {}
        
        recent_rewards = [p['total_reward'] for p in self.performance_history]
        recent_success = [p['success'] for p in self.performance_history]
        recent_lengths = [p['episode_length'] for p in self.performance_history]
        
        analysis = {
            'avg_reward': np.mean(recent_rewards),
            'reward_std': np.std(recent_rewards),
            'success_rate': np.mean(recent_success),
            'avg_episode_length': np.mean(recent_lengths),
            'reward_trend': self._calculate_trend(recent_rewards),
            'consistency': 1.0 / (1.0 + np.std(recent_rewards))  # Higher is more consistent
        }
        
        return analysis
    
    def _analyze_performance_trend(self) -> Dict[str, float]:
        """Analyze longer-term performance trends."""
        if len(self.reward_history) < 10:
            return {'trend': 0.0, 'confidence': 0.0}
        
        rewards = list(self.reward_history)
        
        # Calculate trend using linear regression
        x = np.arange(len(rewards))
        trend_slope = np.polyfit(x, rewards, 1)[0] if len(rewards) > 1 else 0.0
        
        # Calculate confidence based on R-squared
        if len(rewards) > 2:
            correlation = np.corrcoef(x, rewards)[0, 1]
            confidence = correlation ** 2 if not np.isnan(correlation) else 0.0
        else:
            confidence = 0.0
        
        return {
            'trend': trend_slope,
            'confidence': confidence,
            'recent_improvement': rewards[-5:] if len(rewards) >= 5 else rewards
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for a list of values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        return np.polyfit(x, values, 1)[0]
    
    def _linear_adjustment(self) -> Dict[str, Any]:
        """Linear progression strategy."""
        if self.current_difficulty < self.config.max_difficulty:
            return {
                'action': 'increase',
                'reason': 'linear_progression',
                'magnitude': self.config.progression_rate
            }
        return {'action': 'maintain', 'reason': 'max_difficulty_reached', 'magnitude': 0.0}
    
    def _exponential_adjustment(self) -> Dict[str, Any]:
        """Exponential progression strategy."""
        if self.current_difficulty < self.config.max_difficulty:
            # Exponential progression slows down as difficulty increases
            rate = self.config.progression_rate * (1.0 - self.current_difficulty)
            return {
                'action': 'increase',
                'reason': 'exponential_progression',
                'magnitude': rate
            }
        return {'action': 'maintain', 'reason': 'max_difficulty_reached', 'magnitude': 0.0}
    
    def _performance_based_adjustment(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Performance-based adjustment strategy."""
        success_rate = performance.get('success_rate', 0.0)
        avg_reward = performance.get('avg_reward', 0.0)
        consistency = performance.get('consistency', 0.0)
        
        # Increase difficulty if performing well
        if (success_rate >= self.config.success_threshold and 
            consistency > 0.7 and 
            self.stable_episodes >= self.config.stability_threshold):
            
            magnitude = self.config.progression_rate * (1.0 + success_rate - self.config.success_threshold)
            return {
                'action': 'increase',
                'reason': 'good_performance',
                'magnitude': magnitude,
                'metrics': {'success_rate': success_rate, 'consistency': consistency}
            }
        
        # Decrease difficulty if performing poorly
        elif success_rate <= self.config.failure_threshold:
            magnitude = self.config.regression_rate * (self.config.failure_threshold - success_rate + 1.0)
            return {
                'action': 'decrease',
                'reason': 'poor_performance',
                'magnitude': magnitude,
                'metrics': {'success_rate': success_rate}
            }
        
        # Maintain current difficulty
        return {
            'action': 'maintain',
            'reason': 'stable_performance',
            'magnitude': 0.0,
            'metrics': performance
        }
    
    def _adaptive_adjustment(self, performance: Dict[str, float], trend: Dict[str, float]) -> Dict[str, Any]:
        """Adaptive adjustment combining multiple factors."""
        
        # Combine multiple performance indicators
        success_rate = performance.get('success_rate', 0.0)
        reward_trend = trend.get('trend', 0.0)
        trend_confidence = trend.get('confidence', 0.0)
        consistency = performance.get('consistency', 0.0)
        
        # Calculate composite score
        performance_score = (
            success_rate * 0.4 +
            min(max(reward_trend, -1.0), 1.0) * 0.3 +  # Normalize trend
            consistency * 0.2 +
            trend_confidence * 0.1
        )
        
        # Adaptive thresholds based on current difficulty
        adaptive_success_threshold = self.config.success_threshold + (self.current_difficulty * 0.1)
        adaptive_failure_threshold = self.config.failure_threshold - (self.current_difficulty * 0.05)
        
        # Decision logic
        if (performance_score > 0.7 and 
            success_rate >= adaptive_success_threshold and
            self.stable_episodes >= self.config.stability_threshold):
            
            magnitude = self.config.progression_rate * performance_score
            return {
                'action': 'increase',
                'reason': 'adaptive_good_performance',
                'magnitude': magnitude,
                'performance_score': performance_score
            }
        
        elif (performance_score < 0.3 or 
              success_rate <= adaptive_failure_threshold):
            
            magnitude = self.config.regression_rate * (1.0 - performance_score)
            return {
                'action': 'decrease',
                'reason': 'adaptive_poor_performance',
                'magnitude': magnitude,
                'performance_score': performance_score
            }
        
        return {
            'action': 'maintain',
            'reason': 'adaptive_stable',
            'magnitude': 0.0,
            'performance_score': performance_score
        }
    
    def _milestone_based_adjustment(self, performance: Dict[str, float]) -> Dict[str, Any]:
        """Milestone-based adjustment strategy."""
        
        # Check if ready for next milestone
        next_milestone = self._get_next_milestone()
        
        if next_milestone and self._check_milestone_requirements(next_milestone, performance):
            # Progress to next milestone
            target_difficulty = next_milestone.difficulty
            magnitude = min(target_difficulty - self.current_difficulty, self.config.max_progression_per_step)
            
            return {
                'action': 'increase',
                'reason': 'milestone_progression',
                'magnitude': magnitude,
                'target_milestone': next_milestone.name
            }
        
        # Check if need to regress from current milestone
        current_milestone = self._get_current_milestone()
        if current_milestone and not self._check_milestone_requirements(current_milestone, performance):
            magnitude = self.config.regression_rate
            return {
                'action': 'decrease',
                'reason': 'milestone_regression',
                'magnitude': magnitude,
                'current_milestone': current_milestone.name
            }
        
        return {'action': 'maintain', 'reason': 'milestone_stable', 'magnitude': 0.0}
    
    def _apply_safety_constraints(self, adjustment: Dict[str, Any], performance: Dict[str, float]) -> Dict[str, Any]:
        """Apply safety constraints to difficulty adjustment."""
        
        # Emergency regression if performance is extremely poor
        success_rate = performance.get('success_rate', 0.0)
        if success_rate <= self.config.emergency_regression_threshold:
            return {
                'action': 'decrease',
                'reason': 'emergency_regression',
                'magnitude': self.config.max_regression_per_step,
                'original_adjustment': adjustment
            }
        
        # Limit adjustment magnitude
        if adjustment['action'] == 'increase':
            adjustment['magnitude'] = min(adjustment['magnitude'], self.config.max_progression_per_step)
        elif adjustment['action'] == 'decrease':
            adjustment['magnitude'] = min(adjustment['magnitude'], self.config.max_regression_per_step)
        
        return adjustment
    
    def _apply_difficulty_adjustment(self, adjustment: Dict[str, Any]):
        """Apply the calculated difficulty adjustment."""
        
        old_difficulty = self.current_difficulty
        
        if adjustment['action'] == 'increase':
            self.current_difficulty = min(
                self.current_difficulty + adjustment['magnitude'],
                self.config.max_difficulty
            )
            self.stable_episodes = 0  # Reset stability counter
            self.progression_events.append({
                'episode': self.current_episode,
                'old_difficulty': old_difficulty,
                'new_difficulty': self.current_difficulty,
                'reason': adjustment['reason']
            })
            
        elif adjustment['action'] == 'decrease':
            self.current_difficulty = max(
                self.current_difficulty - adjustment['magnitude'],
                self.config.min_difficulty
            )
            self.stable_episodes = 0  # Reset stability counter
            self.regression_events.append({
                'episode': self.current_episode,
                'old_difficulty': old_difficulty,
                'new_difficulty': self.current_difficulty,
                'reason': adjustment['reason']
            })
            
        else:  # maintain
            self.stable_episodes += 1
    
    def _check_milestones(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check and update milestone progress."""
        updates = []
        
        for milestone in self.milestones:
            if milestone.unlocked:
                continue
            
            if self._check_milestone_requirements(milestone, metrics):
                milestone.unlocked = True
                milestone.unlock_episode = self.current_episode
                
                updates.append({
                    'milestone': milestone.name,
                    'difficulty': milestone.difficulty,
                    'episode': self.current_episode,
                    'rewards_bonus': milestone.rewards_bonus
                })
                
                logger.info(f"Milestone unlocked: {milestone.name} at episode {self.current_episode}")
        
        return updates
    
    def _check_milestone_requirements(self, milestone: CurriculumMilestone, metrics: Dict[str, float]) -> bool:
        """Check if milestone requirements are met."""
        
        if len(self.performance_history) < self.config.evaluation_window:
            return False
        
        # Calculate current performance averages
        recent_performance = self._analyze_recent_performance()
        
        # Check each requirement
        for req_name, req_value in milestone.requirements.items():
            if req_name == 'success_rate':
                if recent_performance.get('success_rate', 0.0) < req_value:
                    return False
            elif req_name == 'avg_reward':
                if recent_performance.get('avg_reward', 0.0) < req_value:
                    return False
            elif req_name == 'asset_diversity':
                recent_diversity = [p.get('asset_diversity', 0) for p in self.performance_history]
                if np.mean(recent_diversity) < req_value:
                    return False
        
        return True
    
    def _get_next_milestone(self) -> Optional[CurriculumMilestone]:
        """Get the next unlocked milestone."""
        for milestone in self.milestones:
            if not milestone.unlocked and milestone.difficulty > self.current_difficulty:
                return milestone
        return None
    
    def _get_current_milestone(self) -> Optional[CurriculumMilestone]:
        """Get the current milestone based on difficulty."""
        current_milestone = None
        for milestone in self.milestones:
            if milestone.unlocked and milestone.difficulty <= self.current_difficulty:
                current_milestone = milestone
        return current_milestone
    
    def _update_statistics(self, adjustment: Dict[str, Any], milestone_updates: List[Dict[str, Any]]):
        """Update curriculum statistics."""
        self.statistics['total_episodes'] += 1
        self.statistics[f'{adjustment["action"]}_count'] += 1
        
        if milestone_updates:
            self.statistics['milestones_unlocked'] += len(milestone_updates)
        
        # Calculate progression rate
        if len(self.difficulty_history) > 1:
            total_progression = self.current_difficulty - self.config.initial_difficulty
            self.statistics['avg_progression_rate'] = total_progression / len(self.difficulty_history)
    
    def get_current_difficulty(self) -> float:
        """Get current difficulty level."""
        return self.current_difficulty
    
    def get_milestone_bonus(self) -> float:
        """Get current milestone reward bonus."""
        current_milestone = self._get_current_milestone()
        return current_milestone.rewards_bonus if current_milestone else 0.0
    
    def get_curriculum_statistics(self) -> Dict[str, Any]:
        """Get comprehensive curriculum statistics."""
        stats = dict(self.statistics)
        
        # Add current state
        stats.update({
            'current_difficulty': self.current_difficulty,
            'current_episode': self.current_episode,
            'stable_episodes': self.stable_episodes,
            'strategy': self.config.strategy.value,
            'unlocked_milestones': len([m for m in self.milestones if m.unlocked]),
            'total_milestones': len(self.milestones)
        })
        
        # Add performance summary
        if self.performance_history:
            recent_performance = self._analyze_recent_performance()
            stats['recent_performance'] = recent_performance
        
        # Add progression summary
        if len(self.difficulty_history) > 1:
            stats['difficulty_progression'] = {
                'initial': self.config.initial_difficulty,
                'current': self.current_difficulty,
                'max_reached': max(d[1] for d in self.difficulty_history),
                'total_progressions': len(self.progression_events),
                'total_regressions': len(self.regression_events)
            }
        
        return stats
    
    def save_curriculum_analysis(self, filepath: str):
        """Save comprehensive curriculum analysis."""
        analysis = {
            'config': self.config.__dict__,
            'statistics': self.get_curriculum_statistics(),
            'difficulty_history': self.difficulty_history,
            'progression_events': self.progression_events,
            'regression_events': self.regression_events,
            'milestones': [
                {
                    'name': m.name,
                    'difficulty': m.difficulty,
                    'unlocked': m.unlocked,
                    'unlock_episode': m.unlock_episode,
                    'requirements': m.requirements
                }
                for m in self.milestones
            ],
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"Curriculum analysis saved to {filepath}")
    
    def visualize_progress(self, save_path: Optional[str] = None):
        """Create visualization of curriculum progress."""
        if not self.difficulty_history:
            logger.warning("No difficulty history to visualize")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Difficulty progression
        episodes, difficulties = zip(*self.difficulty_history)
        ax1.plot(episodes, difficulties, 'b-', linewidth=2, label='Difficulty')
        ax1.axhline(y=self.config.max_difficulty, color='r', linestyle='--', alpha=0.7, label='Max Difficulty')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Difficulty')
        ax1.set_title('Curriculum Difficulty Progression')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Reward progression
        if self.reward_history:
            ax2.plot(list(self.reward_history), 'g-', linewidth=1, alpha=0.7)
            # Add moving average
            if len(self.reward_history) > 10:
                window = min(20, len(self.reward_history) // 4)
                moving_avg = np.convolve(list(self.reward_history), np.ones(window)/window, mode='valid')
                ax2.plot(range(window-1, len(self.reward_history)), moving_avg, 'g-', linewidth=2, label='Moving Average')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Reward')
            ax2.set_title('Reward Progression')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Success rate over time
        if self.success_history:
            # Calculate moving success rate
            window = min(20, len(self.success_history))
            success_rates = []
            for i in range(window, len(self.success_history) + 1):
                success_rates.append(np.mean(list(self.success_history)[i-window:i]))
            
            ax3.plot(range(window, len(self.success_history) + 1), success_rates, 'orange', linewidth=2)
            ax3.axhline(y=self.config.success_threshold, color='r', linestyle='--', alpha=0.7, label='Success Threshold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate')
            ax3.set_title('Success Rate Progression')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Milestone progress
        milestone_episodes = [m.unlock_episode for m in self.milestones if m.unlocked and m.unlock_episode]
        milestone_difficulties = [m.difficulty for m in self.milestones if m.unlocked and m.unlock_episode]
        milestone_names = [m.name for m in self.milestones if m.unlocked and m.unlock_episode]
        
        if milestone_episodes:
            ax4.scatter(milestone_episodes, milestone_difficulties, c='red', s=100, zorder=5)
            for i, name in enumerate(milestone_names):
                ax4.annotate(name, (milestone_episodes[i], milestone_difficulties[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot all milestones
        for milestone in self.milestones:
            color = 'green' if milestone.unlocked else 'gray'
            alpha = 1.0 if milestone.unlocked else 0.5
            ax4.axhline(y=milestone.difficulty, color=color, linestyle='-', alpha=alpha, linewidth=1)
        
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Difficulty')
        ax4.set_title('Milestone Progress')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curriculum visualization saved to {save_path}")
        
        return fig

# Example usage
if __name__ == "__main__":
    config = CurriculumConfig(
        strategy=CurriculumStrategy.ADAPTIVE,
        initial_difficulty=0.1,
        progression_rate=0.02
    )
    
    curriculum = CurriculumLearningManager(config)
    
    # Simulate training episodes
    for episode in range(100):
        # Simulate episode data
        episode_data = {
            'total_reward': np.random.normal(10 + episode * 0.1, 2),
            'steps': np.random.randint(100, 300),
            'success': np.random.random() < (0.3 + episode * 0.005),
            'nerf_assets_used': set(np.random.choice(['asset1', 'asset2', 'asset3'], 
                                                   size=np.random.randint(0, 4), replace=False)),
            'performance_metrics': {
                'avg_reward_per_step': np.random.uniform(0.05, 0.15),
                'nerf_reward_ratio': np.random.uniform(0.1, 0.3)
            }
        }
        
        update = curriculum.update(episode_data)
        
        if episode % 20 == 0:
            print(f"Episode {episode}: Difficulty {curriculum.get_current_difficulty():.3f}")
    
    # Get statistics
    stats = curriculum.get_curriculum_statistics()
    print(f"Final statistics: {stats}")
    
    # Save analysis
    curriculum.save_curriculum_analysis("curriculum_analysis_v5.json")
    
    # Create visualization
    curriculum.visualize_progress("curriculum_progress_v5.png")

