"""
Enhanced Reward System for RL Training v3

Improvements based on feedback:
- Configurable reward weighting coefficients from config/CLI
- Comprehensive exception handling for all reward computations
- Outlier reward logging for debugging agent stuck situations
- Incremental reward for steps closer to visual/code/AST match
- Penalty for repetitive or trivial solutions
- Diversity checks on agent outputs
- Anti-gaming measures and detection
"""

import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod
import math
import logging
import json
import hashlib
from dataclasses import dataclass, field
from collections import defaultdict, deque
import warnings
from contextlib import contextmanager

@dataclass
class RewardWeightConfig:
    """Enhanced configuration for reward weighting with validation."""
    # Core reward weights
    gameplay_weight: float = 0.25
    visual_quality_weight: float = 0.30
    code_quality_weight: float = 0.20
    performance_weight: float = 0.15
    learning_progress_weight: float = 0.10
    
    # Sub-component weights for visual quality
    lpips_weight: float = 0.35
    ssim_weight: float = 0.25
    ms_ssim_weight: float = 0.15
    psnr_weight: float = 0.10
    feature_weight: float = 0.10
    edge_weight: float = 0.05
    
    # Anti-gaming weights
    diversity_weight: float = 0.05
    novelty_weight: float = 0.05
    complexity_penalty_weight: float = 0.02
    
    def __post_init__(self):
        """Validate weights and normalize if needed."""
        main_weights = [
            self.gameplay_weight, self.visual_quality_weight, 
            self.code_quality_weight, self.performance_weight, 
            self.learning_progress_weight
        ]
        
        total_main = sum(main_weights)
        if abs(total_main - 1.0) > 1e-6:
            warnings.warn(f"Main reward weights sum to {total_main}, not 1.0. Auto-normalizing.")
            # Auto-normalize
            factor = 1.0 / total_main
            self.gameplay_weight *= factor
            self.visual_quality_weight *= factor
            self.code_quality_weight *= factor
            self.performance_weight *= factor
            self.learning_progress_weight *= factor

@dataclass
class RewardConfig:
    """Enhanced reward system configuration."""
    weights: RewardWeightConfig = field(default_factory=RewardWeightConfig)
    
    # Exception handling
    enable_exception_handling: bool = True
    fallback_reward: float = 0.0
    max_retries: int = 3
    
    # Outlier detection and logging
    enable_outlier_logging: bool = True
    outlier_threshold_low: float = -2.0
    outlier_threshold_high: float = 2.0
    outlier_window_size: int = 100
    
    # Anti-gaming measures
    enable_diversity_checks: bool = True
    enable_repetition_penalty: bool = True
    enable_trivial_solution_detection: bool = True
    
    # Incremental rewards
    enable_incremental_rewards: bool = True
    incremental_reward_scale: float = 0.1
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    reward_computation_timeout: float = 5.0  # seconds

class SafeRewardCalculator(ABC):
    """Enhanced base class with comprehensive error handling."""
    
    def __init__(self, config: RewardConfig, name: str):
        self.config = config
        self.name = name
        self.weight = getattr(config.weights, f"{name}_weight", 1.0)
        self.enabled = True
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Performance tracking
        self.computation_times = deque(maxlen=1000)
        self.error_count = 0
        self.success_count = 0
        
        # Outlier tracking
        self.reward_history = deque(maxlen=config.outlier_window_size)
    
    @abstractmethod
    def _calculate_core(self, state: Dict[str, Any], action: Dict[str, Any], 
                       result: Dict[str, Any]) -> float:
        """Core calculation method to be implemented by subclasses."""
        pass
    
    def calculate(self, state: Dict[str, Any], action: Dict[str, Any], 
                 result: Dict[str, Any]) -> float:
        """Safe calculation with error handling and monitoring."""
        if not self.enabled:
            return 0.0
        
        start_time = time.time()
        
        try:
            with self._timeout_handler():
                reward = self._calculate_with_retries(state, action, result)
                
                # Apply weight
                weighted_reward = reward * self.weight
                
                # Track for outlier detection
                self.reward_history.append(weighted_reward)
                
                # Log outliers if enabled
                if self.config.enable_outlier_logging:
                    self._check_and_log_outliers(weighted_reward, state, action, result)
                
                # Update performance stats
                computation_time = time.time() - start_time
                self.computation_times.append(computation_time)
                self.success_count += 1
                
                return weighted_reward
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Reward calculation failed for {self.name}: {e}")
            
            if self.config.enable_exception_handling:
                return self.config.fallback_reward
            else:
                raise
    
    def _calculate_with_retries(self, state: Dict[str, Any], action: Dict[str, Any], 
                               result: Dict[str, Any]) -> float:
        """Calculate with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                return self._calculate_core(state, action, result)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {self.name}: {e}")
                
                if attempt < self.config.max_retries - 1:
                    # Brief pause before retry
                    time.sleep(0.1 * (attempt + 1))
        
        # All retries failed
        raise last_exception
    
    @contextmanager
    def _timeout_handler(self):
        """Context manager for timeout handling."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Reward calculation timeout for {self.name}")
        
        if self.config.reward_computation_timeout > 0:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.reward_computation_timeout))
            
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            yield
    
    def _check_and_log_outliers(self, reward: float, state: Dict[str, Any], 
                               action: Dict[str, Any], result: Dict[str, Any]):
        """Check for and log outlier rewards."""
        if len(self.reward_history) < 10:
            return  # Need sufficient history
        
        recent_rewards = list(self.reward_history)[-10:]
        mean_reward = np.mean(recent_rewards)
        std_reward = np.std(recent_rewards)
        
        # Z-score based outlier detection
        if std_reward > 0:
            z_score = (reward - mean_reward) / std_reward
            
            if (z_score < self.config.outlier_threshold_low or 
                z_score > self.config.outlier_threshold_high):
                
                self.logger.warning(
                    f"Outlier reward detected for {self.name}: "
                    f"reward={reward:.4f}, z_score={z_score:.2f}, "
                    f"mean={mean_reward:.4f}, std={std_reward:.4f}"
                )
                
                # Log additional context for debugging
                self.logger.debug(f"Outlier context - State keys: {list(state.keys())}")
                self.logger.debug(f"Outlier context - Action keys: {list(action.keys())}")
                self.logger.debug(f"Outlier context - Result keys: {list(result.keys())}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this calculator."""
        avg_time = np.mean(self.computation_times) if self.computation_times else 0
        
        return {
            'name': self.name,
            'enabled': self.enabled,
            'weight': self.weight,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.success_count + self.error_count, 1),
            'avg_computation_time': avg_time,
            'recent_rewards_mean': np.mean(self.reward_history) if self.reward_history else 0,
            'recent_rewards_std': np.std(self.reward_history) if self.reward_history else 0
        }

class EnhancedGameplayRewardCalculator(SafeRewardCalculator):
    """Enhanced gameplay reward calculator with incremental rewards."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config, "gameplay")
        self.previous_scores = deque(maxlen=100)
    
    def _calculate_core(self, state: Dict[str, Any], action: Dict[str, Any], 
                       result: Dict[str, Any]) -> float:
        """Calculate gameplay-based reward with incremental improvements."""
        reward = 0.0
        game_stats = state.get('game_stats', {})
        
        # Base rewards for successful actions
        if result.get('success', False):
            reward += 10.0
        else:
            reward -= 5.0
        
        # Incremental reward for progress towards goals
        if self.config.enable_incremental_rewards:
            reward += self._calculate_incremental_reward(state, action, result)
        
        # Enemy elimination with scaling
        enemies_killed = game_stats.get('enemies_killed_this_step', 0)
        enemy_types = game_stats.get('enemy_types_killed', {})
        
        # Different rewards for different enemy types
        for enemy_type, count in enemy_types.items():
            type_multiplier = {'basic': 1.0, 'elite': 2.0, 'boss': 5.0}.get(enemy_type, 1.0)
            reward += count * 20.0 * type_multiplier
        
        # Penalty for losing lives with escalation
        lives_lost = game_stats.get('lives_lost_this_step', 0)
        if lives_lost > 0:
            # Escalating penalty for multiple life losses
            penalty = lives_lost * 100.0 * (1.5 ** (lives_lost - 1))
            reward -= penalty
        
        # Wave completion with bonus for efficiency
        if game_stats.get('wave_completed_this_step', False):
            base_bonus = 200.0
            efficiency_bonus = game_stats.get('wave_efficiency', 1.0) * 100.0
            reward += base_bonus + efficiency_bonus
        
        # Resource management with sophisticated scoring
        gold_ratio = state.get('player_state', {}).get('gold', 0) / max(state.get('starting_gold', 100), 1)
        if gold_ratio > 0.7:  # Excellent resource management
            reward += 15.0
        elif gold_ratio > 0.5:  # Good resource management
            reward += 5.0
        elif gold_ratio < 0.1:  # Poor resource management
            reward -= 20.0
        
        # Strategic depth reward
        strategy_score = self._calculate_strategy_score(state, action)
        reward += strategy_score
        
        return self._normalize_reward(reward, -500.0, 800.0)
    
    def _calculate_incremental_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                                    result: Dict[str, Any]) -> float:
        """Calculate incremental reward for progress towards objectives."""
        incremental_reward = 0.0
        
        # Progress towards wave completion
        wave_progress = state.get('wave_progress', 0.0)  # 0.0 to 1.0
        if hasattr(self, '_last_wave_progress'):
            progress_delta = wave_progress - self._last_wave_progress
            if progress_delta > 0:
                incremental_reward += progress_delta * 50.0
        self._last_wave_progress = wave_progress
        
        # Progress in tower placement efficiency
        placement_efficiency = result.get('placement_efficiency', 0.0)
        if placement_efficiency > 0.8:
            incremental_reward += 10.0
        elif placement_efficiency > 0.6:
            incremental_reward += 5.0
        
        # Progress in resource optimization
        resource_efficiency = result.get('resource_efficiency', 0.0)
        incremental_reward += resource_efficiency * 15.0
        
        return incremental_reward * self.config.incremental_reward_scale
    
    def _calculate_strategy_score(self, state: Dict[str, Any], action: Dict[str, Any]) -> float:
        """Calculate reward for strategic thinking."""
        strategy_score = 0.0
        
        # Reward for diverse tower types
        tower_types = state.get('tower_types_placed', {})
        diversity_bonus = min(len(tower_types) * 5.0, 25.0)
        strategy_score += diversity_bonus
        
        # Reward for optimal tower positioning
        positioning_score = action.get('positioning_optimality', 0.0)
        strategy_score += positioning_score * 20.0
        
        # Reward for timing of upgrades
        upgrade_timing = action.get('upgrade_timing_score', 0.0)
        strategy_score += upgrade_timing * 15.0
        
        return strategy_score
    
    def _normalize_reward(self, reward: float, min_val: float, max_val: float) -> float:
        """Normalize reward to [0, 1] range."""
        return (reward - min_val) / (max_val - min_val)

class EnhancedVisualQualityRewardCalculator(SafeRewardCalculator):
    """Enhanced visual quality calculator with configurable metrics."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config, "visual_quality")
        self.visual_weights = config.weights
    
    def _calculate_core(self, state: Dict[str, Any], action: Dict[str, Any], 
                       result: Dict[str, Any]) -> float:
        """Calculate visual quality reward with configurable weighting."""
        reward = 0.0
        visual_metrics = result.get('visual_metrics', {})
        
        # Configurable visual metric weights
        lpips_score = visual_metrics.get('lpips_score', 1.0)
        ssim_score = visual_metrics.get('ssim_score', 0.0)
        ms_ssim_score = visual_metrics.get('ms_ssim_score', 0.0)
        psnr_score = visual_metrics.get('psnr_score', 0.0)
        feature_similarity = visual_metrics.get('feature_similarity', 0.0)
        edge_similarity = visual_metrics.get('edge_similarity', 0.0)
        
        # Apply configurable weights
        reward += (1.0 - lpips_score) * 100.0 * self.visual_weights.lpips_weight
        reward += ssim_score * 100.0 * self.visual_weights.ssim_weight
        reward += ms_ssim_score * 100.0 * self.visual_weights.ms_ssim_weight
        reward += min(psnr_score / 40.0, 1.0) * 100.0 * self.visual_weights.psnr_weight
        reward += feature_similarity * 100.0 * self.visual_weights.feature_weight
        reward += edge_similarity * 100.0 * self.visual_weights.edge_weight
        
        # Frame rate consistency with adaptive thresholds
        fps = visual_metrics.get('fps', 60.0)
        target_fps = state.get('target_fps', 60.0)
        
        if fps >= target_fps:
            reward += 20.0
        elif fps >= target_fps * 0.8:
            reward += 10.0
        elif fps < target_fps * 0.5:
            reward -= 30.0
        
        # Rendering quality with detailed breakdown
        render_quality = visual_metrics.get('render_quality', 0.5)
        reward += render_quality * 50.0
        
        # Visual coherence and consistency
        coherence = visual_metrics.get('visual_coherence', 0.5)
        reward += coherence * 40.0
        
        # Incremental visual improvement
        if self.config.enable_incremental_rewards:
            improvement_reward = self._calculate_visual_improvement(visual_metrics)
            reward += improvement_reward
        
        return self._normalize_reward(reward, -100.0, 400.0)
    
    def _calculate_visual_improvement(self, visual_metrics: Dict[str, Any]) -> float:
        """Calculate incremental reward for visual improvements."""
        improvement_reward = 0.0
        
        # Track improvement in key metrics
        current_quality = visual_metrics.get('overall_quality', 0.0)
        
        if hasattr(self, '_last_visual_quality'):
            quality_improvement = current_quality - self._last_visual_quality
            if quality_improvement > 0:
                improvement_reward += quality_improvement * 50.0
        
        self._last_visual_quality = current_quality
        
        return improvement_reward * self.config.incremental_reward_scale
    
    def _normalize_reward(self, reward: float, min_val: float, max_val: float) -> float:
        """Normalize reward to [0, 1] range."""
        return (reward - min_val) / (max_val - min_val)

class AntiGamingRewardCalculator(SafeRewardCalculator):
    """Specialized calculator for detecting and penalizing gaming behaviors."""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config, "anti_gaming")
        self.action_history = deque(maxlen=1000)
        self.output_hashes = deque(maxlen=500)
        self.trivial_patterns = set()
        
    def _calculate_core(self, state: Dict[str, Any], action: Dict[str, Any], 
                       result: Dict[str, Any]) -> float:
        """Calculate penalties for gaming behaviors."""
        penalty = 0.0
        
        # Track current action and output
        action_hash = self._hash_action(action)
        output_hash = self._hash_output(result)
        
        self.action_history.append(action_hash)
        self.output_hashes.append(output_hash)
        
        # Repetition penalty
        if self.config.enable_repetition_penalty:
            penalty += self._calculate_repetition_penalty()
        
        # Trivial solution detection
        if self.config.enable_trivial_solution_detection:
            penalty += self._detect_trivial_solutions(result)
        
        # Diversity penalty
        if self.config.enable_diversity_checks:
            penalty += self._calculate_diversity_penalty()
        
        return -penalty  # Return negative penalty as reward reduction
    
    def _hash_action(self, action: Dict[str, Any]) -> str:
        """Create hash of action for repetition detection."""
        # Simplified action hashing
        action_str = json.dumps(action, sort_keys=True, default=str)
        return hashlib.md5(action_str.encode()).hexdigest()[:16]
    
    def _hash_output(self, result: Dict[str, Any]) -> str:
        """Create hash of output for diversity checking."""
        # Focus on key output characteristics
        output_features = {
            'code_length': len(str(result.get('generated_code', ''))),
            'visual_complexity': result.get('visual_complexity', 0),
            'performance_score': round(result.get('performance_score', 0), 2)
        }
        
        output_str = json.dumps(output_features, sort_keys=True)
        return hashlib.md5(output_str.encode()).hexdigest()[:16]
    
    def _calculate_repetition_penalty(self) -> float:
        """Calculate penalty for repetitive actions."""
        if len(self.action_history) < 10:
            return 0.0
        
        recent_actions = list(self.action_history)[-10:]
        unique_actions = len(set(recent_actions))
        
        # Penalty increases as diversity decreases
        repetition_ratio = 1.0 - (unique_actions / len(recent_actions))
        penalty = repetition_ratio * 50.0
        
        # Extra penalty for identical consecutive actions
        consecutive_identical = 0
        for i in range(1, len(recent_actions)):
            if recent_actions[i] == recent_actions[i-1]:
                consecutive_identical += 1
        
        penalty += consecutive_identical * 10.0
        
        return penalty
    
    def _detect_trivial_solutions(self, result: Dict[str, Any]) -> float:
        """Detect and penalize trivial solutions."""
        penalty = 0.0
        
        # Check for minimal code solutions
        code = result.get('generated_code', '')
        if len(code.strip()) < 10:
            penalty += 30.0
            self.logger.warning("Trivial solution detected: minimal code")
        
        # Check for blank or near-blank outputs
        visual_complexity = result.get('visual_complexity', 0)
        if visual_complexity < 0.1:
            penalty += 25.0
            self.logger.warning("Trivial solution detected: minimal visual output")
        
        # Check for copy-paste solutions
        if self._is_likely_copy_paste(result):
            penalty += 40.0
            self.logger.warning("Trivial solution detected: likely copy-paste")
        
        return penalty
    
    def _is_likely_copy_paste(self, result: Dict[str, Any]) -> bool:
        """Detect if result is likely a copy-paste solution."""
        code = result.get('generated_code', '')
        
        # Simple heuristics for copy-paste detection
        if len(code) > 100:
            # Check for common copy-paste patterns
            copy_paste_indicators = [
                'TODO', 'FIXME', 'placeholder', 'example',
                'lorem ipsum', 'test test', 'hello world'
            ]
            
            code_lower = code.lower()
            for indicator in copy_paste_indicators:
                if indicator in code_lower:
                    return True
        
        return False
    
    def _calculate_diversity_penalty(self) -> float:
        """Calculate penalty for lack of output diversity."""
        if len(self.output_hashes) < 20:
            return 0.0
        
        recent_outputs = list(self.output_hashes)[-20:]
        unique_outputs = len(set(recent_outputs))
        
        # Penalty for low diversity
        diversity_ratio = unique_outputs / len(recent_outputs)
        if diversity_ratio < 0.5:
            penalty = (0.5 - diversity_ratio) * 60.0
            self.logger.warning(f"Low output diversity detected: {diversity_ratio:.2f}")
            return penalty
        
        return 0.0

class EnhancedCompositeRewardSystem:
    """Enhanced composite reward system with comprehensive improvements."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize calculators
        self.calculators = {}
        self._initialize_calculators()
        
        # Performance monitoring
        self.total_calculations = 0
        self.total_errors = 0
        self.calculation_times = deque(maxlen=1000)
        
        # Reward history for analysis
        self.reward_history = deque(maxlen=10000)
        
        self.logger.info("Enhanced composite reward system initialized")
    
    def _initialize_calculators(self):
        """Initialize all reward calculators with error handling."""
        try:
            self.calculators['gameplay'] = EnhancedGameplayRewardCalculator(self.config)
            self.calculators['visual_quality'] = EnhancedVisualQualityRewardCalculator(self.config)
            
            if self.config.enable_diversity_checks or self.config.enable_repetition_penalty:
                self.calculators['anti_gaming'] = AntiGamingRewardCalculator(self.config)
            
            self.logger.info(f"Initialized {len(self.calculators)} reward calculators")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize reward calculators: {e}")
            if not self.config.enable_exception_handling:
                raise
    
    def calculate_total_reward(self, state: Dict[str, Any], action: Dict[str, Any], 
                              result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate total reward with comprehensive error handling and monitoring.
        
        Returns:
            Dictionary with individual rewards, total, and metadata
        """
        start_time = time.time()
        
        try:
            rewards = {}
            total_reward = 0.0
            errors = []
            
            # Calculate rewards from each calculator
            for name, calculator in self.calculators.items():
                try:
                    reward = calculator.calculate(state, action, result)
                    rewards[name] = reward
                    total_reward += reward
                    
                except Exception as e:
                    error_msg = f"Calculator {name} failed: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
                    
                    if self.config.enable_exception_handling:
                        rewards[name] = self.config.fallback_reward
                    else:
                        raise
            
            # Apply global normalization
            rewards['total'] = total_reward
            rewards['normalized_total'] = self._normalize_total_reward(total_reward)
            
            # Add metadata
            calculation_time = time.time() - start_time
            self.calculation_times.append(calculation_time)
            self.total_calculations += 1
            
            if errors:
                self.total_errors += len(errors)
            
            # Store in history
            reward_record = {
                'timestamp': start_time,
                'rewards': rewards.copy(),
                'errors': errors,
                'calculation_time': calculation_time
            }
            self.reward_history.append(reward_record)
            
            # Add performance metadata
            rewards['_metadata'] = {
                'calculation_time': calculation_time,
                'errors': errors,
                'total_calculations': self.total_calculations,
                'error_rate': self.total_errors / max(self.total_calculations, 1)
            }
            
            return rewards
            
        except Exception as e:
            self.logger.error(f"Total reward calculation failed: {e}")
            self.total_errors += 1
            
            if self.config.enable_exception_handling:
                return {
                    'total': self.config.fallback_reward,
                    'normalized_total': 0.0,
                    '_metadata': {'error': str(e)}
                }
            else:
                raise
    
    def _normalize_total_reward(self, total_reward: float) -> float:
        """Normalize total reward to a standard range."""
        # Use adaptive normalization based on recent history
        if len(self.reward_history) >= 100:
            recent_rewards = [r['rewards']['total'] for r in list(self.reward_history)[-100:]]
            mean_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            
            if std_reward > 0:
                # Z-score normalization
                normalized = (total_reward - mean_reward) / std_reward
                # Clamp to reasonable range
                return np.clip(normalized, -3.0, 3.0)
        
        # Fallback to simple clipping
        return np.clip(total_reward, -1.0, 1.0)
    
    def update_weights(self, weight_updates: Dict[str, float]):
        """Update calculator weights with validation."""
        for name, weight in weight_updates.items():
            if name in self.calculators:
                old_weight = self.calculators[name].weight
                self.calculators[name].weight = weight
                self.logger.info(f"Updated {name} weight: {old_weight:.3f} -> {weight:.3f}")
            else:
                self.logger.warning(f"Unknown calculator for weight update: {name}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all calculators."""
        stats = {
            'system_stats': {
                'total_calculations': self.total_calculations,
                'total_errors': self.total_errors,
                'error_rate': self.total_errors / max(self.total_calculations, 1),
                'avg_calculation_time': np.mean(self.calculation_times) if self.calculation_times else 0,
                'config': {
                    'exception_handling': self.config.enable_exception_handling,
                    'outlier_logging': self.config.enable_outlier_logging,
                    'diversity_checks': self.config.enable_diversity_checks,
                    'incremental_rewards': self.config.enable_incremental_rewards
                }
            },
            'calculator_stats': {}
        }
        
        # Get stats from each calculator
        for name, calculator in self.calculators.items():
            stats['calculator_stats'][name] = calculator.get_performance_stats()
        
        # Recent reward statistics
        if self.reward_history:
            recent_totals = [r['rewards']['total'] for r in list(self.reward_history)[-100:]]
            stats['recent_rewards'] = {
                'mean': np.mean(recent_totals),
                'std': np.std(recent_totals),
                'min': np.min(recent_totals),
                'max': np.max(recent_totals),
                'count': len(recent_totals)
            }
        
        return stats
    
    def export_reward_history(self, filepath: str):
        """Export reward history for analysis."""
        try:
            history_data = {
                'config': self.config.__dict__,
                'history': list(self.reward_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(history_data, f, indent=2, default=str)
                
            self.logger.info(f"Exported reward history to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export reward history: {e}")

# Factory function for easy creation
def create_enhanced_reward_system(
    weights: Optional[Dict[str, float]] = None,
    enable_exception_handling: bool = True,
    enable_outlier_logging: bool = True,
    enable_diversity_checks: bool = True,
    enable_incremental_rewards: bool = True
) -> EnhancedCompositeRewardSystem:
    """
    Factory function to create enhanced reward system.
    
    Args:
        weights: Custom weight configuration
        enable_exception_handling: Enable comprehensive error handling
        enable_outlier_logging: Enable outlier detection and logging
        enable_diversity_checks: Enable anti-gaming diversity checks
        enable_incremental_rewards: Enable incremental progress rewards
        
    Returns:
        Configured EnhancedCompositeRewardSystem
    """
    config = RewardConfig(
        enable_exception_handling=enable_exception_handling,
        enable_outlier_logging=enable_outlier_logging,
        enable_diversity_checks=enable_diversity_checks,
        enable_incremental_rewards=enable_incremental_rewards
    )
    
    # Apply custom weights if provided
    if weights:
        for key, value in weights.items():
            if hasattr(config.weights, key):
                setattr(config.weights, key, value)
    
    return EnhancedCompositeRewardSystem(config)

# Import time for timeout handling
import time

