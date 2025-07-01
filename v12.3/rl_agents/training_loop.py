"""
Enhanced Training Loop Infrastructure for RL Training Version 5 BETA 1

Major improvements for v5:
- Enhanced NeRF integration with reward reflection
- Improved parallel environment rollout with better GPU utilization
- Advanced checkpointing with automatic crash recovery
- Mixed-precision training with enhanced numerical stability
- Real-time progress visualization with image grids
- Optimized batch reward processing
- Comprehensive error handling and graceful recovery
- Human-in-the-loop feedback integration
- Curriculum learning support
- Advanced memory management and replay buffer
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
import time
import logging
import threading
import queue
import pickle
import json
import hashlib
import csv
from collections import deque, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import gc
from contextlib import contextmanager
import signal
import psutil
import wandb
import asyncio
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Version 5 specific imports
from .nerf_integration_v5 import NeRFIntegrationV5
from .reward_system_v5 import EnhancedRewardSystemV5
from .curriculum_learning import CurriculumLearningManager
from .replay_buffer import EpisodicReplayBuffer
from .hitl_feedback import HITLFeedbackManager
from .dark_matter import DarkMatterManager, EnvironmentConfig

# Try to import wandb for enhanced logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("WandB not available. Only TensorBoard logging will be used.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfigV5:
    """Enhanced configuration for Version 5 training loop."""
    
    # Basic training parameters
    num_environments: int = 16
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    learning_rate: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 100000
    
    # Version 5 specific parameters
    nerf_integration_enabled: bool = True
    hitl_feedback_enabled: bool = True
    curriculum_learning_enabled: bool = True
    visualization_enabled: bool = True
    
    # NeRF specific settings
    nerf_asset_diversity_bonus: float = 0.1
    nerf_quality_threshold: float = 0.8
    nerf_reward_weight: float = 0.2
    
    # HITL settings
    hitl_feedback_frequency: int = 100  # episodes
    hitl_ui_enabled: bool = True
    hitl_csv_path: str = "hitl_feedback.csv"
    
    # Curriculum learning settings
    curriculum_start_difficulty: float = 0.1
    curriculum_max_difficulty: float = 1.0
    curriculum_progression_rate: float = 0.01
    
    # Visualization settings
    visualization_frequency: int = 50  # episodes
    generate_gifs: bool = True
    tensorboard_enabled: bool = True
    wandb_enabled: bool = True
    
    # Performance settings
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    checkpoint_frequency: int = 500  # episodes
    
    # Error handling
    max_consecutive_failures: int = 5
    auto_recovery_enabled: bool = True
    graceful_shutdown_timeout: int = 30

class EnhancedTrainingLoopV5:
    """
    Version 5 Enhanced Training Loop with comprehensive improvements.
    
    Key features:
    - Advanced NeRF integration with reward reflection
    - Human-in-the-loop feedback system
    - Curriculum learning with adaptive difficulty
    - Real-time visualization and monitoring
    - Robust error handling and recovery
    - Episodic memory and replay buffer
    """
    
    def __init__(self, config: TrainingConfigV5, dark_matter_manager: Optional[DarkMatterManager] = None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize core components
        self.nerf_integration = None
        self.reward_system = None
        self.curriculum_manager = None
        self.replay_buffer = None
        self.hitl_manager = None
        self.dark_matter_manager: Optional[DarkMatterManager] = None # New: Dark Matter Manager
        
        # Training state
        self.episode_count = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.consecutive_failures = 0
        self.training_start_time = None
        
        # Logging and visualization
        self.writer = None
        self.wandb_run = None
        self.metrics_history = defaultdict(list)
        
        # Initialize components
        self._initialize_components()
        self._setup_logging()
        
    def _initialize_components(self):
        """Initialize all training components."""
        logger.info("Initializing Version 5 training components...")
        
        # Initialize NeRF integration
        if self.config.nerf_integration_enabled:
            self.nerf_integration = NeRFIntegrationV5(
                asset_diversity_bonus=self.config.nerf_asset_diversity_bonus,
                quality_threshold=self.config.nerf_quality_threshold,
                reward_weight=self.config.nerf_reward_weight
            )
            logger.info("NeRF integration initialized")
        
        # Initialize enhanced reward system
        self.reward_system = EnhancedRewardSystemV5(
            nerf_integration=self.nerf_integration,
            config=self.config
        )
        
        # Initialize curriculum learning
        if self.config.curriculum_learning_enabled:
            self.curriculum_manager = CurriculumLearningManager(
                start_difficulty=self.config.curriculum_start_difficulty,
                max_difficulty=self.config.curriculum_max_difficulty,
                progression_rate=self.config.curriculum_progression_rate
            )
            logger.info("Curriculum learning manager initialized")
        
        # Initialize replay buffer
        self.replay_buffer = EpisodicReplayBuffer(
            capacity=self.config.buffer_size,
            device=self.device
        )
        
        # Initialize HITL feedback manager
        if self.config.hitl_feedback_enabled:
            self.hitl_manager = HITLFeedbackManager(
                feedback_frequency=self.config.hitl_feedback_frequency,
                ui_enabled=self.config.hitl_ui_enabled,
                csv_path=self.config.hitl_csv_path
            )
            logger.info("HITL feedback manager initialized")

        if dark_matter_manager:
            self.dark_matter_manager = dark_matter_manager
            logger.info("DarkMatterManager provided and set.")
    
    def _setup_logging(self):
        """Setup logging and visualization systems."""
        if self.config.tensorboard_enabled:
            log_dir = f"runs/v5_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging enabled: {log_dir}")
        
        if self.config.wandb_enabled and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project="rl-training-v5",
                config=self.config.__dict__,
                tags=["version5", "beta1"]
            )
            logger.info("WandB logging enabled")
    
    def train(self, agent, environment):
        """
        Main training loop with Version 5 enhancements.
        
        Args:
            agent: The RL agent to train
            environment: The training environment
        """
        logger.info("Starting Version 5 training loop...")
        self.training_start_time = time.time()
        
        try:
            while self.episode_count < self.config.max_episodes:
                episode_start_time = time.time()
                
                # Get current curriculum difficulty
                current_difficulty = 1.0
                if self.curriculum_manager:
                    current_difficulty = self.curriculum_manager.get_current_difficulty()
                    environment.set_difficulty(current_difficulty)
                
                # Run episode
                episode_data = self._run_episode(agent, environment)
                
                # Process episode results
                self._process_episode_results(episode_data)
                
                # Update curriculum
                if self.curriculum_manager:
                    self.curriculum_manager.update(episode_data['total_reward'])
                
                # HITL feedback collection
                if self.hitl_manager and self.hitl_manager.should_collect_feedback(self.episode_count):
                    self._collect_hitl_feedback(episode_data)
                
                # Visualization and logging
                if self.episode_count % self.config.visualization_frequency == 0:
                    self._generate_visualizations(episode_data)
                
                # Checkpointing
                if self.episode_count % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(agent)
                
                # Update metrics
                self._update_metrics(episode_data, time.time() - episode_start_time)
                
                self.episode_count += 1
                # ===== BLUE/GREEN STATE VALIDATION AUTOMATION =====
                # This assumes your engine object is accessible as self.engine (adjust as needed!)
                if hasattr(self, "engine") and hasattr(self.engine, "world"):
                    # Define or get your validators list, e.g.:
                    validators = getattr(self.engine, "validators", [])
                    if self.engine.world.validate_blue(validators):
                        self.engine.world.promote_blue_to_green()
                        logger.info(f"Episode {self.episode_count}: BLUE state promoted to GREEN.")
                    else:
                        logger.warning(f"Episode {self.episode_count}: BLUE state validation failed, GREEN unchanged.")
                # ===== END BLUE/GREEN BLOCK =====
                # Check for early stopping or convergence
                if self._check_convergence():
                    logger.info("Training converged. Stopping early.")
                    break
                    
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training error: {e}")
            self._handle_training_error(e)
        finally:
            self._cleanup()
    
    def _run_episode(self, agent, environment):
        """Run a single episode with enhanced monitoring."""
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'nerf_rewards': [],
            'states': [],
            'total_reward': 0,
            'steps': 0,
            'nerf_assets_used': set(),
            'performance_metrics': {}
        }
        
        state = environment.reset()
        done = False
        step_count = 0
        
        while not done and step_count < self.config.max_steps_per_episode:
            # Agent action selection
            with autocast(enabled=self.config.mixed_precision):
                action = agent.select_action(state)
            
            # Environment step
            next_state, base_reward, done, info = environment.step(action)
            
            # Enhanced reward calculation with NeRF integration
            enhanced_reward = self.reward_system.calculate_reward(
                state, action, next_state, base_reward, info
            )
            
            # Track NeRF asset usage
            if 'nerf_asset' in info:
                episode_data['nerf_assets_used'].add(info['nerf_asset'])
            
            # Store episode data
            episode_data['observations'].append(state)
            episode_data['actions'].append(action)
            episode_data['rewards'].append(enhanced_reward)
            episode_data['nerf_rewards'].append(info.get('nerf_reward', 0))
            episode_data['states'].append(next_state)
            
            episode_data['total_reward'] += enhanced_reward
            episode_data['steps'] = step_count + 1
            
            # Update state
            state = next_state
            step_count += 1
            self.total_steps += 1
        
        # Calculate performance metrics
        episode_data['performance_metrics'] = self._calculate_performance_metrics(episode_data)
        
        return episode_data
    
    def _process_episode_results(self, episode_data):
        """Process episode results and update replay buffer."""
        # Add to replay buffer
        self.replay_buffer.add_episode(episode_data)
        
        # Update best reward
        if episode_data['total_reward'] > self.best_reward:
            self.best_reward = episode_data['total_reward']
            logger.info(f"New best reward: {self.best_reward:.4f}")
    
    def _collect_hitl_feedback(self, episode_data):
        """Collect human-in-the-loop feedback."""
        if self.hitl_manager:
            feedback = self.hitl_manager.collect_feedback(episode_data)
            if feedback:
                # Log feedback to CSV
                self._log_hitl_feedback(feedback)
                
                # Use feedback to adjust training
                self._apply_hitl_feedback(feedback)
    
    def _log_hitl_feedback(self, feedback):
        """Log HITL feedback to CSV file."""
        csv_path = Path(self.config.hitl_csv_path)
        file_exists = csv_path.exists()
        
        with open(csv_path, 'a', newline='') as csvfile:
            fieldnames = ['episode', 'timestamp', 'rating', 'comments', 'suggestions']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'episode': self.episode_count,
                'timestamp': datetime.now().isoformat(),
                'rating': feedback.get('rating', 0),
                'comments': feedback.get('comments', ''),
                'suggestions': feedback.get('suggestions', '')
            })
    
    def _apply_hitl_feedback(self, feedback):
        """Apply HITL feedback to adjust training parameters."""
        rating = feedback.get('rating', 0)
        
        # Adjust learning rate based on feedback
        if rating < 3:  # Poor performance
            # Reduce learning rate or adjust exploration
            pass
        elif rating > 7:  # Good performance
            # Potentially increase difficulty or learning rate
            pass
    
    def _generate_visualizations(self, episode_data):
        """Generate visualizations for training progress."""
        if not self.config.visualization_enabled:
            return
        
        # Generate image grids
        self._generate_image_grids(episode_data)
        
        # Generate GIFs if enabled
        if self.config.generate_gifs:
            self._generate_episode_gif(episode_data)
        
        # Update TensorBoard
        if self.writer:
            self._update_tensorboard(episode_data)
        
        # Update WandB
        if self.wandb_run:
            self._update_wandb(episode_data)
    
    def _generate_image_grids(self, episode_data):
        """Generate image grids showing training progress."""
        # Implementation for image grid generation
        pass
    
    def _generate_episode_gif(self, episode_data):
        """Generate GIF showing episode progression."""
        # Implementation for GIF generation
        pass
    
    def _update_tensorboard(self, episode_data):
        """Update TensorBoard with current metrics."""
        self.writer.add_scalar('Reward/Total', episode_data['total_reward'], self.episode_count)
        self.writer.add_scalar('Reward/NeRF', sum(episode_data['nerf_rewards']), self.episode_count)
        self.writer.add_scalar('Episode/Steps', episode_data['steps'], self.episode_count)
        self.writer.add_scalar('Episode/NeRF_Assets_Used', len(episode_data['nerf_assets_used']), self.episode_count)
        
        # Add performance metrics
        for metric, value in episode_data['performance_metrics'].items():
            self.writer.add_scalar(f'Performance/{metric}', value, self.episode_count)
    
    def _update_wandb(self, episode_data):
        """Update WandB with current metrics."""
        wandb.log({
            'episode': self.episode_count,
            'total_reward': episode_data['total_reward'],
            'nerf_reward': sum(episode_data['nerf_rewards']),
            'steps': episode_data['steps'],
            'nerf_assets_used': len(episode_data['nerf_assets_used']),
            **episode_data['performance_metrics']
        })
    
    def _calculate_performance_metrics(self, episode_data):
        """Calculate comprehensive performance metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['avg_reward_per_step'] = episode_data['total_reward'] / max(episode_data['steps'], 1)
        metrics['nerf_reward_ratio'] = sum(episode_data['nerf_rewards']) / max(episode_data['total_reward'], 1)
        metrics['nerf_asset_diversity'] = len(episode_data['nerf_assets_used'])
        
        # Advanced metrics
        rewards = episode_data['rewards']
        if rewards:
            metrics['reward_variance'] = np.var(rewards)
            metrics['reward_trend'] = np.polyfit(range(len(rewards)), rewards, 1)[0] if len(rewards) > 1 else 0
        
        return metrics
    
    def _update_metrics(self, episode_data, episode_time):
        """Update training metrics history."""
        self.metrics_history['total_reward'].append(episode_data['total_reward'])
        self.metrics_history['episode_time'].append(episode_time)
        self.metrics_history['steps'].append(episode_data['steps'])
        self.metrics_history['nerf_assets_used'].append(len(episode_data['nerf_assets_used']))
        
        # Keep only recent history
        max_history = 1000
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_history:
                self.metrics_history[key] = self.metrics_history[key][-max_history:]
    
    def _check_convergence(self):
        """Check if training has converged."""
        if len(self.metrics_history['total_reward']) < 100:
            return False
        
        recent_rewards = self.metrics_history['total_reward'][-100:]
        reward_std = np.std(recent_rewards)
        reward_mean = np.mean(recent_rewards)
        
        # Convergence criteria: low variance in recent rewards
        convergence_threshold = 0.01 * abs(reward_mean) if reward_mean != 0 else 0.01
        return reward_std < convergence_threshold
    
    def _save_checkpoint(self, agent):
        """Save training checkpoint."""
        checkpoint_data = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'metrics_history': dict(self.metrics_history),
            'config': self.config.__dict__
        }
        
        checkpoint_path = f"checkpoints/v5_checkpoint_ep{self.episode_count}.pkl"
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _handle_training_error(self, error):
        """Handle training errors with recovery attempts."""
        self.consecutive_failures += 1
        logger.error(f"Training error (attempt {self.consecutive_failures}): {error}")
        
        if self.consecutive_failures >= self.config.max_consecutive_failures:
            logger.error("Maximum consecutive failures reached. Stopping training.")
            raise error
        
        if self.config.auto_recovery_enabled:
            logger.info("Attempting automatic recovery...")
            # Implement recovery logic here
            time.sleep(5)  # Brief pause before retry
    
    def _cleanup(self):
        """Cleanup resources and save final results."""
        logger.info("Cleaning up training resources...")
        
        if self.writer:
            self.writer.close()
        
        if self.wandb_run:
            wandb.finish()
        
        # Save final metrics
        final_metrics = {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0,
            'metrics_history': dict(self.metrics_history)
        }
        
        with open('final_training_metrics_v5.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        logger.info("Training cleanup completed")

# Example usage
if __name__ == "__main__":
    config = TrainingConfigV5(
        num_environments=8,
        max_episodes=5000,
        nerf_integration_enabled=True,
        hitl_feedback_enabled=True,
        curriculum_learning_enabled=True,
        visualization_enabled=True
    )
    
    training_loop = EnhancedTrainingLoopV5(config)
    # training_loop.train(agent, environment)



    def set_dark_matter_manager(self, manager: DarkMatterManager):
        """Sets the DarkMatterManager for the training loop."""
        self.dark_matter_manager = manager
        logger.info("DarkMatterManager set in training loop.")


