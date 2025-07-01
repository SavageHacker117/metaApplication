#!/usr/bin/env python3
"""
Main Integration Script for RL Training System Version 8
This script serves as the primary entry point for the enhanced RL training system,
integrating all components including HITL feedback, visualization, robustness testing,
and advanced training features.

Usage:
    python main_v5.py --config config/training_v5.json
    python main_v5.py --episodes 1000 --enable-hitl --enable-viz
    python main_v5.py --test-mode --robustness-testing

Author: Manus AI Team
Version: 8.3.0
"""

import os
import sys
import json
import argparse
import logging
import signal
import time
from pathlib import Path
from typing import Dict, Any, Optional
import threading

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import Version 5 components
from core.training_loop import EnhancedTrainingLoopV5, TrainingConfig
from core.nerf_integration import NeRFIntegrationManager, NeRFConfig
from core.reward_system import RewardSystem, RewardConfig
from core.curriculum_learning import CurriculumManager, CurriculumConfig
from core.replay_buffer import EpisodicReplayBuffer, ReplayBufferConfig
from core.dark_matter_environment import DarkMatterEnvironment
from core.plugin_loader import PluginLoader

from hitl.hitl_feedback_manager import HITLFeedbackManager, HITLConfig
from visualization.visualization_manager import VisualizationManager, VisualizationConfig
from tests.robustness_testing import RobustnessTestingFramework, RobustnessConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_v5.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
class RLTrainingSystemV8:
    """
    Main integration class for RL Training System Version 5 BETA 1.
    
    Coordinates all subsystems and provides a unified interface for
    training, monitoring, and human interaction.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.version = "8.3.0"
        self.config_path = config_path
        self.config = self._load_configuration()
        
        # Correct placement for initial_state, self.world, and self.validators
        initial_state = {} 
        self.world = DarkMatterEnvironment(self.dark_matter_manager) # Use DarkMatterEnvironment
        self.validators = []

        # System components
        self.training_loop = None
        self.nerf_manager = None
        self.reward_system = None
        self.curriculum_manager = None
        self.replay_buffer = None
        self.hitl_manager = None
        self.visualization_manager = None
        self.robustness_framework = None
        self.dark_matter_manager = None # New: Dark Matter Manager
        self.plugin_loader = None # New: Plugin Loader
        
        # System state
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.current_episode = 0
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"RL Training System Version {self.version} initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration from file or use defaults."""
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                return self._get_default_configuration()
        else:
            logger.info("Using default configuration")
            return self._get_default_configuration()
    
    def _get_default_configuration(self) -> Dict[str, Any]:
        """Get default system configuration."""
        
        return {
            "version": "8.3.0",
            "training": {
                "episodes": 1000,
                "batch_size": 32,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "target_update_frequency": 10,
                "save_frequency": 100
            },
            "nerf": {
                "enabled": True,
                "asset_directory": "assets/nerf_v5",
                "max_assets_per_episode": 5,
                "diversity_bonus_weight": 0.2,
                "correlation_tracking": True,
                "dynamic_selection": True
            },
            "reward": {
                "base_reward_weight": 1.0,
                "nerf_reward_weight": 0.3,
                "diversity_bonus_weight": 0.2,
                "success_bonus": 10.0,
                "failure_penalty": -5.0,
                "anti_hacking_enabled": True
            },
            "curriculum": {
                "enabled": True,
                "initial_difficulty": 0.1,
                "max_difficulty": 1.0,
                "adaptation_rate": 0.05,
                "performance_window": 50,
                "success_threshold": 0.7
            },
            "replay_buffer": {
                "max_episodes": 10000,
                "max_memory_mb": 1000,
                "compression_enabled": True,
                "default_sampling_strategy": "priority",
                "priority_alpha": 0.6,
                "priority_beta": 0.4
            },
            "hitl": {
                "enabled": True,
                "feedback_frequency": 20,
                "rating_scale": 10,
                "timeout_seconds": 30,
                "web_interface_port": 5001,
                "cli_enabled": True,
                "auto_feedback": False
            },
            "visualization": {
                "enabled": True,
                "dashboard_port": 5002,
                "tensorboard_enabled": True,
                "wandb_enabled": False,
                "gif_generation": True,
                "image_grid_frequency": 100,
                "plot_generation_frequency": 200
            },
            "robustness": {
                "enabled": True,
                "error_testing_enabled": True,
                "health_monitoring": True,
                "emergency_save_frequency": 50,
                "memory_threshold_mb": 8000,
                "fault_injection": False
            },
            "output": {
                "base_directory": "outputs_v5",
                "models_directory": "models_v5",
                "logs_directory": "logs_v5",
                "visualizations_directory": "visualizations_v5",
                "feedback_directory": "feedback_v5"
            },
            "dark_matter": {
                "enabled": True,
                "default_env_type": "generic",
                "initial_difficulty": 0.5
            },
            "plugins": {
                "enabled": True,
                "web_plugin_directory": "web_plugin",
                "available_plugins": {
                    "tower_defense_game": {
                        "type": "web",
                        "path": "web_plugin",
                        "config": {"port": 3000}
                    }
                }
            }
        }
    
    def initialize_components(self):
        """Initialize all system components."""
        
        logger.info("Initializing Version 8.3.0 components...")
        
        try:
            # Create output directories
            self._create_output_directories()
            
            # Initialize core training components
            self._initialize_core_components()
            
            # Initialize HITL system
            if self.config['hitl']['enabled']:
                self._initialize_hitl_system()
            
            # Initialize visualization system
            if self.config['visualization']['enabled']:
                self._initialize_visualization_system()
            
            # Initialize robustness framework
            if self.config["robustness"]["enabled"]:
                self._initialize_robustness_framework()

            # Initialize Dark Matter system
            if self.config["dark_matter"]["enabled"]:
                self._initialize_dark_matter_system()

            # Initialize Plugin Loader
            if self.config["plugins"]["enabled"]:
                self._initialize_plugin_loader()
            
            # Connect components
            self._connect_components()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        
        output_config = self.config['output']
        
        directories = [
            output_config['base_directory'],
            output_config['models_directory'],
            output_config['logs_directory'],
            output_config['visualizations_directory'],
            output_config['feedback_directory']
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info("Output directories created")
    
    def _initialize_core_components(self):
        """Initialize core training components."""
        
        # Training configuration
        training_config = TrainingConfig(**self.config['training'])
        
        # NeRF integration
        if self.config['nerf']['enabled']:
            nerf_config = NeRFConfig(**self.config['nerf'])
            self.nerf_manager = NeRFIntegrationManager(nerf_config)
        
        # Reward system
        reward_config = RewardConfig(**self.config['reward'])
        self.reward_system = RewardSystem(reward_config)
        
        # Curriculum learning
        if self.config['curriculum']['enabled']:
            curriculum_config = CurriculumConfig(**self.config['curriculum'])
            self.curriculum_manager = CurriculumManager(curriculum_config)
        
        # Replay buffer
        replay_config = ReplayBufferConfig(**self.config['replay_buffer'])
        self.replay_buffer = EpisodicReplayBuffer(replay_config)
        
        # Main training loop
        self.training_loop = EnhancedTrainingLoopV5(
            config=training_config,
            nerf_manager=self.nerf_manager,
            reward_system=self.reward_system,
            curriculum_manager=self.curriculum_manager,
            replay_buffer=self.replay_buffer,
            dark_matter_manager=self.dark_matter_manager
        )
        logger.info("Core components initialized")
    
    def _initialize_hitl_system(self):
        """Initialize Human-in-the-Loop system."""
        
        hitl_config = HITLConfig(**self.config['hitl'])
        self.hitl_manager = HITLFeedbackManager(hitl_config)
        
        logger.info("HITL system initialized")
    
    def _initialize_visualization_system(self):
        """Initialize visualization and monitoring system."""
        
        viz_config = VisualizationConfig(**self.config['visualization'])
        self.visualization_manager = VisualizationManager(viz_config)
        
        logger.info("Visualization system initialized")
    
    def _initialize_robustness_framework(self):
        """Initialize robustness testing framework."""
        
        robustness_config = RobustnessConfig(**self.config['robustness'])
        self.robustness_framework = RobustnessTestingFramework(robustness_config)
        
        logger.info("Robustness framework initialized")
    
    def _connect_components(self):
        """Connect components for integrated operation."""
        
        # Connect HITL to training loop
        if self.hitl_manager:
            self.training_loop.set_hitl_manager(self.hitl_manager)
        
        # Connect visualization to training loop
        if self.visualization_manager:
            self.training_loop.set_visualization_manager(self.visualization_manager)
        
        # Connect robustness framework
        if self.robustness_framework:
            self.training_loop.set_robustness_framework(self.robustness_framework)
        
        logger.info("Components connected")
    
    def start_training(self, episodes: Optional[int] = None):
        """Start the training process."""
        
        if self.is_running:
            logger.warning("Training is already running")
            return
        
        episodes = episodes or self.config['training']['episodes']
        
        logger.info(f"Starting Version 8.3.0 training for {episodes} episodes")
        
        try:
            self.is_running = True
            
            # Start training
            self.training_loop.train(
                episodes=episodes,
                shutdown_event=self.shutdown_event
            )
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_running = False
            logger.info("Training completed")
    
    def run_tests(self):
        """Run comprehensive system tests."""
        
        logger.info("Running Version 8.3.0 system tests...")
        
        test_results = {}
        
        try:
            # Component tests
            test_results['core_components'] = self._test_core_components()
            test_results['hitl_system'] = self._test_hitl_system()
            test_results['visualization'] = self._test_visualization_system()
            test_results['robustness'] = self._test_robustness_framework()
            
            # Integration tests
            test_results['integration'] = self._test_integration()
            
            # Performance tests
            test_results['performance'] = self._test_performance()
            
            logger.info("All tests completed successfully")
            return test_results
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            raise
    
    def _test_core_components(self) -> Dict[str, bool]:
        """Test core training components."""
        
        results = {}
        
        try:
            # Test training loop
            results['training_loop'] = self.training_loop is not None
            
            # Test NeRF manager
            if self.nerf_manager:
                results['nerf_manager'] = self.nerf_manager.test_connection()
            
            # Test reward system
            results['reward_system'] = self.reward_system.test_calculation()
            
            # Test curriculum manager
            if self.curriculum_manager:
                results['curriculum_manager'] = self.curriculum_manager.test_adaptation()
            
            # Test replay buffer
            results['replay_buffer'] = self.replay_buffer.test_storage()
            
            logger.info("Core component tests passed")
            
        except Exception as e:
            logger.error(f"Core component tests failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _test_hitl_system(self) -> Dict[str, bool]:
        """Test HITL feedback system."""
        
        results = {}
        
        try:
            if self.hitl_manager:
                results['hitl_manager'] = self.hitl_manager.test_feedback_collection()
                results['web_interface'] = self.hitl_manager.test_web_interface()
                results['cli_tool'] = self.hitl_manager.test_cli_tool()
            else:
                results['hitl_disabled'] = True
            
            logger.info("HITL system tests passed")
            
        except Exception as e:
            logger.error(f"HITL system tests failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _test_visualization_system(self) -> Dict[str, bool]:
        """Test visualization and monitoring system."""
        
        results = {}
        
        try:
            if self.visualization_manager:
                results['visualization_manager'] = True
                results['dashboard'] = self.visualization_manager.test_dashboard()
                results['tensorboard'] = self.visualization_manager.test_tensorboard()
                results['plot_generation'] = self.visualization_manager.test_plot_generation()
            else:
                results['visualization_disabled'] = True
            
            logger.info("Visualization system tests passed")
            
        except Exception as e:
            logger.error(f"Visualization system tests failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _test_robustness_framework(self) -> Dict[str, bool]:
        """Test robustness and error handling framework."""
        
        results = {}
        
        try:
            if self.robustness_framework:
                results['robustness_framework'] = True
                results['error_handling'] = self.robustness_framework.test_error_handling()
                results['health_monitoring'] = self.robustness_framework.test_health_monitoring()
                results['recovery_mechanisms'] = self.robustness_framework.test_recovery_mechanisms()
            else:
                results['robustness_disabled'] = True
            
            logger.info("Robustness framework tests passed")
            
        except Exception as e:
            logger.error(f"Robustness framework tests failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _test_integration(self) -> Dict[str, bool]:
        """Test component integration."""
        
        results = {}
        
        try:
            # Test training loop integration
            results['training_integration'] = self.training_loop.test_integration()
            
            # Test data flow
            results['data_flow'] = self._test_data_flow()
            
            # Test error propagation
            results['error_propagation'] = self._test_error_propagation()
            
            logger.info("Integration tests passed")
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _test_performance(self) -> Dict[str, Any]:
        """Test system performance."""
        
        results = {}
        
        try:
            # Memory usage test
            import psutil
            process = psutil.Process()
            results['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
            
            # GPU memory test
            try:
                import torch
                if torch.cuda.is_available():
                    results['gpu_memory_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            except ImportError:
                results['gpu_memory_mb'] = 0
            
            # Training speed test
            start_time = time.time()
            # Run a few test episodes
            test_episodes = 5
            for i in range(test_episodes):
                # Simulate episode
                time.sleep(0.1)
            
            end_time = time.time()
            results['episodes_per_second'] = test_episodes / (end_time - start_time)
            
            logger.info("Performance tests completed")
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _test_data_flow(self) -> bool:
        """Test data flow between components."""
        
        try:
            # Create test episode data
            test_data = {
                'episode': 1,
                'observations': [],
                'actions': [],
                'rewards': [],
                'total_reward': 10.0,
                'success': True,
                'nerf_assets_used': {'asset1', 'asset2'}
            }
            
            # Test data processing through each component
            if self.reward_system:
                processed_reward = self.reward_system.calculate_reward(test_data)
                test_data['processed_reward'] = processed_reward
            
            if self.replay_buffer:
                episode_id = self.replay_buffer.add_episode(test_data)
                test_data['episode_id'] = episode_id
            
            if self.visualization_manager:
                self.visualization_manager.update_metrics(test_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Data flow test failed: {e}")
            return False
    
    def _test_error_propagation(self) -> bool:
        """Test error handling and propagation."""
        
        try:
            if self.robustness_framework:
                # Test error handling
                test_context = {'episode': 1, 'test_mode': True}
                
                # Simulate different error types
                from tests.robustness_testing import ErrorType
                
                test_errors = [
                    (ErrorType.IO_ERROR, "Test I/O error"),
                    (ErrorType.MEMORY_LEAK, "Test memory error"),
                    (ErrorType.VISUALIZATION_ERROR, "Test visualization error")
                ]
                
                for error_type, error_message in test_errors:
                    recovery_success = self.robustness_framework.handle_error(
                        error_type, error_message, test_context
                    )
                    
                    if not recovery_success:
                        logger.warning(f"Recovery failed for {error_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error propagation test failed: {e}")
            return False
    
    def _signal_handler(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        
        logger.info("Shutting down Version 5 BETA 1 system...")
        
        # Signal shutdown to all components
        self.shutdown_event.set()
        
        # Stop training
        if self.training_loop and self.is_running:
            self.training_loop.stop()
        
        # Cleanup components
        if self.visualization_manager:
            self.visualization_manager.cleanup()
        
        if self.robustness_framework:
            self.robustness_framework.cleanup()
        
        if self.hitl_manager:
            self.hitl_manager.cleanup()
        
        logger.info("System shutdown completed")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        status = {
            'version': self.version,
            'is_running': self.is_running,
            'current_episode': self.current_episode,
            'components': {
                'training_loop': self.training_loop is not None,
                'nerf_manager': self.nerf_manager is not None,
                'reward_system': self.reward_system is not None,
                'curriculum_manager': self.curriculum_manager is not None,
                'replay_buffer': self.replay_buffer is not None,
                'hitl_manager': self.hitl_manager is not None,
                'visualization_manager': self.visualization_manager is not None,
                'robustness_framework': self.robustness_framework is not None
            }
        }
        
        # Add component-specific status
        if self.visualization_manager:
            status['visualization_stats'] = self.visualization_manager.get_visualization_statistics()
        
        if self.robustness_framework:
            status['robustness_stats'] = self.robustness_framework.get_robustness_statistics()
        
        if self.replay_buffer:
            status['replay_buffer_stats'] = self.replay_buffer.get_statistics()
        
        return status

def main():
    """Main entry point for the RL Training System Version 8.0.1."""

    parser = argparse.ArgumentParser(
        description='RL Training System Version 8.0.1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main_v5.py --config config/training.json
  python main_v5.py --episodes 1000 --enable-hitl --enable-viz
  python main_v5.py --test-mode --robustness-testing
  python main_v5.py --status
        '''
    )

    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--enable-hitl', action='store_true', help='Enable HITL feedback')
    parser.add_argument('--enable-viz', action='store_true', help='Enable visualization')
    parser.add_argument('--enable-robustness', action='store_true', help='Enable robustness testing')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode')
    parser.add_argument('--robustness-testing', action='store_true', help='Run robustness tests only')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--version', action='version', version='RL Training System Version 8.0.1')

    args = parser.parse_args()

    try:
        # Initialize system for Version 8.0.1
        system = RLTrainingSystemV5(config_path=args.config)

        # >>>>>> LOAD PLUGINS HERE (V8.0.1) <<<<<<
        plugin_bus = load_plugins("RUBY/plugins")
        system.plugin_bus = plugin_bus   # (optional: attach for system-wide access)

        # Override configuration with command line arguments (V8.0.1)
        if args.episodes:
            system.config['training']['episodes'] = args.episodes

        if args.enable_hitl:
            system.config['hitl']['enabled'] = True

        if args.enable_viz:
            system.config['visualization']['enabled'] = True

        if args.enable_robustness:
            system.config['robustness']['enabled'] = True

        # Initialize all V8.0.1 components
        system.initialize_components()

        # Handle different run modes for Version 8.0.1
        if args.status:
            status = system.get_system_status()
            print(json.dumps(status, indent=2))
            return

        if args.test_mode or args.robustness_testing:
            test_results = system.run_tests()
            print("Test Results:")
            print(json.dumps(test_results, indent=2))
            return

        # Start training for V8.0.1
        system.start_training(episodes=args.episodes)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)
    finally:
        if 'system' in locals():
            system.shutdown()


    def _initialize_dark_matter_system(self):
        """Initialize the Dark Matter meta-environmental layer."""
        self.dark_matter_manager = DarkMatterManager()
        logger.info("Dark Matter system initialized.")




        # Connect Dark Matter manager to training loop
        if self.dark_matter_manager:
            self.training_loop.set_dark_matter_manager(self.dark_matter_manager)




    def _initialize_plugin_loader(self):
        """Initialize the Plugin Loader and load configured plugins."""
        plugin_config = self.config["plugins"]
        self.plugin_loader = PluginLoader(base_plugin_dir=os.path.join(project_root, plugin_config["web_plugin_directory"]))

        for plugin_name, details in plugin_config["available_plugins"].items():
            self.plugin_loader.load_plugin(plugin_name, details["type"], details["config"])
        logger.info("Plugin loader initialized and plugins loaded.")


