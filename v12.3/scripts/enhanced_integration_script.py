#!/usr/bin/env python3
"""
Enhanced RL Training Integration Script v3

This script demonstrates the integration of all enhanced components:
- Enhanced configuration management
- Enhanced training loop with parallel processing
- Enhanced transformer agent
- Enhanced visual assessment and reward system
- Enhanced rendering pipeline
- Enhanced testing framework

Improvements based on feedback:
- Centralized configuration management
- Better error handling and recovery
- Performance monitoring and optimization
- Comprehensive logging and debugging
"""

import sys
import os
import logging
import time
import signal
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import enhanced components
from enhanced_config_manager import (
    ConfigurationManager, ConfigEnvironment, get_config_manager,
    get_training_config, get_environment_config, get_render_config
)
from training_loop_enhanced import (
    EnhancedTrainingLoop, EnhancedTrainingConfig, create_enhanced_training_loop
)
from transformer_agent_enhanced import (
    EnhancedTransformerRLAgent, EnhancedAgentConfig, create_enhanced_transformer_agent
)
from visual_assessment_gpu_enhanced import (
    EnhancedVisualAssessmentGPU, create_enhanced_visual_assessor
)
from reward_system_enhanced import (
    EnhancedRewardSystem, create_enhanced_reward_system
)
from threejs_renderer_enhanced import (
    EnhancedThreeJSRenderer, create_enhanced_renderer
)
from async_rendering_pipeline_enhanced import (
    EnhancedAsyncRenderingPipeline, create_enhanced_pipeline
)
from enhanced_test_framework import (
    EnhancedTestFramework, TestConfig
)

class EnhancedRLTrainingSystem:
    """
    Enhanced RL Training System integrating all improved components.
    """
    
    def __init__(self, environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT):
        self.environment = environment
        self.logger = self._setup_logging()
        
        # Initialize configuration manager
        self.config_manager = get_config_manager(environment=environment)
        self.config = self.config_manager.get_config()
        
        # Initialize components
        self.training_loop = None
        self.agent = None
        self.visual_assessor = None
        self.reward_system = None
        self.renderer = None
        self.rendering_pipeline = None
        
        # System state
        self.is_running = False
        self.current_episode = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info(f"Enhanced RL Training System initialized for {environment.value}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        logger = logging.getLogger(__name__)
        
        # Configure logging after config is loaded
        if hasattr(self, 'config'):
            log_config = self.config.logging
            logger.setLevel(getattr(logging, log_config.level))
            
            # Create log directory
            log_dir = Path(log_config.log_dir)
            log_dir.mkdir(exist_ok=True)
            
            # File handler
            if log_config.save_logs:
                log_file = log_dir / f"training_{int(time.time())}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
        
        return logger
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown()
    
    def initialize_components(self):
        """Initialize all system components."""
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize visual assessor
            self.visual_assessor = create_enhanced_visual_assessor(
                use_gpu=True,
                enable_caching=True,
                cache_size=1000
            )
            self.logger.info("✅ Visual assessor initialized")
            
            # Initialize reward system
            reward_config = self.config.reward
            self.reward_system = create_enhanced_reward_system(
                gameplay_weight=reward_config.gameplay_weight,
                visual_weight=reward_config.visual_quality_weight,
                code_weight=reward_config.code_quality_weight,
                performance_weight=reward_config.performance_weight,
                enable_diversity_bonus=reward_config.enable_diversity_bonus
            )
            self.logger.info("✅ Reward system initialized")
            
            # Initialize renderer
            render_config = self.config.render
            self.renderer = create_enhanced_renderer(
                use_headless_gpu=render_config.use_headless_gpu,
                enable_context_pool=render_config.enable_context_pool,
                enable_mock_renderer=render_config.enable_mock_renderer,
                pool_size=render_config.pool_size,
                render_timeout=render_config.render_timeout
            )
            self.logger.info("✅ Renderer initialized")
            
            # Initialize rendering pipeline
            self.rendering_pipeline = create_enhanced_pipeline(
                min_workers=2,
                max_workers=render_config.pool_size * 2,
                enable_proxy_renderer=render_config.enable_mock_renderer,
                enable_render_cache=True,
                enable_profiling=True
            )
            self.logger.info("✅ Rendering pipeline initialized")
            
            # Initialize agent
            training_config = self.config.training
            env_config = self.config.environment
            
            self.agent = create_enhanced_transformer_agent(
                state_dim=env_config.grid_size[0] * env_config.grid_size[1] + 10,  # Grid + metadata
                action_dim=len(env_config.tower_types) + 3,  # Tower types + actions
                code_vocab_size=10000,
                image_feature_dim=512
            )
            self.logger.info("✅ Agent initialized")
            
            # Initialize training loop
            enhanced_training_config = EnhancedTrainingConfig(
                max_episodes=training_config.max_episodes,
                max_steps_per_episode=training_config.max_steps_per_episode,
                batch_size=training_config.batch_size,
                learning_rate=training_config.learning_rate,
                num_parallel_envs=training_config.num_parallel_envs,
                use_mixed_precision=training_config.use_mixed_precision,
                checkpoint_frequency=training_config.checkpoint_frequency,
                auto_resume=training_config.auto_resume,
                checkpoint_dir=training_config.checkpoint_dir
            )
            
            # Create dummy environment factory for now
            def env_factory():
                # This would create the actual environment
                return DummyEnvironment()
            
            # Create dummy optimizer
            import torch
            optimizer = torch.optim.Adam(self.agent.parameters(), lr=training_config.learning_rate)
            
            self.training_loop = EnhancedTrainingLoop(
                enhanced_training_config,
                env_factory,
                self.agent,
                self.reward_system,
                optimizer
            )
            self.logger.info("✅ Training loop initialized")
            
            self.logger.info("🎉 All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def run_training(self):
        """Run the enhanced training process."""
        try:
            self.logger.info("Starting enhanced RL training...")
            self.is_running = True
            
            # Set episode for rendering pipeline
            self.rendering_pipeline.set_episode(0)
            
            # Start training
            self.training_loop.train()
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
        finally:
            self.is_running = False
    
    def run_tests(self, test_mode: str = "smoke"):
        """Run enhanced testing framework."""
        try:
            self.logger.info(f"Running {test_mode} tests...")
            
            # Configure test framework
            test_config = TestConfig(
                smoke_test_mode=(test_mode == "smoke"),
                full_test_mode=(test_mode == "full"),
                performance_test_mode=(test_mode == "performance"),
                enable_random_agent_tests=True,
                verbose=True
            )
            
            # Run tests
            test_framework = EnhancedTestFramework(test_config)
            report = test_framework.run_all_tests()
            
            # Save and display results
            test_framework.save_report(report)
            
            self.logger.info("Test Results Summary:")
            self.logger.info(f"  Total Tests: {report['summary']['total_tests']}")
            self.logger.info(f"  Passed: {report['summary']['passed']}")
            self.logger.info(f"  Failed: {report['summary']['failed']}")
            self.logger.info(f"  Success Rate: {report['summary']['success_rate']:.1%}")
            
            return report['summary']['failed'] == 0
            
        except Exception as e:
            self.logger.error(f"Testing failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "environment": self.environment.value,
            "is_running": self.is_running,
            "current_episode": self.current_episode,
            "configuration": self.config_manager.get_config_summary(),
            "components": {
                "visual_assessor": self.visual_assessor is not None,
                "reward_system": self.reward_system is not None,
                "renderer": self.renderer is not None,
                "rendering_pipeline": self.rendering_pipeline is not None,
                "agent": self.agent is not None,
                "training_loop": self.training_loop is not None
            }
        }
        
        # Add performance stats if available
        if self.renderer:
            try:
                status["renderer_stats"] = self.renderer.get_performance_stats()
            except:
                pass
        
        if self.rendering_pipeline:
            try:
                status["pipeline_stats"] = self.rendering_pipeline.get_performance_stats()
            except:
                pass
        
        return status
    
    def shutdown(self):
        """Gracefully shutdown the system."""
        try:
            self.logger.info("Shutting down enhanced RL training system...")
            self.is_running = False
            
            # Cleanup components
            if self.training_loop:
                self.training_loop.cleanup()
            
            if self.renderer:
                self.renderer.cleanup()
            
            if self.rendering_pipeline:
                self.rendering_pipeline.cleanup()
            
            self.logger.info("✅ System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

class DummyEnvironment:
    """Dummy environment for testing."""
    
    def __init__(self):
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        return {"state": [0] * 10}
    
    def step(self, action):
        self.step_count += 1
        next_state = {"state": [self.step_count] * 10}
        reward = 1.0
        done = self.step_count >= 10
        info = {"step": self.step_count}
        return next_state, reward, done, info
    
    def close(self):
        pass

def main():
    """Main entry point for enhanced RL training system."""
    parser = argparse.ArgumentParser(description="Enhanced RL Training System v3")
    parser.add_argument("--env", choices=["development", "testing", "staging", "production"],
                       default="development", help="Environment to run in")
    parser.add_argument("--mode", choices=["train", "test", "status"],
                       default="train", help="Mode to run in")
    parser.add_argument("--test-type", choices=["smoke", "full", "performance"],
                       default="smoke", help="Type of tests to run")
    parser.add_argument("--create-configs", action="store_true",
                       help="Create default configuration files")
    
    args = parser.parse_args()
    
    try:
        # Get environment
        environment = ConfigEnvironment(args.env)
        
        # Create default configs if requested
        if args.create_configs:
            config_manager = ConfigurationManager(environment=environment)
            config_manager.create_default_configs()
            print("✅ Default configuration files created")
            return 0
        
        # Initialize system
        system = EnhancedRLTrainingSystem(environment)
        
        if args.mode == "status":
            # Show system status
            status = system.get_system_status()
            print("🔍 System Status:")
            print(f"  Environment: {status['environment']}")
            print(f"  Running: {status['is_running']}")
            print(f"  Components Initialized: {sum(status['components'].values())}/{len(status['components'])}")
            return 0
        
        elif args.mode == "test":
            # Run tests
            system.initialize_components()
            success = system.run_tests(args.test_type)
            return 0 if success else 1
        
        elif args.mode == "train":
            # Run training
            system.initialize_components()
            system.run_training()
            return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

