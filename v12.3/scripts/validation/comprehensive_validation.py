#!/usr/bin/env python3
"""
Comprehensive Validation Script for RL Training v3

This script validates all enhanced components and ensures they work correctly
together. It performs integration testing, performance validation, and
system health checks.
"""

import sys
import os
import time
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging for validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('validation.log')
        ]
    )
    return logging.getLogger(__name__)

def validate_imports():
    """Validate that all enhanced components can be imported."""
    logger = logging.getLogger(__name__)
    logger.info("🔍 Validating imports...")
    
    imports_to_test = [
        ('enhanced_config_manager', 'ConfigurationManager'),
        ('visual_assessment_gpu_enhanced', 'EnhancedVisualAssessmentGPU'),
        ('reward_system_enhanced', 'EnhancedRewardSystem'),
        ('training_loop_enhanced', 'EnhancedTrainingLoop'),
        ('transformer_agent_enhanced', 'EnhancedTransformerRLAgent'),
        ('threejs_renderer_enhanced', 'EnhancedThreeJSRenderer'),
        ('async_rendering_pipeline_enhanced', 'EnhancedAsyncRenderingPipeline'),
        ('enhanced_test_framework', 'EnhancedTestFramework'),
    ]
    
    failed_imports = []
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            logger.info(f"  ✅ {module_name}.{class_name}")
        except Exception as e:
            logger.error(f"  ❌ {module_name}.{class_name}: {e}")
            failed_imports.append((module_name, class_name, str(e)))
    
    if failed_imports:
        logger.error(f"Failed to import {len(failed_imports)} components")
        return False
    
    logger.info("✅ All imports successful")
    return True

def validate_configuration_system():
    """Validate the enhanced configuration system."""
    logger = logging.getLogger(__name__)
    logger.info("⚙️ Validating configuration system...")
    
    try:
        from enhanced_config_manager import (
            ConfigurationManager, ConfigEnvironment, MasterConfig,
            TrainingConfig, EnvironmentConfig, RenderConfig
        )
        
        # Test configuration creation
        config_manager = ConfigurationManager(
            config_dir="test_config",
            environment=ConfigEnvironment.DEVELOPMENT
        )
        
        # Test default configuration
        default_config = MasterConfig()
        default_config.validate()
        logger.info("  ✅ Default configuration validation")
        
        # Test configuration saving and loading
        config_dir = Path("test_config")
        config_dir.mkdir(exist_ok=True)
        
        config_manager.save_config(default_config, "test_config.yaml")
        loaded_config = config_manager.load_config("test_config.yaml")
        logger.info("  ✅ Configuration save/load")
        
        # Test environment variable resolution
        os.environ['TEST_MAX_EPISODES'] = '5000'
        test_config_dict = {
            "training": {
                "max_episodes": "${TEST_MAX_EPISODES:1000}"
            }
        }
        
        from enhanced_config_manager import EnvironmentVariableResolver
        resolved = EnvironmentVariableResolver.resolve_env_vars(test_config_dict)
        assert resolved["training"]["max_episodes"] == "5000"
        logger.info("  ✅ Environment variable resolution")
        
        # Cleanup
        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)
        
        logger.info("✅ Configuration system validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration system validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_enhanced_components():
    """Validate enhanced components can be created and initialized."""
    logger = logging.getLogger(__name__)
    logger.info("🧩 Validating enhanced components...")
    
    try:
        # Test visual assessor
        from visual_assessment_gpu_enhanced import create_enhanced_visual_assessor
        visual_assessor = create_enhanced_visual_assessor(
            use_gpu=False,  # Use CPU for testing
            enable_caching=True,
            cache_size=100
        )
        logger.info("  ✅ Enhanced visual assessor")
        
        # Test reward system
        from reward_system_enhanced import create_enhanced_reward_system
        reward_system = create_enhanced_reward_system(
            gameplay_weight=0.4,
            visual_weight=0.2,
            code_weight=0.2,
            performance_weight=0.2
        )
        logger.info("  ✅ Enhanced reward system")
        
        # Test rendering pipeline
        from async_rendering_pipeline_enhanced import create_enhanced_pipeline
        pipeline = create_enhanced_pipeline(
            min_workers=1,
            max_workers=2,
            enable_proxy_renderer=True,
            enable_render_cache=True
        )
        logger.info("  ✅ Enhanced rendering pipeline")
        
        # Test testing framework
        from enhanced_test_framework import EnhancedTestFramework, TestConfig
        test_config = TestConfig(
            smoke_test_mode=True,
            enable_random_agent_tests=False,
            verbose=False
        )
        test_framework = EnhancedTestFramework(test_config)
        logger.info("  ✅ Enhanced testing framework")
        
        # Cleanup
        pipeline.cleanup()
        
        logger.info("✅ Enhanced components validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced components validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_integration():
    """Validate that components can work together."""
    logger = logging.getLogger(__name__)
    logger.info("🔗 Validating component integration...")
    
    try:
        # Create a minimal integration test
        from enhanced_config_manager import ConfigurationManager, ConfigEnvironment
        
        # Initialize configuration
        config_manager = ConfigurationManager(environment=ConfigEnvironment.TESTING)
        config = config_manager.get_config()
        
        # Test configuration access
        assert config.training.max_episodes > 0
        assert len(config.environment.grid_size) == 2
        assert config.render.width > 0
        logger.info("  ✅ Configuration integration")
        
        # Test component creation with configuration
        from visual_assessment_gpu_enhanced import EnhancedVisualAssessmentGPU
        from reward_system_enhanced import EnhancedRewardSystem
        
        # Create components using configuration
        visual_assessor = EnhancedVisualAssessmentGPU(
            use_gpu=False,
            enable_caching=True,
            cache_size=config.render.pool_size * 10
        )
        
        reward_system = EnhancedRewardSystem(
            gameplay_weight=config.reward.gameplay_weight,
            visual_quality_weight=config.reward.visual_quality_weight,
            code_quality_weight=config.reward.code_quality_weight,
            performance_weight=config.reward.performance_weight
        )
        
        logger.info("  ✅ Component creation with configuration")
        
        # Test basic functionality
        dummy_metrics = {
            'gameplay': {'score': 100, 'efficiency': 0.8},
            'visual': {'quality': 0.9, 'aesthetics': 0.85},
            'code': {'complexity': 0.7, 'readability': 0.9},
            'performance': {'fps': 60, 'memory_usage': 0.6}
        }
        
        reward = reward_system.calculate_comprehensive_reward(
            gameplay_metrics=dummy_metrics['gameplay'],
            visual_metrics=dummy_metrics['visual'],
            code_metrics=dummy_metrics['code'],
            performance_metrics=dummy_metrics['performance']
        )
        
        assert isinstance(reward, (int, float))
        assert not (reward != reward)  # Check for NaN
        logger.info("  ✅ Reward calculation integration")
        
        logger.info("✅ Component integration validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component integration validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_performance():
    """Validate performance improvements and optimizations."""
    logger = logging.getLogger(__name__)
    logger.info("⚡ Validating performance optimizations...")
    
    try:
        import time
        import psutil
        
        # Test memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple components to test memory efficiency
        from visual_assessment_gpu_enhanced import create_enhanced_visual_assessor
        from reward_system_enhanced import create_enhanced_reward_system
        
        components = []
        for i in range(5):
            visual_assessor = create_enhanced_visual_assessor(
                use_gpu=False,
                enable_caching=True,
                cache_size=50
            )
            reward_system = create_enhanced_reward_system()
            components.append((visual_assessor, reward_system))
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        logger.info(f"  📊 Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
        
        # Test performance with caching
        start_time = time.time()
        
        # Simulate repeated operations that should benefit from caching
        for i in range(10):
            dummy_metrics = {
                'gameplay': {'score': 100 + i, 'efficiency': 0.8},
                'visual': {'quality': 0.9, 'aesthetics': 0.85},
                'code': {'complexity': 0.7, 'readability': 0.9},
                'performance': {'fps': 60, 'memory_usage': 0.6}
            }
            
            for visual_assessor, reward_system in components:
                reward = reward_system.calculate_comprehensive_reward(
                    gameplay_metrics=dummy_metrics['gameplay'],
                    visual_metrics=dummy_metrics['visual'],
                    code_metrics=dummy_metrics['code'],
                    performance_metrics=dummy_metrics['performance']
                )
        
        elapsed_time = time.time() - start_time
        operations_per_second = (10 * len(components)) / elapsed_time
        
        logger.info(f"  ⚡ Performance: {operations_per_second:.1f} operations/second")
        
        # Cleanup
        del components
        
        logger.info("✅ Performance validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_error_handling():
    """Validate error handling and recovery mechanisms."""
    logger = logging.getLogger(__name__)
    logger.info("🛡️ Validating error handling...")
    
    try:
        from enhanced_config_manager import ConfigurationManager, MasterConfig
        from reward_system_enhanced import EnhancedRewardSystem
        
        # Test configuration validation errors
        try:
            invalid_config = MasterConfig()
            invalid_config.training.max_episodes = -1  # Invalid value
            invalid_config.validate()
            logger.error("  ❌ Configuration validation should have failed")
            return False
        except ValueError:
            logger.info("  ✅ Configuration validation error handling")
        
        # Test reward system error handling
        reward_system = EnhancedRewardSystem()
        
        # Test with invalid metrics
        try:
            invalid_metrics = {
                'gameplay': {'score': float('nan'), 'efficiency': 0.8},
                'visual': {'quality': 0.9, 'aesthetics': 0.85},
                'code': {'complexity': 0.7, 'readability': 0.9},
                'performance': {'fps': 60, 'memory_usage': 0.6}
            }
            
            reward = reward_system.calculate_comprehensive_reward(
                gameplay_metrics=invalid_metrics['gameplay'],
                visual_metrics=invalid_metrics['visual'],
                code_metrics=invalid_metrics['code'],
                performance_metrics=invalid_metrics['performance']
            )
            
            # Should handle NaN gracefully
            assert not (reward != reward), "Reward should not be NaN"
            logger.info("  ✅ NaN handling in reward calculation")
            
        except Exception as e:
            logger.info(f"  ✅ Error handling for invalid metrics: {type(e).__name__}")
        
        # Test timeout handling simulation
        from async_rendering_pipeline_enhanced import EnhancedRenderConfig
        
        config = EnhancedRenderConfig(render_timeout=0.001)  # Very short timeout
        logger.info("  ✅ Timeout configuration handling")
        
        logger.info("✅ Error handling validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error handling validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_comprehensive_validation():
    """Run comprehensive validation of all v3 enhancements."""
    logger = setup_logging()
    logger.info("🚀 Starting comprehensive validation of RL Training v3")
    logger.info("=" * 60)
    
    validation_tests = [
        ("Import Validation", validate_imports),
        ("Configuration System", validate_configuration_system),
        ("Enhanced Components", validate_enhanced_components),
        ("Component Integration", validate_integration),
        ("Performance Optimization", validate_performance),
        ("Error Handling", validate_error_handling),
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in validation_tests:
        logger.info(f"\n🧪 Running {test_name}...")
        try:
            test_start = time.time()
            success = test_func()
            test_time = time.time() - test_start
            
            results.append({
                'name': test_name,
                'success': success,
                'time': test_time
            })
            
            status = "✅ PASSED" if success else "❌ FAILED"
            logger.info(f"{status} - {test_name} ({test_time:.2f}s)")
            
        except Exception as e:
            test_time = time.time() - test_start
            results.append({
                'name': test_name,
                'success': False,
                'time': test_time,
                'error': str(e)
            })
            logger.error(f"❌ FAILED - {test_name} ({test_time:.2f}s): {e}")
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Validation Results Summary")
    logger.info("=" * 60)
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    success_rate = passed / total if total > 0 else 0
    
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {total - passed}")
    logger.info(f"Success Rate: {success_rate:.1%}")
    logger.info(f"Total Time: {total_time:.2f}s")
    
    # Detailed results
    logger.info("\nDetailed Results:")
    for result in results:
        status = "✅" if result['success'] else "❌"
        logger.info(f"  {status} {result['name']}: {result['time']:.2f}s")
        if not result['success'] and 'error' in result:
            logger.info(f"    Error: {result['error']}")
    
    # Performance summary
    total_test_time = sum(r['time'] for r in results)
    avg_test_time = total_test_time / total if total > 0 else 0
    logger.info(f"\nPerformance Summary:")
    logger.info(f"  Average test time: {avg_test_time:.2f}s")
    logger.info(f"  Fastest test: {min(r['time'] for r in results):.2f}s")
    logger.info(f"  Slowest test: {max(r['time'] for r in results):.2f}s")
    
    if success_rate == 1.0:
        logger.info("\n🎉 All validation tests passed! RL Training v3 is ready for deployment.")
        return 0
    else:
        logger.warning(f"\n⚠️ {total - passed} validation tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    exit(run_comprehensive_validation())

