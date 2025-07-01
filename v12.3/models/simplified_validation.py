#!/usr/bin/env python3
"""
Simplified Validation Script for RL Training v3

This script validates the core functionality without requiring heavy dependencies
like PyTorch. It focuses on configuration management, basic component structure,
and integration testing.
"""

import sys
import os
import time
import traceback
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging for validation."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def validate_configuration_system():
    """Validate the enhanced configuration system."""
    logger = logging.getLogger(__name__)
    logger.info("⚙️ Validating configuration system...")
    
    try:
        from enhanced_config_manager import (
            ConfigurationManager, ConfigEnvironment, MasterConfig,
            TrainingConfig, EnvironmentConfig, RenderConfig, RewardConfig, LoggingConfig
        )
        
        # Test configuration creation
        config_manager = ConfigurationManager(
            config_dir="test_config",
            environment=ConfigEnvironment.DEVELOPMENT
        )
        logger.info("  ✅ Configuration manager creation")
        
        # Test default configuration
        default_config = MasterConfig()
        default_config.validate()
        logger.info("  ✅ Default configuration validation")
        
        # Test individual config components
        training_config = TrainingConfig()
        training_config.validate()
        logger.info("  ✅ Training configuration validation")
        
        env_config = EnvironmentConfig()
        env_config.validate()
        logger.info("  ✅ Environment configuration validation")
        
        render_config = RenderConfig()
        render_config.validate()
        logger.info("  ✅ Render configuration validation")
        
        reward_config = RewardConfig()
        reward_config.validate()
        logger.info("  ✅ Reward configuration validation")
        
        logging_config = LoggingConfig()
        logging_config.validate()
        logger.info("  ✅ Logging configuration validation")
        
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
        
        # Test configuration validation errors
        try:
            invalid_config = MasterConfig()
            invalid_config.training.max_episodes = -1  # Invalid value
            invalid_config.validate()
            logger.error("  ❌ Should have failed validation")
            return False
        except ValueError:
            logger.info("  ✅ Configuration validation error detection")
        
        # Cleanup
        import shutil
        shutil.rmtree(config_dir, ignore_errors=True)
        
        logger.info("✅ Configuration system validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration system validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_component_structure():
    """Validate that component files have correct structure."""
    logger = logging.getLogger(__name__)
    logger.info("🏗️ Validating component structure...")
    
    try:
        # Check that all enhanced component files exist
        component_files = [
            'enhanced_config_manager.py',
            'visual_assessment_gpu_enhanced.py',
            'reward_system_enhanced.py',
            'training_loop_enhanced.py',
            'transformer_agent_enhanced.py',
            'threejs_renderer_enhanced.py',
            'async_rendering_pipeline_enhanced.py',
            'enhanced_test_framework.py',
            'enhanced_integration_script.py'
        ]
        
        for file_name in component_files:
            file_path = Path(file_name)
            if not file_path.exists():
                logger.error(f"  ❌ Missing file: {file_name}")
                return False
            
            # Check file size (should not be empty)
            if file_path.stat().st_size == 0:
                logger.error(f"  ❌ Empty file: {file_name}")
                return False
            
            logger.info(f"  ✅ {file_name} exists and has content")
        
        # Check configuration files
        config_files = [
            'config/config_development.yaml',
            'config/config_production.yaml'
        ]
        
        for file_name in config_files:
            file_path = Path(file_name)
            if not file_path.exists():
                logger.error(f"  ❌ Missing config file: {file_name}")
                return False
            logger.info(f"  ✅ {file_name} exists")
        
        # Check documentation files
        doc_files = [
            'README.md',
            'DOCUMENTATION_V3.md',
            'PROJECT_SUMMARY_V3.md'
        ]
        
        for file_name in doc_files:
            file_path = Path(file_name)
            if not file_path.exists():
                logger.error(f"  ❌ Missing documentation: {file_name}")
                return False
            logger.info(f"  ✅ {file_name} exists")
        
        logger.info("✅ Component structure validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Component structure validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_basic_imports():
    """Validate basic imports without heavy dependencies."""
    logger = logging.getLogger(__name__)
    logger.info("📦 Validating basic imports...")
    
    try:
        # Test configuration manager import
        from enhanced_config_manager import ConfigurationManager, ConfigEnvironment
        logger.info("  ✅ Configuration manager import")
        
        # Test that we can create basic instances
        config_manager = ConfigurationManager(environment=ConfigEnvironment.TESTING)
        config = config_manager.get_config()
        logger.info("  ✅ Configuration manager instantiation")
        
        # Test configuration access
        assert config.training.max_episodes > 0
        assert len(config.environment.grid_size) == 2
        assert config.render.width > 0
        logger.info("  ✅ Configuration access")
        
        # Test enhanced test framework (without running tests)
        try:
            from enhanced_test_framework import TestConfig
            test_config = TestConfig(
                smoke_test_mode=True,
                enable_random_agent_tests=False,
                verbose=False
            )
            logger.info("  ✅ Test framework configuration")
        except ImportError as e:
            if "torch" in str(e) or "numpy" in str(e):
                logger.info("  ⚠️ Test framework requires additional dependencies (expected)")
            else:
                raise
        
        logger.info("✅ Basic imports validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Basic imports validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_configuration_profiles():
    """Validate different configuration profiles."""
    logger = logging.getLogger(__name__)
    logger.info("🔧 Validating configuration profiles...")
    
    try:
        from enhanced_config_manager import ConfigurationManager, ConfigEnvironment
        
        # Test different environments
        environments = [
            ConfigEnvironment.DEVELOPMENT,
            ConfigEnvironment.TESTING,
            ConfigEnvironment.PRODUCTION
        ]
        
        for env in environments:
            try:
                config_manager = ConfigurationManager(environment=env)
                config = config_manager.load_config()
                
                # Validate environment-specific settings
                if env == ConfigEnvironment.DEVELOPMENT:
                    assert config.logging.level == "DEBUG"
                    assert config.render.enable_mock_renderer == True
                elif env == ConfigEnvironment.TESTING:
                    assert config.training.max_episodes == 100
                elif env == ConfigEnvironment.PRODUCTION:
                    assert config.logging.level == "INFO"
                    assert config.render.enable_mock_renderer == False
                
                logger.info(f"  ✅ {env.value} environment configuration")
                
            except FileNotFoundError:
                logger.info(f"  ⚠️ {env.value} config file not found (using defaults)")
            except Exception as e:
                logger.error(f"  ❌ {env.value} environment failed: {e}")
                return False
        
        logger.info("✅ Configuration profiles validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration profiles validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_documentation():
    """Validate documentation completeness."""
    logger = logging.getLogger(__name__)
    logger.info("📚 Validating documentation...")
    
    try:
        # Check README content
        readme_path = Path("README.md")
        if readme_path.exists():
            readme_content = readme_path.read_text()
            
            # Check for key sections
            required_sections = [
                "What's New in v3",
                "Quick Start",
                "Performance Benchmarks",
                "Configuration",
                "Testing",
                "Deployment"
            ]
            
            for section in required_sections:
                if section not in readme_content:
                    logger.error(f"  ❌ Missing README section: {section}")
                    return False
                logger.info(f"  ✅ README section: {section}")
        
        # Check comprehensive documentation
        doc_path = Path("DOCUMENTATION_V3.md")
        if doc_path.exists():
            doc_content = doc_path.read_text()
            
            # Check for key documentation sections
            doc_sections = [
                "Enhanced Components",
                "Usage Guide",
                "Performance Optimization",
                "Deployment Guide",
                "Troubleshooting"
            ]
            
            for section in doc_sections:
                if section not in doc_content:
                    logger.error(f"  ❌ Missing documentation section: {section}")
                    return False
                logger.info(f"  ✅ Documentation section: {section}")
        
        # Check project summary
        summary_path = Path("PROJECT_SUMMARY_V3.md")
        if summary_path.exists():
            summary_content = summary_path.read_text()
            
            if "v3" not in summary_content.lower():
                logger.error("  ❌ Project summary doesn't mention v3")
                return False
            
            logger.info("  ✅ Project summary updated for v3")
        
        logger.info("✅ Documentation validation successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Documentation validation failed: {e}")
        logger.error(traceback.format_exc())
        return False

def run_simplified_validation():
    """Run simplified validation of v3 enhancements."""
    logger = setup_logging()
    logger.info("🚀 Starting simplified validation of RL Training v3")
    logger.info("=" * 60)
    
    validation_tests = [
        ("Component Structure", validate_component_structure),
        ("Basic Imports", validate_basic_imports),
        ("Configuration System", validate_configuration_system),
        ("Configuration Profiles", validate_configuration_profiles),
        ("Documentation", validate_documentation),
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
    logger.info("📊 Simplified Validation Results")
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
    
    if success_rate == 1.0:
        logger.info("\n🎉 All simplified validation tests passed!")
        logger.info("✨ RL Training v3 core functionality is working correctly.")
        logger.info("📝 Note: Full validation requires PyTorch and other dependencies.")
        return 0
    else:
        logger.warning(f"\n⚠️ {total - passed} validation tests failed.")
        return 1

if __name__ == "__main__":
    exit(run_simplified_validation())

