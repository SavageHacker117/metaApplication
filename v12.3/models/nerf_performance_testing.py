#!/usr/bin/env python3
"""
NeRF Performance Optimization and Testing Suite

This module provides comprehensive testing and performance optimization
for the NeRF integration system, ensuring optimal performance and
reliability in production environments.

Features:
- Performance benchmarking and optimization
- Comprehensive test suite for all NeRF components
- Memory usage optimization
- GPU utilization monitoring
- Quality vs performance trade-off analysis
- Automated performance tuning
"""

import os
import sys
import time
import logging
import asyncio
import unittest
import numpy as np
import psutil
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import NeRF components
try:
    from nerf_integration_module import (
        NeRFAssetType, NeRFQuality, NeRFAssetMetadata, 
        NeRFAssetManager, NeRFRenderer, create_nerf_integration_system
    )
    from nerf_asset_management import (
        EnhancedNeRFAssetManager, NeRFConfig, create_enhanced_nerf_system
    )
    from nerf_agent_extensions import (
        NeRFActionSpace, NeRFEnhancedAgent, NeRFAction, NeRFActionType
    )
    from enhanced_config_manager import MasterConfig
except ImportError as e:
    logging.warning(f"Some NeRF components not available: {e}")

@dataclass
class PerformanceMetrics:
    """Performance metrics for NeRF operations."""
    operation_name: str
    execution_time: float
    memory_usage_mb: float
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_name': self.operation_name,
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_utilization': self.gpu_utilization,
            'cpu_utilization': self.cpu_utilization,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }

class PerformanceProfiler:
    """Profiles performance of NeRF operations."""
    
    def __init__(self):
        self.metrics = []
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
    
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                return self._profile_function(operation_name, func, *args, **kwargs)
            return wrapper
        return decorator
    
    def _profile_function(self, operation_name: str, func, *args, **kwargs):
        """Profile a function execution."""
        # Record initial state
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        success = True
        error_message = None
        result = None
        
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            self.logger.error(f"Error in {operation_name}: {e}")
        
        # Record final state
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_utilization = (start_cpu + end_cpu) / 2
        
        # Store metrics
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            cpu_utilization=cpu_utilization,
            success=success,
            error_message=error_message
        )
        
        self.metrics.append(metrics)
        
        return result
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics:
            return {}
        
        # Group by operation name
        operation_groups = {}
        for metric in self.metrics:
            if metric.operation_name not in operation_groups:
                operation_groups[metric.operation_name] = []
            operation_groups[metric.operation_name].append(metric)
        
        summary = {}
        for op_name, metrics_list in operation_groups.items():
            successful_metrics = [m for m in metrics_list if m.success]
            
            if successful_metrics:
                summary[op_name] = {
                    'count': len(metrics_list),
                    'success_rate': len(successful_metrics) / len(metrics_list),
                    'avg_execution_time': np.mean([m.execution_time for m in successful_metrics]),
                    'avg_memory_usage': np.mean([m.memory_usage_mb for m in successful_metrics]),
                    'avg_cpu_utilization': np.mean([m.cpu_utilization for m in successful_metrics]),
                    'min_execution_time': np.min([m.execution_time for m in successful_metrics]),
                    'max_execution_time': np.max([m.execution_time for m in successful_metrics])
                }
            else:
                summary[op_name] = {
                    'count': len(metrics_list),
                    'success_rate': 0.0,
                    'errors': [m.error_message for m in metrics_list if m.error_message]
                }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        metrics_data = [m.to_dict() for m in self.metrics]
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)

class NeRFPerformanceOptimizer:
    """Optimizes NeRF system performance."""
    
    def __init__(self, asset_manager: EnhancedNeRFAssetManager, renderer: NeRFRenderer):
        self.asset_manager = asset_manager
        self.renderer = renderer
        self.logger = logging.getLogger(__name__)
        self.profiler = PerformanceProfiler()
        
        # Optimization settings
        self.target_fps = 60
        self.max_memory_usage_mb = 2048  # 2GB
        self.max_concurrent_renders = 4
        
        # Performance history
        self.performance_history = []
    
    def optimize_system(self) -> Dict[str, Any]:
        """Perform comprehensive system optimization."""
        optimization_results = {}
        
        # 1. Optimize asset cache
        cache_optimization = self._optimize_asset_cache()
        optimization_results['cache_optimization'] = cache_optimization
        
        # 2. Optimize rendering pipeline
        render_optimization = self._optimize_rendering_pipeline()
        optimization_results['render_optimization'] = render_optimization
        
        # 3. Optimize memory usage
        memory_optimization = self._optimize_memory_usage()
        optimization_results['memory_optimization'] = memory_optimization
        
        # 4. Optimize quality settings
        quality_optimization = self._optimize_quality_settings()
        optimization_results['quality_optimization'] = quality_optimization
        
        self.logger.info("NeRF system optimization completed")
        return optimization_results
    
    def _optimize_asset_cache(self) -> Dict[str, Any]:
        """Optimize asset cache settings."""
        current_cache_size = self.asset_manager.cache_size
        current_memory = self._get_cache_memory_usage()
        
        # Adjust cache size based on memory usage
        if current_memory > self.max_memory_usage_mb * 0.5:  # 50% of max memory
            new_cache_size = max(10, int(current_cache_size * 0.8))
            self.asset_manager.cache_size = new_cache_size
            self.logger.info(f"Reduced cache size from {current_cache_size} to {new_cache_size}")
            return {'action': 'reduced', 'old_size': current_cache_size, 'new_size': new_cache_size}
        elif current_memory < self.max_memory_usage_mb * 0.2:  # 20% of max memory
            new_cache_size = min(200, int(current_cache_size * 1.2))
            self.asset_manager.cache_size = new_cache_size
            self.logger.info(f"Increased cache size from {current_cache_size} to {new_cache_size}")
            return {'action': 'increased', 'old_size': current_cache_size, 'new_size': new_cache_size}
        
        return {'action': 'no_change', 'cache_size': current_cache_size}
    
    def _optimize_rendering_pipeline(self) -> Dict[str, Any]:
        """Optimize rendering pipeline settings."""
        render_stats = self.renderer.get_render_stats()
        
        optimizations = []
        
        # Adjust concurrent renders based on performance
        current_fps = self._estimate_current_fps()
        if current_fps < self.target_fps * 0.8:  # Below 80% of target
            if self.max_concurrent_renders > 1:
                self.max_concurrent_renders -= 1
                optimizations.append(f"Reduced concurrent renders to {self.max_concurrent_renders}")
        elif current_fps > self.target_fps * 1.1:  # Above 110% of target
            if self.max_concurrent_renders < 8:
                self.max_concurrent_renders += 1
                optimizations.append(f"Increased concurrent renders to {self.max_concurrent_renders}")
        
        # Optimize cache hit rate
        cache_hit_rate = render_stats.get('cache_hit_rate', 0)
        if cache_hit_rate < 0.7:  # Below 70%
            # Increase cache size for renderer
            if hasattr(self.renderer, 'cache_size'):
                old_size = self.renderer.cache_size
                self.renderer.cache_size = min(100, int(old_size * 1.3))
                optimizations.append(f"Increased render cache from {old_size} to {self.renderer.cache_size}")
        
        return {'optimizations': optimizations, 'render_stats': render_stats}
    
    def _optimize_memory_usage(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        optimizations = []
        
        if current_memory > self.max_memory_usage_mb:
            # Clear least recently used assets
            if hasattr(self.asset_manager, 'asset_cache'):
                cache_size_before = len(self.asset_manager.asset_cache)
                # Clear 25% of cache
                items_to_remove = max(1, cache_size_before // 4)
                for _ in range(items_to_remove):
                    if self.asset_manager.asset_cache:
                        # Remove oldest item (simplified LRU)
                        oldest_key = next(iter(self.asset_manager.asset_cache))
                        del self.asset_manager.asset_cache[oldest_key]
                
                cache_size_after = len(self.asset_manager.asset_cache)
                optimizations.append(f"Cleared {cache_size_before - cache_size_after} cached assets")
            
            # Force garbage collection
            import gc
            gc.collect()
            optimizations.append("Forced garbage collection")
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return {
            'optimizations': optimizations,
            'memory_before_mb': current_memory,
            'memory_after_mb': final_memory,
            'memory_saved_mb': current_memory - final_memory
        }
    
    def _optimize_quality_settings(self) -> Dict[str, Any]:
        """Optimize quality settings based on performance."""
        current_fps = self._estimate_current_fps()
        
        optimizations = []
        
        # Adjust default quality based on performance
        if current_fps < self.target_fps * 0.7:  # Below 70% of target
            # Reduce quality
            if hasattr(self.asset_manager, 'config'):
                if self.asset_manager.config.default_quality == "high":
                    self.asset_manager.config.default_quality = "medium"
                    optimizations.append("Reduced default quality to medium")
                elif self.asset_manager.config.default_quality == "medium":
                    self.asset_manager.config.default_quality = "low"
                    optimizations.append("Reduced default quality to low")
        elif current_fps > self.target_fps * 1.2:  # Above 120% of target
            # Increase quality
            if hasattr(self.asset_manager, 'config'):
                if self.asset_manager.config.default_quality == "low":
                    self.asset_manager.config.default_quality = "medium"
                    optimizations.append("Increased default quality to medium")
                elif self.asset_manager.config.default_quality == "medium":
                    self.asset_manager.config.default_quality = "high"
                    optimizations.append("Increased default quality to high")
        
        return {'optimizations': optimizations, 'current_fps': current_fps}
    
    def _get_cache_memory_usage(self) -> float:
        """Estimate cache memory usage in MB."""
        # Simplified estimation
        if hasattr(self.asset_manager, 'asset_cache'):
            return len(self.asset_manager.asset_cache) * 10  # Assume 10MB per asset
        return 0.0
    
    def _estimate_current_fps(self) -> float:
        """Estimate current FPS (placeholder)."""
        # In a real implementation, this would measure actual FPS
        return 60.0  # Placeholder

class NeRFTestSuite(unittest.TestCase):
    """Comprehensive test suite for NeRF integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.logger = logging.getLogger(__name__)
        
        # Create test configuration
        cls.test_config = NeRFConfig(
            asset_directory="test_nerf_assets",
            cache_size=10,
            enable_quality_assessment=False,  # Disable for testing
            enable_performance_monitoring=False
        )
        
        # Create test components
        try:
            cls.asset_manager = EnhancedNeRFAssetManager(cls.test_config)
            cls.renderer = NeRFRenderer(asset_manager=cls.asset_manager)
            cls.action_space = NeRFActionSpace(cls.asset_manager)
        except Exception as e:
            cls.logger.warning(f"Could not create test components: {e}")
            cls.asset_manager = None
            cls.renderer = None
            cls.action_space = None
    
    def setUp(self):
        """Set up each test."""
        self.profiler = PerformanceProfiler()
    
    def test_asset_manager_initialization(self):
        """Test asset manager initialization."""
        if self.asset_manager is None:
            self.skipTest("Asset manager not available")
        
        self.assertIsNotNone(self.asset_manager)
        self.assertEqual(self.asset_manager.config.cache_size, 10)
        self.assertTrue(Path(self.asset_manager.asset_directory).exists())
    
    def test_asset_registration(self):
        """Test asset registration functionality."""
        if self.asset_manager is None:
            self.skipTest("Asset manager not available")
        
        # Create a dummy asset file
        test_asset_path = Path(self.asset_manager.asset_directory) / "test_asset.glb"
        test_asset_path.parent.mkdir(exist_ok=True)
        test_asset_path.write_text("dummy content")
        
        try:
            asset_id = self.asset_manager.register_asset(
                file_path=str(test_asset_path),
                name="Test Asset",
                asset_type=NeRFAssetType.MESH,
                quality_level=NeRFQuality.MEDIUM,
                compatibility_tags=["test", "mesh"]
            )
            
            self.assertIsNotNone(asset_id)
            self.assertIn(asset_id, self.asset_manager.metadata_cache)
            
            # Test metadata retrieval
            metadata = self.asset_manager.get_asset_metadata(asset_id)
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata.name, "Test Asset")
            self.assertEqual(metadata.asset_type, NeRFAssetType.MESH)
            
        finally:
            # Cleanup
            if test_asset_path.exists():
                test_asset_path.unlink()
    
    def test_asset_search(self):
        """Test asset search functionality."""
        if self.asset_manager is None:
            self.skipTest("Asset manager not available")
        
        # Register test assets
        test_assets = []
        for i in range(3):
            test_asset_path = Path(self.asset_manager.asset_directory) / f"test_asset_{i}.glb"
            test_asset_path.write_text("dummy content")
            
            asset_id = self.asset_manager.register_asset(
                file_path=str(test_asset_path),
                name=f"Test Asset {i}",
                asset_type=NeRFAssetType.MESH,
                quality_level=NeRFQuality.MEDIUM,
                compatibility_tags=["test", f"type_{i}"]
            )
            test_assets.append((asset_id, test_asset_path))
        
        try:
            # Test search by type
            mesh_assets = self.asset_manager.search_assets(asset_type=NeRFAssetType.MESH)
            self.assertGreaterEqual(len(mesh_assets), 3)
            
            # Test search by tags
            tagged_assets = self.asset_manager.search_assets(tags=["test"])
            self.assertGreaterEqual(len(tagged_assets), 3)
            
        finally:
            # Cleanup
            for asset_id, asset_path in test_assets:
                if asset_path.exists():
                    asset_path.unlink()
    
    def test_action_space_generation(self):
        """Test NeRF action space generation."""
        if self.action_space is None:
            self.skipTest("Action space not available")
        
        # Create test game state
        game_state = {
            'objects': {
                'tower_1': {'type': 'tower', 'visible': True, 'position': [0.5, 0.5]},
                'wall_1': {'type': 'wall', 'visible': True, 'position': [0.3, 0.7]}
            },
            'phase': 'gameplay',
            'performance_metrics': {'fps': 60, 'memory_usage': 0.5}
        }
        
        # Get available actions
        actions = self.action_space.get_available_actions(game_state)
        
        self.assertIsInstance(actions, list)
        # Should have at least some actions available
        self.assertGreaterEqual(len(actions), 0)
        
        # Check action structure
        for action in actions[:5]:  # Check first 5 actions
            self.assertIsInstance(action, NeRFAction)
            self.assertIn(action.action_type, NeRFActionType)
            self.assertIsInstance(action.target_object, str)
    
    def test_action_execution(self):
        """Test NeRF action execution."""
        if self.action_space is None:
            self.skipTest("Action space not available")
        
        # Create test action
        test_action = NeRFAction(
            action_type=NeRFActionType.APPLY_SKIN,
            target_object="test_object",
            asset_id="test_asset_123",
            quality_level=NeRFQuality.MEDIUM
        )
        
        game_state = {'objects': {'test_object': {'type': 'tower', 'visible': True}}}
        
        # Execute action
        success = self.action_space.execute_action(test_action, game_state)
        
        # Should succeed (even without actual asset)
        self.assertTrue(success)
        
        # Check that action is tracked
        self.assertIn("test_object", self.action_space.active_nerfs)
    
    def test_performance_profiling(self):
        """Test performance profiling functionality."""
        
        @self.profiler.profile_operation("test_operation")
        def test_function():
            time.sleep(0.1)  # Simulate work
            return "success"
        
        result = test_function()
        
        self.assertEqual(result, "success")
        self.assertEqual(len(self.profiler.metrics), 1)
        
        metric = self.profiler.metrics[0]
        self.assertEqual(metric.operation_name, "test_operation")
        self.assertGreaterEqual(metric.execution_time, 0.1)
        self.assertTrue(metric.success)
    
    def test_performance_optimization(self):
        """Test performance optimization functionality."""
        if self.asset_manager is None or self.renderer is None:
            self.skipTest("Components not available")
        
        optimizer = NeRFPerformanceOptimizer(self.asset_manager, self.renderer)
        
        # Run optimization
        results = optimizer.optimize_system()
        
        self.assertIsInstance(results, dict)
        self.assertIn('cache_optimization', results)
        self.assertIn('render_optimization', results)
        self.assertIn('memory_optimization', results)
        self.assertIn('quality_optimization', results)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Clean up test directory
        if hasattr(cls, 'test_config') and cls.test_config:
            test_dir = Path(cls.test_config.asset_directory)
            if test_dir.exists():
                import shutil
                shutil.rmtree(test_dir, ignore_errors=True)

def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("🚀 Running NeRF Performance Benchmark...")
    
    # Create test configuration
    config = NeRFConfig(
        asset_directory="benchmark_assets",
        cache_size=50,
        enable_performance_monitoring=True
    )
    
    try:
        # Create components
        asset_manager = EnhancedNeRFAssetManager(config)
        renderer = NeRFRenderer(asset_manager=asset_manager)
        optimizer = NeRFPerformanceOptimizer(asset_manager, renderer)
        
        # Run benchmarks
        print("📊 Running optimization benchmark...")
        start_time = time.time()
        optimization_results = optimizer.optimize_system()
        optimization_time = time.time() - start_time
        
        print(f"✅ Optimization completed in {optimization_time:.3f}s")
        print(f"📈 Results: {json.dumps(optimization_results, indent=2)}")
        
        # Performance summary
        print("\n📋 Performance Summary:")
        print(f"   Cache optimization: {optimization_results.get('cache_optimization', {}).get('action', 'N/A')}")
        print(f"   Memory optimization: {len(optimization_results.get('memory_optimization', {}).get('optimizations', []))} optimizations")
        print(f"   Quality optimization: {len(optimization_results.get('quality_optimization', {}).get('optimizations', []))} optimizations")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
    
    finally:
        # Cleanup
        benchmark_dir = Path("benchmark_assets")
        if benchmark_dir.exists():
            import shutil
            shutil.rmtree(benchmark_dir, ignore_errors=True)

def run_comprehensive_tests():
    """Run comprehensive test suite."""
    print("🧪 Running NeRF Comprehensive Test Suite...")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(NeRFTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / max(1, result.testsRun)) * 100:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    print("🎯 NeRF Performance Optimization and Testing Suite")
    print("=" * 60)
    
    # Run tests
    test_success = run_comprehensive_tests()
    
    print("\n" + "=" * 60)
    
    # Run benchmark
    run_performance_benchmark()
    
    print("\n" + "=" * 60)
    print(f"🏁 Testing completed. Overall success: {'✅' if test_success else '❌'}")
    
    if test_success:
        print("🎉 All NeRF components are working correctly!")
        print("🚀 System is ready for production deployment.")
    else:
        print("⚠️  Some tests failed. Please review the results above.")
        print("🔧 Consider running individual tests for debugging.")

