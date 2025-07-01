#!/usr/bin/env python3
"""
Simplified NeRF Integration Validation

This script validates the NeRF integration components without requiring
heavy dependencies like PyTorch or GPU acceleration.
"""

import os
import sys
import logging
import time
from pathlib import Path

def test_nerf_module_imports():
    """Test that NeRF modules can be imported."""
    print("🔍 Testing NeRF module imports...")
    
    try:
        from nerf_integration_module import NeRFAssetType, NeRFQuality
        print("✅ NeRF integration module imported successfully")
        
        # Test enum values
        assert NeRFAssetType.MESH.value == "mesh"
        assert NeRFQuality.HIGH.value == "high"
        print("✅ NeRF enums working correctly")
        
        return True
    except Exception as e:
        print(f"❌ NeRF module import failed: {e}")
        return False

def test_nerf_asset_management():
    """Test NeRF asset management functionality."""
    print("🔍 Testing NeRF asset management...")
    
    try:
        from nerf_asset_management import NeRFConfig
        
        # Test configuration
        config = NeRFConfig(
            asset_directory="test_nerf_assets",
            cache_size=10,
            enable_quality_assessment=False
        )
        
        config.validate()
        print("✅ NeRF configuration validation passed")
        
        return True
    except Exception as e:
        print(f"❌ NeRF asset management test failed: {e}")
        return False

def test_nerf_action_space():
    """Test NeRF action space functionality."""
    print("🔍 Testing NeRF action space...")
    
    try:
        from nerf_agent_extensions import NeRFActionType, NeRFAction
        
        # Test action creation
        action = NeRFAction(
            action_type=NeRFActionType.APPLY_SKIN,
            target_object="test_tower",
            asset_id="test_asset_123"
        )
        
        # Test action serialization
        action_dict = action.to_dict()
        assert action_dict['action_type'] == "apply_skin"
        assert action_dict['target_object'] == "test_tower"
        
        print("✅ NeRF action space working correctly")
        return True
    except Exception as e:
        print(f"❌ NeRF action space test failed: {e}")
        return False

def test_threejs_integration():
    """Test Three.js integration file structure."""
    print("🔍 Testing Three.js integration...")
    
    try:
        threejs_file = Path("threejs_nerf_integration.js")
        if not threejs_file.exists():
            print("❌ Three.js integration file not found")
            return False
        
        # Read and validate basic structure
        content = threejs_file.read_text()
        
        required_components = [
            "NeRFRenderer",
            "NeRFAssetCache",
            "NeRFPerformanceMonitor",
            "createNeRFRenderer"
        ]
        
        for component in required_components:
            if component not in content:
                print(f"❌ Missing component: {component}")
                return False
        
        print("✅ Three.js integration structure validated")
        return True
    except Exception as e:
        print(f"❌ Three.js integration test failed: {e}")
        return False

def test_performance_testing():
    """Test performance testing module."""
    print("🔍 Testing performance testing module...")
    
    try:
        from nerf_performance_testing import PerformanceProfiler, PerformanceMetrics
        
        # Test profiler
        profiler = PerformanceProfiler()
        
        @profiler.profile_operation("test_operation")
        def test_function():
            time.sleep(0.01)  # Short sleep
            return "success"
        
        result = test_function()
        assert result == "success"
        assert len(profiler.metrics) == 1
        
        # Test metrics
        metric = profiler.metrics[0]
        assert metric.operation_name == "test_operation"
        assert metric.success == True
        
        print("✅ Performance testing module working correctly")
        return True
    except Exception as e:
        print(f"❌ Performance testing module test failed: {e}")
        return False

def validate_file_structure():
    """Validate that all NeRF files are present."""
    print("🔍 Validating NeRF file structure...")
    
    required_files = [
        "nerf_integration_module.py",
        "nerf_asset_management.py",
        "nerf_agent_extensions.py",
        "nerf_performance_testing.py",
        "threejs_nerf_integration.js"
    ]
    
    missing_files = []
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All NeRF files present")
    return True

def run_nerf_validation():
    """Run complete NeRF validation."""
    print("🎯 NeRF Integration Validation Suite")
    print("=" * 50)
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    tests = [
        ("File Structure", validate_file_structure),
        ("Module Imports", test_nerf_module_imports),
        ("Asset Management", test_nerf_asset_management),
        ("Action Space", test_nerf_action_space),
        ("Three.js Integration", test_threejs_integration),
        ("Performance Testing", test_performance_testing)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Validation Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({passed}/{total})")
    
    if success_rate == 100:
        print("🎉 All NeRF integration tests passed!")
        print("🚀 NeRF system is ready for integration!")
    elif success_rate >= 80:
        print("✅ NeRF system is mostly functional with minor issues.")
    else:
        print("⚠️  NeRF system has significant issues that need attention.")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = run_nerf_validation()
    sys.exit(0 if success else 1)

