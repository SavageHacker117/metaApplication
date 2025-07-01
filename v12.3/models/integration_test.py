#!/usr/bin/env python3
"""
Simple Integration Test for RL Training Super Script

This script performs basic integration testing to verify that all systems
can be imported and initialized without errors.
"""

import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all major components can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Core components
        from core.environment.base_environment import TowerDefenseEnvironment
        from core.rewards.reward_system import RewardSystem
        from core.actions.action_space import ActionSpace
        from core.observations.observation_space import ObservationSpace
        from core.training.training_loop import TrainingLoop
        print("  ✅ Core RL components imported successfully")
        
        # Tower generation
        from systems.tower_generation.tower_generator import TowerGenerator
        from systems.tower_generation.threejs_renderer import ThreeJSRenderer
        print("  ✅ Tower generation components imported successfully")
        
        # Placement system
        from systems.placement.placement_system import TowerPlacementSystem
        from systems.placement.ui_components import UIComponentManager
        print("  ✅ Placement system components imported successfully")
        
        # Dialogue and content
        from systems.dialogue.npc_dialogue import NPCDialogueSystem
        from systems.content_generation.procedural_content import ProceduralContentAPI
        print("  ✅ Dialogue and content components imported successfully")
        
        # Game master
        from systems.game_master.real_time_gm import RealTimeGameMaster
        print("  ✅ Game master components imported successfully")
        
        # Visual assessment
        from systems.visual_assessment.visual_fidelity import VisualFidelityAssessment
        print("  ✅ Visual assessment components imported successfully")
        
        # Feedback loop
        from systems.feedback_loop.rl_agent_feedback import RLAgentFeedbackLoop
        print("  ✅ Feedback loop components imported successfully")
        
        # Utilities
        from utils.logging import setup_logger
        from utils.math import normalize_value
        print("  ✅ Utility components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_basic_initialization():
    """Test basic initialization of key components."""
    print("🚀 Testing basic initialization...")
    
    try:
        # Test environment
        env = TowerDefenseEnvironment(grid_size=[5, 5], max_towers=3, max_enemies=10)
        print("  ✅ Environment initialized successfully")
        
        # Test reward system
        reward_system = RewardSystem()
        print("  ✅ Reward system initialized successfully")
        
        # Test tower generator
        tower_generator = TowerGenerator()
        print("  ✅ Tower generator initialized successfully")
        
        # Test placement system
        placement_system = TowerPlacementSystem(grid_size=[5, 5])
        print("  ✅ Placement system initialized successfully")
        
        # Test visual assessor
        visual_assessor = VisualFidelityAssessment()
        print("  ✅ Visual assessor initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Initialization failed: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of key components."""
    print("⚙️ Testing basic functionality...")
    
    try:
        # Test tower generation
        from systems.tower_generation.tower_generator import TowerGenerator, TowerTheme
        generator = TowerGenerator()
        tower = generator.generate_single_tower(TowerTheme.MEDIEVAL, 1)
        print(f"  ✅ Generated tower: {tower.name}")
        
        # Test placement validation
        from systems.placement.placement_system import TowerPlacementSystem
        placement = TowerPlacementSystem(grid_size=[5, 5])
        is_valid = placement.validate_placement(2, 2, tower)
        print(f"  ✅ Placement validation: {is_valid}")
        
        # Test reward calculation
        from core.rewards.reward_system import RewardSystem
        reward_system = RewardSystem()
        reward = reward_system.calculate_placement_reward(2, 2, tower, [])
        print(f"  ✅ Reward calculation: {reward:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading."""
    print("📋 Testing configuration...")
    
    try:
        # Test default configuration
        from rl_training_super_script import RLTrainingSuperScript
        
        # This should work with default config
        script = RLTrainingSuperScript()
        print("  ✅ Default configuration loaded successfully")
        
        # Test configuration access
        grid_size = script.config["environment"]["grid_size"]
        print(f"  ✅ Configuration access: grid_size = {grid_size}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("🧪 RL Training Super Script - Integration Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Initialization Test", test_basic_initialization),
        ("Functionality Test", test_basic_functionality),
        ("Configuration Test", test_configuration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All integration tests passed! System is ready for deployment.")
        return 0
    else:
        print("⚠️ Some integration tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())

