import pytest
import tempfile
import json
import os
from v12.curriculum_learning_system import (
    DifficultyDimension,
    SceneConfiguration,
    LearningProgress,
    CurriculumManager,
    AdaptiveDifficultyScaler,
)

@pytest.fixture
def example_scene_config():
    return SceneConfiguration(
        scene_id="scene1",
        name="Scene 1",
        description="Test scene",
        difficulty_level=0.5,
        difficulty_dimensions={
            DifficultyDimension.ENVIRONMENTAL_COMPLEXITY: 0.5,
            DifficultyDimension.TASK_OBJECTIVES: 0.5,
        },
        prerequisites=[],
        learning_objectives=["move", "pick"],
        success_criteria={"completion_rate": 0.9},
        environment_config={"size": "small"},
        reward_config={"bonus": 1},
        estimated_training_time=100,
    )

def test_scene_configuration_serialization(example_scene_config):
    d = example_scene_config.to_dict()
    restored = SceneConfiguration.from_dict(d)
    assert restored == example_scene_config

def test_curriculum_manager_load_and_scene_selection(tmp_path):
    # Create a fake curriculum config file
    scene = {
        "scene_id": "basic",
        "name": "Basic",
        "description": "desc",
        "difficulty_level": 0.1,
        "difficulty_dimensions": {
            "env_complexity": 0.1,
            "task_objectives": 0.1,
        },
        "prerequisites": [],
        "learning_objectives": ["a"],
        "success_criteria": {"rate": 1.0},
        "environment_config": {},
        "reward_config": {},
        "estimated_training_time": 10,
    }
    config = {"scenes": [scene]}
    config_path = tmp_path / "curriculum.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    mgr = CurriculumManager(str(config_path))
    # New agent: should get only the basic scene available
    available = mgr.get_available_scenes("new_agent")
    assert available == ["basic"]
    # Selecting next scene returns the same
    assert mgr.select_next_scene("new_agent") == "basic"

def test_update_progress_and_mastery(tmp_path):
    # Set up curriculum manager
    scene = {
        "scene_id": "s1",
        "name": "S1",
        "description": "",
        "difficulty_level": 0.1,
        "difficulty_dimensions": {"env_complexity": 0.1, "task_objectives": 0.2},
        "prerequisites": [],
        "learning_objectives": ["obj"],
        "success_criteria": {"metric": 1.0},
        "environment_config": {},
        "reward_config": {},
        "estimated_training_time": 10,
    }
    config = {"scenes": [scene]}
    config_path = tmp_path / "curriculum.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    mgr = CurriculumManager(str(config_path))
    agent_id = "a1"
    # Before progress: nothing completed
    status = mgr.get_curriculum_status(agent_id)
    assert status["completed_scenes"] == 0
    # Update progress below mastery threshold (should not be completed)
    mgr.update_progress(agent_id, "s1", 0.5, 10)
    status = mgr.get_curriculum_status(agent_id)
    assert "s1" not in status["mastery_levels"] or status["mastery_levels"]["s1"] < mgr.mastery_threshold
    # Update with high performance (should be completed)
    mgr.update_progress(agent_id, "s1", 0.95, 10)
    status = mgr.get_curriculum_status(agent_id)
    assert "s1" in status["mastery_levels"] and status["mastery_levels"]["s1"] >= mgr.mastery_threshold
    assert status["completed_scenes"] == 1

def test_adaptive_difficulty_scaler(example_scene_config):
    scaler = AdaptiveDifficultyScaler(adaptation_rate=0.2)
    agent_id = "agentX"
    # Not enough performance data: should not change
    adjusted = scaler.adjust_scene_difficulty(example_scene_config, agent_id, [0.9, 0.8])
    assert adjusted.difficulty_level == example_scene_config.difficulty_level
    # Good performance: should increase difficulty
    perf = [0.7, 0.85, 0.9, 0.95, 1.0, 0.95, 0.9, 0.95, 0.9, 0.93]
    adjusted2 = scaler.adjust_scene_difficulty(example_scene_config, agent_id, perf)
    assert adjusted2.difficulty_level > example_scene_config.difficulty_level
    # Poor performance: should decrease difficulty
    perf_bad = [0.1, 0.3, 0.2, 0.35, 0.4, 0.2, 0.25, 0.3, 0.22, 0.18]
    adjusted3 = scaler.adjust_scene_difficulty(example_scene_config, agent_id, perf_bad)
    assert adjusted3.difficulty_level < example_scene_config.difficulty_level

def test_save_and_load_progress(tmp_path):
    # Set up
    scene = {
        "scene_id": "foo",
        "name": "Foo",
        "description": "",
        "difficulty_level": 0.2,
        "difficulty_dimensions": {"env_complexity": 0.2, "task_objectives": 0.3},
        "prerequisites": [],
        "learning_objectives": ["x"],
        "success_criteria": {"win": 1.0},
        "environment_config": {},
        "reward_config": {},
        "estimated_training_time": 5,
    }
    config = {"scenes": [scene]}
    config_path = tmp_path / "curriculum.json"
    with open(config_path, "w") as f:
        json.dump(config, f)
    mgr = CurriculumManager(str(config_path))
    # Simulate progress
    mgr.update_progress("agent_z", "foo", 0.99, 50)
    progress_path = tmp_path / "progress.json"
    mgr.save_progress(str(progress_path))
    # Reload to new manager
    mgr2 = CurriculumManager(str(config_path))
    mgr2.load_progress(str(progress_path))
    assert "agent_z" in mgr2.progression_history
    assert mgr2.progression_history["agent_z"].current_scene == "foo"
