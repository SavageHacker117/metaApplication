import os
from pydantic import ValidationError

from core.config.config_schema import MainConfig
from training.td_rl_llm_trainer import TDRL_LLMTrainer
from game_env.tower_defense_env import TowerDefenseEnv
from game_env.game_renderer import GameRenderer
from game_env.event_manager import EventManager
from game_env.game_loop import GameLoop
from game_env.interactive_manager import InteractiveManager

def run_training_mode():
    print("\n--- Starting Training Mode ---")
    example_config_data = {
        "llm": {"model_name": "gpt-3.5-turbo", "api_key": os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")},
        "rl": {"algorithm": "PPO", "learning_rate": 0.0001, "gamma": 0.99, "num_episodes": 5},
        "system": {"environment_name": "tower_defense", "log_level": "INFO", "output_dir": "./td_outputs", "debug_mode": False}
    }

    try:
        validated_config = MainConfig(**example_config_data)
        trainer = TDRL_LLMTrainer(validated_config)
        trainer.train(num_training_episodes=validated_config.rl.num_episodes)
    except ValidationError as e:
        print(f"Configuration validation error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def run_interactive_mode():
    print("\n--- Starting Interactive Mode ---")
    grid_size = (10, 10)
    cell_size = 60
    env = TowerDefenseEnv(grid_size=grid_size)
    renderer = GameRenderer(grid_size=grid_size, cell_size=cell_size)
    event_manager = EventManager(cell_size=cell_size)
    
    game_loop = GameLoop(env, renderer, event_manager)
    game_loop.start()

def run_interactive_manager_mode():
    print("\n--- Starting Interactive Manager Mode (CLI) ---")
    grid_size = (10, 10)
    env = TowerDefenseEnv(grid_size=grid_size)
    visualizer = GameVisualizer(grid_size=grid_size)
    interactive_session = InteractiveManager(env, visualizer)
    interactive_session.start_interactive_session()

if __name__ == "__main__":
    # Example of how to run different modes
    # You can modify this to accept command-line arguments
    # or choose a mode based on a configuration setting.

    # To run training:
    # run_training_mode()

    # To run interactive game with Pygame (graphical):
    # run_interactive_mode()

    # To run interactive game with CLI (text-based):
    run_interactive_manager_mode()


