
import argparse
import os
from pydantic import ValidationError

from core.config.config_schema import MainConfig
from core.config.rl_llm_td_config import TDRL_LLMMainConfig
from training.td_rl_llm_trainer import TDRL_LLMTrainer
from game_env.tower_defense_env import TowerDefenseEnv
from game_env.game_renderer import GameRenderer
from game_env.event_manager import EventManager
from game_env.game_loop import GameLoop
from game_env.interactive_manager import InteractiveManager
from utils.config_loader import load_config

def main():
    parser = argparse.ArgumentParser(description="Run RL-LLM Tower Defense system in different modes.")
    parser.add_argument("--mode", type=str, default="interactive_cli",
                        choices=["train", "interactive_graphical", "interactive_cli"],
                        help="Mode to run the system in: train, interactive_graphical, or interactive_cli.")
    parser.add_argument("--config", type=str, default="./config/td_config.json",
                        help="Path to the configuration file (JSON or YAML).")

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config, TDRL_LLMMainConfig)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    if args.mode == "train":
        print("\n--- Starting Training Mode ---")
        trainer = TDRL_LLMTrainer(config)
        trainer.train(num_training_episodes=config.rl.num_episodes)
    elif args.mode == "interactive_graphical":
        print("\n--- Starting Interactive Graphical Mode ---")
        grid_size = config.game_specific.grid_size
        cell_size = 60 # Fixed cell size for graphical rendering
        env = TowerDefenseEnv(grid_size=grid_size,
                              initial_cash=config.game_specific.initial_cash,
                              initial_lives=config.game_specific.initial_lives)
        renderer = GameRenderer(grid_size=grid_size, cell_size=cell_size)
        event_manager = EventManager(cell_size=cell_size)
        
        game_loop = GameLoop(env, renderer, event_manager)
        game_loop.start()
    elif args.mode == "interactive_cli":
        print("\n--- Starting Interactive CLI Mode ---")
        grid_size = config.game_specific.grid_size
        env = TowerDefenseEnv(grid_size=grid_size,
                              initial_cash=config.game_specific.initial_cash,
                              initial_lives=config.game_specific.initial_lives)
        visualizer = GameVisualizer(grid_size=grid_size)
        interactive_session = InteractiveManager(env, visualizer)
        interactive_session.start_interactive_session()

if __name__ == "__main__":
    main()

