

import json
import os
from typing import Dict, Any

from game_env.tower_defense_env import TowerDefenseEnv
from game_env.threejs_scene_generator import ThreeJSSceneGenerator
from game_env.game_state_to_3d_converter import GameStateTo3DConverter
from game_env.procedural_generator import ProceduralGenerator
from nerf_integration.nerf_scene_renderer import NeRFSceneRenderer
from nerf_integration.nerf_asset_pipeline import NeRFAssetPipeline
from llm_integration.llm_api_interface import LLMAPIInterface
from llm_integration.llm_game_designer import LLMGameDesigner
from llm_integration.llm_game_analyzer import GoLLMAnalyzer # Re-using for general analysis
from rl_agent.rl_3d_game_tester import RL3DGameTester
from utils.config_loader import load_config
from core.config.rl_llm_td_config import TDRL_LLMMainConfig # Re-using the TD config

class TD3DOrchestrator:
    """
    Orchestrates the entire pipeline for procedurally generating, rendering,
    testing, and refining 3D Tower Defense games using RL-LLM and NeRF.
    """
    def __init__(self, config_path: str):
        self.config = load_config(config_path, TDRL_LLMMainConfig)
        
        # Initialize core components
        self.llm_api = LLMAPIInterface(api_key=self.config.llm.api_key, model_name=self.config.llm.model_name)
        self.procedural_generator = ProceduralGenerator(grid_size=self.config.game_specific.grid_size)
        self.llm_game_designer = LLMGameDesigner(self.llm_api, self.procedural_generator)
        self.nerf_renderer = NeRFSceneRenderer() # Assuming NeRF is ready
        self.nerf_asset_pipeline = NeRFAssetPipeline(self.nerf_renderer, self.llm_game_designer)
        self.game_state_converter = GameStateTo3DConverter()
        self.threejs_scene_generator = ThreeJSSceneGenerator(grid_size=self.config.game_specific.grid_size)
        self.llm_game_analyzer = GoLLMAnalyzer(self.llm_api, board_size=self.config.game_specific.grid_size[0]) # Re-using

        # Game environment and tester
        self.game_env = TowerDefenseEnv(grid_size=self.config.game_specific.grid_size,
                                        initial_cash=self.config.game_specific.initial_cash,
                                        initial_lives=self.config.game_specific.initial_lives)
        self.rl_tester = RL3DGameTester(self.game_env, self.nerf_renderer, self.llm_game_analyzer)

        # Ensure output directories exist
        os.makedirs(self.config.system.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.system.output_dir, "web_frontend"), exist_ok=True)
        os.makedirs(os.path.join(self.config.system.output_dir, "nerf_renders"), exist_ok=True)
        os.makedirs(os.path.join(self.config.system.output_dir, "nerf_assets"), exist_ok=True)

    def generate_and_test_game(self, iteration: int):
        print(f"\n--- Orchestration Cycle {iteration}: Generating and Testing Game ---")

        # 1. LLM designs the game level (high-level)
        game_design_params = self.llm_game_designer.design_game_level(
            theme="sci-fi", difficulty="hard" if iteration % 2 == 0 else "medium"
        )
        print("LLM Game Design Parameters:", json.dumps(game_design_params, indent=2))

        # 2. Generate 3D assets using NeRF based on LLM descriptions
        generated_assets = self.nerf_asset_pipeline.generate_assets_for_game(game_design_params)
        print("Generated 3D Assets:", generated_assets)

        # 3. Initialize game environment with procedural elements
        # The game_env needs to be able to load this generated config
        # For now, we'll pass the generated grid and path directly
        initial_game_state = {
            "grid": game_design_params["grid"],
            "path": game_design_params["path"],
            "initial_cash": game_design_params["initial_cash"],
            "initial_lives": game_design_params["initial_lives"],
            "tower_types": game_design_params["tower_types"],
            "enemy_waves": game_design_params["enemy_waves"],
            "towers": {},
            "enemies": [],
            "wave_number": 0,
            "game_over": False
        }
        # The TowerDefenseEnv needs to be updated to accept this full initial_game_state
        # For now, we'll just pass the grid and path to reset, and update other params manually
        self.game_env.reset(initial_game_state) # Assuming reset can take initial_game_state

        # 4. Convert game state to 3D representation and generate Three.js scene
        converted_3d_state = self.game_state_converter.convert_to_3d_representation(self.game_env.get_state())
        threejs_scene_config = self.threejs_scene_generator.generate_scene_config(converted_3d_state)
        
        # Save Three.js scene config for web frontend
        scene_config_path = os.path.join(self.config.system.output_dir, "web_frontend", "scene_config.json")
        self.threejs_scene_generator.save_scene_config(threejs_scene_config, scene_config_path)
        print(f"Three.js scene config saved to {scene_config_path}")

        # 5. RL-LLM tests the generated game level
        test_report = self.rl_tester.test_game_level(initial_game_state, num_test_episodes=self.config.rl.num_test_episodes)
        print("RL Tester Report:", json.dumps(test_report["test_results"], indent=2))
        print("LLM Analysis of Test Results:\n", test_report["llm_analysis"])

        # 6. Use LLM analysis to refine future generation parameters (meta-reasoning)
        # This part would involve feeding test_report["llm_analysis"] back to a strategy refiner
        # to update the config. For simplicity, we'll just print the analysis here.
        print("\n--- Orchestration Cycle Complete ---")

# Example Usage:
# if __name__ == "__main__":
#     # Create a dummy config file if it doesn't exist
#     config_dir = "./config"
#     os.makedirs(config_dir, exist_ok=True)
#     config_path = os.path.join(config_dir, "rl_llm_td_config.json")
#     if not os.path.exists(config_path):
#         from core.config.rl_llm_td_config import TDRL_LLMMainConfig
#         dummy_config = TDRL_LLMMainConfig(
#             llm={"model_name": "gpt-3.5-turbo", "api_key": "YOUR_OPENAI_API_KEY"},
#             rl={"num_test_episodes": 2},
#             game_specific={"grid_size": [10,10]},
#             system={"output_dir": "./orchestrator_output"}
#         )
#         with open(config_path, "w") as f:
#             f.write(dummy_config.json(indent=4))
#
#     orchestrator = TD3DOrchestrator(config_path)
#     orchestrator.generate_and_test_game(iteration=1)


