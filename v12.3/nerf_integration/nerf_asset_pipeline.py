
from typing import Dict, Any, List
from .nerf_scene_renderer import NeRFSceneRenderer
from llm_integration.llm_game_designer import LLMGameDesigner

class NeRFAssetPipeline:
    """
    Manages the generation of 3D assets (towers, enemies, environment props)
    using NeRF based on LLM-generated descriptions.
    """
    def __init__(self, nerf_renderer: NeRFSceneRenderer, llm_game_designer: LLMGameDesigner):
        self.nerf_renderer = nerf_renderer
        self.llm_game_designer = llm_game_designer
        self.generated_assets = {}

    def generate_assets_for_game(self, game_design_params: Dict[str, Any]) -> Dict[str, str]:
        """
        Generates 3D assets for towers and enemies based on the game design parameters
        provided by the LLMGameDesigner.
        """
        print("\n--- NeRF Asset Pipeline: Generating Assets ---")
        
        # Generate Tower Assets
        tower_concepts = game_design_params.get("llm_tower_concepts", [])
        for i, concept in enumerate(tower_concepts):
            asset_id = f"tower_{i}_{concept.replace(" ", "_")}"
            model_path = self.nerf_renderer.generate_3d_asset_from_description(concept, asset_id)
            self.generated_assets[asset_id] = model_path
            print(f"Generated tower asset: {asset_id} at {model_path}")

        # Generate Enemy Assets
        enemy_concepts = game_design_params.get("llm_enemy_concepts", [])
        for i, concept in enumerate(enemy_concepts):
            asset_id = f"enemy_{i}_{concept.replace(" ", "_")}"
            model_path = self.nerf_renderer.generate_3d_asset_from_description(concept, asset_id)
            self.generated_assets[asset_id] = model_path
            print(f"Generated enemy asset: {asset_id} at {model_path}")

        # Potentially generate environment props based on path_description
        path_description = game_design_params.get("llm_path_description")
        if path_description:
            env_asset_id = f"environment_props_{path_description.replace(" ", "_")}"
            env_asset_description = f"3D models for a {path_description} environment, including trees, rocks, and terrain features."
            model_path = self.nerf_renderer.generate_3d_asset_from_description(env_asset_description, env_asset_id)
            self.generated_assets[env_asset_id] = model_path
            print(f"Generated environment asset: {env_asset_id} at {model_path}")

        return self.generated_assets

# Example Usage:
# if __name__ == "__main__":
#     from llm_api_interface import LLMAPIInterface
#     from game_env.procedural_generator import ProceduralGenerator
#
#     class MockLLMAPI:
#         def generate_response(self, prompt, max_tokens, temperature):
#             if "design a new level" in prompt:
#                 return json.dumps({
#                     "grid_size": [10, 10],
#                     "num_waves": 5,
#                     "initial_cash": 150,
#                     "initial_lives": 20,
#                     "path_description": "a winding river valley",
#                     "tower_concept_descriptions": ["water elemental tower", "stone golem turret"],
#                     "enemy_concept_descriptions": ["river sprites", "swamp monsters"]
#                 })
#             return ""
#
#     mock_llm_api = MockLLMAPI()
#     nerf_renderer = NeRFSceneRenderer()
#     procedural_generator = ProceduralGenerator()
#     llm_game_designer = LLMGameDesigner(mock_llm_api, procedural_generator)
#
#     # Simulate getting game design parameters from LLM
#     game_design_params = llm_game_designer.design_game_level(theme="river", difficulty="easy")
#
#     asset_pipeline = NeRFAssetPipeline(nerf_renderer, llm_game_designer)
#     generated_assets = asset_pipeline.generate_assets_for_game(game_design_params)
#     print("\nAll Generated Assets:", generated_assets)


