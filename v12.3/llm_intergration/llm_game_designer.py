

import json
from typing import Dict, Any, List, Tuple

from .llm_api_interface import LLMAPIInterface
from game_env.procedural_generator import ProceduralGenerator

class LLMGameDesigner:
    """
    Leverages the LLM to act as a game designer, generating high-level game concepts,
    specific level parameters, or even narrative elements for the Tower Defense game.
    It can guide the procedural generator.
    """
    def __init__(self, llm_api_interface: LLMAPIInterface, procedural_generator: ProceduralGenerator):
        self.llm_api = llm_api_interface
        self.procedural_generator = procedural_generator

    def design_game_level(self, theme: str = "fantasy", difficulty: str = "medium") -> Dict[str, Any]:
        """
        Asks the LLM to design a game level based on a theme and difficulty.
        The LLM provides high-level parameters which are then used to guide procedural generation.
        """
        prompt = f"""You are a game designer for a Tower Defense game. 
        Design a new level with a \"{theme}\" theme and \"{difficulty}\" difficulty. 
        Provide the following parameters in JSON format:
        - grid_size: [width, height] (e.g., [15, 15])
        - num_waves: integer (number of enemy waves)
        - initial_cash: integer
        - initial_lives: integer
        - path_description: string (a textual description of the desired enemy path, e.g., "a winding path through a forest")
        - tower_concept_descriptions: list of strings (e.g., ["a magical arcane tower", "a sturdy stone turret"])
        - enemy_concept_descriptions: list of strings (e.g., ["small goblins", "armored trolls"])
        
        Ensure the JSON is valid and contains only the specified keys.
        """
        llm_response = self.llm_api.generate_response(prompt, max_tokens=500, temperature=0.8)
        
        try:
            design_params = json.loads(llm_response)
            
            # Use LLM suggestions to guide procedural generation
            grid_size = tuple(design_params.get("grid_size", self.procedural_generator.grid_size))
            self.procedural_generator.grid_size = grid_size # Update generator's grid size

            # Path generation (simplified: LLM description is not directly used for pathfinding)
            # In a more advanced system, LLM description could guide a pathfinding algorithm
            path = self.procedural_generator.generate_random_path() 
            grid_layout = self.procedural_generator.generate_grid_layout(path)

            enemy_waves = self.procedural_generator.generate_enemy_waves(design_params.get("num_waves", 5))
            tower_types = self.procedural_generator.generate_tower_types(len(design_params.get("tower_concept_descriptions", ["basic"])))

            return {
                "grid_size": grid_size,
                "initial_cash": design_params.get("initial_cash", 150),
                "initial_lives": design_params.get("initial_lives", 20),
                "grid": grid_layout,
                "path": path,
                "enemy_waves": enemy_waves,
                "tower_types": tower_types,
                "llm_path_description": design_params.get("path_description"),
                "llm_tower_concepts": design_params.get("tower_concept_descriptions"),
                "llm_enemy_concepts": design_params.get("enemy_concept_descriptions")
            }
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"LLM Response: {llm_response}")
            return self._default_game_design() # Fallback
        except Exception as e:
            print(f"Error processing LLM design: {e}")
            return self._default_game_design()

    def _default_game_design(self) -> Dict[str, Any]:
        """
        Provides a default game design as a fallback.
        """
        path = self.procedural_generator.generate_random_path()
        grid_layout = self.procedural_generator.generate_grid_layout(path)
        return {
            "grid_size": self.procedural_generator.grid_size,
            "initial_cash": 150,
            "initial_lives": 20,
            "grid": grid_layout,
            "path": path,
            "enemy_waves": self.procedural_generator.generate_enemy_waves(),
            "tower_types": self.procedural_generator.generate_tower_types(),
            "llm_path_description": "default winding path",
            "llm_tower_concepts": ["basic tower"],
            "llm_enemy_concepts": ["basic enemy"]
        }

# Example Usage:
# if __name__ == "__main__":
#     from llm_api_interface import LLMAPIInterface
#     # Mock LLM API for testing
#     class MockLLMAPI:
#         def generate_response(self, prompt, max_tokens, temperature):
#             if "design a new level" in prompt:
#                 return json.dumps({
#                     "grid_size": [12, 12],
#                     "num_waves": 7,
#                     "initial_cash": 200,
#                     "initial_lives": 25,
#                     "path_description": "a serpentine path through ancient ruins",
#                     "tower_concept_descriptions": ["ancient guardian statue", "crystal turret"],
#                     "enemy_concept_descriptions": ["skeletal warriors", "shadow beasts"]
#                 })
#             return ""
#
#     mock_llm = MockLLMAPI()
#     proc_gen = ProceduralGenerator()
#     designer = LLMGameDesigner(mock_llm, proc_gen)
#     
#     game_design = designer.design_game_level(theme="ancient ruins", difficulty="hard")
#     print("\nGenerated Game Design:", json.dumps(game_design, indent=2))


