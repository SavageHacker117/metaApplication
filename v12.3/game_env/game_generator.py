

from typing import Dict, Any, List, Tuple
from ..llm_integration.llm_api_interface import LLMAPIInterface
import json

class GameGenerator:
    """
    Uses an LLM to generate configurations for the Tower Defense game environment.
    This includes grid layouts, enemy paths, and initial game parameters.
    """
    def __init__(self, llm_api_interface: LLMAPIInterface):
        self.llm_api = llm_api_interface

    def generate_game_config(self, complexity_level: str = "medium") -> Dict[str, Any]:
        """
        Generates a new game configuration based on a complexity level.
        The LLM will output a JSON string representing the game setup.
        """
        prompt = f"""Generate a JSON configuration for a Tower Defense game environment. 
        The configuration should include: 
        - 'grid_size': [width, height] (e.g., [10, 10])
        - 'initial_cash': integer
        - 'initial_lives': integer
        - 'path': A list of [x, y] coordinates defining the enemy path from spawn to exit. 
                  The path must be continuous and within grid boundaries.
        - 'spawn_point': [x, y] coordinate, must be the first point in 'path'.
        - 'exit_point': [x, y] coordinate, must be the last point in 'path'.
        - 'tower_types': A dictionary of available tower types with their base cost, range, and damage.
        - 'enemy_waves': A list of dictionaries, each defining a wave with 'num_enemies' and 'enemy_health'.

        Ensure the path is valid and all coordinates are within the grid_size. 
        Complexity level: {complexity_level}. Make it challenging but fair.
        """
        
        llm_response = self.llm_api.generate_response(prompt, max_tokens=1000, temperature=0.8)
        
        try:
            game_config = json.loads(llm_response)
            # Basic validation (more robust validation would be needed)
            if not all(k in game_config for k in ["grid_size", "initial_cash", "initial_lives", "path", "spawn_point", "exit_point", "tower_types", "enemy_waves"]):
                raise ValueError("Missing essential keys in generated config.")
            return game_config
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            print(f"LLM Response: {llm_response}")
            return self._default_game_config() # Fallback to a default config
        except ValueError as e:
            print(f"Validation error for generated config: {e}")
            return self._default_game_config()

    def _default_game_config(self) -> Dict[str, Any]:
        """
        Provides a default, hardcoded game configuration as a fallback.
        """
        return {
            "grid_size": [10, 10],
            "initial_cash": 150,
            "initial_lives": 20,
            "path": [[0,0], [0,1], [0,2], [1,2], [2,2], [2,3], [2,4], [3,4], [4,4], [4,5], [4,6], [5,6], [6,6], [6,7], [6,8], [7,8], [8,8], [9,8], [9,9]],
            "spawn_point": [0,0],
            "exit_point": [9,9],
            "tower_types": {
                "basic": {"cost": 50, "range": 2, "damage": 10},
                "archer": {"cost": 75, "range": 3, "damage": 15},
                "cannon": {"cost": 120, "range": 2, "damage": 30}
            },
            "enemy_waves": [
                {"num_enemies": 5, "enemy_health": 50},
                {"num_enemies": 8, "enemy_health": 70},
                {"num_enemies": 10, "enemy_health": 100}
            ]
        }

# Example usage:
# if __name__ == "__main__":
#     # Dummy LLMAPIInterface for testing
#     class MockLLMAPIInterface:
#         def generate_response(self, prompt, max_tokens, temperature):
#             # Simulate a valid JSON response
#             return json.dumps({
#                 "grid_size": [8, 8],
#                 "initial_cash": 200,
#                 "initial_lives": 15,
#                 "path": [[0,0], [1,0], [2,0], [2,1], [2,2], [3,2], [4,2], [4,3], [4,4], [5,4], [6,4], [6,5], [6,6], [7,6], [7,7]],
#                 "spawn_point": [0,0],
#                 "exit_point": [7,7],
#                 "tower_types": {
#                     "laser": {"cost": 100, "range": 4, "damage": 20}
#                 },
#                 "enemy_waves": [
#                     {"num_enemies": 7, "enemy_health": 60}
#                 ]
#             })

#     llm_mock = MockLLMAPIInterface()
#     generator = GameGenerator(llm_mock)
#     config = generator.generate_game_config("hard")
#     print(json.dumps(config, indent=2))


