

import json
from typing import Dict, Any, List

from .llm_api_interface import LLMAPIInterface
from game_env.enemy_attributes import EnemyAttributes

class LLMEnemyDesigner:
    """
    Leverages the LLM to design detailed enemy attributes and behaviors,
    including their interaction with different terrain types and special abilities.
    """
    def __init__(self, llm_api_interface: LLMAPIInterface):
        self.llm_api = llm_api_interface
        self.enemy_attributes_base = EnemyAttributes()

    def design_enemy_type(self, enemy_concept: str, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asks the LLM to design a new enemy type based on a concept and game context.
        """
        prompt = f"""You are an enemy designer for a Tower Defense game. 
        Design a new enemy type based on the concept: \"{enemy_concept}\". 
        Consider the following game context: {json.dumps(game_context, indent=2)}.
        
        Provide the following parameters in JSON format:
        - name: string (e.g., "Armored Orc", "Flying Drone")
        - health: float
        - speed: float
        - damage: float
        - type: string ("air", "land", "water")
        - model: string (path to 3D model, e.g., "armored_orc.obj")
        - can_traverse: list of strings (e.g., ["land", "water"])
        - special_ability: string (optional, e.g., "heals_nearby_enemies", "explodes_on_death")
        
        Ensure the JSON is valid and contains only the specified keys.
        """
        llm_response = self.llm_api.generate_response(prompt, max_tokens=300, temperature=0.8)
        try:
            designed_enemy = json.loads(llm_response)
            # Validate against base attributes or add defaults
            base_attrs = self.enemy_attributes_base.get_attributes("grunt") # Use grunt as a base template
            for key, default_val in base_attrs.items():
                if key not in designed_enemy:
                    designed_enemy[key] = default_val
            return designed_enemy
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            return {}

# Example Usage:
# if __name__ == "__main__":
#     from llm_api_interface import LLMAPIInterface
#     class MockLLMAPI:
#         def generate_response(self, prompt, max_tokens, temperature):
#             if "design a new enemy type" in prompt:
#                 return json.dumps({
#                     "name": "Giant Spider",
#                     "health": 180.0,
#                     "speed": 1.8,
#                     "damage": 15.0,
#                     "type": "land",
#                     "model": "giant_spider.obj",
#                     "can_traverse": ["land", "mountain"],
#                     "special_ability": "web_slow"
#                 })
#             return "{}"
#
#     mock_llm = MockLLMAPI()
#     designer = LLMEnemyDesigner(mock_llm)
#
#     game_context = {"current_wave": 5, "terrain_type": "forest"}
#     enemy_design = designer.design_enemy_type("a stealthy forest creature", game_context)
#     print("Designed Enemy:", enemy_design)


