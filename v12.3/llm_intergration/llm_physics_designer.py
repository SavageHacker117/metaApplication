
import json
from typing import Dict, Any, List

from .llm_api_interface import LLMAPIInterface

class LLMPhysicsDesigner:
    """
    Leverages the LLM to design physics-related parameters for game elements,
    such as projectile properties, material densities, or environmental forces.
    """
    def __init__(self, llm_api_interface: LLMAPIInterface):
        self.llm_api = llm_api_interface

    def design_projectile_properties(self, projectile_type: str, context: str) -> Dict[str, Any]:
        """
        Asks the LLM to suggest realistic or interesting physics properties for a projectile.
        """
        prompt = f"""You are a physics designer for a game. 
        Design the physics properties for a \"{projectile_type}\" projectile. 
        Consider the following context: {context}.
        Provide the following parameters in JSON format:
        - speed: float (initial speed in units/second)
        - mass: float (mass in kg)
        - damage: float (impact damage)
        - explosion_radius: float (radius of explosion, 0 if no explosion)
        - visual_effect: string (e.g., "smoke_trail", "fiery_explosion")
        
        Ensure the JSON is valid and contains only the specified keys.
        """
        llm_response = self.llm_api.generate_response(prompt, max_tokens=200, temperature=0.7)
        try:
            properties = json.loads(llm_response)
            return properties
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            return {}

    def suggest_environmental_forces(self, biome_type: str, current_game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asks the LLM to suggest environmental forces (e.g., wind, current) based on biome.
        """
        prompt = f"""Given a \"{biome_type}\" environment and the current game state:
        {json.dumps(current_game_state, indent=2)}

        Suggest any environmental forces that should be applied to objects in this biome.
        Provide your suggestions as a JSON object with force vectors (x, y, z) and their duration.
        Example: {{"wind": [0.5, 0, 0.1], "duration": 10.0}}.
        """
        llm_response = self.llm_api.generate_response(prompt, max_tokens=150, temperature=0.6)
        try:
            forces = json.loads(llm_response)
            return forces
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM response: {e}")
            return {}

# Example Usage:
# if __name__ == "__main__":
#     from llm_api_interface import LLMAPIInterface
#     class MockLLMAPI:
#         def generate_response(self, prompt, max_tokens, temperature):
#             if "design the physics properties for a \"missile\"" in prompt:
#                 return json.dumps({"speed": 50.0, "mass": 2.0, "damage": 100.0, "explosion_radius": 5.0, "visual_effect": "fiery_explosion"})
#             elif "environmental forces" in prompt:
#                 return json.dumps({"wind": [0.2, 0.1, 0], "duration": 60.0})
#             return "{}"
#
#     mock_llm = MockLLMAPI()
#     designer = LLMPhysicsDesigner(mock_llm)
#
#     missile_props = designer.design_projectile_properties("missile", "long-range, high-impact")
#     print("Missile Properties:", missile_props)
#
#     env_forces = designer.suggest_environmental_forces("desert", {"time_of_day": "noon"})
#     print("Environmental Forces:", env_forces)