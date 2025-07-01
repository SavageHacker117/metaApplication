
import json
from typing import Dict, Any

from .llm_api_interface import LLMAPIInterface

class LLMFeedbackLoop:
    """
    Manages the feedback loop where LLM analyzes game test results and provides
    actionable insights or parameter adjustments for the next iteration of game generation.
    """
    def __init__(self, llm_api_interface: LLMAPIInterface):
        self.llm_api = llm_api_interface

    def get_refinement_suggestions(self, test_report: Dict[str, Any], current_game_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sends the test report to the LLM and asks for suggestions to refine game parameters.
        """
        prompt = f"""Based on the following game test report:
{json.dumps(test_report, indent=2)}

And the current game generation parameters:
{json.dumps(current_game_params, indent=2)}

Provide specific, actionable suggestions to improve the game. Focus on adjusting parameters related to:
- Terrain generation (e.g., heightmap_complexity, water_coverage, buildable_area_density)
- Enemy attributes (e.g., health, speed, special_ability)
- Projectile physics (e.g., speed, damage, explosion_radius)
- Game balance (e.g., initial_cash, initial_lives, wave compositions)

Respond with a JSON object containing only the parameters you suggest changing and their new values.
Example: {{"terrain_generation": {{"heightmap_complexity": 0.7}}, "enemy_attributes": {{"tank": {{"health": 250}}}}}}
"""
        llm_response = self.llm_api.generate_response(prompt, max_tokens=500, temperature=0.5)
        try:
            suggestions = json.loads(llm_response)
            return suggestions
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from LLM feedback: {e}")
            return {}

    def get_qualitative_feedback(self, test_report: Dict[str, Any]) -> str:
        """
        Asks the LLM for qualitative feedback on the game, focusing on fun, engagement, and overall design.
        """
        prompt = f"""Review the following game test report:
{json.dumps(test_report, indent=2)}

Provide qualitative feedback on the game. Discuss aspects like:
- Overall fun factor and engagement.
- Strategic depth and player choices.
- Visual appeal (based on descriptions).
- Any emergent behaviors or unexpected outcomes.

Keep the feedback concise and constructive.
"""
        llm_response = self.llm_api.generate_response(prompt, max_tokens=400, temperature=0.7)
        return llm_response

# Example Usage:
# if __name__ == "__main__":
#     from llm_api_interface import LLMAPIInterface
#     class MockLLMAPI:
#         def generate_response(self, prompt, max_tokens, temperature):
#             if "refinement suggestions" in prompt:
#                 return json.dumps({
#                     "terrain_generation": {"heightmap_complexity": 0.6},
#                     "enemy_attributes": {"plane": {"speed": 6.0}},
#                     "game_balance": {"initial_cash": 180}
#                 })
#             elif "qualitative feedback" in prompt:
#                 return "The game is challenging but fair. The terrain feels a bit flat, and enemies could use more varied movement patterns."
#             return "{}"
#
#     mock_llm = MockLLMAPI()
#     feedback_loop = LLMFeedbackLoop(mock_llm)
#
#     dummy_test_report = {"test_results": [{"final_lives": 0, "waves_survived": 3}], "llm_analysis": "Level too hard."}
#     dummy_game_params = {"terrain_generation": {"heightmap_complexity": 0.8}, "enemy_attributes": {"plane": {"speed": 5.0}}}
#
#     suggestions = feedback_loop.get_refinement_suggestions(dummy_test_report, dummy_game_params)
#     print("Refinement Suggestions:", suggestions)
#
#     qualitative_feedback = feedback_loop.get_qualitative_feedback(dummy_test_report)
#     print("Qualitative Feedback:", qualitative_feedback)