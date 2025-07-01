
from typing import Dict, Any, List

class LLMPromptEngineer:
    """
    Responsible for constructing detailed and effective prompts for the LLM
    to guide its generation of game elements, actions, or narrative.
    """
    def __init__(self):
        pass

    def generate_game_design_prompt(self, desired_complexity: str, game_type: str = "Tower Defense") -> str:
        """
        Generates a prompt for the LLM to design a game configuration.
        """
        prompt = f"""You are an expert game designer for a {game_type} game. 
        Your task is to generate a detailed JSON configuration for a new game level. 
        The level should be of {desired_complexity} complexity. 
        
        Include the following keys in your JSON output:
        - 'grid_size': [width, height] (e.g., [10, 10])
        - 'initial_cash': integer (starting money for the player)
        - 'initial_lives': integer (starting lives for the player)
        - 'path': A list of [x, y] coordinates defining the enemy path from spawn to exit. 
                  The path must be continuous, within grid boundaries, and challenging.
        - 'spawn_point': [x, y] coordinate, must be the first point in 'path'.
        - 'exit_point': [x, y] coordinate, must be the last point in 'path'.
        - 'tower_types': A dictionary of available tower types with their base cost, range, and damage. 
                         Example: {{"basic": {{"cost": 50, "range": 2, "damage": 10}}}}
        - 'enemy_waves': A list of dictionaries, each defining a wave with 'num_enemies' and 'enemy_type' (e.g., 'grunt', 'fast', 'tank').

        Ensure the path is valid and all coordinates are within the grid_size. 
        Make sure the generated configuration is a valid JSON object and nothing else.
        """
        return prompt

    def generate_action_command_prompt(self, game_state_text: str, rl_suggestion: str) -> str:
        """
        Generates a prompt for the LLM to convert an RL agent's suggestion
        into a precise, executable game command.
        """
        prompt = f"""Given the current Tower Defense game state:
{game_state_text}

The Reinforcement Learning agent suggests the following high-level action: {rl_suggestion}.

Translate this high-level suggestion into a precise, executable game command in natural language. 

Examples:
- If the suggestion is to 'place basic tower', respond with 'place basic tower at (X,Y)' where X and Y are valid, empty grid coordinates.
- If the suggestion is to 'start wave', respond with 'start wave'.
- If the suggestion is to 'upgrade tower', respond with 'upgrade tower at (X,Y)'.
- If the suggestion is 'no_op', respond with 'do nothing'.

Your response should be only the command, without any additional explanation or preamble.
"""
        return prompt

# Example Usage:
# if __name__ == "__main__":
#     engineer = LLMPromptEngineer()
#     
#     # Test game design prompt
#     design_prompt = engineer.generate_game_design_prompt("hard")
#     print("\n--- Game Design Prompt ---")
#     print(design_prompt)
#
#     # Test action command prompt
#     dummy_game_state = "Grid: [[0,0],[0,0]], Cash: 100, Lives: 10"
#     rl_suggest = "place basic tower"
#     action_prompt = engineer.generate_action_command_prompt(dummy_game_state, rl_suggest)
#     print("\n--- Action Command Prompt ---")
#     print(action_prompt)


