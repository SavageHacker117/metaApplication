
from typing import Dict, Any, List
from .llm_api_interface import LLMAPIInterface

class LLMStrategyRefiner:
    """
    Uses the LLM to refine RL agent strategies or game generation parameters
    based on analysis and feedback.
    """
    def __init__(self, llm_api_interface: LLMAPIInterface):
        self.llm_api = llm_api_interface

    def refine_rl_parameters(self, analysis_report: str, current_rl_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asks the LLM to suggest refinements to RL training parameters.
        """
        prompt = f"""Based on the following analysis report of an RL agent's performance in Go:
{analysis_report}

And the current RL training configuration:
{current_rl_config}

Suggest specific adjustments to the RL parameters (e.g., learning_rate, mcts_n_playout, buffer_size, epochs). 
Provide your suggestions as a JSON object with only the parameters to change. 
Example: {{"learning_rate": 0.0005, "mcts_n_playout": 2000}}.
"""
        response = self.llm_api.generate_response(prompt, max_tokens=100, temperature=0.2)
        try:
            suggested_changes = json.loads(response)
            return suggested_changes
        except json.JSONDecodeError:
            print(f"LLM returned unparseable JSON for RL parameter refinement: {response}")
            return {}

    def suggest_curriculum_shift(self, game_analysis: str) -> str:
        """
        Asks the LLM to suggest changes to the training curriculum (e.g., board size, opponent strength).
        """
        prompt = f"""Based on the following analysis of game performance:
{game_analysis}

Suggest a suitable next step for the training curriculum. Should the board size increase? 
Should the opponent strength be adjusted? Provide a brief explanation.
"""
        response = self.llm_api.generate_response(prompt, max_tokens=150, temperature=0.7)
        return response

    def propose_new_reward_scheme(self, game_outcome_feedback: str) -> str:
        """
        Asks the LLM to propose modifications to the reward function.
        """
        prompt = f"""Given the following feedback on game outcomes:
{game_outcome_feedback}

Propose a new or modified reward scheme for the RL agent to encourage desired behaviors 
(e.g., capturing more stones, securing territory, early resignation). Explain your reasoning.
"""
        response = self.llm_api.generate_response(prompt, max_tokens=200, temperature=0.8)
        return response

import json


