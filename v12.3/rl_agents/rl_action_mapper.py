
from typing import Dict, Any, List

class RLActionMapper:
    """
    Maps the discrete action output from the RL agent to a meaningful
    high-level game action or LLM prompt strategy.
    """
    def __init__(self, action_space_definition: List[str]):
        self.action_space_definition = action_space_definition
        self.action_to_idx = {action: i for i, action in enumerate(action_space_definition)}
        self.idx_to_action = {i: action for i, action in enumerate(action_space_definition)}

    def map_idx_to_action(self, action_idx: int) -> str:
        """
        Converts an integer action index from the RL agent to its string representation.
        """
        if action_idx < 0 or action_idx >= len(self.action_space_definition):
            return "no_op" # Fallback for invalid index
        return self.idx_to_action.get(action_idx, "no_op")

    def map_action_to_idx(self, action_str: str) -> int:
        """
        Converts a string action to its integer index.
        """
        return self.action_to_idx.get(action_str, -1) # -1 for unknown action

    def get_action_space_size(self) -> int:
        """
        Returns the total number of discrete actions in the action space.
        """
        return len(self.action_space_definition)

# Example Usage:
# if __name__ == "__main__":
#     # Define the possible high-level actions the RL agent can choose from
#     td_actions = [
#         "place basic tower",
#         "place archer tower",
#         "place cannon tower",
#         "start wave",
#         "upgrade tower",
#         "do nothing" # Corresponds to no_op
#     ]
#     mapper = RLActionMapper(td_actions)

#     # RL agent outputs index 0
#     rl_output_idx = 0
#     mapped_action = mapper.map_idx_to_action(rl_output_idx)
#     print(f"RL agent output {rl_output_idx} maps to: {mapped_action}")

#     # LLM generates a command, we want to know its index
#     llm_command_str = "start wave"
#     mapped_idx = mapper.map_action_to_idx(llm_command_str)
#     print(f"LLM command '{llm_command_str}' maps to index: {mapped_idx}")


