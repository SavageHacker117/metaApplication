

import torch
import numpy as np
from typing import Tuple, Any, List

from .go_policy_value_net import GoPolicyValueNet
from .mcts import MCTS, state_to_nn_input
from go_env.minigo_go_env import MinigoGoEnv # For type hinting and env interaction

class GoAgent:
    """
    A unified Go agent that combines the neural network and MCTS for decision making.
    Can operate in self-play mode or play against other agents/humans.
    """
    def __init__(self, policy_value_net: GoPolicyValueNet, board_size: int, is_self_play: bool = False, n_playout: int = 1600, c_puct: float = 1.0):
        self.policy_value_net = policy_value_net
        self.board_size = board_size
        self.is_self_play = is_self_play
        self.mcts = MCTS(policy_value_net, c_puct=c_puct, n_playout=n_playout)

    def get_action(self, env: MinigoGoEnv, temp: float = 1e-3) -> Tuple[int, int]:
        """
        Chooses an action based on MCTS search.
        Args:
            env: The current Go environment.
            temp: Temperature parameter for exploration (higher means more exploration).
        Returns:
            Tuple[int, int]: The chosen move (row, col) or (None, None) for pass.
        """
        # Convert current board state to NN input format
        nn_input = state_to_nn_input(env.board, env.current_player, self.board_size)
        
        # Run MCTS to get action probabilities
        action_probs, legal_actions = self.mcts.get_action_probs(env, temp=temp)

        # Choose action based on probabilities
        if self.is_self_play: # During self-play, add noise for exploration
            # Dirichlet noise for exploration during self-play
            # This is a simplified version, typically applied to root node priors
            # For now, just choose proportionally to MCTS probabilities
            action_idx = np.random.choice(len(action_probs), p=action_probs)
        else: # During evaluation, choose the best action
            action_idx = np.argmax(action_probs)

        # Map action_idx back to (row, col) or (None, None)
        if action_idx == self.board_size * self.board_size: # Pass move
            action = (None, None)
        else:
            r = action_idx // self.board_size
            c = action_idx % self.board_size
            action = (r, c)
        
        # Update MCTS tree with the chosen move
        self.mcts.update_with_move(action)

        return action

    def reset_mcts(self):
        """
        Resets the MCTS tree for a new game.
        """
        self.mcts.update_with_move(-1) # -1 indicates a new game, resetting the tree

# Example Usage:
# if __name__ == "__main__":
#     board_size = 9
#     policy_value_net = GoPolicyValueNet(board_size)
#     agent = GoAgent(policy_value_net, board_size, is_self_play=True)
#     env = MinigoGoEnv(board_size=board_size)
#     env.reset()
#
#     print("Initial board:\n", env.render())
#
#     for _ in range(5): # Simulate a few moves
#         action = agent.get_action(env, temp=1.0)
#         print(f"Agent chooses action: {action}")
#         state, reward, done, info = env.step(action)
#         print("\nBoard after move:\n", env.render())
#         if done:
#             print("Game Over!")
#             break


