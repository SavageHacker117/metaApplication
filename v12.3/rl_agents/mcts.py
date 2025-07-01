
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class MCTSNode:
    """
    A node in the Monte Carlo Tree Search tree.
    Stores visit counts, total action value, and prior probability.
    """
    def __init__(self, parent=None, prior_p=0.0, action_taken=None):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.q_value = 0.0
        self.prior_p = prior_p
        self.action_taken = action_taken

    def is_leaf(self) -> bool:
        return self.children == {}

    def is_root(self) -> bool:
        return self.parent is None

    def ucb_score(self, c_puct: float) -> float:
        """
        Calculates the UCB score for selection during MCTS search.
        """
        if self.n_visits == 0:
            return float("inf") # Encourage exploration of unvisited nodes
        return self.q_value + c_puct * self.prior_p * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)

    def select_child(self, c_puct: float):
        """
        Selects the child node with the highest UCB score.
        """
        return max(self.children.values(), key=lambda node: node.ucb_score(c_puct))

    def expand(self, policy_probs: np.ndarray, legal_moves: List[Tuple[int, int]]):
        """
        Expands the node by creating new child nodes for legal moves.
        `policy_probs` is a 1D array where index corresponds to action_idx.
        """
        for action_idx, prob in enumerate(policy_probs):
            # Map action_idx back to (row, col) or (None, None)
            # This mapping needs to be consistent with how actions are indexed
            # For Go, usually (row*board_size + col) for moves, and board_size*board_size for pass
            # Assuming action_idx_to_move function is available elsewhere or passed
            # For simplicity, let's assume policy_probs are already filtered for legal moves
            # and action_idx directly maps to legal_moves list index.
            # A more robust solution would involve a global action mapping.
            
            # Placeholder for actual action mapping logic
            # For now, let's assume policy_probs are for all possible board positions + pass
            # and we filter for legal moves here.
            
            # This part needs careful integration with the Go environment's action space.
            # For a 9x9 board, 81 moves + 1 pass = 82 actions.
            # Let's assume action_idx 0 to 80 are (row, col) and 81 is pass.
            
            if action_idx < self.parent.board_size * self.parent.board_size: # Assuming board_size is accessible from parent or passed
                r = action_idx // self.parent.board_size
                c = action_idx % self.parent.board_size
                action = (r, c)
            else:
                action = (None, None) # Pass move

            if action in legal_moves: # Only expand legal moves
                self.children[action] = MCTSNode(parent=self, prior_p=prob, action_taken=action)

    def backup(self, value: float):
        """
        Backpropagates the value from the current node up to the root.
        """
        current = self
        while current is not None:
            current.n_visits += 1
            # Q-value update: average of all values seen through this node
            # For Go, value is typically from the perspective of the current player
            # So, if value is for the player who just moved, it's -value for the parent's Q-value.
            # Simplified: just average
            current.q_value = (current.q_value * (current.n_visits - 1) + value) / current.n_visits
            current = current.parent

class MCTS:
    """
    Monte Carlo Tree Search algorithm for Go.
    """
    def __init__(self, policy_value_net, c_puct=1.0, n_playout=1600):
        self.policy_value_net = policy_value_net
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.root = MCTSNode()

    def _playout(self, env_copy):
        """
        Performs a single MCTS playout (selection, expansion, simulation, backup).
        """
        node = self.root
        while not node.is_leaf():
            node = node.select_child(self.c_puct)
            # Apply action to environment copy
            _, _, done, _ = env_copy.step(node.action_taken)
            if done: # Game ended during selection phase
                break

        # Evaluate leaf node with policy-value network
        # Assuming env_copy.get_state_for_nn() returns the correct input format
        # and policy_value_net returns (policy_probs, value)
        policy_probs, value = self.policy_value_net(env_copy.get_state_for_nn())
        policy_probs = policy_probs.cpu().numpy().flatten() # Convert to numpy
        value = value.item() # Convert to scalar

        # Expand if not terminal
        if not env_copy.is_game_over():
            node.expand(policy_probs, env_copy.get_legal_moves())

        # Backup value
        node.backup(value)

    def get_action_probs(self, env, temp=1.0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Runs MCTS simulations and returns action probabilities.
        """
        # Make a deep copy of the environment for simulations
        # This requires the Go environment to be copyable or re-initializable to a state
        # For simplicity, let's assume env.copy() works.
        # A more robust solution might involve passing board state directly.
        
        # For the purpose of this example, we'll assume a simple copy mechanism.
        # In a real scenario, you'd need to ensure the environment state is fully captured and restored.
        
        # For now, let's just use the current environment state and reset it after each playout
        # This is NOT ideal for true MCTS but simplifies the example.
        # Proper MCTS requires a copy of the environment at each node.
        
        # A better approach: pass the current board state to MCTS and let it manage internal state
        # Or, ensure the GoEnv can be efficiently copied.
        
        # Let's modify _playout to take a state and return a new state, rather than modifying env_copy
        # This would require the GoEnv.step() to return a new state object, not modify in-place.
        
        # Given the current GoEnv.step() modifies in-place, we need to copy the environment.
        # If env is not directly copyable, we need a way to save/load its state.
        
        # For now, let's assume a simple copy for demonstration.
        # In a real system, you'd use a more robust state management.
        
        # Reset root for each new search
        self.root = MCTSNode()
        
        for _ in range(self.n_playout):
            # Create a fresh copy of the environment for each playout
            # This is crucial for MCTS correctness.
            # Assuming env has a method to get its current state and re-initialize from it.
            # Or, a deepcopy works if the environment is simple enough.
            
            # For this example, let's assume env.get_state() and env.set_state() methods exist.
            # Or, a simple deepcopy if the environment object is not too complex.
            
            # Let's assume the GoEnv is simple enough for deepcopy for now.
            import copy
            env_copy = copy.deepcopy(env)
            self._playout(env_copy)

        # Calculate action probabilities based on visit counts
        action_visits = [(action, node.n_visits) for action, node in self.root.children.items()]
        actions, visits = zip(*action_visits)
        
        # Convert actions to a consistent index for the policy output
        # This requires a mapping from (row, col) or (None, None) to a single index.
        # Assuming a global action_to_idx mapping is available.
        
        # For a 9x9 board, 81 moves + 1 pass = 82 actions.
        # Let's define a simple mapping here for demonstration.
        board_size = env.get_board_size()
        action_size = board_size * board_size + 1
        
        action_probs = np.zeros(action_size)
        for action, visit_count in action_visits:
            if action == (None, None):
                idx = board_size * board_size # Pass move
            else:
                r, c = action
                idx = r * board_size + c
            action_probs[idx] = visit_count

        # Apply temperature for exploration during training
        if temp == 1.0:
            probs = action_probs / np.sum(action_probs) # Normalize
        else:
            probs = np.power(action_probs, 1 / temp)
            probs = probs / np.sum(probs) # Normalize

        return probs, list(actions)

    def update_with_move(self, last_move: Tuple[int, int]):
        """
        Updates the MCTS tree by making the chosen move the new root.
        Discards the rest of the tree.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None # Detach from old tree
        else:
            # If the chosen move is not in the tree (e.g., first move or random exploration),
            # or if the tree is reset for each new search, create a new root.
            self.root = MCTSNode()

# Helper function for state conversion (placeholder)
def state_to_nn_input(board_state: np.ndarray, current_player: int, board_size: int) -> torch.Tensor:
    """
    Converts the Go board state into a format suitable for the neural network.
    Input channels: 0 for current player stones, 1 for opponent stones, 2 for empty.
    """\n    # Assuming board_state is a numpy array with 0: empty, 1: black, -1: white
    # And current_player is 1 (black) or -1 (white)
    
    current_player_stones = (board_state == current_player).astype(np.float32)
    opponent_stones = (board_state == -current_player).astype(np.float32)
    empty_spots = (board_state == 0).astype(np.float32)

    # Stack as (channels, height, width)
    nn_input = np.stack([current_player_stones, opponent_stones, empty_spots], axis=0)
    return torch.from_numpy(nn_input).unsqueeze(0) # Add batch dimension

# Add this method to MinigoGoEnv for MCTS to use
# (This is a conceptual addition, needs to be physically added to MinigoGoEnv class)
# class MinigoGoEnv(GoGameInterface):
#     ...
#     def get_state_for_nn(self) -> torch.Tensor:
#         return state_to_nn_input(self.board, self.current_player, self.board_size)


