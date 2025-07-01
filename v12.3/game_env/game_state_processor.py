
import numpy as np
from typing import Dict, Any, List

class GameStateProcessor:
    """
    Processes the raw game state from TowerDefenseEnv into a numerical format
    suitable for an RL agent. This can involve flattening the grid, encoding
    tower/enemy information, and normalizing values.
    """
    def __init__(self, grid_size: Tuple[int, int] = (10, 10)):
        self.grid_size = grid_size
        self.state_dim = (grid_size[0] * grid_size[1]) + 4 # Grid + cash, lives, wave, game_over

    def process_state(self, raw_state: Dict[str, Any]) -> np.ndarray:
        """
        Converts the dictionary-based raw game state into a flattened numpy array.
        """
        grid_flat = np.array(raw_state["grid"]).flatten()
        
        # Normalize cash and lives (example: divide by max possible values)
        cash_normalized = raw_state["cash"] / 1000.0 # Assuming max cash is 1000
        lives_normalized = raw_state["lives"] / 20.0 # Assuming max lives is 20
        wave_normalized = raw_state["wave_number"] / 100.0 # Assuming max waves is 100
        game_over_int = 1.0 if raw_state["game_over"] else 0.0

        # Combine all features into a single array
        processed_state = np.concatenate([
            grid_flat,
            [cash_normalized],
            [lives_normalized],
            [wave_normalized],
            [game_over_int]
        ])
        return processed_state

    def get_state_dimension(self) -> int:
        """
        Returns the dimension of the processed state vector.
        """
        return self.state_dim

# Example usage:
# if __name__ == "__main__":
#     # Dummy raw state for demonstration
#     dummy_raw_state = {
#         "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
#         "cash": 150,
#         "lives": 10,
#         "towers": {}, # Not directly used in this simple processor, but part of raw_state
#         "enemies": [], # Not directly used
#         "wave_number": 1,
#         "game_over": False
#     }
#     processor = GameStateProcessor(grid_size=(3,3))
#     processed = processor.process_state(dummy_raw_state)
#     print(f"Processed state: {processed}")
#     print(f"Processed state shape: {processed.shape}")
#     print(f"Expected state dimension: {processor.get_state_dimension()}")


