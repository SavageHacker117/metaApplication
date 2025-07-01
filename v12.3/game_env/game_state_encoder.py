
import numpy as np
from typing import Dict, Any, Tuple

class GameStateEncoder:
    """
    Encodes the raw game state from TowerDefenseEnv into a numerical vector
    that serves as the observation for the RL agent. This involves representing
    the grid, player stats, and enemy/tower positions in a consistent format.
    """
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size
        # Define the size of the observation space
        # Grid (flattened) + Cash + Lives + Wave Number + Game Over status
        self.observation_dim = (grid_size[0] * grid_size[1]) + 4

    def encode(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Converts the dictionary-based game state into a fixed-size numpy array.
        """
        grid_flat = np.array(game_state["grid"]).flatten()
        
        # Normalize numerical values to be between 0 and 1
        cash_normalized = game_state["cash"] / 1000.0  # Assuming max cash is 1000
        lives_normalized = game_state["lives"] / 20.0  # Assuming max lives is 20
        wave_normalized = game_state["wave_number"] / 50.0 # Assuming max waves is 50
        game_over_encoded = 1.0 if game_state["game_over"] else 0.0

        # Combine all features into a single observation vector
        observation = np.concatenate([
            grid_flat,
            [cash_normalized],
            [lives_normalized],
            [wave_normalized],
            [game_over_encoded]
        ])
        return observation

    def get_observation_dimension(self) -> int:
        """
        Returns the dimension of the encoded observation vector.
        """
        return self.observation_dim

# Example usage:
# if __name__ == "__main__":
#     # Dummy game state for demonstration
#     dummy_game_state = {
#         "grid": [[0, 1, 0], [2, 0, 0], [0, 0, 4]],
#         "cash": 120,
#         "lives": 15,
#         "towers": {}, 
#         "enemies": [], 
#         "wave_number": 3,
#         "game_over": False
#     }
#     encoder = GameStateEncoder(grid_size=(3,3))
#     encoded_state = encoder.encode(dummy_game_state)
#     print(f"Encoded state: {encoded_state}")
#     print(f"Encoded state shape: {encoded_state.shape}")
#     print(f"Observation dimension: {encoder.get_observation_dimension()}")


