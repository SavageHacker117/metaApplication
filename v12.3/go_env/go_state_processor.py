
import numpy as np
import torch
from typing import Any, Tuple

class GoStateProcessor:
    """
    Processes the raw Go board state into a numerical format suitable for the
    neural network (GoPolicyValueNet). This typically involves creating channels
    for current player stones, opponent stones, and empty spaces.
    """
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.num_channels = 3 # Current player, opponent, empty

    def process_state(self, board_state: np.ndarray, current_player: int) -> torch.Tensor:
        """
        Converts a numpy array representation of the Go board into a PyTorch tensor
        with appropriate channels for the neural network input.

        Args:
            board_state (np.ndarray): A 2D numpy array representing the Go board.
                                      Values: 0 (empty), 1 (black stone), -1 (white stone).
            current_player (int): The player whose turn it is (1 for black, -1 for white).

        Returns:
            torch.Tensor: A PyTorch tensor of shape (1, num_channels, board_size, board_size)
                          ready for input to the neural network.
        """
        # Create channels based on current player's perspective
        current_player_stones = (board_state == current_player).astype(np.float32)
        opponent_stones = (board_state == -current_player).astype(np.float32)
        empty_spots = (board_state == 0).astype(np.float32)

        # Stack the channels to form the input tensor
        # Resulting shape: (num_channels, board_size, board_size)
        processed_state = np.stack([current_player_stones, opponent_stones, empty_spots], axis=0)

        # Convert to PyTorch tensor and add a batch dimension
        return torch.from_numpy(processed_state).unsqueeze(0)

    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Returns the expected input shape for the neural network: (channels, height, width).
        """
        return (self.num_channels, self.board_size, self.board_size)

# Example Usage:
# if __name__ == "__main__":
#     board_size = 9
#     processor = GoStateProcessor(board_size)
#
#     # Example board state (simplified)
#     dummy_board = np.zeros((board_size, board_size), dtype=int)
#     dummy_board[2, 2] = 1 # Black stone
#     dummy_board[3, 3] = -1 # White stone
#
#     current_player = 1 # Black's turn
#     nn_input = processor.process_state(dummy_board, current_player)
#
#     print(f"Processed state shape: {nn_input.shape}")
#     print(f"Expected input shape: {processor.get_input_shape()}")
#     print("Sample processed state (first channel - current player stones):\n", nn_input[0, 0, :, :])


