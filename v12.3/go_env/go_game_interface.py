
from abc import ABC, abstractmethod
from typing import List, Tuple, Any

class GoGameInterface(ABC):
    """
    Abstract base class defining the interface for interacting with a Go game environment.
    This allows different Go engine implementations (e.g., python-go, KataGo) to be used interchangeably.
    """

    @abstractmethod
    def reset(self) -> Any:
        """
        Resets the Go game to its initial state.
        Returns the initial board state.
        """
        pass

    @abstractmethod
    def step(self, action: Tuple[int, int]) -> Tuple[Any, float, bool, Any]:
        """
        Applies an action (move) to the Go environment.
        Args:
            action: A tuple (row, col) representing the move, or (None, None) for a pass.
        Returns:
            Tuple: (next_state, reward, done, info)
            - next_state: The new board state after the action.
            - reward: The reward received from the environment.
            - done: True if the episode has ended, False otherwise.
            - info: A dictionary containing additional information (e.g., current player, legal moves).
        """
        pass

    @abstractmethod
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """
        Returns a list of legal moves for the current player.
        """
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        """
        Returns the current player (e.g., 1 for Black, -1 for White).
        """
        pass

    @abstractmethod
    def get_board_size(self) -> int:
        """
        Returns the size of the Go board (e.g., 19 for 19x19).
        """
        pass

    @abstractmethod
    def render(self) -> str:
        """
        Returns a string representation of the current board state for display.
        """
        pass

    @abstractmethod
    def get_state_as_json(self) -> str:
        """
        Returns the current board state as a JSON string.
        """
        pass

    @abstractmethod
    def get_score(self) -> float:
        """
        Returns the current score of the game.
        """
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """
        Checks if the game has ended.
        """
        pass


