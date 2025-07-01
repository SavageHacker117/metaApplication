

import json
from typing import List, Tuple, Dict, Any

class GoLLMFormatter:
    """
    Formats Go board states and game information into various representations
    suitable for LLM input, such as JSON, SGF (Simplified Go Format), or text diagrams.
    """
    def __init__(self, board_size: int):
        self.board_size = board_size

    def format_to_json(self, board_state: List[List[int]], current_player: int, last_move: Tuple[int, int] = None) -> str:
        """
        Formats the Go board state and relevant game info into a JSON string.
        """
        data = {
            "board_size": self.board_size,
            "board_state": board_state, # 0: empty, 1: black, -1: white
            "current_player": "black" if current_player == 1 else "white",
            "last_move": {"row": last_move[0], "col": last_move[1]} if last_move else None
        }
        return json.dumps(data, indent=2)

    def format_to_sgf(self, moves: List[Tuple[int, int]], board_size: int, initial_player: int = 1) -> str:
        """
        Generates a basic SGF (Smart Game Format) string from a list of moves.
        This is a simplified SGF generator.
        """
        sgf_string = f"(;GM[1]FF[4]CA[UTF-8]AP[GoLLM]SZ[{board_size}]"\
                     f"PB[Black]PW[White]KM[6.5]" # Basic SGF header
        
        current_player = initial_player
        for move in moves:
            if move == (None, None): # Pass move
                sgf_string += f";{chr(ord("W") if current_player == -1 else ord("B"))}[tt]" # 'tt' for pass
            else:
                row, col = move
                # SGF coordinates are 0-indexed, a-z, aa-zz
                sgf_col = chr(ord("a") + col)
                sgf_row = chr(ord("a") + row)
                sgf_string += f";{chr(ord("W") if current_player == -1 else ord("B"))}[{sgf_col}{sgf_row}]"
            current_player *= -1 # Switch player
        
        sgf_string += ")"
        return sgf_string

    def format_to_text_diagram(self, board_state: List[List[int]]) -> str:
        """
        Generates a simple ASCII text diagram of the Go board.
        """
        diagram = []
        # Column headers
        diagram.append("  " + " ".join([chr(ord("A") + i) for i in range(self.board_size)]))
        diagram.append(" +" + "--" * self.board_size + "-")

        for r_idx, row in enumerate(board_state):
            row_str = f"{r_idx % 10}|"
            for cell_val in row:
                if cell_val == 1: # Black
                    row_str += "X "
                elif cell_val == -1: # White
                    row_str += "O "
                else:
                    row_str += ". "
            diagram.append(row_str + "|")
        diagram.append(" +" + "--" * self.board_size + "-")
        return "\n".join(diagram)

# Example Usage:
# if __name__ == "__main__":
#     formatter = GoLLMFormatter(board_size=9)
#     
#     # Example board state (simplified)
#     dummy_board = [
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, -1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ]
#
#     # Test JSON formatting
#     json_output = formatter.format_to_json(dummy_board, 1, (2,2))
#     print("\n--- JSON Format ---")
#     print(json_output)
#
#     # Test SGF formatting
#     moves_history = [(2,2), (3,3), (None, None)] # Black plays (2,2), White plays (3,3), Black passes
#     sgf_output = formatter.format_to_sgf(moves_history, 9)
#     print("\n--- SGF Format ---")
#     print(sgf_output)
#
#     # Test Text Diagram formatting
#     text_diagram = formatter.format_to_text_diagram(dummy_board)
#     print("\n--- Text Diagram Format ---")
#     print(text_diagram)


