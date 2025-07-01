
from typing import Dict, Any, List, Tuple
from .llm_api_interface import LLMAPIInterface
from .go_llm_formatter import GoLLMFormatter
import json

class GoLLMAnalyzer:
    """
    Interacts with the LLM to analyze Go game states, moves, and strategies.
    It uses the GoLLMFormatter to prepare inputs for the LLM.
    """
    def __init__(self, llm_api_interface: LLMAPIInterface, board_size: int):
        self.llm_api = llm_api_interface
        self.formatter = GoLLMFormatter(board_size)

    def explain_move(self, board_state: List[List[int]], current_player: int, move: Tuple[int, int]) -> str:
        """
        Asks the LLM to explain a specific move in the context of the current board state.
        """
        board_diagram = self.formatter.format_to_text_diagram(board_state)
        move_str = f"({chr(ord("A") + move[1])}{move[0] + 1})" if move != (None, None) else "pass"
        player_str = "Black" if current_player == 1 else "White"

        prompt = f"""Given the following Go board state:
{board_diagram}

{player_str} just played {move_str}. Explain the strategic reasoning behind this move. 
Consider potential threats, territory, and influence. Keep the explanation concise and insightful.
"""
        response = self.llm_api.generate_response(prompt, max_tokens=200, temperature=0.7)
        return response

    def summarize_strategy(self, game_history_sgf: str) -> str:
        """
        Asks the LLM to summarize the overall strategy of a player or both players in a game.
        """
        prompt = f"""Analyze the following Go game (SGF format) and summarize the key strategic approaches 
        used by both Black and White players. Identify any turning points or notable tactical sequences.

        {game_history_sgf}
"""
        response = self.llm_api.generate_response(prompt, max_tokens=500, temperature=0.6)
        return response

    def spot_weaknesses(self, game_sequence_sgf: str) -> str:
        """
        Asks the LLM to identify weaknesses in a specific game sequence.
        """
        prompt = f"""Examine the following Go game sequence (SGF format). 
        Identify any weaknesses, mistakes, or missed opportunities for either player within this sequence.
        Provide actionable suggestions for improvement.

        {game_sequence_sgf}
"""
        response = self.llm_api.generate_response(prompt, max_tokens=300, temperature=0.7)
        return response

    def get_llm_move_suggestion(self, board_state: List[List[int]], current_player: int, legal_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """
        Asks the LLM for a move suggestion given the current board state and legal moves.
        The LLM should respond with a move in (row, col) format or 'pass'.
        """
        board_diagram = self.formatter.format_to_text_diagram(board_state)
        player_str = "Black" if current_player == 1 else "White"
        legal_moves_str = ", ".join([f"({chr(ord("A") + c)}{r + 1})" for r, c in legal_moves if (r,c) != (None,None)])
        if (None, None) in legal_moves:
            legal_moves_str += ", pass"

        prompt = f"""Given the following Go board state:
{board_diagram}

It is {player_str}'s turn. The legal moves are: {legal_moves_str}.

Suggest the best move for {player_str}. Respond only with the move in (row, col) format (e.g., (3,4)) or 'pass'.
"""
        response = self.llm_api.generate_response(prompt, max_tokens=20, temperature=0.1) # Low temperature for precise output
        
        # Attempt to parse the LLM's response
        response = response.strip().lower()
        if response == "pass":
            return (None, None)
        
        match = re.match(r'\((\d+),\s*(\d+)\)', response)
        if match:
            row = int(match.group(1))
            col = int(match.group(2))
            if (row, col) in legal_moves:
                return (row, col)
        
        print(f"LLM returned unparseable or illegal move: {response}. Defaulting to pass.")
        return (None, None) # Fallback to pass if LLM response is invalid

import re


