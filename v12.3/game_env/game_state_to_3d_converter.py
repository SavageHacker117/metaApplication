

from typing import Dict, Any, Tuple, List
import numpy as np

class GameStateTo3DConverter:
    """
    Converts the abstract game state (grid, towers, enemies) into a format
    that can be consumed by the 3D rendering engine (e.g., Three.js scene generator).
    This involves mapping 2D grid coordinates to 3D space and preparing object properties.
    """
    def __init__(self, cell_size: float = 10.0, base_height: float = 0.0):
        self.cell_size = cell_size
        self.base_height = base_height

    def convert_to_3d_representation(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts the game_state dictionary into a 3D-ready representation.
        """
        grid_3d = []
        for r_idx, row in enumerate(game_state["grid"]):
            row_3d = []
            for c_idx, cell_val in enumerate(row):
                # Map 2D grid cell to a 3D position (center of the cell)
                pos_x = c_idx * self.cell_size + self.cell_size / 2
                pos_y = r_idx * self.cell_size + self.cell_size / 2
                pos_z = self.base_height
                
                cell_3d_info = {
                    "type": "grid_cell",
                    "grid_coords": (r_idx, c_idx),
                    "value": cell_val,
                    "position": [pos_x, pos_y, pos_z]
                }
                row_3d.append(cell_3d_info)
            grid_3d.append(row_3d)

        towers_3d = []
        for (x, y), tower_info in game_state["towers"].items():
            pos_x = y * self.cell_size + self.cell_size / 2
            pos_y = x * self.cell_size + self.cell_size / 2
            pos_z = self.base_height + self.cell_size / 2 # Place tower above ground
            towers_3d.append({
                "type": "tower",
                "grid_coords": (x, y),
                "tower_type": tower_info["type"],
                "position": [pos_x, pos_y, pos_z]
            })

        enemies_3d = []
        for enemy_info in game_state["enemies"]:
            pos_x_grid, pos_y_grid = enemy_info["position"]
            pos_x = pos_y_grid * self.cell_size + self.cell_size / 2
            pos_y = pos_x_grid * self.cell_size + self.cell_size / 2
            pos_z = self.base_height + self.cell_size / 4 # Place enemy slightly above ground
            enemies_3d.append({
                "type": "enemy",
                "grid_coords": (pos_x_grid, pos_y_grid),
                "health": enemy_info["health"],
                "position": [pos_x, pos_y, pos_z]
            })

        return {
            "grid_3d": grid_3d,
            "towers_3d": towers_3d,
            "enemies_3d": enemies_3d,
            "cash": game_state["cash"],
            "lives": game_state["lives"],
            "wave_number": game_state["wave_number"],
            "game_over": game_state["game_over"]
        }

# Example Usage:
# if __name__ == "__main__":
#     converter = GameStateTo3DConverter()
#     # Dummy game state
#     dummy_game_state = {
#         "grid": [[0, 1, 0], [2, 0, 0], [0, 0, 4]],
#         "cash": 100,
#         "lives": 10,
#         "towers": {(1, 0): {"type": "basic"}},
#         "enemies": [{"position": (0, 1), "health": 50}],
#         "wave_number": 1,
#         "game_over": False
#     }
#
#     converted_3d_state = converter.convert_to_3d_representation(dummy_game_state)
#     import json
#     print(json.dumps(converted_3d_state, indent=2))


