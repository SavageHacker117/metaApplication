

from typing import Dict, Any, List, Tuple

class GameVisualDataProcessor:
    """
    Processes the raw game state into a format optimized for visual rendering,
    including positions, health bars, and model references for 3D objects.
    """
    def __init__(self, cell_size: float = 10.0):
        self.cell_size = cell_size

    def process_for_rendering(self, game_state: Dict[str, Any], terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts the game state and terrain data into a format suitable for the 3D renderer.
        This includes calculating 3D positions, scaling, and adding visual attributes like health bars.
        """
        render_data = {
            "player_stats": {
                "cash": game_state["cash"],
                "lives": game_state["lives"],
                "score": game_state["score"]
            },
            "terrain": {
                "heightmap": terrain_data["heightmap"],
                "biome_map": terrain_data["biome_map"],
                "grid_size": terrain_data["grid_size"] # Assuming terrain_data has grid_size
            },
            "objects": []
        }

        # Add grid cells with terrain information
        for r_idx, row in enumerate(game_state["grid"]):
            for c_idx, cell_val in enumerate(row):
                pos_x = c_idx * self.cell_size + self.cell_size / 2
                pos_y = r_idx * self.cell_size + self.cell_size / 2
                pos_z = terrain_data["heightmap"][r_idx][c_idx] # Use terrain height
                
                cell_type = "empty"
                if cell_val == 1: cell_type = "path"
                elif cell_val == 2: cell_type = "buildable"
                elif cell_val == 3: cell_type = "spawn"
                elif cell_val == 4: cell_type = "exit"

                render_data["objects"].append({
                    "id": f"grid_cell_{r_idx}_{c_idx}",
                    "type": "grid_cell",
                    "grid_coords": (r_idx, c_idx),
                    "position": [pos_x, pos_y, pos_z],
                    "cell_type": cell_type,
                    "biome": terrain_data["biome_map"][r_idx][c_idx]
                })

        # Add towers
        for (x, y), tower_info in game_state["towers"].items():
            pos_x = y * self.cell_size + self.cell_size / 2
            pos_y = x * self.cell_size + self.cell_size / 2
            pos_z = terrain_data["heightmap"][x][y] + self.cell_size * 0.5 # Place tower above terrain
            render_data["objects"].append({
                "id": f"tower_{x}_{y}",
                "type": "tower",
                "position": [pos_x, pos_y, pos_z],
                "tower_type": tower_info["type"],
                "health": tower_info["health"],
                "max_health": tower_info["max_health"],
                "model_ref": tower_info.get("model", "default_tower_model.obj") # Reference to 3D model
            })

        # Add enemies
        for enemy_info in game_state["enemies"]:
            pos_x_grid, pos_y_grid = enemy_info["position"]
            pos_x = pos_y_grid * self.cell_size + self.cell_size / 2
            pos_y = pos_x_grid * self.cell_size + self.cell_size / 2
            pos_z = terrain_data["heightmap"][pos_x_grid][pos_y_grid] + self.cell_size * 0.2 # Place enemy above terrain
            render_data["objects"].append({
                "id": enemy_info["id"],
                "type": "enemy",
                "position": [pos_x, pos_y, pos_z],
                "enemy_type": enemy_info["type"],
                "health": enemy_info["health"],
                "max_health": enemy_info["max_health"],
                "model_ref": enemy_info.get("model", "default_enemy_model.obj") # Reference to 3D model
            })

        # Add interactive elements (e.g., walls)
        for (x, y), element_info in game_state["interactive_elements"].items():
            pos_x = y * self.cell_size + self.cell_size / 2
            pos_y = x * self.cell_size + self.cell_size / 2
            pos_z = terrain_data["heightmap"][x][y] + self.cell_size * 0.1 # Place element above terrain
            render_data["objects"].append({
                "id": element_info["id"],
                "type": element_info["type"],
                "position": [pos_x, pos_y, pos_z],
                "health": element_info["health"],
                "max_health": element_info["max_health"],
                "model_ref": element_info.get("model", "default_wall_model.obj") # Reference to 3D model
            })

        # Add projectiles (if any)
        # This would require the GameLoopManager to expose active projectiles
        # For now, assuming projectiles are handled directly by the renderer based on physics updates

        return render_data

# Example Usage:
# if __name__ == "__main__":
#     # Dummy game state and terrain data
#     dummy_game_state = {
#         "grid_size": (10, 10),
#         "grid": [[0 for _ in range(10)] for _ in range(10)],
#         "cash": 100, "lives": 10, "score": 0,
#         "towers": {(2, 2): {"type": "basic", "health": 100, "max_health": 100}},
#         "enemies": [{
#             "id": "enemy_0", "type": "tank", "position": (0, 0),
#             "health": 200, "max_health": 200, "model": "tank_model.obj"
#         }],
#         "interactive_elements": {(5, 5): {"id": "wall_0", "type": "destroyable_wall", "position": (5, 5), "health": 50, "max_health": 100}},
#         "wave_number": 1, "game_over": False
#     }
#     dummy_terrain_data = {
#         "heightmap": [[float(i + j) for j in range(10)] for i in range(10)],
#         "biome_map": [["land" for _ in range(10)] for _ in range(10)],
#         "grid_size": (10, 10)
#     }
#     
#     processor = GameVisualDataProcessor()
#     visual_data = processor.process_for_rendering(dummy_game_state, dummy_terrain_data)
#     import json
#     print(json.dumps(visual_data, indent=2))


