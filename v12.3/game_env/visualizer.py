

from typing import Dict, Any, List, Tuple

class GameVisualizer:
    """
    Renders the Tower Defense game state for human readability.
    Can be extended to use graphical libraries (e.g., Pygame, Matplotlib) for richer visualization.
    """
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size

    def render(self, game_state: Dict[str, Any]) -> str:
        """
        Generates a text-based representation of the game grid and key statistics.
        """
        grid = game_state["grid"]
        cash = game_state["cash"]
        lives = game_state["lives"]
        wave_number = game_state["wave_number"]
        game_over = game_state["game_over"]

        display_grid = []
        for r_idx, row in enumerate(grid):
            display_row = []
            for c_idx, cell_val in enumerate(row):
                if cell_val == 0: # Empty
                    display_row.append(". ")
                elif cell_val == 1: # Path
                    display_row.append("# ")
                elif cell_val == 2: # Tower
                    display_row.append("T ")
                elif cell_val == 3: # Spawn
                    display_row.append("S ")
                elif cell_val == 4: # Exit
                    display_row.append("E ")
                else:
                    display_row.append("? ") # Unknown
            display_grid.append("".join(display_row))
        
        # Overlay enemies on the grid (simple representation)
        for enemy_status in game_state["enemies"]:
            pos_x, pos_y = enemy_status["position"]
            if 0 <= pos_x < self.grid_size[0] and 0 <= pos_y < self.grid_size[1]:
                row_list = list(display_grid[pos_x])
                row_list[pos_y * 2] = 'X' # Mark enemy with 'X'
                display_grid[pos_x] = "".join(row_list)

        output = ["" for _ in range(self.grid_size[0] + 5)] # Add space for stats
        output[0] = "Tower Defense Game State:"
        output[1] = "------------------------"
        for i, row_str in enumerate(display_grid):
            output[i+2] = row_str
        
        output[self.grid_size[0] + 2] = f"Cash: {cash} | Lives: {lives} | Wave: {wave_number}"
        output[self.grid_size[0] + 3] = f"Game Over: {game_over}"
        output[self.grid_size[0] + 4] = "------------------------"

        return "\n".join(output)

# Example usage:
# if __name__ == "__main__":
#     from tower_defense_env import TowerDefenseEnv
#     env = TowerDefenseEnv(grid_size=(5,5))
#     state = env.reset()
#     visualizer = GameVisualizer(grid_size=(5,5))
#     print(visualizer.render(state))

#     env.place_tower(1,1,"basic")
#     env.spawn_wave(2)
#     state = env.get_state()
#     print(visualizer.render(state))


