import pygame
import numpy as np
from typing import Dict, Any, Tuple

class GameRenderer:
    """
    Renders the Tower Defense game environment using Pygame for a graphical output.
    """
    def __init__(self, grid_size: Tuple[int, int], cell_size: int = 60):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width, self.height = grid_size[1] * cell_size, grid_size[0] * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tower Defense RL-LLM")
        self.colors = {
            "background": (0, 0, 0),
            "path": (100, 100, 100),
            "tower": (0, 0, 255),
            "enemy": (255, 0, 0),
            "spawn": (0, 255, 0),
            "exit": (255, 255, 0),
            "grid_lines": (50, 50, 50)
        }

    def _draw_grid(self, grid: np.ndarray):
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = self.colors["background"]
                if grid[r, c] == 1: # Path
                    color = self.colors["path"]
                elif grid[r, c] == 2: # Tower
                    color = self.colors["tower"]
                elif grid[r, c] == 3: # Spawn
                    color = self.colors["spawn"]
                elif grid[r, c] == 4: # Exit
                    color = self.colors["exit"]
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.colors["grid_lines"], rect, 1) # Grid lines

    def _draw_towers(self, towers: Dict[Tuple[int, int], Dict[str, Any]]):
        for (x, y), tower_info in towers.items():
            center_x = y * self.cell_size + self.cell_size // 2
            center_y = x * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.colors["tower"], (center_x, center_y), self.cell_size // 3)
            font = pygame.font.Font(None, 20)
            text = font.render(tower_info["type"][0].upper(), True, (255, 255, 255))
            text_rect = text.get_rect(center=(center_x, center_y))
            self.screen.blit(text, text_rect)

    def _draw_enemies(self, enemies: List[Dict[str, Any]]):
        for enemy_info in enemies:
            pos_x, pos_y = enemy_info["position"]
            center_x = pos_y * self.cell_size + self.cell_size // 2
            center_y = pos_x * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.screen, self.colors["enemy"], (center_x, center_y), self.cell_size // 4)

    def render_game(self, game_state: Dict[str, Any]):
        self.screen.fill(self.colors["background"])
        grid_array = np.array(game_state["grid"])
        self._draw_grid(grid_array)
        self._draw_towers(game_state["towers"])
        self._draw_enemies(game_state["enemies"])
        pygame.display.flip()

    def close(self):
        pygame.quit()

# Example usage:
# if __name__ == "__main__":
#     from tower_defense_env import TowerDefenseEnv
#     env = TowerDefenseEnv(grid_size=(10,10))
#     renderer = GameRenderer(grid_size=(10,10))
#     
#     running = True
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#         
#         current_state = env.get_state()
#         renderer.render_game(current_state)
#         pygame.time.Clock().tick(30) # Limit to 30 FPS
#     
#     renderer.close()


