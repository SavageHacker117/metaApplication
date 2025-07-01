
import pygame
from typing import Dict, Any, Tuple

from .tower_defense_env import TowerDefenseEnv
from .game_renderer import GameRenderer
from .event_manager import EventManager

class GameLoop:
    """
    Manages the main game loop for the interactive Tower Defense environment.
    Handles rendering, event processing, and game state updates.
    """
    def __init__(self, env: TowerDefenseEnv, renderer: GameRenderer, event_manager: EventManager, fps: int = 60):
        self.env = env
        self.renderer = renderer
        self.event_manager = event_manager
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.running = False

    def start(self):
        self.running = True
        current_state = self.env.reset()
        
        while self.running:
            # Event handling
            click_coords = self.event_manager.get_mouse_click_grid_coords()
            if click_coords == "QUIT":
                self.running = False
            elif click_coords:
                row, col = click_coords
                print(f"Clicked at grid: ({row}, {col})")
                # Example: Try to place a basic tower on click
                action = {"type": "place_tower", "x": row, "y": col, "tower_type": "basic"}
                next_state, reward, done, info = self.env.step(action)
                current_state = next_state
                if done:
                    self.running = False

            if self.event_manager.check_quit_event():
                self.running = False

            # Game state update (e.g., enemies move, towers attack)
            # This can be triggered by a timer or specific game events
            # For simplicity, we'll call it every frame for now
            if not current_state["game_over"]:
                # Simulate a game tick without a specific player action
                current_state, _, done, _ = self.env.step({"type": "no_op"})
                if done:
                    self.running = False

            # Rendering
            self.renderer.render_game(current_state)

            # Cap the frame rate
            self.clock.tick(self.fps)

        self.renderer.close()
        print("Game loop ended.")

# Example Usage:
# if __name__ == "__main__":
#     grid_size = (10, 10)
#     cell_size = 60
#     env = TowerDefenseEnv(grid_size=grid_size)
#     renderer = GameRenderer(grid_size=grid_size, cell_size=cell_size)
#     event_manager = EventManager(cell_size=cell_size)
#     
#     game_loop = GameLoop(env, renderer, event_manager)
#     game_loop.start()


