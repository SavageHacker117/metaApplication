

import pygame
from typing import Tuple, Optional

class EventManager:
    """
    Handles user input events (mouse clicks, keyboard presses) for the Pygame-based game.
    """
    def __init__(self, cell_size: int):
        self.cell_size = cell_size

    def get_mouse_click_grid_coords(self) -> Optional[Tuple[int, int]]:
        """
        Checks for a mouse click event and returns the grid coordinates if a click occurred.
        Returns (row, col) or None.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left click
                    mouse_x, mouse_y = event.pos
                    grid_col = mouse_x // self.cell_size
                    grid_row = mouse_y // self.cell_size
                    return (grid_row, grid_col)
        return None

    def check_quit_event(self) -> bool:
        """
        Checks if the user has requested to quit the game.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

# Example Usage:
# if __name__ == "__main__":
#     pygame.init()
#     screen = pygame.display.set_mode((600, 600))
#     pygame.display.set_caption("Event Manager Test")
#     
#     event_manager = EventManager(cell_size=60)
#     running = True
#     while running:
#         click_coords = event_manager.get_mouse_click_grid_coords()
#         if click_coords == "QUIT":
#             running = False
#         elif click_coords:
#             print(f"Mouse clicked at grid coordinates: {click_coords}")
#         
#         if event_manager.check_quit_event():
#             running = False
# 
#         pygame.time.Clock().tick(60)
#     pygame.quit()
