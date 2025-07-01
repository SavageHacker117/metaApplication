

from typing import Dict, Any, List, Tuple
from game_env.enemy_attributes import EnemyAttributes
from game_env.interactive_elements import InteractiveElement, DestroyableWall

class GameStateManager:
    """
    Manages the entire game state, including player stats, enemies, towers,
    interactive elements, and their interactions within the environment.
    """
    def __init__(self, initial_game_config: Dict[str, Any]):
        self.grid_size = initial_game_config["grid_size"]
        self.grid = initial_game_config["grid"]
        self.path = initial_game_config["path"]
        self.enemy_waves = initial_game_config["enemy_waves"]
        self.tower_types = initial_game_config["tower_types"]

        self.cash = initial_game_config["initial_cash"]
        self.lives = initial_game_config["initial_lives"]
        self.score = 0
        self.wave_number = 0
        self.game_over = False

        self.towers: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.enemies: List[Dict[str, Any]] = []
        self.interactive_elements: Dict[Tuple[int, int], InteractiveElement] = {}

        self.enemy_attributes = EnemyAttributes()

    def get_state(self) -> Dict[str, Any]:
        """
        Returns the current comprehensive game state.
        """
        return {
            "grid_size": self.grid_size,
            "grid": self.grid,
            "path": self.path,
            "cash": self.cash,
            "lives": self.lives,
            "score": self.score,
            "wave_number": self.wave_number,
            "game_over": self.game_over,
            "towers": self.towers,
            "enemies": self.enemies,
            "interactive_elements": {pos: el.get_status() for pos, el in self.interactive_elements.items()}
        }

    def place_tower(self, x: int, y: int, tower_type: str) -> bool:
        """
        Attempts to place a tower at (x, y).
        """
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return False # Out of bounds
        if self.grid[x][y] != 2: # Not a buildable spot
            return False
        if (x, y) in self.towers: # Already a tower here
            return False
        
        tower_cost = self.tower_types.get(tower_type, {}).get("cost", float("inf"))
        if self.cash < tower_cost:
            return False # Not enough cash

        self.towers[(x, y)] = {"type": tower_type, "health": 100, "max_health": 100, **self.tower_types.get(tower_type, {})}
        self.cash -= tower_cost
        print(f"Placed {tower_type} tower at ({x},{y}). Cash: {self.cash}")
        return True

    def spawn_wave(self) -> bool:
        """
        Spawns the next wave of enemies.
        """
        if self.wave_number >= len(self.enemy_waves):
            print("All waves completed!")
            return False
        
        self.wave_number += 1
        wave_info = self.enemy_waves[self.wave_number - 1]
        print(f"Spawning Wave {self.wave_number} with {len(wave_info["enemies"])} enemies.")
        
        spawn_point = self.path[0]
        for enemy_data in wave_info["enemies"]:
            enemy_type = enemy_data["type"]
            attrs = self.enemy_attributes.get_attributes(enemy_type)
            self.enemies.append({
                "id": f"enemy_{self.wave_number}_{len(self.enemies)}",
                "type": enemy_type,
                "position": spawn_point, # Current grid position
                "health": attrs["health"] * enemy_data.get("health_multiplier", 1.0),
                "max_health": attrs["health"] * enemy_data.get("health_multiplier", 1.0),
                "speed": attrs["speed"],
                "damage": attrs["damage"],
                "path_index": 0,
                "active": True,
                "model": attrs["model"],
                "can_traverse": attrs["can_traverse"]
            })
        return True

    def update_enemies(self, terrain_map: List[List[str]]) -> None:
        """
        Updates enemy positions and handles enemies reaching the exit.
        """
        enemies_to_remove = []
        for enemy in self.enemies:
            if not enemy["active"]: continue

            current_path_index = enemy["path_index"]
            if current_path_index + 1 < len(self.path):
                next_pos = self.path[current_path_index + 1]
                current_terrain_type = terrain_map[enemy["position"][0]][enemy["position"][1]]
                next_terrain_type = terrain_map[next_pos[0]][next_pos[1]]

                # Check if enemy can traverse current and next terrain
                if not self.enemy_attributes.can_enemy_traverse(enemy["type"], current_terrain_type):
                    # Enemy stuck or cannot traverse, remove or penalize
                    enemy["active"] = False
                    enemies_to_remove.append(enemy)
                    print(f"Enemy {enemy["id"]} stuck on {current_terrain_type} terrain.")
                    continue
                if not self.enemy_attributes.can_enemy_traverse(enemy["type"], next_terrain_type):
                    # Cannot move to next terrain, wait or find alternative path
                    # For simplicity, they just stay put for this turn
                    continue

                # Check for interactive elements blocking path
                if next_pos in self.interactive_elements and not self.interactive_elements[next_pos].is_destroyed:
                    # Enemy hits a wall, apply damage to wall
                    self.interactive_elements[next_pos].take_damage(enemy["damage"])
                    continue # Enemy stops at wall

                # Move enemy along path
                enemy["position"] = next_pos
                enemy["path_index"] += 1
            else:
                # Enemy reached the exit
                self.lives -= 1
                enemy["active"] = False
                enemies_to_remove.append(enemy)
                print(f"Enemy {enemy["id"]} reached exit. Lives remaining: {self.lives}")
                if self.lives <= 0:
                    self.game_over = True
                    print("Game Over - No lives left!")

        self.enemies = [e for e in self.enemies if e["active"]]

    def apply_damage_to_enemy(self, enemy_id: str, damage_amount: float) -> None:
        """
        Applies damage to a specific enemy.
        """
        for enemy in self.enemies:
            if enemy["id"] == enemy_id and enemy["active"]:
                enemy["health"] -= damage_amount
                if enemy["health"] <= 0:
                    enemy["health"] = 0
                    enemy["active"] = False
                    self.score += 10 # Example score for defeating enemy
                    print(f"Enemy {enemy_id} defeated! Score: {self.score}")
                break

    def place_destroyable_wall(self, x: int, y: int) -> bool:
        """
        Places a destroyable wall at the specified grid coordinates.
        """
        if not (0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1]):
            return False # Out of bounds
        if self.grid[x][y] != 2: # Not a buildable spot
            return False
        if (x, y) in self.interactive_elements: # Already an element here
            return False
        
        wall_cost = 50 # Example cost
        if self.cash < wall_cost:
            return False

        wall_id = f"wall_{x}_{y}"
        self.interactive_elements[(x, y)] = DestroyableWall(wall_id, (x, y))
        self.cash -= wall_cost
        print(f"Placed destroyable wall at ({x},{y}). Cash: {self.cash}")
        return True

    def update_interactive_elements(self) -> None:
        """
        Removes destroyed interactive elements.
        """
        self.interactive_elements = {pos: el for pos, el in self.interactive_elements.items() if not el.is_destroyed}

    def is_game_over(self) -> bool:
        return self.game_over

    def all_enemies_defeated(self) -> bool:
        return self.wave_number >= len(self.enemy_waves) and not self.enemies

# Example Usage:
# if __name__ == "__main__":
#     # Dummy initial config
#     initial_config = {
#         "grid_size": (5, 5),
#         "grid": [[0, 1, 0, 0, 0],
#                  [0, 1, 0, 0, 0],
#                  [0, 1, 0, 0, 0],
#                  [0, 1, 0, 0, 0],
#                  [0, 1, 0, 0, 0]],
#         "path": [(0,1), (1,1), (2,1), (3,1), (4,1)],
#         "enemy_waves": [
#             {"wave_number": 1, "enemies": [{"type": "grunt"}, {"type": "grunt"}]},
#             {"wave_number": 2, "enemies": [{"type": "tank"}]}
#         ],
#         "tower_types": {"basic": {"cost": 50, "damage": 20}},
#         "initial_cash": 100,
#         "initial_lives": 5
#     }
#     gsm = GameStateManager(initial_config)
#     
#     print("Initial State:", gsm.get_state())
#
#     gsm.place_tower(0,0, "basic")
#     gsm.spawn_wave()
#     
#     # Simulate game loop
#     dummy_terrain = [["land" for _ in range(5)] for _ in range(5)]
#     for _ in range(10): # Simulate turns
#         gsm.update_enemies(dummy_terrain)
#         if gsm.is_game_over() or gsm.all_enemies_defeated():
#             break
#         print("Current Lives:", gsm.lives, "Enemies left:", len(gsm.enemies))
#
#     print("Final State:", gsm.get_state())


