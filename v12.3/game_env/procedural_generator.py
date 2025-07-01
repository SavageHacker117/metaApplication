
import random
from typing import Dict, Any, List, Tuple

class ProceduralGenerator:
    """
    Generates procedural game elements for Tower Defense, such as:
    - Grid layouts (paths, buildable areas)
    - Enemy wave compositions
    - Tower types and properties
    
    This generator can be guided by LLM suggestions or RL agent exploration.
    """
    def __init__(self, grid_size: Tuple[int, int] = (10, 10)):
        self.grid_size = grid_size

    def generate_random_path(self) -> List[Tuple[int, int]]:
        """
        Generates a simple random path from top-left to bottom-right.
        """
        path = []
        current_pos = [0, 0]
        path.append(tuple(current_pos))

        while current_pos[0] < self.grid_size[0] - 1 or current_pos[1] < self.grid_size[1] - 1:
            possible_moves = []
            if current_pos[0] < self.grid_size[0] - 1: # Can move down
                possible_moves.append((current_pos[0] + 1, current_pos[1]))
            if current_pos[1] < self.grid_size[1] - 1: # Can move right
                possible_moves.append((current_pos[0], current_pos[1] + 1))
            
            if not possible_moves: # Should not happen if target is reachable
                break

            next_pos = random.choice(possible_moves)
            current_pos = list(next_pos)
            path.append(tuple(current_pos))
        
        return path

    def generate_grid_layout(self, path: List[Tuple[int, int]]) -> List[List[int]]:
        """
        Generates a grid with path, spawn, and exit points.
        0: empty, 1: path, 2: tower placement, 3: spawn, 4: exit
        """
        grid = [[0 for _ in range(self.grid_size[1])] for _ in range(self.grid_size[0])]

        for r, c in path:
            grid[r][c] = 1 # Mark path
        
        if path:
            grid[path[0][0]][path[0][1]] = 3 # Spawn point
            grid[path[-1][0]][path[-1][1]] = 4 # Exit point

        # Mark all non-path cells as potential tower placement areas (2)
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                if grid[r][c] == 0:
                    grid[r][c] = 2
        
        return grid

    def generate_enemy_waves(self, num_waves: int = 5) -> List[Dict[str, Any]]:
        """
        Generates a list of enemy waves with varying compositions.
        """
        enemy_types = ["grunt", "fast", "tank"]
        waves = []
        for i in range(num_waves):
            wave = {"wave_number": i + 1, "enemies": []}
            num_enemies = random.randint(5, 15) + i * 2 # More enemies in later waves
            for _ in range(num_enemies):
                enemy_type = random.choice(enemy_types)
                wave["enemies"].append({"type": enemy_type, "health_multiplier": random.uniform(0.8, 1.2)})
            waves.append(wave)
        return waves

    def generate_tower_types(self, num_types: int = 3) -> Dict[str, Any]:
        """
        Generates a set of random tower types with basic properties.
        """
        tower_names = ["basic", "laser", "frost", "fire", "sniper"]
        tower_properties = {}
        for i in range(num_types):
            name = random.choice(tower_names) + str(i) # Ensure unique names
            tower_properties[name] = {
                "cost": random.randint(50, 200),
                "range": random.randint(2, 5),
                "damage": random.randint(10, 50),
                "attack_speed": random.uniform(0.5, 2.0)
            }
        return tower_properties

# Example Usage:
# if __name__ == "__main__":
#     generator = ProceduralGenerator(grid_size=(15, 15))
#     
#     random_path = generator.generate_random_path()
#     print("\nGenerated Path:", random_path)
#
#     grid_layout = generator.generate_grid_layout(random_path)
#     print("\nGenerated Grid Layout:")
#     for row in grid_layout:
#         print(row)
#
#     enemy_waves = generator.generate_enemy_waves(num_waves=3)
#     print("\nGenerated Enemy Waves:", json.dumps(enemy_waves, indent=2))
#
#     tower_types = generator.generate_tower_types(num_types=4)
#     print("\nGenerated Tower Types:", json.dumps(tower_types, indent=2))


