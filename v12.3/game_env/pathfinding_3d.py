

from typing import List, Tuple, Dict, Any
import heapq
import numpy as np

class Pathfinding3D:
    """
    Implements A* pathfinding algorithm for enemies in a 3D grid, considering
    terrain types and interactive obstacles (like destroyable walls).
    """
    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int],
                  grid: List[List[int]], terrain_map: List[List[str]],
                  enemy_type: str, enemy_attributes: Any, interactive_elements: Dict[Tuple[int, int], Any]) -> List[Tuple[int, int]]:
        """
        Finds the shortest path from start to end using A* algorithm.
        Args:
            start (Tuple[int, int]): Starting grid coordinates (row, col).
            end (Tuple[int, int]): Ending grid coordinates (row, col).
            grid (List[List[int]]): The game grid (0: empty, 1: path, 2: buildable, etc.).
            terrain_map (List[List[str]]): 2D map of terrain types (e.g., 'land', 'water', 'air').
            enemy_type (str): Type of enemy (e.g., 'plane', 'tank').
            enemy_attributes (Any): An instance of EnemyAttributes to check traversal.
            interactive_elements (Dict[Tuple[int, int], Any]): Dictionary of active interactive elements.
        Returns:
            List[Tuple[int, int]]: A list of (row, col) tuples representing the path, or empty list if no path.
        """
        open_set = []
        heapq.heappush(open_set, (0, start)) # (f_score, node)

        came_from = {}
        g_score = { (r, c): float("inf") for r in range(self.grid_size[0]) for c in range(self.grid_size[1]) }
        g_score[start] = 0
        f_score = { (r, c): float("inf") for r in range(self.grid_size[0]) for c in range(self.grid_size[1]) }
        f_score[start] = self._heuristic(start, end)

        while open_set:
            current_f_score, current_node = heapq.heappop(open_set)

            if current_node == end:
                return self._reconstruct_path(came_from, current_node)

            for neighbor in self._get_neighbors(current_node):
                if not (0 <= neighbor[0] < self.grid_size[0] and 0 <= neighbor[1] < self.grid_size[1]):
                    continue # Out of bounds

                # Check terrain traversability
                neighbor_terrain_type = terrain_map[neighbor[0]][neighbor[1]]
                if not enemy_attributes.can_enemy_traverse(enemy_type, neighbor_terrain_type):
                    continue # Cannot traverse this terrain

                # Check for interactive elements (e.g., walls)
                if neighbor in interactive_elements and not interactive_elements[neighbor].get("is_destroyed", True):
                    continue # Wall is blocking path

                tentative_g_score = g_score[current_node] + 1 # Assuming uniform cost for now

                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_g_score
                    f_score[neighbor] = g_score[neighbor] + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return [] # No path found

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Manhattan distance heuristic.
        """\n        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns direct (up, down, left, right) neighbors.
        """\n        r, c = node
        return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

# Example Usage:
# if __name__ == "__main__":
#     from game_env.enemy_attributes import EnemyAttributes
#     pathfinder = Pathfinding3D(grid_size=(10, 10))
#     
#     # Dummy grid (1 for path, 2 for buildable, 0 for empty)
#     dummy_grid = [[0 for _ in range(10)] for _ in range(10)]
#     dummy_grid[0][0] = 3 # Spawn
#     dummy_grid[9][9] = 4 # Exit
#     for i in range(1, 9): dummy_grid[i][i] = 1 # Diagonal path
#     dummy_grid[5][4] = 2 # Buildable spot
#
#     # Dummy terrain map
#     dummy_terrain_map = [["land" for _ in range(10)] for _ in range(10)]
#     dummy_terrain_map[5][5] = "water" # Water spot
#
#     # Dummy interactive elements
#     dummy_interactive_elements = {}
#     # Add a wall at (3,3)
#     dummy_interactive_elements[(3,3)] = {"id": "wall_0", "is_destroyed": False}
#
#     enemy_attrs = EnemyAttributes()
#
#     # Test path for a tank (can only traverse land)
#     path_tank = pathfinder.find_path((0,0), (9,9), dummy_grid, dummy_terrain_map, "tank", enemy_attrs, dummy_interactive_elements)
#     print("Path for Tank:", path_tank)
#
#     # Test path for a plane (can traverse air, but our terrain is land/water)
#     # This would ideally need a 3D grid or a different representation for air paths
#     # For now, it will fail if only land/water is present
#     path_plane = pathfinder.find_path((0,0), (9,9), dummy_grid, dummy_terrain_map, "plane", enemy_attrs, dummy_interactive_elements)
#     print("Path for Plane (should be empty if no air path):", path_plane)
#
#     # Test path with a destroyed wall
#     dummy_interactive_elements[(3,3)]["is_destroyed"] = True
#     path_tank_after_destroy = pathfinder.find_path((0,0), (9,9), dummy_grid, dummy_terrain_map, "tank", enemy_attrs, dummy_interactive_elements)
#     print("Path for Tank (wall destroyed):", path_tank_after_destroy)


