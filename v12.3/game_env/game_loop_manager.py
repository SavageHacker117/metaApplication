
from typing import Dict, Any, List, Tuple
import time

from game_env.game_state_manager import GameStateManager
from game_env.projectile_manager import ProjectileManager
from game_env.physics_engine_interface import DarkMatterPhysicsEngine
from game_env.terrain_generator import TerrainGenerator

class GameLoopManager:
    """
    Manages the main game loop, updating game state, physics, and handling events.
    """
    def __init__(self, initial_game_config: Dict[str, Any]):
        self.game_state_manager = GameStateManager(initial_game_config)
        self.physics_engine = DarkMatterPhysicsEngine()
        self.projectile_manager = ProjectileManager(self.physics_engine)
        self.terrain_generator = TerrainGenerator(grid_size=initial_game_config["grid_size"])
        self.terrain_map = self.terrain_generator.generate_full_terrain()["biome_map"]

        self.last_update_time = time.time()
        self.time_step = 0.1 # Fixed time step for physics updates

    def update(self) -> Dict[str, Any]:
        """
        Performs one step of the game loop.
        """
        current_time = time.time()
        delta_time = current_time - self.last_update_time
        self.last_update_time = current_time

        # Update physics simulation (for projectiles, etc.)
        self.physics_engine.update_simulation(delta_time)

        # Update projectiles and handle hits
        hit_projectiles = self.projectile_manager.update_projectiles(delta_time)
        for proj_info in hit_projectiles:
            # In a real game, determine what the projectile hit and apply damage
            # For now, let's assume it hits an enemy or a wall
            print(f"Projectile {proj_info["id"]} hit something! Damage: {proj_info["damage"]}")
            # Example: apply damage to a random enemy for demonstration
            if self.game_state_manager.enemies:
                target_enemy = self.game_state_manager.enemies[0] # Just pick first one
                self.game_state_manager.apply_damage_to_enemy(target_enemy["id"], proj_info["damage"])

        # Update enemies (movement, interaction with terrain/walls)
        self.game_state_manager.update_enemies(self.terrain_map)

        # Update interactive elements (remove destroyed ones)
        self.game_state_manager.update_interactive_elements()

        # Check game over conditions
        if self.game_state_manager.is_game_over():
            print("Game Over!")

        return self.game_state_manager.get_state()

    def get_current_game_state(self) -> Dict[str, Any]:
        """
        Returns the current game state from the manager.
        """
        return self.game_state_manager.get_state()

    def get_terrain_data(self) -> Dict[str, Any]:
        """
        Returns the generated terrain data.
        """
        return self.terrain_generator.generate_full_terrain()

    def handle_user_action(self, action: Dict[str, Any]) -> bool:
        """
        Handles user actions like placing towers or walls, or spawning waves.
        """
        action_type = action.get("type")
        if action_type == "place_tower":
            return self.game_state_manager.place_tower(action["x"], action["y"], action["tower_type"])
        elif action_type == "place_wall":
            return self.game_state_manager.place_destroyable_wall(action["x"], action["y"])
        elif action_type == "spawn_wave":
            return self.game_state_manager.spawn_wave()
        elif action_type == "fire_projectile":
            # Example: fire a projectile from a tower at an enemy
            start_pos = action["start_pos"]
            target_pos = action["target_pos"]
            projectile_type = action.get("projectile_type", "missile")
            damage = action.get("damage", 25.0)
            speed = action.get("speed", 10.0)
            self.projectile_manager.create_projectile(start_pos, target_pos, speed, projectile_type, damage)
            return True
        return False

# Example Usage:
# if __name__ == "__main__":
#     # Dummy initial config
#     initial_config = {
#         "grid_size": (10, 10),
#         "grid": [[0 for _ in range(10)] for _ in range(10)],
#         "path": [(0,0), (0,1), (0,2), (1,2), (2,2), (2,3), (2,4), (3,4), (4,4), (5,4), (6,4), (7,4), (8,4), (9,4)],
#         "enemy_waves": [
#             {"wave_number": 1, "enemies": [{"type": "grunt"}, {"type": "grunt"}]},
#             {"wave_number": 2, "enemies": [{"type": "tank"}]}
#         ],
#         "tower_types": {"basic": {"cost": 50, "damage": 20}},
#         "initial_cash": 100,
#         "initial_lives": 5
#     }
#     # Mark some grid cells as buildable (2) and path (1)
#     for r in range(10):
#         for c in range(10):
#             if (r,c) in initial_config["path"]:
#                 initial_config["grid"][r][c] = 1
#             else:
#                 initial_config["grid"][r][c] = 2
#     initial_config["grid"][initial_config["path"][0][0]][initial_config["path"][0][1]] = 3 # Spawn
#     initial_config["grid"][initial_config["path"][-1][0]][initial_config["path"][-1][1]] = 4 # Exit
#
#     game_loop = GameLoopManager(initial_config)
#
#     # Simulate some actions
#     game_loop.handle_user_action({"type": "place_tower", "x": 1, "y": 1, "tower_type": "basic"})
#     game_loop.handle_user_action({"type": "place_wall", "x": 3, "y": 3})
#     game_loop.handle_user_action({"type": "spawn_wave"})
#
#     # Simulate game running for a few updates
#     for i in range(20):
#         state = game_loop.update()
#         print(f"Update {i+1}: Lives={state["lives"]}, Score={state["score"]}, Enemies={len(state["enemies"])}, Walls={len(state["interactive_elements"])}")
#         if state["game_over"] or state["wave_number"] >= len(initial_config["enemy_waves"]) and not state["enemies"]:
#             break
#         time.sleep(0.1)
#
#     print("Final Game State:", game_loop.get_current_game_state())


