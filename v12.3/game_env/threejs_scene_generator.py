
import json
from typing import Dict, Any, List, Tuple

class ThreeJSSceneGenerator:
    """
    Generates JSON configurations for a Three.js scene, representing the 3D Tower Defense environment.
    This includes grid layout, tower positions, enemy paths, and basic visual properties.
    """
    def __init__(self, grid_size: Tuple[int, int], cell_size: float = 10.0):
        self.grid_size = grid_size
        self.cell_size = cell_size

    def generate_scene_config(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates a Three.js scene configuration based on the current game state.
        """
        scene_config = {
            "camera": {
                "position": [self.grid_size[1] * self.cell_size / 2, self.grid_size[0] * self.cell_size / 2, 50],
                "lookAt": [self.grid_size[1] * self.cell_size / 2, self.grid_size[0] * self.cell_size / 2, 0]
            },
            "lights": [
                {"type": "AmbientLight", "color": 0x404040},
                {"type": "DirectionalLight", "color": 0xffffff, "intensity": 0.5, "position": [1, 1, 1]}
            ],
            "objects": []
        }

        # Add ground plane
        scene_config["objects"].append({
            "type": "PlaneGeometry",
            "name": "ground",
            "width": self.grid_size[1] * self.cell_size,
            "height": self.grid_size[0] * self.cell_size,
            "material": {"type": "MeshPhongMaterial", "color": 0x228B22},
            "position": [self.grid_size[1] * self.cell_size / 2, self.grid_size[0] * self.cell_size / 2, -0.1],
            "rotation": [-Math.PI / 2, 0, 0]
        })

        # Add grid cells (simplified representation)
        for r in range(self.grid_size[0]):
            for c in range(self.grid_size[1]):
                cell_type = game_state["grid"][r][c]
                color = 0x808080 # Default empty
                if cell_type == 1: # Path
                    color = 0x8B4513 # Brown
                elif cell_type == 2: # Tower placement area
                    color = 0x6A5ACD # SlateBlue
                elif cell_type == 3: # Spawn
                    color = 0x00FF00 # Green
                elif cell_type == 4: # Exit
                    color = 0xFF0000 # Red

                scene_config["objects"].append({
                    "type": "BoxGeometry",
                    "name": f"cell_{r}_{c}",
                    "width": self.cell_size * 0.9,
                    "height": self.cell_size * 0.9,
                    "depth": 0.1,
                    "material": {"type": "MeshPhongMaterial", "color": color},
                    "position": [c * self.cell_size + self.cell_size / 2, r * self.cell_size + self.cell_size / 2, 0]
                })

        # Add towers
        for (x, y), tower_info in game_state["towers"].items():
            scene_config["objects"].append({
                "type": "CylinderGeometry",
                "name": f"tower_{x}_{y}",
                "radiusTop": self.cell_size * 0.2,
                "radiusBottom": self.cell_size * 0.3,
                "height": self.cell_size * 0.8,
                "radialSegments": 16,
                "material": {"type": "MeshPhongMaterial", "color": 0x0000FF},
                "position": [y * self.cell_size + self.cell_size / 2, x * self.cell_size + self.cell_size / 2, self.cell_size * 0.4]
            })

        # Add enemies (simplified spheres)
        for enemy_info in game_state["enemies"]:
            pos_x, pos_y = enemy_info["position"]
            scene_config["objects"].append({
                "type": "SphereGeometry",
                "name": f"enemy_{pos_x}_{pos_y}",
                "radius": self.cell_size * 0.2,
                "widthSegments": 16,
                "heightSegments": 16,
                "material": {"type": "MeshPhongMaterial", "color": 0xFF0000},
                "position": [pos_y * self.cell_size + self.cell_size / 2, pos_x * self.cell_size + self.cell_size / 2, self.cell_size * 0.2]
            })

        return scene_config

    def save_scene_config(self, scene_config: Dict[str, Any], file_path: str):
        """
        Saves the generated scene configuration to a JSON file.
        """
        with open(file_path, "w") as f:
            json.dump(scene_config, f, indent=4)
        print(f"Three.js scene config saved to {file_path}")

# Example Usage:
# if __name__ == "__main__":
#     generator = ThreeJSSceneGenerator(grid_size=(10, 10))
#     # Dummy game state
#     dummy_game_state = {
#         "grid": [[0 for _ in range(10)] for _ in range(10)],
#         "cash": 100,
#         "lives": 10,
#         "towers": {(2, 2): {"type": "basic"}, (5, 5): {"type": "archer"}},
#         "enemies": [{"position": (0, 0)}, {"position": (0, 1)}],
#         "wave_number": 1,
#         "game_over": False
#     }
#     dummy_game_state["grid"][0][0] = 3 # Spawn
#     dummy_game_state["grid"][9][9] = 4 # Exit
#     dummy_game_state["grid"][0][1] = 1 # Path
#     dummy_game_state["grid"][0][2] = 1 # Path
#
#     scene_config = generator.generate_scene_config(dummy_game_state)
#     generator.save_scene_config(scene_config, "./web_frontend/scene_config.json")


