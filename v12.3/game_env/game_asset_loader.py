

from typing import Dict, Any, List
import os

class GameAssetLoader:
    """
    Manages loading of 3D models, textures, and other visual assets.
    It can map abstract asset names (e.g., "tank_model") to actual file paths.
    """
    def __init__(self, base_asset_path: str = "./assets"):
        self.base_asset_path = base_asset_path
        self.asset_map: Dict[str, str] = {}
        self._load_default_assets()

    def _load_default_assets(self):
        """
        Loads default asset mappings. In a real system, this would read from a config file.
        """
        self.asset_map = {
            "plane_model": os.path.join(self.base_asset_path, "models", "plane.obj"),
            "tank_model": os.path.join(self.base_asset_path, "models", "tank.obj"),
            "boat_model": os.path.join(self.base_asset_path, "models", "boat.obj"),
            "grunt_model": os.path.join(self.base_asset_path, "models", "grunt.obj"),
            "basic_tower_model": os.path.join(self.base_asset_path, "models", "basic_tower.obj"),
            "destroyable_wall_model": os.path.join(self.base_asset_path, "models", "wall.obj"),
            "missile_model": os.path.join(self.base_asset_path, "models", "missile.obj"),
            "rocket_model": os.path.join(self.base_asset_path, "models", "rocket.obj"),
            "terrain_texture_land": os.path.join(self.base_asset_path, "textures", "land.png"),
            "terrain_texture_water": os.path.join(self.base_asset_path, "textures", "water.png"),
            "terrain_texture_mountain": os.path.join(self.base_asset_path, "textures", "mountain.png"),
        }

    def get_asset_path(self, asset_name: str) -> str:
        """
        Returns the file path for a given asset name.
        """
        return self.asset_map.get(asset_name, "")

    def add_custom_asset(self, asset_name: str, file_path: str) -> None:
        """
        Adds or updates a custom asset mapping.
        """
        self.asset_map[asset_name] = file_path

# Example Usage:
# if __name__ == "__main__":
#     loader = GameAssetLoader()
#     print("Tank model path:", loader.get_asset_path("tank_model"))
#     loader.add_custom_asset("my_custom_tower", "./assets/models/custom_tower.obj")
#     print("Custom tower path:", loader.get_asset_path("my_custom_tower"))


