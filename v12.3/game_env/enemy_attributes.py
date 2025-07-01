

from typing import Dict, Any, Tuple

class EnemyAttributes:
    """
    Defines attributes for different enemy types (planes, tanks, boats)
    and how they interact with terrain (water, land, air).
    """
    def __init__(self):
        self.attributes = {
            "plane": {
                "health": 100,
                "speed": 5.0,
                "damage": 10,
                "type": "air",
                "model": "plane_model_path.obj",
                "can_traverse": ["air"]
            },
            "tank": {
                "health": 200,
                "speed": 2.0,
                "damage": 20,
                "type": "land",
                "model": "tank_model_path.obj",
                "can_traverse": ["land"]
            },
            "boat": {
                "health": 150,
                "speed": 3.0,
                "damage": 15,
                "type": "water",
                "model": "boat_model_path.obj",
                "can_traverse": ["water"]
            },
            "grunt": {
                "health": 50,
                "speed": 1.5,
                "damage": 5,
                "type": "land",
                "model": "grunt_model_path.obj",
                "can_traverse": ["land"]
            }
        }

    def get_attributes(self, enemy_type: str) -> Dict[str, Any]:
        """
        Returns the attributes for a given enemy type.
        """
        return self.attributes.get(enemy_type, self.attributes["grunt"])

    def can_enemy_traverse(self, enemy_type: str, terrain_type: str) -> bool:
        """
        Checks if an enemy type can traverse a given terrain type.
        """
        attrs = self.get_attributes(enemy_type)
        return terrain_type in attrs["can_traverse"]

# Example Usage:
# if __name__ == "__main__":
#     enemy_attrs = EnemyAttributes()
#     
#     plane_info = enemy_attrs.get_attributes("plane")
#     print("Plane Info:", plane_info)
#     print("Can plane traverse land?", enemy_attrs.can_enemy_traverse("plane", "land"))
#     print("Can plane traverse air?", enemy_attrs.can_enemy_traverse("plane", "air"))
#
#     tank_info = enemy_attrs.get_attributes("tank")
#     print("Tank Info:", tank_info)
#     print("Can tank traverse water?", enemy_attrs.can_enemy_traverse("tank", "water"))


