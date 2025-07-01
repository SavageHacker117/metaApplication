
from typing import Dict, Any, Tuple

class InteractiveElement:
    """
    Base class for interactive elements in the game environment, such as destroyable walls.
    """
    def __init__(self, element_id: str, position: Tuple[int, int], health: float, max_health: float, element_type: str):
        self.id = element_id
        self.position = position # Grid coordinates
        self.health = health
        self.max_health = max_health
        self.element_type = element_type
        self.is_destroyed = False

    def take_damage(self, damage_amount: float) -> None:
        """
        Applies damage to the element.
        """
        if not self.is_destroyed:
            self.health -= damage_amount
            if self.health <= 0:
                self.health = 0
                self.is_destroyed = True
                print(f"Interactive element {self.id} ({self.element_type}) at {self.position} destroyed!")

    def get_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the element.
        """
        return {
            "id": self.id,
            "position": self.position,
            "health": self.health,
            "max_health": self.max_health,
            "type": self.element_type,
            "is_destroyed": self.is_destroyed
        }

class DestroyableWall(InteractiveElement):
    """
    A destroyable wall that can block enemy movement temporarily.
    """
    def __init__(self, wall_id: str, position: Tuple[int, int], health: float = 100.0):
        super().__init__(wall_id, position, health, health, "destroyable_wall")
        print(f"Created Destroyable Wall {self.id} at {self.position} with health {self.health}")

# Example Usage:
# if __name__ == "__main__":
#     wall = DestroyableWall("wall_001", (5, 5))
#     print(wall.get_status())
#
#     wall.take_damage(30)
#     print(wall.get_status())
#
#     wall.take_damage(80) # Overkill
#     print(wall.get_status())


