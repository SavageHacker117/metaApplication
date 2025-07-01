
from typing import Dict, Any, Callable

class GameEventHandler:
    """
    Manages game events and their corresponding callbacks.
    This allows for a decoupled system where different game components
    can react to events without direct dependencies.
    """
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribes a callback function to a specific event type.
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribes a callback function from a specific event type.
        """
        if event_type in self._listeners and callback in self._listeners[event_type]:
            self._listeners[event_type].remove(callback)

    def publish(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Publishes an event, triggering all subscribed callbacks.
        """
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    callback(event_data)
                except Exception as e:
                    print(f"Error in event handler for {event_type}: {e}")

# Example Usage:
# if __name__ == "__main__":
#     event_handler = GameEventHandler()
#
#     def on_enemy_hit(data):
#         print(f"Enemy {data["enemy_id"]} hit! Remaining health: {data["health"]}")
#
#     def on_game_over(data):
#         print(f"Game Over! Reason: {data["reason"]}")
#
#     event_handler.subscribe("enemy_hit", on_enemy_hit)
#     event_handler.subscribe("game_over", on_game_over)
#
#     event_handler.publish("enemy_hit", {"enemy_id": "goblin_1", "health": 50})
#     event_handler.publish("game_over", {"reason": "lives_depleted"})
#
#     event_handler.unsubscribe("enemy_hit", on_enemy_hit)
#     event_handler.publish("enemy_hit", {"enemy_id": "goblin_2", "health": 20}) # This won't trigger on_enemy_hit


