

from typing import Dict, Any, Tuple
from .tower_defense_env import TowerDefenseEnv
from .visualizer import GameVisualizer

class InteractiveManager:
    """
    Manages interactive sessions with the Tower Defense game environment.
    Allows human input for actions and visualizes the game state.
    """
    def __init__(self, env: TowerDefenseEnv, visualizer: GameVisualizer):
        self.env = env
        self.visualizer = visualizer

    def start_interactive_session(self):
        """
        Starts a loop for interactive gameplay.
        """
        print("\n--- Starting Interactive Tower Defense Session ---")
        current_state = self.env.reset()
        print(self.visualizer.render(current_state))

        while not current_state["game_over"]:
            action_input = input("Enter action (e.g., place basic 2 3, start wave, upgrade 1 1, quit): ").strip().lower()
            
            if action_input == "quit":
                print("Exiting interactive session.")
                break

            parsed_action = self._parse_human_input(action_input)
            if parsed_action:
                next_state, reward, done, info = self.env.step(parsed_action)
                current_state = next_state
                print(self.visualizer.render(current_state))
                print(f"Action Reward: {reward:.2f}")
                if "error" in info:
                    print(f"Error: {info["error"]}")
            else:
                print("Invalid input format. Please try again.")

        if current_state["game_over"]:
            print("Game Over! Thanks for playing.")

    def _parse_human_input(self, input_str: str) -> Dict[str, Any]:
        """
        Parses human-readable input into a structured action dictionary.
        """
        parts = input_str.split()
        action_type = parts[0]

        if action_type == "place" and len(parts) >= 4:
            tower_type = parts[1]
            try:
                x = int(parts[2])
                y = int(parts[3])
                return {"type": "place_tower", "x": x, "y": y, "tower_type": tower_type}
            except ValueError:
                return None
        elif action_type == "start" and len(parts) >= 2 and parts[1] == "wave":
            num_enemies = 5
            if len(parts) == 4 and parts[2].isdigit():
                num_enemies = int(parts[2])
            return {"type": "start_wave", "num_enemies": num_enemies}
        elif action_type == "upgrade" and len(parts) >= 3:
            try:
                x = int(parts[1])
                y = int(parts[2])
                return {"type": "upgrade_tower", "x": x, "y": y}
            except ValueError:
                return None
        elif action_type == "no_op":
            return {"type": "no_op"}
        
        return None

# Example Usage:
# if __name__ == "__main__":
#     env = TowerDefenseEnv(grid_size=(10,10))
#     visualizer = GameVisualizer(grid_size=(10,10))
#     interactive_session = InteractiveManager(env, visualizer)
#     interactive_session.start_interactive_session()


