
import json
from typing import Dict, Any, List
from datetime import datetime

class GameRecorder:
    """
    Records detailed logs of game episodes, including states, actions, rewards,
    and LLM interactions, for later analysis and replay.
    """
    def __init__(self, output_dir: str = "./game_records"):
        self.output_dir = output_dir
        self.records = []
        self.current_episode_data = {}

    def start_episode_recording(self, episode_id: int, initial_config: Dict[str, Any]):
        """
        Initializes recording for a new episode.
        """
        self.current_episode_data = {
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "initial_config": initial_config,
            "turns": []
        }

    def record_turn(self, turn_data: Dict[str, Any]):
        """
        Records data for a single turn within the current episode.
        Turn data should include: game_state, rl_action, llm_prompt, llm_response, parsed_action, reward.
        """
        if not self.current_episode_data:
            raise RuntimeError("No episode recording started. Call start_episode_recording first.")
        self.current_episode_data["turns"].append(turn_data)

    def end_episode_recording(self, final_game_state: Dict[str, Any], total_reward: float):
        """
        Finalizes recording for the current episode and saves it.
        """
        if not self.current_episode_data:
            raise RuntimeError("No episode recording started.")
        
        self.current_episode_data["final_game_state"] = final_game_state
        self.current_episode_data["total_reward"] = total_reward
        self.records.append(self.current_episode_data)
        self.current_episode_data = {}
        print(f"Episode {self.records[-1]["episode_id"]} recorded.")

    def save_all_records(self, filename: str = "game_records.json"):
        """
        Saves all recorded episodes to a JSON file.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, "w") as f:
            json.dump(self.records, f, indent=4)
        print(f"All game records saved to {file_path}")

    def load_records(self, filename: str = "game_records.json") -> List[Dict[str, Any]]:
        """
        Loads game records from a JSON file.
        """
        file_path = os.path.join(self.output_dir, filename)
        try:
            with open(file_path, "r") as f:
                self.records = json.load(f)
            print(f"Loaded {len(self.records)} game records from {file_path}")
            return self.records
        except FileNotFoundError:
            print(f"No records file found at {file_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return []

# Example Usage:
# if __name__ == "__main__":
#     recorder = GameRecorder(output_dir="./my_game_logs")
#     
#     # Simulate an episode
#     recorder.start_episode_recording(episode_id=1, initial_config={"grid_size": [5,5]})
#     recorder.record_turn({"game_state": {"cash": 100}, "rl_action": "place_tower", "reward": 10})
#     recorder.record_turn({"game_state": {"cash": 50}, "rl_action": "start_wave", "reward": 5})
#     recorder.end_episode_recording(final_game_state={"cash": 20, "lives": 5}, total_reward=15)
#
#     recorder.save_all_records()
#     loaded_records = recorder.load_records()
#     print(f"First loaded record: {loaded_records[0]}")


