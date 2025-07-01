
from typing import Dict, Any, List
from collections import defaultdict

class GameEvaluator:
    """
    Evaluates the performance of the RL-LLM system in generating and playing Tower Defense games.
    Collects and analyzes metrics such as game completion rate, average lives remaining, 
    tower placement efficiency, and enemy defeat rate.
    """
    def __init__(self):
        self.episode_results = []

    def record_episode_results(self, episode_data: Dict[str, Any]):
        """
        Records the results of a single game episode.
        Expected episode_data keys: 'episode_id', 'total_reward', 'game_over', 'final_lives', 'towers_placed', 'enemies_defeated'
        """
        self.episode_results.append(episode_data)
        print(f"Recorded results for episode {episode_data.get("episode_id", "N/A")}")

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generates a summary report of all recorded episodes.
        """
        if not self.episode_results:
            return {"message": "No episodes recorded for evaluation.", "summary": {}}

        total_episodes = len(self.episode_results)
        completed_games = sum(1 for res in self.episode_results if res.get("game_over") and res.get("final_lives", 0) > 0)
        total_rewards = [res.get("total_reward", 0) for res in self.episode_results]
        final_lives = [res.get("final_lives", 0) for res in self.episode_results]
        towers_placed = [res.get("towers_placed", 0) for res in self.episode_results]
        enemies_defeated = [res.get("enemies_defeated", 0) for res in self.episode_results]

        summary = {
            "total_episodes": total_episodes,
            "game_completion_rate": (completed_games / total_episodes) * 100 if total_episodes > 0 else 0,
            "average_total_reward": sum(total_rewards) / total_episodes if total_episodes > 0 else 0,
            "average_final_lives": sum(final_lives) / total_episodes if total_episodes > 0 else 0,
            "average_towers_placed": sum(towers_placed) / total_episodes if total_episodes > 0 else 0,
            "average_enemies_defeated": sum(enemies_defeated) / total_episodes if total_episodes > 0 else 0,
        }
        print("Evaluation summary generated.")
        return summary

    def save_results(self, file_path: str):
        """
        Saves the raw episode results to a JSON file.
        """
        import json
        with open(file_path, "w") as f:
            json.dump(self.episode_results, f, indent=4)
        print(f"Episode results saved to {file_path}")

# Example Usage:
# if __name__ == "__main__":
#     evaluator = GameEvaluator()
#     evaluator.record_episode_results({
#         "episode_id": 1, "total_reward": 50, "game_over": True, 
#         "final_lives": 5, "towers_placed": 3, "enemies_defeated": 10
#     })
#     evaluator.record_episode_results({
#         "episode_id": 2, "total_reward": -10, "game_over": True, 
#         "final_lives": 0, "towers_placed": 1, "enemies_defeated": 2
#     })
#     evaluator.record_episode_results({
#         "episode_id": 3, "total_reward": 120, "game_over": True, 
#         "final_lives": 10, "towers_placed": 5, "enemies_defeated": 25
#     })

#     summary = evaluator.generate_summary_report()
#     print("\nSummary Report:")
#     import json
#     print(json.dumps(summary, indent=2))

#     evaluator.save_results("./game_evaluation_results.json")

