
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

class DataVisualizer:
    """
    Provides utilities for visualizing various data points from the RL-LLM training
    and game simulation, such as rewards, game statistics, and performance metrics.
    """
    def __init__(self):
        pass

    def plot_episode_rewards(self, rewards: List[float], title: str = "Episode Rewards Over Time", save_path: str = None):
        """
        Plots the total reward obtained in each episode.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(rewards)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_game_statistics(self, stats: Dict[str, List[Any]], title: str = "Game Statistics Over Episodes", save_path: str = None):
        """
        Plots various game statistics (e.g., lives remaining, towers placed).
        `stats` should be a dictionary where keys are statistic names and values are lists of values per episode.
        """
        plt.figure(figsize=(14, 7))
        for stat_name, values in stats.items():
            plt.plot(values, label=stat_name)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()

    def plot_distribution(self, data: List[Any], title: str = "Data Distribution", bins: int = 20, save_path: str = None):
        """
        Plots a histogram to show the distribution of a given dataset.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, edgecolor=\



'black')
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        plt.show()

# Example Usage:
# if __name__ == "__main__":
#     rewards = [np.random.rand() * 100 for _ in range(50)]
#     visualizer = DataVisualizer()
#     visualizer.plot_episode_rewards(rewards, save_path="./rewards_plot.png")

#     game_stats = {
#         "Lives Remaining": [np.random.randint(0, 20) for _ in range(50)],
#         "Towers Placed": [np.random.randint(1, 10) for _ in range(50)]
#     }
#     visualizer.plot_game_statistics(game_stats, save_path="./game_stats_plot.png")

#     data_dist = [np.random.normal(0, 1, 1000)]
#     visualizer.plot_distribution(data_dist, title="Normal Distribution", save_path="./normal_dist.png")


