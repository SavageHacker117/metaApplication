

import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards: list, title="Episode Rewards", save_path=None):
    """
    Plots the rewards per episode.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

def plot_metrics(metrics: dict, title="Training Metrics", save_path=None):
    """
    Plots various training metrics.
    `metrics` should be a dictionary where keys are metric names and values are lists of metric values.
    """
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()

# Example usage:
# if __name__ == "__main__":
#     sample_rewards = [np.random.rand() for _ in range(100)]
#     plot_rewards(sample_rewards, save_path="./rewards_plot.png")

#     sample_metrics = {
#         "Average Reward": [np.random.rand() * 10 for _ in range(100)],
#         "Loss": [np.random.rand() * 0.1 for _ in range(100)]
#     }
#     plot_metrics(sample_metrics, save_path="./metrics_plot.png")


