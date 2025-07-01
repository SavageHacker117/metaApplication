

import numpy as np
from typing import List, Dict

def calculate_average_reward(rewards: List[float]) -> float:
    """
    Calculates the average reward from a list of rewards.
    """
    if not rewards:
        return 0.0
    return np.mean(rewards)

def calculate_dialogue_length_metrics(dialogue_lengths: List[int]) -> Dict[str, float]:
    """
    Calculates average, min, and max dialogue lengths.
    """
    if not dialogue_lengths:
        return {"average_length": 0.0, "min_length": 0.0, "max_length": 0.0}
    return {
        "average_length": np.mean(dialogue_lengths),
        "min_length": np.min(dialogue_lengths),
        "max_length": np.max(dialogue_lengths)
    }

def calculate_success_rate(successes: List[bool]) -> float:
    """
    Calculates the success rate from a list of boolean outcomes.
    """
    if not successes:
        return 0.0
    return np.mean(successes) * 100

# Example usage:
# if __name__ == "__main__":
#     rewards = [0.8, 0.9, 0.7, 0.95, 0.6]
#     avg_reward = calculate_average_reward(rewards)
#     print(f"Average Reward: {avg_reward:.2f}")

#     lengths = [5, 7, 3, 8, 6]
#     length_metrics = calculate_dialogue_length_metrics(lengths)
#     print(f"Dialogue Length Metrics: {length_metrics}")

#     outcomes = [True, True, False, True, False]
#     success_rate = calculate_success_rate(outcomes)
#     print(f"Success Rate: {success_rate:.2f}%")


