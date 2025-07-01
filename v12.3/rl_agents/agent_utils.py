
import torch
import numpy as np

def preprocess_state(state_text: str, llm_utils_instance) -> torch.Tensor:
    """
    Converts a text state into a numerical embedding using the LLMUtils.
    """
    embedding = llm_utils_instance.get_embedding(state_text)
    return torch.FloatTensor(embedding).unsqueeze(0) # Add batch dimension

def sample_action_from_policy(action_probs: torch.Tensor) -> int:
    """
    Samples an action from a probability distribution.
    """
    dist = torch.distributions.Categorical(action_probs)
    action = dist.sample()
    return action.item()

def calculate_gae(rewards: list, values: list, dones: list, gamma: float, lambda_gae: float) -> np.ndarray:
    """
    Calculates Generalized Advantage Estimation (GAE).
    """
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lambda_gae * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    return np.array(advantages)

# Example usage:
# if __name__ == "__main__":
#     # This requires an LLMUtils instance, which depends on transformers.
#     # For a standalone test, you might mock it or run it in an environment
#     # where transformers is installed.
#     from llm_integration.llm_utils import LLMUtils
#     llm_utils = LLMUtils()
#     state_text = "The user said hello."
#     processed_state = preprocess_state(state_text, llm_utils)
#     print(f"Processed state shape: {processed_state.shape}")

#     action_probs = torch.tensor([0.1, 0.2, 0.7])
#     action = sample_action_from_policy(action_probs)
#     print(f"Sampled action: {action}")

#     rewards = [1, 1, 0]
#     values = [0.5, 0.6, 0.7, 0.0] # values[t+1] for terminal state is 0
#     dones = [0, 0, 1]
#     advantages = calculate_gae(rewards, values, dones, 0.99, 0.95)
#     print(f"Advantages: {advantages}")


