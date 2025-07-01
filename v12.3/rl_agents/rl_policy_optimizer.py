

import torch
import torch.optim as optim
from typing import List, Tuple

class RLPolicyOptimizer:
    """
    Manages the optimization process for the RL agent's policy and value networks.
    Encapsulates the training loop for the agent, applying gradients based on collected experiences.
    """
    def __init__(self, agent, learning_rate: float, clip_param: float = 0.2, num_epochs: int = 10, batch_size: int = 64):
        self.agent = agent
        self.optimizer = optim.Adam(self.agent.actor_critic.parameters(), lr=learning_rate)
        self.clip_param = clip_param
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def optimize_policy(self, states: List[torch.Tensor], actions: List[torch.Tensor], old_log_probs: List[torch.Tensor], advantages: List[torch.Tensor], returns: List[torch.Tensor]):
        """
        Performs policy optimization using the collected experience.
        """
        # Convert lists to tensors
        states = torch.cat(states)
        actions = torch.cat(actions)
        old_log_probs = torch.cat(old_log_probs)
        advantages = torch.cat(advantages)
        returns = torch.cat(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        dataset_size = states.size(0)
        for _ in range(self.num_epochs):
            # Shuffle and create mini-batches
            indices = torch.randperm(dataset_size)
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get new action probabilities and values
                action_probs, values = self.agent.actor_critic(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)

                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = (values - batch_returns).pow(2).mean()

                # Total loss
                loss = policy_loss + 0.5 * value_loss # 0.5 is a common coefficient for value loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# Example Usage:
# if __name__ == "__main__":
#     from td_agent import TDAgent
#     state_dim = 10 # Example
#     action_dim = 3 # Example
#     agent = TDAgent(state_dim, action_dim)
#     optimizer = RLPolicyOptimizer(agent, learning_rate=0.001)

#     # Dummy data for optimization
#     states = [torch.randn(state_dim) for _ in range(100)]
#     actions = [torch.randint(0, action_dim, (1,)) for _ in range(100)]
#     old_log_probs = [torch.randn(1) for _ in range(100)]
#     advantages = [torch.randn(1) for _ in range(100)]
#     returns = [torch.randn(1) for _ in range(100)]

#     optimizer.optimize_policy(states, actions, old_log_probs, advantages, returns)
#     print("Policy optimization complete.")
