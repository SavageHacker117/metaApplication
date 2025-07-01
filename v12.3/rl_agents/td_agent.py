

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        return self.actor(state), self.critic(state)

class TDAgent:
    """
    Reinforcement Learning Agent specifically designed for the Tower Defense game.
    This agent learns to make decisions like placing towers, upgrading, or starting waves.
    """
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs, value = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        self.values.append(value)
        return action.item()

    def update(self):
        # Calculate discounted rewards
        returns = []
        R = 0
        for r, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
            if is_terminal:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Calculate advantages
        values = torch.cat(self.values).squeeze()
        advantages = returns - values

        # Calculate actor and critic loss
        log_probs = torch.cat(self.log_probs)
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.MSELoss()(values, returns)

        # Backpropagate and optimize
        self.optimizer.zero_grad()
        loss = actor_loss + critic_loss
        loss.backward()
        self.optimizer.step()

        # Clear memory
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []

# Example usage (requires state_dim and action_dim):
# if __name__ == "__main__":
#     state_dim = 100 # Example, should match GameStateProcessor output
#     action_dim = 3 # Example: place_tower, upgrade_tower, start_wave
#     agent = TDAgent(state_dim, action_dim)

#     dummy_state = torch.randn(state_dim)
#     action = agent.select_action(dummy_state)
#     print(f"Selected action: {action}")

#     agent.rewards.append(0.5)
#     agent.is_terminals.append(False)
#     agent.update()
#     print("TD Agent updated.")


