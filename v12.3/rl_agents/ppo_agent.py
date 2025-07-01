

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.policy_old = Actor(state_dim, action_dim)
        self.policy_old.load_state_dict(self.actor.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.policy_old(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.squeeze(torch.stack(memory.states).detach(), 1)
        old_actions = torch.squeeze(torch.stack(memory.actions).detach(), 1)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs).detach(), 1)

        for _ in range(self.k_epochs):
            logprobs = self.actor(old_states).log()
            state_values = self.critic(old_states)

            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss_actor = -torch.min(surr1, surr2).mean()
            loss_critic = self.MseLoss(state_values, rewards)

            self.optimizer_actor.zero_grad()
            loss_actor.backward()
            self.optimizer_actor.step()

            self.optimizer_critic.zero_grad()
            loss_critic.backward()
            self.optimizer_critic.step()

        self.policy_old.load_state_dict(self.actor.state_dict())


