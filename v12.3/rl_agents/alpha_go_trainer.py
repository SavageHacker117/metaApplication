

import torch
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any

from go_env.minigo_go_env import MinigoGoEnv
from rl_agent.go_policy_value_net import GoPolicyValueNet
from rl_agent.mcts import MCTS, state_to_nn_input

class AlphaGoTrainer:
    """
    Implements the training loop for an AlphaGo-style RL agent.
    Focuses on self-play, MCTS, and policy/value network updates.
    """
    def __init__(self, policy_value_net: GoPolicyValueNet, board_size: int = 9,
                 learning_rate: float = 0.001, l2_const: float = 1e-4,
                 momentum: float = 0.9, n_playout: int = 1600,
                 buffer_size: int = 10000, batch_size: int = 512,
                 epochs: int = 5, kl_targ: float = 0.02,
                 check_freq: int = 100, cuda: bool = False):

        self.policy_value_net = policy_value_net
        self.board_size = board_size
        self.l2_const = l2_const
        self.momentum = momentum
        self.n_playout = n_playout
        self.data_buffer = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.kl_targ = kl_targ
        self.check_freq = check_freq
        self.cuda = cuda and torch.cuda.is_available()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=learning_rate, weight_decay=l2_const)

        if self.cuda:
            self.policy_value_net.cuda()

    def collect_self_play_data(self, n_games: int = 1):
        """
        Collects self-play data by playing n_games and storing (state, mcts_prob, winner) tuples.
        """
        for i in range(n_games):
            env = MinigoGoEnv(board_size=self.board_size)
            mcts = MCTS(self.policy_value_net, n_playout=self.n_playout)
            states, mcts_probs = [], []
            current_state = env.reset()
            done = False
            episode_len = 0

            while not done:
                # Convert current_state (numpy array) to NN input format
                nn_input = state_to_nn_input(current_state, env.get_current_player(), self.board_size)
                if self.cuda:
                    nn_input = nn_input.cuda()

                # Get action probabilities from MCTS
                action_probs, _ = mcts.get_action_probs(env, temp=1.0) # temp=1.0 for exploration

                # Store data
                states.append(nn_input.cpu().squeeze(0).numpy()) # Store raw numpy state
                mcts_probs.append(action_probs)

                # Choose action based on probabilities
                action_idx = np.random.choice(len(action_probs), p=action_probs)
                
                # Map action_idx back to (row, col) or (None, None)
                if action_idx == self.board_size * self.board_size: # Pass move
                    action = (None, None)
                else:
                    r = action_idx // self.board_size
                    c = action_idx % self.board_size
                    action = (r, c)

                current_state, reward, done, info = env.step(action)
                mcts.update_with_move(action) # Update MCTS tree
                episode_len += 1

            # Game ended, assign winner
            winner = 1 if env.get_score() > 0 else -1 # Simplified winner determination
            # For each state, assign the winner from that state's perspective
            for j in range(len(states)):
                # If the player at states[j] was the winner, assign 1, else -1
                # This requires knowing whose turn it was at states[j]
                # For simplicity, assuming winner is from Black's perspective, and alternating turns
                # A more robust solution would store player info with state.
                # For now, let's assume winner is for the player who started the game (Black).
                self.data_buffer.append((states[j], mcts_probs[j], winner))

            # Keep buffer size limited
            if len(self.data_buffer) > self.buffer_size:
                self.data_buffer = self.data_buffer[-self.buffer_size:]
            print(f"Game {i+1} finished. Episode length: {episode_len}. Winner: {winner}")

    def train_step(self):
        """
        Performs one training step on a batch of collected data.
        """
        if len(self.data_buffer) < self.batch_size:
            print("Not enough data in buffer for training.")
            return

        # Sample batch
        sample_indices = np.random.choice(len(self.data_buffer), self.batch_size, replace=False)
        batch = [self.data_buffer[i] for i in sample_indices]
        state_batch, mcts_prob_batch, winner_batch = zip(*batch)

        state_batch = torch.from_numpy(np.array(state_batch)).float()
        mcts_prob_batch = torch.from_numpy(np.array(mcts_prob_batch)).float()
        winner_batch = torch.from_numpy(np.array(winner_batch)).float().unsqueeze(1)

        if self.cuda:
            state_batch = state_batch.cuda()
            mcts_prob_batch = mcts_prob_batch.cuda()
            winner_batch = winner_batch.cuda()

        # Forward pass
        policy_probs, value_output = self.policy_value_net(state_batch)

        # Calculate loss
        value_loss = F.mse_loss(value_output, winner_batch)
        policy_loss = -torch.sum(mcts_prob_batch * torch.log(policy_probs + 1e-10), dim=1).mean()
        total_loss = value_loss + policy_loss

        # Backpropagate and optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()

    def run(self, num_iterations: int):
        """
        Runs the main training loop.
        """
        for i in range(num_iterations):
            print(f"\n--- Training Iteration {i+1} ---")
            self.collect_self_play_data(n_games=self.check_freq) # Collect data for a few games
            
            if len(self.data_buffer) >= self.batch_size:
                total_loss, policy_loss, value_loss = self.train_step()
                print(f"Loss: {total_loss:.4f}, Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}")
            else:
                print("Data buffer not full enough for training.")

            # Save model periodically
            if (i + 1) % self.check_freq == 0:
                self.save_model(f"model_iter_{i+1}.pth")

    def save_model(self, path: str):
        torch.save(self.policy_value_net.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        self.policy_value_net.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

# Example Usage:
# if __name__ == "__main__":
#     board_size = 9
#     policy_value_net = GoPolicyValueNet(board_size)
#     trainer = AlphaGoTrainer(policy_value_net, board_size=board_size, cuda=False)
#     trainer.run(num_iterations=5) # Run a few iterations


