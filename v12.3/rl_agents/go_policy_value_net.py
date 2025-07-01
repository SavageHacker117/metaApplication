

import torch
import torch.nn as nn
import torch.nn.functional as F

class GoPolicyValueNet(nn.Module):
    """
    A neural network for Go, inspired by AlphaZero, that outputs both a policy
    (probability distribution over moves) and a value (win probability).
    """
    def __init__(self, board_size: int, num_filters: int = 128):
        super(GoPolicyValueNet, self).__init__()
        self.board_size = board_size
        self.input_channels = 3 # Current player, opponent, empty (simplified)
        self.action_size = board_size * board_size + 1 # +1 for pass move

        # Common convolutional layers
        self.conv1 = nn.Conv2d(self.input_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, num_filters)
        self.value_fc2 = nn.Linear(num_filters, 1)

    def forward(self, state_batch):
        # state_batch shape: (batch_size, input_channels, board_size, board_size)
        x = F.relu(self.bn1(self.conv1(state_batch)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Policy head
        policy_x = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_x = policy_x.view(-1, 2 * self.board_size * self.board_size)
        policy_logits = self.policy_fc(policy_x)
        policy_probs = F.softmax(policy_logits, dim=1)

        # Value head
        value_x = F.relu(self.value_bn(self.value_conv(x)))
        value_x = value_x.view(-1, 1 * self.board_size * self.board_size)
        value_x = F.relu(self.value_fc1(value_x))
        value_output = torch.tanh(self.value_fc2(value_x)) # Value between -1 and 1

        return policy_probs, value_output

# Example Usage:
# if __name__ == "__main__":
#     board_size = 9
#     model = GoPolicyValueNet(board_size)
#     
#     # Dummy input state (batch_size, channels, height, width)
#     # Channels: 0 for current player stones, 1 for opponent stones, 2 for empty
#     dummy_state = torch.randn(1, 3, board_size, board_size)
#     
#     policy_probs, value_output = model(dummy_state)
#     
#     print(f"Policy probabilities shape: {policy_probs.shape}") # (1, board_size*board_size + 1)
#     print(f"Value output shape: {value_output.shape}") # (1, 1)
#     print(f"Sample policy probabilities: {policy_probs[0, :5]}")
#     print(f"Sample value output: {value_output.item()}")


