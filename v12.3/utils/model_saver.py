
import torch
import os

def save_model(model, path, filename="model.pt"):
    """
    Saves a PyTorch model to the specified path.
    """
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    torch.save(model.state_dict(), full_path)
    print(f"Model saved to {full_path}")

def load_model(model, path, filename="model.pt"):
    """
    Loads a PyTorch model from the specified path.
    """
    full_path = os.path.join(path, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found at {full_path}")
    model.load_state_dict(torch.load(full_path))
    model.eval() # Set model to evaluation mode
    print(f"Model loaded from {full_path}")
    return model

# Example usage:
# if __name__ == "__main__":
#     import torch.nn as nn
#     class SimpleModel(nn.Module):
#         def __init__(self):
#             super(SimpleModel, self).__init__()
#             self.linear = nn.Linear(10, 1)
#         def forward(self, x):
#             return self.linear(x)

#     model = SimpleModel()
#     save_model(model, "./saved_models", "my_simple_model.pt")

#     loaded_model = SimpleModel()
#     loaded_model = load_model(loaded_model, "./saved_models", "my_simple_model.pt")
#     print("Model loaded successfully.")


