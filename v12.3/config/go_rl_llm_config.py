

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple

class LLMConfig(BaseModel):
    model_name: str = Field("gpt-3.5-turbo", description="Name of the LLM model to use.")
    api_key: str = Field(..., description="API key for the LLM service.")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature for LLM generation.")
    max_tokens: int = Field(500, ge=1, description="Maximum number of tokens to generate in LLM response.")

class RLConfig(BaseModel):
    algorithm: str = Field("AlphaZero", description="RL algorithm to use (e.g., AlphaZero, PPO).")
    learning_rate: float = Field(0.001, ge=0.0, description="Learning rate for the RL agent.")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor for rewards.")
    l2_reg: float = Field(1e-4, ge=0.0, description="L2 regularization constant for the neural network.")
    mcts_n_playout: int = Field(1600, ge=1, description="Number of MCTS playouts per move.")
    buffer_size: int = Field(10000, ge=1, description="Size of the experience replay buffer.")
    batch_size: int = Field(512, ge=1, description="Batch size for training the neural network.")
    epochs: int = Field(5, ge=1, description="Number of training epochs per update.")
    num_iterations: int = Field(100, ge=1, description="Number of self-play iterations for training.")
    save_freq: int = Field(10, ge=1, description="Frequency (in iterations) to save the model.")
    load_model_path: Optional[str] = Field(None, description="Path to a pre-trained model to load.")

class GameSpecificConfig(BaseModel):
    board_size: int = Field(9, ge=5, le=19, description="Size of the Go board (e.g., 9 for 9x9, 19 for 19x19).")
    komi: float = Field(6.5, description="Komi value for the game.")

class SystemConfig(BaseModel):
    environment_name: str = Field("minigo_go", description="Name of the Go environment implementation.")
    log_level: str = Field("INFO", description="Logging level (e.g., DEBUG, INFO, WARNING).")
    output_dir: str = Field("./output", description="Directory for saving logs, models, and game data.")
    use_cuda: bool = Field(False, description="Whether to use CUDA for GPU acceleration.")

class GoRL_LLMMainConfig(BaseModel):
    llm: LLMConfig = Field(..., description="Configuration for LLM integration.")
    rl: RLConfig = Field(..., description="Configuration for Reinforcement Learning.")
    game_specific: GameSpecificConfig = Field(..., description="Game-specific configurations for Go.")
    system: SystemConfig = Field(..., description="System-wide configurations.")

# Example Usage:
# if __name__ == "__main__":
#     from pydantic import ValidationError
#     try:
#         config_data = {
#             "llm": {"model_name": "gpt-4", "api_key": "sk-YOUR_API_KEY"},
#             "rl": {"learning_rate": 0.0005, "num_iterations": 200},
#             "game_specific": {"board_size": 13},
#             "system": {"use_cuda": True}
#         }
#         validated_config = GoRL_LLMMainConfig(**config_data)
#         print("Configuration validated successfully!")
#         print(validated_config.json(indent=2))
#     except ValidationError as e:
#         print(f"Configuration validation error: {e}")


