
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class LLMConfig(BaseModel):
    model_name: str = Field(..., description="Name of the large language model to use.")
    api_key: str = Field(..., description="API key for the LLM service.")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature for text generation.")
    max_tokens: int = Field(150, ge=1, description="Maximum number of tokens to generate.")

class RLConfig(BaseModel):
    algorithm: str = Field(..., description="Reinforcement learning algorithm to use (e.g., PPO, A2C).")
    learning_rate: float = Field(0.0001, ge=0.0, description="Learning rate for the RL agent.")
    gamma: float = Field(0.99, ge=0.0, le=1.0, description="Discount factor for future rewards.")
    num_episodes: int = Field(1000, ge=1, description="Number of training episodes.")

class SystemConfig(BaseModel):
    environment_name: str = Field(..., description="Name of the simulation environment.")
    log_level: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR).")
    output_dir: str = Field("./outputs", description="Directory for saving logs and models.")
    debug_mode: bool = Field(False, description="Enable or disable debug mode.")

class MainConfig(BaseModel):
    llm: LLMConfig
    rl: RLConfig
    system: SystemConfig
    plugins: Optional[Dict[str, Any]] = Field(None, description="Configuration for various plugins.")

# Example usage:
# config_data = {
#     "llm": {"model_name": "gpt-4", "api_key": "your_key"},
#     "rl": {"algorithm": "PPO", "learning_rate": 0.0005},
#     "system": {"environment_name": "text_env", "output_dir": "./runs"}
# }
# try:
#     validated_config = MainConfig(**config_data)
#     print("Configuration validated successfully!")
# except ValidationError as e:
#     print(f"Configuration validation error: {e}")


