

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple

from .config_schema import MainConfig, LLMConfig, RLConfig, SystemConfig

class GameSpecificConfig(BaseModel):
    grid_size: Tuple[int, int] = Field((10, 10), description="Dimensions of the game grid (width, height).")
    initial_cash: int = Field(150, ge=0, description="Starting cash for the player.")
    initial_lives: int = Field(20, ge=1, description="Starting lives for the player.")
    tower_types: Dict[str, Dict[str, Any]] = Field(
        {
            "basic": {"cost": 50, "range": 2, "damage": 10},
            "archer": {"cost": 75, "range": 3, "damage": 15},
            "cannon": {"cost": 120, "range": 2, "damage": 30}
        },
        description="Definitions of available tower types."
    )
    enemy_types: Dict[str, Dict[str, Any]] = Field(
        {
            "grunt": {"health": 50, "speed": 1},
            "fast": {"health": 30, "speed": 2},
            "tank": {"health": 150, "speed": 0.5}
        },
        description="Definitions of available enemy types."
    )
    max_waves: int = Field(10, ge=1, description="Maximum number of waves in a game.")

class TDRL_LLMMainConfig(MainConfig):
    game_specific: GameSpecificConfig = Field(..., description="Game-specific configurations for Tower Defense.")

# Example usage:
# if __name__ == "__main__":
#     from pydantic import ValidationError
#     try:
#         td_config_data = {
#             "llm": {"model_name": "gpt-3.5-turbo", "api_key": "YOUR_OPENAI_API_KEY"},
#             "rl": {"algorithm": "PPO", "learning_rate": 0.0001, "gamma": 0.99, "num_episodes": 10},
#             "system": {"environment_name": "tower_defense", "log_level": "INFO", "output_dir": "./td_outputs", "debug_mode": False},
#             "game_specific": {
#                 "grid_size": [15, 15],
#                 "initial_cash": 200,
#                 "tower_types": {
#                     "laser": {"cost": 100, "range": 4, "damage": 20}
#                 }
#             }
#         }
#         validated_td_config = TDRL_LLMMainConfig(**td_config_data)
#         print("Tower Defense configuration validated successfully!")
#     except ValidationError as e:
#         print(f"Tower Defense configuration validation error: {e}")


