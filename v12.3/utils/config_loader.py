
import json
import yaml
from pydantic import ValidationError
from typing import Type, TypeVar

T = TypeVar("T", bound=BaseModel)

def load_config(file_path: str, config_model: Type[T]) -> T:
    """
    Loads configuration from a JSON or YAML file and validates it against a Pydantic model.
    """
    file_extension = file_path.split(".")[-1].lower()
    config_data = {}

    try:
        with open(file_path, "r") as f:
            if file_extension == "json":
                config_data = json.load(f)
            elif file_extension in ["yaml", "yml"]:
                config_data = yaml.safe_load(f)
            else:
                raise ValueError("Unsupported configuration file format. Use .json, .yaml, or .yml.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found at {file_path}")
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise ValueError(f"Error parsing configuration file {file_path}: {e}")

    try:
        validated_config = config_model(**config_data)
        print(f"Configuration loaded and validated successfully from {file_path}")
        return validated_config
    except ValidationError as e:
        raise ValueError(f"Configuration validation error in {file_path}: {e}")

# Example Usage:
# if __name__ == "__main__":
#     from core.config.rl_llm_td_config import TDRL_LLMMainConfig
#     import os

#     # Create a dummy config file for testing
#     dummy_config_content = {
#         "llm": {"model_name": "test-llm", "api_key": "dummy_key"},
#         "rl": {"algorithm": "PPO", "learning_rate": 0.0001, "gamma": 0.99, "num_episodes": 1},
#         "system": {"environment_name": "test_env", "log_level": "INFO", "output_dir": "./test_outputs", "debug_mode": False},
#         "game_specific": {"grid_size": [5,5], "initial_cash": 100}
#     }
#     with open("test_config.json", "w") as f:
#         json.dump(dummy_config_content, f, indent=4)

#     try:
#         loaded_config = load_config("test_config.json", TDRL_LLMMainConfig)
#         print(f"Loaded LLM model name: {loaded_config.llm.model_name}")
#     except Exception as e:
#         print(f"Failed to load config: {e}")

#     os.remove("test_config.json")


