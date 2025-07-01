
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class LLMModelConfig(BaseModel):
    name: str = Field(..., description="Name of the LLM model (e.g., GPT-4, Llama-2).")
    version: str = Field(..., description="Version of the LLM model.")
    parameters: Dict[str, Any] = Field({}, description="Specific parameters for the LLM model.")
    finetuning_data_path: Optional[str] = Field(None, description="Path to finetuning dataset for the LLM.")

class RLModelConfig(BaseModel):
    name: str = Field(..., description="Name of the RL model/algorithm (e.g., PPO, SAC).")
    version: str = Field(..., description="Version of the RL model.")
    hyperparameters: Dict[str, Any] = Field({}, description="Hyperparameters for the RL model.")
    checkpoint_path: Optional[str] = Field(None, description="Path to a pre-trained RL model checkpoint.")

class ModelConfig(BaseModel):
    llm_model: LLMModelConfig
    rl_model: RLModelConfig
    embedding_model_name: str = Field("bert-base-uncased", description="Name of the model used for generating embeddings.")

# Example usage:
# config_data = {
#     "llm_model": {"name": "GPT-4", "version": "1.0", "parameters": {"temperature": 0.7}},
#     "rl_model": {"name": "PPO", "version": "2.0", "hyperparameters": {"learning_rate": 0.0001}},
#     "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2"
# }
# try:
#     validated_config = ModelConfig(**config_data)
#     print("Model configuration validated successfully!")
# except ValidationError as e:
#     print(f"Model configuration validation error: {e}")


