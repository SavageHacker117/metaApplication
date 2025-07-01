
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

class EnvironmentMetadata(BaseModel):
    """Metadata associated with an environment."""
    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0.0"
    author: str = "Manus AI"
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

class EnvironmentConfig(BaseModel):
    """Configuration parameters for an environment."""
    env_type: str = "generic"
    difficulty: float = 0.5
    reward_function_id: str = "default"
    nerf_model_path: Optional[str] = None
    asset_pack_id: Optional[str] = None
    llm_prompt_template: Optional[str] = None
    base_env_id: Optional[str] = None
    mutations: Dict[str, Any] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)

class Transaction(BaseModel):
    """Represents a transaction on the blockchain."""
    timestamp: datetime
    sender: str
    recipient: str
    payload: Dict[str, Any]
    signature: str
    transaction_id: str

class Block(BaseModel):
    """Represents a block in the blockchain."""
    index: int
    timestamp: datetime
    transactions: List[Transaction]
    proof: int
    previous_hash: str
    hash: str

class BlockchainStatus(BaseModel):
    """Current status of the blockchain."""
    chain_length: int
    latest_block_hash: str
    pending_transactions: int
    nodes: List[str]
    consensus_status: str

class MultiverseGraph(BaseModel):
    """Represents the graph structure of environments and their relationships."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


