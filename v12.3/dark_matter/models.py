
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class WorldLine:
    """Represents an evolutionary path of an environment through the multiverse."""
    id: str
    name: str
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # e.g., 'active', 'archived', 'terminated'
    parent_id: Optional[str] = None  # For forked environments
    merged_from: List[str] = field(default_factory=list) # For merged environments
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvironmentMetadata:
    """Metadata associated with an environment."""
    name: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    author: str = "Manus AI"
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class EnvironmentConfig:
    """Configuration parameters for an environment."""
    # Basic environment parameters
    env_type: str = "generic" # e.g., 'tower_defense', 'robotics_sim'
    difficulty: float = 0.5
    reward_function_id: str = "default"
    
    # NeRF/3D rendering specific parameters
    nerf_model_path: Optional[str] = None
    asset_pack_id: Optional[str] = None
    
    # LLM integration parameters
    llm_prompt_template: Optional[str] = None
    
    # Dark Matter specific parameters
    base_env_id: Optional[str] = None # Used for forking
    mutations: Dict[str, Any] = field(default_factory=dict) # Changes from base
    
    # Any other environment-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Transaction:
    """Represents a transaction on the blockchain, e.g., state promotion, termination."""
    timestamp: datetime
    sender: str
    recipient: str
    payload: Dict[str, Any]
    signature: str
    transaction_id: str

@dataclass
class Block:
    """Represents a block in the blockchain."""
    index: int
    timestamp: datetime
    transactions: List[Transaction]
    proof: int
    previous_hash: str
    hash: str

@dataclass
class BlockchainStatus:
    """Current status of the blockchain."""
    chain_length: int
    latest_block_hash: str
    pending_transactions: int
    nodes: List[str]
    consensus_status: str

@dataclass
class MultiverseGraph:
    """Represents the graph structure of environments and their relationships."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]




