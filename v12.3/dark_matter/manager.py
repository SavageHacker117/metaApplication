
import logging
from typing import Dict, Any, Optional, List
from ..blockchain.chain import Blockchain
from ..green_state import GreenState
from ..blue_state import BlueState
from ..data_contracts.dm_schemas import EnvironmentMetadata, EnvironmentConfig

logger = logging.getLogger(__name__)

class DarkMatterManager:
    """
    The central orchestration engine for the DARK MATTER meta-environmental layer.
    Handles environment creation, listing, termination, fork/merge operations,
    and integration with the blockchain for state management.
    """

    def __init__(self):
        self.blockchain = Blockchain()
        self.green_state = GreenState(self.blockchain)
        self.blue_state = BlueState()
        self.environments: Dict[str, Any] = {}
        logger.info("DarkMatterManager initialized.")

    def create_env(self, config: EnvironmentConfig, base_env_id: Optional[str] = None) -> str:
        """
        Creates a new experimental (Blue State) environment.
        If a base_env_id is provided, it forks from an existing environment.
        """
        env_id = self.blue_state.create_environment(config, base_env_id)
        self.environments[env_id] = {
            "status": "blue",
            "config": config.dict(),
            "metadata": EnvironmentMetadata(name=f"Environment-{env_id[:8]}", description="New experimental environment").dict()
        }
        logger.info(f"Created new Blue State environment: {env_id}")
        return env_id

    def fork_env(self, source_env_id: str) -> Optional[str]:
        """
        Forks an existing environment (Blue or Green) into a new Blue State environment.
        """
        if source_env_id not in self.environments:
            logger.error(f"Source environment {source_env_id} not found for forking.")
            return None

        source_env_config = EnvironmentConfig(**self.environments[source_env_id]["config"])
        new_env_id = self.blue_state.create_environment(source_env_config, source_env_id)
        self.environments[new_env_id] = {
            "status": "blue",
            "config": source_env_config.dict(),
            "metadata": EnvironmentMetadata(name=f"Forked-Env-{new_env_id[:8]}", description=f"Forked from {source_env_id}").dict()
        }
        logger.info(f"Forked environment {source_env_id} to new Blue State environment: {new_env_id}")
        return new_env_id

    def merge_envs(self, env_ids: List[str], new_env_name: str) -> Optional[str]:
        """
        Merges multiple experimental (Blue State) environments into a new Blue State environment.
        """
        for env_id in env_ids:
            if env_id not in self.environments or self.environments[env_id]["status"] != "blue":
                logger.error(f"Environment {env_id} not found or not a Blue State environment for merging.")
                return None
        
        merged_config = self.blue_state.merge_environments(env_ids)
        if merged_config:
            new_env_id = self.blue_state.create_environment(merged_config)
            self.environments[new_env_id] = {
                "status": "blue",
                "config": merged_config.dict(),
                "metadata": EnvironmentMetadata(name=new_env_name, description=f"Merged from {', '.join(env_ids)}").dict()
            }
            logger.info(f"Merged environments {', '.join(env_ids)} into new Blue State environment: {new_env_id}")
            return new_env_id
        return None

    def terminate_env(self, env_id: str) -> bool:
        """
        Terminates an environment, removing it from the manager.
        Green State environments require blockchain transaction for termination.
        """
        if env_id not in self.environments:
            logger.warning(f"Attempted to terminate non-existent environment: {env_id}")
            return False

        if self.environments[env_id]["status"] == "green":
            # For Green State, a blockchain transaction is required
            success = self.green_state.terminate_environment(env_id)
            if success:
                del self.environments[env_id]
                logger.info(f"Terminated Green State environment: {env_id}")
            return success
        else:
            # For Blue State, simply remove
            self.blue_state.terminate_environment(env_id)
            del self.environments[env_id]
            logger.info(f"Terminated Blue State environment: {env_id}")
            return True

    def list_environments(self) -> Dict[str, Any]:
        """
        Lists all managed environments with their status and metadata.
        """
        all_envs = {}
        all_envs.update(self.blue_state.list_environments())
        all_envs.update(self.green_state.list_environments())
        
        # Update internal tracking with latest from states
        for env_id, env_data in all_envs.items():
            self.environments[env_id] = {
                "status": env_data["status"],
                "config": env_data["config"],
                "metadata": env_data["metadata"]
            }
        return self.environments

    def promote_to_green(self, blue_env_id: str, metadata: EnvironmentMetadata) -> Optional[str]:
        """
        Promotes a Blue State environment to a Green State, securing it on the blockchain.
        """
        if blue_env_id not in self.environments or self.environments[blue_env_id]["status"] != "blue":
            logger.error(f"Environment {blue_env_id} not found or not a Blue State for promotion.")
            return None

        config = EnvironmentConfig(**self.environments[blue_env_id]["config"])
        green_env_id = self.green_state.promote_blue_to_green(blue_env_id, config, metadata)
        if green_env_id:
            # Update status in internal tracking
            self.environments[green_env_id] = {
                "status": "green",
                "config": config.dict(),
                "metadata": metadata.dict()
            }
            # Remove from blue state tracking if successful
            self.blue_state.terminate_environment(blue_env_id) # Remove from blue state after promotion
            del self.environments[blue_env_id] # Remove old blue entry
            logger.info(f"Promoted Blue State environment {blue_env_id} to Green State: {green_env_id}")
            return green_env_id
        return None

    def get_multiverse_graph(self) -> Dict[str, Any]:
        """
        Generates a graph representation of the multiverse, showing relationships
        between environments (forks, merges, promotions).
        """
        nodes = []
        edges = []

        all_envs = self.list_environments() # Ensure internal state is up-to-date

        for env_id, data in all_envs.items():
            nodes.append({
                "id": env_id,
                "name": data["metadata"]["name"],
                "status": data["status"],
                "type": "environment"
            })
            if "base_env_id" in data["config"] and data["config"]["base_env_id"]:
                edges.append({
                    "source": data["config"]["base_env_id"],
                    "target": env_id,
                    "type": "fork"
                })
            # Add more edge types for merges, promotions if tracked in config/metadata

        # Add blockchain nodes/edges if relevant for visualization
        blockchain_status = self.blockchain.get_status()
        nodes.append({"id": "blockchain_root", "name": "Blockchain Ledger", "status": "active", "type": "system"})
        for block in blockchain_status.get("chain", []):
            block_id = f"block_{block['index']}"
            nodes.append({"id": block_id, "name": f"Block {block['index']}", "status": "sealed", "type": "blockchain_block"})
            edges.append({"source": "blockchain_root", "target": block_id, "type": "contains"})
            if block['index'] > 0:
                edges.append({"source": f"block_{block['index']-1}", "target": block_id, "type": "next_block"})

        return {"nodes": nodes, "edges": edges}

    def get_blockchain_status(self) -> Dict[str, Any]:
        """
        Retrieves the current status of the underlying blockchain.
        """
        return self.blockchain.get_status()

    def rollback_green_state(self, block_index: int) -> bool:
        """
        Initiates a rollback of the Green State to a specified block index.
        This is a critical operation and should be used with caution.
        """
        logger.warning(f"Initiating Green State rollback to block index: {block_index}")
        return self.green_state.rollback(block_index)

    def audit_logs(self) -> List[Dict[str, Any]]:
        """
        Retrieves the audit logs from the blockchain.
        """
        return self.blockchain.get_audit_logs()




