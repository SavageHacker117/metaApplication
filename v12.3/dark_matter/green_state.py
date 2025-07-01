
import logging
from typing import Dict, Any, Optional
from uuid import uuid4

from .blockchain.chain import Blockchain
from .data_contracts.dm_schemas import EnvironmentConfig, EnvironmentMetadata

logger = logging.getLogger(__name__)

class GreenState:
    """
    Manages the canonical, blockchain-guarded "Green State" of the multiverse.
    Environments in this state are considered immutable and their changes are
    recorded on the blockchain.
    """

    def __init__(self, blockchain: Blockchain):
        self.blockchain = blockchain
        self.green_environments: Dict[str, Dict[str, Any]] = {}
        self._load_from_blockchain()
        logger.info("GreenState manager initialized.")

    def _load_from_blockchain(self):
        """
        Loads the current state of green environments by replaying blockchain transactions.
        This ensures consistency with the immutable ledger.
        """
        logger.info("Replaying blockchain to reconstruct Green State...")
        # In a real scenario, this would parse blockchain transactions
        # to reconstruct the current set of green environments.
        # For simplicity, we'll assume the blockchain provides a direct way
        # to query current green states or we parse specific transaction types.
        # For now, this will be a placeholder.
        # Example: Iterate through blocks and apply 'promote_environment' transactions
        for block in self.blockchain.chain:
            for tx in block.transactions:
                if tx.payload.get("action") == "promote_environment":
                    env_id = tx.payload.get("env_id")
                    config_data = tx.payload.get("config")
                    metadata_data = tx.payload.get("metadata")
                    if env_id and config_data and metadata_data:
                        self.green_environments[env_id] = {
                            "status": "green",
                            "config": config_data,
                            "metadata": metadata_data
                        }
                elif tx.payload.get("action") == "terminate_environment":
                    env_id = tx.payload.get("env_id")
                    if env_id in self.green_environments:
                        del self.green_environments[env_id]
        logger.info(f"Green State reconstructed. {len(self.green_environments)} environments loaded.")

    def promote_blue_to_green(self, blue_env_id: str, config: EnvironmentConfig, metadata: EnvironmentMetadata) -> Optional[str]:
        """
        Promotes a Blue State environment to the Green State by recording the promotion
        as a transaction on the blockchain.
        """
        green_env_id = str(uuid4())
        payload = {
            "action": "promote_environment",
            "env_id": green_env_id,
            "blue_env_id": blue_env_id,
            "config": config.dict(),
            "metadata": metadata.dict()
        }
        # In a real system, signature would come from an authorized validator
        signature = f"signed_by_validator_{str(uuid4())}"

        try:
            self.blockchain.new_transaction(
                sender="blue_state_promoter",
                recipient="green_state_ledger",
                payload=payload,
                signature=signature
            )
            self.blockchain.new_block(proof=len(self.blockchain.chain) + 1, validator="system_promoter")
            self.green_environments[green_env_id] = {
                "status": "green",
                "config": config.dict(),
                "metadata": metadata.dict()
            }
            logger.info(f"Successfully promoted {blue_env_id} to Green State: {green_env_id}")
            return green_env_id
        except Exception as e:
            logger.error(f"Failed to promote Blue State {blue_env_id} to Green State: {e}")
            return None

    def terminate_environment(self, env_id: str) -> bool:
        """
        Terminates a Green State environment by recording the termination
        as a transaction on the blockchain.
        """
        if env_id not in self.green_environments:
            logger.warning(f"Attempted to terminate non-existent Green State environment: {env_id}")
            return False

        payload = {
            "action": "terminate_environment",
            "env_id": env_id
        }
        signature = f"signed_by_validator_{str(uuid4())}"

        try:
            self.blockchain.new_transaction(
                sender="green_state_terminator",
                recipient="green_state_ledger",
                payload=payload,
                signature=signature
            )
            self.blockchain.new_block(proof=len(self.blockchain.chain) + 1, validator="system_terminator")
            del self.green_environments[env_id]
            logger.info(f"Successfully terminated Green State environment: {env_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to terminate Green State environment {env_id}: {e}")
            return False

    def list_environments(self) -> Dict[str, Any]:
        """
        Lists all environments currently in the Green State.
        """
        return self.green_environments

    def rollback(self, block_index: int) -> bool:
        """
        Initiates a rollback of the Green State by rolling back the underlying blockchain.
        This is a critical operation and will re-sync the green_environments state.
        """
        logger.warning(f"Initiating Green State rollback to block index: {block_index}")
        success = self.blockchain.rollback_chain(block_index)
        if success:
            self.green_environments.clear()
            self._load_from_blockchain() # Reconstruct state after rollback
            logger.info("Green State successfully rolled back and re-synced.")
        return success




