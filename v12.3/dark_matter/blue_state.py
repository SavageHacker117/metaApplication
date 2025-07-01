
import logging
from typing import Dict, Any, List, Optional
from uuid import uuid4
import copy

from .data_contracts.dm_schemas import EnvironmentConfig, EnvironmentMetadata

logger = logging.getLogger(__name__)

class BlueState:
    """
    Manages experimental, provisional "Blue State" environments.
    These environments are in-memory and allow for rapid iteration and experimentation
    without affecting the canonical Green State.
    """

    def __init__(self):
        self.blue_environments: Dict[str, Dict[str, Any]] = {}
        logger.info("BlueState manager initialized.")

    def create_environment(self, config: EnvironmentConfig, base_env_id: Optional[str] = None) -> str:
        """
        Creates a new Blue State environment.
        :param config: The configuration for the new environment.
        :param base_env_id: Optional ID of a base environment if this is a fork.
        :return: The ID of the newly created environment.
        """
        env_id = str(uuid4())
        new_config = copy.deepcopy(config) # Ensure deep copy to avoid mutation issues
        new_config.base_env_id = base_env_id # Link to parent if forked

        self.blue_environments[env_id] = {
            "status": "blue",
            "config": new_config.dict(),
            "metadata": EnvironmentMetadata(name=f"BlueEnv-{env_id[:8]}", description="Experimental environment").dict()
        }
        logger.info(f"Created Blue State environment: {env_id}")
        return env_id

    def terminate_environment(self, env_id: str) -> bool:
        """
        Terminates a Blue State environment.
        :param env_id: The ID of the environment to terminate.
        :return: True if successful, False otherwise.
        """
        if env_id in self.blue_environments:
            del self.blue_environments[env_id]
            logger.info(f"Terminated Blue State environment: {env_id}")
            return True
        logger.warning(f"Attempted to terminate non-existent Blue State environment: {env_id}")
        return False

    def list_environments(self) -> Dict[str, Any]:
        """
        Lists all environments currently in the Blue State.
        :return: A dictionary of Blue State environments.
        """
        return self.blue_environments

    def get_environment_config(self, env_id: str) -> Optional[EnvironmentConfig]:
        """
        Retrieves the configuration of a specific Blue State environment.
        :param env_id: The ID of the environment.
        :return: The EnvironmentConfig if found, None otherwise.
        """
        env_data = self.blue_environments.get(env_id)
        if env_data:
            return EnvironmentConfig(**env_data["config"])
        return None

    def merge_environments(self, env_ids: List[str]) -> Optional[EnvironmentConfig]:
        """
        Merges configurations of multiple Blue State environments into a new combined configuration.
        This is a simplified merge, in a real system, complex merge strategies would be applied.
        :param env_ids: List of environment IDs to merge.
        :return: A new EnvironmentConfig representing the merged state, or None if any env_id is invalid.
        """
        if not env_ids:
            logger.warning("No environment IDs provided for merging.")
            return None

        configs = []
        for env_id in env_ids:
            config = self.get_environment_config(env_id)
            if not config:
                logger.error(f"Environment {env_id} not found in Blue State for merging.")
                return None
            configs.append(config)

        # Simplified merge logic: take the first config and apply mutations from others
        # In a real scenario, this would involve more sophisticated conflict resolution and merging rules
        merged_config = copy.deepcopy(configs[0])
        merged_config.base_env_id = None # Merged environments don't have a single base
        merged_config.mutations = {}

        for i, config in enumerate(configs):
            # Example: merge parameters, with later configs overriding earlier ones
            merged_config.params.update(config.params)
            # Example: combine tags
            merged_config.tags.extend(tag for tag in config.tags if tag not in merged_config.tags)
            # Example: simple average of difficulty
            merged_config.difficulty = (merged_config.difficulty * i + config.difficulty) / (i + 1)
            # Merge mutations (simple union for now)
            merged_config.mutations.update(config.mutations)

        logger.info(f"Merged configurations for environments: {env_ids}")
        return merged_config




