
import logging
from typing import Dict, Any, Optional

from .dark_matter import DarkMatterManager, EnvironmentConfig, EnvironmentMetadata

logger = logging.getLogger(__name__)

class DarkMatterEnvironment:
    """
    An RL environment interface that leverages the DarkMatterManager
    to create, manage, and interact with environments.
    """

    def __init__(self, dm_manager: DarkMatterManager, initial_env_config: Optional[EnvironmentConfig] = None):
        self.dm_manager = dm_manager
        self.current_env_id: Optional[str] = None
        self.current_env_config: Optional[EnvironmentConfig] = None
        self.difficulty: float = 0.5

        if initial_env_config:
            self.current_env_id = self.dm_manager.create_env(initial_env_config)
            self.current_env_config = initial_env_config
            logger.info(f"Initialized DarkMatterEnvironment with initial environment: {self.current_env_id}")
        else:
            logger.warning("DarkMatterEnvironment initialized without an initial environment config. Call reset() to create one.")

    def reset(self, env_config: Optional[EnvironmentConfig] = None) -> Dict[str, Any]:
        """
        Resets the environment, optionally creating a new one based on config.
        If no config is provided, it attempts to reset the current environment
        or create a default one.
        """
        if self.current_env_id:
            self.dm_manager.terminate_env(self.current_env_id)
            self.current_env_id = None
            self.current_env_config = None

        if env_config:
            self.current_env_id = self.dm_manager.create_env(env_config)
            self.current_env_config = env_config
        else:
            # Create a default environment if none exists
            default_config = EnvironmentConfig(
                env_type="default_rl_env",
                difficulty=self.difficulty,
                params={"initial_state": "random"}
            )
            self.current_env_id = self.dm_manager.create_env(default_config)
            self.current_env_config = default_config

        logger.info(f"Environment reset. Current active environment: {self.current_env_id}")
        # In a real scenario, this would return the initial observation of the environment
        return {"env_id": self.current_env_id, "state": "initial_state_data", "config": self.current_env_config.dict()}

    def step(self, action: Any) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Performs a step in the current active environment.
        This is a placeholder for actual environment interaction.
        """
        if not self.current_env_id:
            logger.error("Cannot step: no active environment. Call reset() first.")
            return {}, 0.0, True, {"error": "no_active_environment"}

        # Simulate environment interaction
        # In a real system, this would involve sending the action to the actual environment
        # and receiving observation, reward, done, and info.
        
        # For demonstration, we'll simulate a simple step
        observation = {"env_id": self.current_env_id, "state": "new_state_data"}
        reward = 1.0 # Placeholder reward
        done = False # Environment never terminates for simplicity
        info = {"action_taken": action, "difficulty": self.difficulty}

        logger.debug(f"Stepped in env {self.current_env_id} with action {action}")
        return observation, reward, done, info

    def set_difficulty(self, difficulty: float):
        """
        Sets the difficulty for the current environment.
        This would typically involve updating the environment's internal state or config.
        """
        self.difficulty = difficulty
        if self.current_env_config:
            self.current_env_config.difficulty = difficulty
            # In a real system, you might need to re-create or re-configure the environment
            # self.dm_manager.update_env_config(self.current_env_id, self.current_env_config)
        logger.info(f"Set environment difficulty to: {difficulty}")

    def get_current_environment_info(self) -> Optional[Dict[str, Any]]:
        """
        Retrieves information about the currently active environment.
        """
        if self.current_env_id:
            return self.dm_manager.list_environments().get(self.current_env_id)
        return None

    def fork_current_environment(self) -> Optional[str]:
        """
        Forks the current environment into a new Blue State environment.
        """
        if self.current_env_id:
            new_env_id = self.dm_manager.fork_env(self.current_env_id)
            if new_env_id:
                logger.info(f"Forked current environment {self.current_env_id} to {new_env_id}")
                return new_env_id
        logger.warning("No current environment to fork.")
        return None

    def promote_current_environment_to_green(self, metadata: EnvironmentMetadata) -> Optional[str]:
        """
        Promotes the current Blue State environment to Green State.
        """
        if self.current_env_id and self.current_env_config:
            promoted_id = self.dm_manager.promote_to_green(self.current_env_id, metadata)
            if promoted_id:
                self.current_env_id = promoted_id # Update to the new Green State ID
                logger.info(f"Promoted environment to Green State: {promoted_id}")
                return promoted_id
        logger.warning("No current Blue State environment to promote.")
        return None

    def __del__(self):
        """
        Ensures the current environment is terminated when the object is deleted.
        """
        if self.current_env_id:
            self.dm_manager.terminate_env(self.current_env_id)
            logger.info(f"Cleaned up DarkMatterEnvironment: terminated {self.current_env_id}")




