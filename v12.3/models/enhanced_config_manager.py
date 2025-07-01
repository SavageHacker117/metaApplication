"""
Enhanced Configuration Management System v3

Improvements based on feedback:
- Centralized config management with validation
- Environment variable support with fallbacks
- Configuration profiles for different environments
- Dynamic configuration updates
- Configuration versioning and migration
- Type-safe configuration with validation
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union, List, Type, get_type_hints, Callable
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
import logging
from enum import Enum
import warnings
from copy import deepcopy
import hashlib
import time
from contextlib import contextmanager

class ConfigEnvironment(Enum):
    """Configuration environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigFormat(Enum):
    """Supported configuration formats."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"

@dataclass
class TrainingConfig:
    """Training configuration with validation."""
    # Core training parameters
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    batch_size: int = 32
    learning_rate: float = 3e-4
    gamma: float = 0.99
    
    # Environment settings
    num_parallel_envs: int = 8
    use_vectorized_env: bool = True
    
    # Performance settings
    use_mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Checkpointing
    checkpoint_frequency: int = 100
    auto_resume: bool = True
    checkpoint_dir: str = "checkpoints"
    
    def validate(self):
        """Validate configuration values."""
        if self.max_episodes <= 0:
            raise ValueError("max_episodes must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")

@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    grid_size: List[int] = field(default_factory=lambda: [10, 10])
    max_towers: int = 20
    max_enemies: int = 50
    enemy_spawn_rate: float = 0.1
    tower_types: List[str] = field(default_factory=lambda: ["basic", "cannon", "archer"])
    
    def validate(self):
        """Validate environment configuration."""
        if len(self.grid_size) != 2:
            raise ValueError("grid_size must have exactly 2 dimensions")
        if any(size <= 0 for size in self.grid_size):
            raise ValueError("grid_size dimensions must be positive")
        if self.max_towers <= 0:
            raise ValueError("max_towers must be positive")
        if self.max_enemies <= 0:
            raise ValueError("max_enemies must be positive")

@dataclass
class RenderConfig:
    """Rendering configuration."""
    width: int = 1024
    height: int = 768
    use_headless_gpu: bool = True
    enable_context_pool: bool = True
    pool_size: int = 4
    render_timeout: float = 30.0
    enable_mock_renderer: bool = False
    mock_render_probability: float = 0.5
    
    def validate(self):
        """Validate render configuration."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be positive")
        if self.render_timeout <= 0:
            raise ValueError("render_timeout must be positive")

@dataclass
class RewardConfig:
    """Reward system configuration."""
    gameplay_weight: float = 0.4
    visual_quality_weight: float = 0.2
    code_quality_weight: float = 0.2
    performance_weight: float = 0.2
    enable_diversity_bonus: bool = True
    diversity_threshold: float = 0.8
    penalty_for_repetition: float = -0.1
    
    def validate(self):
        """Validate reward configuration."""
        total_weight = (self.gameplay_weight + self.visual_quality_weight + 
                       self.code_quality_weight + self.performance_weight)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Reward weights must sum to 1.0, got {total_weight}")
        
        if not 0 <= self.diversity_threshold <= 1:
            raise ValueError("diversity_threshold must be between 0 and 1")

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    log_dir: str = "logs"
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    log_frequency: int = 10
    save_logs: bool = True
    
    def validate(self):
        """Validate logging configuration."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.level not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        if self.log_frequency <= 0:
            raise ValueError("log_frequency must be positive")

@dataclass
class MasterConfig:
    """Master configuration containing all subsystem configs."""
    # Subsystem configurations
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Meta configuration
    config_version: str = "3.0.0"
    environment_name: str = ConfigEnvironment.DEVELOPMENT.value
    created_at: float = field(default_factory=time.time)
    
    def validate(self):
        """Validate all configurations."""
        self.training.validate()
        self.environment.validate()
        self.render.validate()
        self.reward.validate()
        self.logging.validate()

class ConfigValidator:
    """Validates configuration values and types."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type) -> bool:
        """Validate that value matches expected type."""
        if expected_type == Any:
            return True
        
        # Handle Union types (e.g., Optional[int])
        if hasattr(expected_type, '__origin__'):
            if expected_type.__origin__ is Union:
                return any(ConfigValidator.validate_type(value, t) for t in expected_type.__args__)
        
        # Handle List types
        if hasattr(expected_type, '__origin__') and expected_type.__origin__ is list:
            if not isinstance(value, list):
                return False
            if expected_type.__args__:
                item_type = expected_type.__args__[0]
                return all(ConfigValidator.validate_type(item, item_type) for item in value)
            return True
        
        return isinstance(value, expected_type)
    
    @staticmethod
    def validate_config_object(config_obj: Any) -> List[str]:
        """Validate a configuration object against its type hints."""
        errors = []
        
        if hasattr(config_obj, '__dataclass_fields__'):
            type_hints = get_type_hints(type(config_obj))
            
            for field_name, field_info in config_obj.__dataclass_fields__.items():
                if field_name in type_hints:
                    expected_type = type_hints[field_name]
                    actual_value = getattr(config_obj, field_name)
                    
                    if not ConfigValidator.validate_type(actual_value, expected_type):
                        errors.append(
                            f"Field '{field_name}' expected {expected_type}, "
                            f"got {type(actual_value)} with value {actual_value}"
                        )
        
        return errors

class EnvironmentVariableResolver:
    """Resolves environment variables in configuration."""
    
    @staticmethod
    def resolve_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively resolve environment variables in configuration."""
        resolved = {}
        
        for key, value in config_dict.items():
            if isinstance(value, dict):
                resolved[key] = EnvironmentVariableResolver.resolve_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Environment variable format: ${VAR_NAME:default_value}
                env_spec = value[2:-1]  # Remove ${ and }
                
                if ":" in env_spec:
                    env_var, default_value = env_spec.split(":", 1)
                    resolved[key] = os.getenv(env_var, default_value)
                else:
                    env_var = env_spec
                    env_value = os.getenv(env_var)
                    if env_value is None:
                        raise ValueError(f"Required environment variable '{env_var}' not found")
                    resolved[key] = env_value
            else:
                resolved[key] = value
        
        return resolved
    
    @staticmethod
    def get_env_with_type(env_var: str, default: Any = None, var_type: Type = str) -> Any:
        """Get environment variable with type conversion."""
        value = os.getenv(env_var)
        
        if value is None:
            if default is not None:
                return default
            else:
                raise ValueError(f"Required environment variable '{env_var}' not found")
        
        # Type conversion
        try:
            if var_type == bool:
                return value.lower() in ('true', '1', 'yes', 'on')
            elif var_type == int:
                return int(value)
            elif var_type == float:
                return float(value)
            elif var_type == list:
                return value.split(',')
            else:
                return var_type(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert environment variable '{env_var}' to {var_type}: {e}")

class ConfigurationManager:
    """
    Enhanced configuration manager with all improvements.
    """
    
    def __init__(self, 
                 config_dir: str = "config",
                 environment: ConfigEnvironment = ConfigEnvironment.DEVELOPMENT):
        self.config_dir = Path(config_dir)
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Configuration state
        self._config: Optional[MasterConfig] = None
        self._config_hash: Optional[str] = None
        self._config_watchers: List[Callable] = []
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Configuration manager initialized for {environment.value}")
    
    def load_config(self, 
                   config_file: Optional[str] = None,
                   format: ConfigFormat = ConfigFormat.YAML) -> MasterConfig:
        """
        Load configuration from file with environment variable resolution.
        
        Args:
            config_file: Path to configuration file
            format: Configuration file format
            
        Returns:
            Loaded and validated configuration
        """
        if config_file is None:
            config_file = f"config_{self.environment.value}.{format.value}"
        
        config_path = self.config_dir / config_file
        
        try:
            # Load base configuration
            if config_path.exists():
                config_dict = self._load_config_file(config_path, format)
            else:
                self.logger.warning(f"Config file {config_path} not found, using defaults")
                config_dict = {}
            
            # Resolve environment variables
            config_dict = EnvironmentVariableResolver.resolve_env_vars(config_dict)
            
            # Override with environment-specific settings
            config_dict = self._apply_environment_overrides(config_dict)
            
            # Create configuration object
            config = self._dict_to_config(config_dict)
            
            # Validate configuration
            config.validate()
            
            # Type validation
            validation_errors = ConfigValidator.validate_config_object(config)
            if validation_errors:
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            
            # Store configuration and hash
            self._config = config
            self._config_hash = self._calculate_config_hash(config)
            
            self.logger.info(f"Configuration loaded successfully from {config_path}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_config_file(self, config_path: Path, format: ConfigFormat) -> Dict[str, Any]:
        """Load configuration from file based on format."""
        with open(config_path, 'r') as f:
            if format == ConfigFormat.JSON:
                return json.load(f)
            elif format == ConfigFormat.YAML:
                return yaml.safe_load(f) or {}
            else:
                raise ValueError(f"Unsupported config format: {format}")
    
    def _apply_environment_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration overrides."""
        # Environment-specific overrides
        env_overrides = {
            ConfigEnvironment.DEVELOPMENT: {
                "training": {
                    "max_episodes": EnvironmentVariableResolver.get_env_with_type(
                        "DEV_MAX_EPISODES", 1000, int
                    ),
                    "checkpoint_frequency": 10
                },
                "logging": {
                    "level": "DEBUG",
                    "enable_tensorboard": True
                },
                "render": {
                    "enable_mock_renderer": True
                }
            },
            ConfigEnvironment.TESTING: {
                "training": {
                    "max_episodes": 100,
                    "max_steps_per_episode": 50
                },
                "logging": {
                    "level": "WARNING",
                    "save_logs": False
                },
                "render": {
                    "enable_mock_renderer": True,
                    "mock_render_probability": 1.0
                }
            },
            ConfigEnvironment.PRODUCTION: {
                "training": {
                    "use_mixed_precision": True,
                    "checkpoint_frequency": 1000
                },
                "logging": {
                    "level": "INFO",
                    "enable_wandb": True
                },
                "render": {
                    "enable_mock_renderer": False,
                    "use_headless_gpu": True
                }
            }
        }
        
        if self.environment in env_overrides:
            config_dict = self._deep_merge(config_dict, env_overrides[self.environment])
        
        return config_dict
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> MasterConfig:
        """Convert dictionary to configuration object."""
        # Extract subsystem configurations
        training_dict = config_dict.get("training", {})
        environment_dict = config_dict.get("environment", {})
        render_dict = config_dict.get("render", {})
        reward_dict = config_dict.get("reward", {})
        logging_dict = config_dict.get("logging", {})
        
        # Create configuration objects
        training_config = TrainingConfig(**training_dict)
        environment_config = EnvironmentConfig(**environment_dict)
        render_config = RenderConfig(**render_dict)
        reward_config = RewardConfig(**reward_dict)
        logging_config = LoggingConfig(**logging_dict)
        
        # Create master configuration
        master_config = MasterConfig(
            training=training_config,
            environment=environment_config,
            render=render_config,
            reward=reward_config,
            logging=logging_config,
            environment_name=self.environment.value
        )
        
        return master_config
    
    def save_config(self, 
                   config: MasterConfig,
                   config_file: Optional[str] = None,
                   format: ConfigFormat = ConfigFormat.YAML):
        """Save configuration to file."""
        if config_file is None:
            config_file = f"config_{self.environment.value}.{format.value}"
        
        config_path = self.config_dir / config_file
        
        try:
            # Convert to dictionary
            config_dict = asdict(config)
            
            # Save to file
            with open(config_path, 'w') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_dict, f, indent=2)
                elif format == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    raise ValueError(f"Unsupported config format: {format}")
            
            self.logger.info(f"Configuration saved to {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def get_config(self) -> MasterConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        if self._config is None:
            self._config = self.load_config()
        
        # Apply updates
        current_dict = asdict(self._config)
        updated_dict = self._deep_merge(current_dict, updates)
        
        # Create new configuration
        new_config = self._dict_to_config(updated_dict)
        new_config.validate()
        
        # Update stored configuration
        old_hash = self._config_hash
        self._config = new_config
        self._config_hash = self._calculate_config_hash(new_config)
        
        # Notify watchers if configuration changed
        if old_hash != self._config_hash:
            self._notify_config_watchers()
        
        self.logger.info("Configuration updated successfully")
    
    def _calculate_config_hash(self, config: MasterConfig) -> str:
        """Calculate hash of configuration for change detection."""
        config_str = json.dumps(asdict(config), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def watch_config_changes(self, callback: Callable[[MasterConfig], None]):
        """Register callback for configuration changes."""
        self._config_watchers.append(callback)
    
    def _notify_config_watchers(self):
        """Notify all registered watchers of configuration changes."""
        for callback in self._config_watchers:
            try:
                callback(self._config)
            except Exception as e:
                self.logger.warning(f"Config watcher callback failed: {e}")
    
    def create_default_configs(self):
        """Create default configuration files for all environments."""
        for env in ConfigEnvironment:
            self.environment = env
            default_config = MasterConfig()
            
            # Environment-specific adjustments
            if env == ConfigEnvironment.DEVELOPMENT:
                default_config.training.max_episodes = 1000
                default_config.logging.level = "DEBUG"
                default_config.render.enable_mock_renderer = True
            elif env == ConfigEnvironment.TESTING:
                default_config.training.max_episodes = 100
                default_config.training.max_steps_per_episode = 50
                default_config.logging.level = "WARNING"
                default_config.render.enable_mock_renderer = True
            elif env == ConfigEnvironment.PRODUCTION:
                default_config.training.max_episodes = 50000
                default_config.logging.level = "INFO"
                default_config.render.enable_mock_renderer = False
            
            self.save_config(default_config)
        
        self.logger.info("Default configuration files created for all environments")
    
    @contextmanager
    def temporary_config(self, updates: Dict[str, Any]):
        """Context manager for temporary configuration changes."""
        if self._config is None:
            self._config = self.load_config()
        
        # Store original configuration
        original_config = deepcopy(self._config)
        original_hash = self._config_hash
        
        try:
            # Apply temporary updates
            self.update_config(updates)
            yield self._config
        finally:
            # Restore original configuration
            self._config = original_config
            self._config_hash = original_hash
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration."""
        if self._config is None:
            self._config = self.load_config()
        
        return {
            "environment": self.environment.value,
            "config_version": self._config.config_version,
            "config_hash": self._config_hash,
            "created_at": self._config.created_at,
            "training": {
                "max_episodes": self._config.training.max_episodes,
                "batch_size": self._config.training.batch_size,
                "learning_rate": self._config.training.learning_rate
            },
            "environment_settings": {
                "grid_size": self._config.environment.grid_size,
                "max_towers": self._config.environment.max_towers,
                "max_enemies": self._config.environment.max_enemies
            },
            "render": {
                "resolution": f"{self._config.render.width}x{self._config.render.height}",
                "use_headless_gpu": self._config.render.use_headless_gpu,
                "enable_mock_renderer": self._config.render.enable_mock_renderer
            }
        }

# Global configuration manager instance
_config_manager: Optional[ConfigurationManager] = None

def get_config_manager(
    config_dir: str = "config",
    environment: Optional[ConfigEnvironment] = None
) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        if environment is None:
            # Determine environment from environment variable
            env_name = os.getenv("RL_ENVIRONMENT", "development").lower()
            try:
                environment = ConfigEnvironment(env_name)
            except ValueError:
                environment = ConfigEnvironment.DEVELOPMENT
                warnings.warn(f"Unknown environment '{env_name}', using development")
        
        _config_manager = ConfigurationManager(config_dir, environment)
    
    return _config_manager

def get_config() -> MasterConfig:
    """Get current configuration."""
    return get_config_manager().get_config()

# Convenience functions for accessing specific configurations
def get_training_config() -> TrainingConfig:
    """Get training configuration."""
    return get_config().training

def get_environment_config() -> EnvironmentConfig:
    """Get environment configuration."""
    return get_config().environment

def get_render_config() -> RenderConfig:
    """Get render configuration."""
    return get_config().render

def get_reward_config() -> RewardConfig:
    """Get reward configuration."""
    return get_config().reward

def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_config().logging

