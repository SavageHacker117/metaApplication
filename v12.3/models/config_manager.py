"""
Configuration Manager for RL-LLM System

This module provides centralized configuration management with validation,
environment variable support, and hierarchical configuration merging.
"""

import os
import json
import yaml
import argparse
from typing import Dict, Any, Optional, Union, List, Type
from pathlib import Path
import logging
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import copy
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """Configuration schema definition."""
    name: str
    type: Type
    default: Any = None
    required: bool = False
    description: str = ""
    choices: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None


class ConfigValidator:
    """Configuration validator with schema support."""
    
    def __init__(self):
        """Initialize configuration validator."""
        self.schemas: Dict[str, ConfigSchema] = {}
        self.validation_errors: List[str] = []
    
    def register_schema(self, path: str, schema: ConfigSchema):
        """
        Register configuration schema.
        
        Args:
            path: Configuration path (e.g., 'training.learning_rate')
            schema: Schema definition
        """
        self.schemas[path] = schema
    
    def validate(self, config: Dict[str, Any], path_prefix: str = "") -> bool:
        """
        Validate configuration against registered schemas.
        
        Args:
            config: Configuration dictionary
            path_prefix: Current path prefix
            
        Returns:
            True if validation passes
        """
        self.validation_errors.clear()
        
        # Validate current level
        for key, value in config.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            
            if current_path in self.schemas:
                if not self._validate_value(current_path, value, self.schemas[current_path]):
                    return False
            
            # Recursively validate nested dictionaries
            if isinstance(value, dict):
                if not self.validate(value, current_path):
                    return False
        
        # Check for required fields
        for schema_path, schema in self.schemas.items():
            if schema.required and not self._path_exists(config, schema_path):
                self.validation_errors.append(f"Required field '{schema_path}' is missing")
        
        return len(self.validation_errors) == 0
    
    def _validate_value(self, path: str, value: Any, schema: ConfigSchema) -> bool:
        """Validate a single value against schema."""
        try:
            # Type validation
            if not isinstance(value, schema.type):
                # Try type conversion
                try:
                    value = schema.type(value)
                except (ValueError, TypeError):
                    self.validation_errors.append(
                        f"Field '{path}': expected {schema.type.__name__}, got {type(value).__name__}"
                    )
                    return False
            
            # Choices validation
            if schema.choices and value not in schema.choices:
                self.validation_errors.append(
                    f"Field '{path}': value '{value}' not in allowed choices {schema.choices}"
                )
                return False
            
            # Range validation
            if schema.min_value is not None and value < schema.min_value:
                self.validation_errors.append(
                    f"Field '{path}': value {value} below minimum {schema.min_value}"
                )
                return False
            
            if schema.max_value is not None and value > schema.max_value:
                self.validation_errors.append(
                    f"Field '{path}': value {value} above maximum {schema.max_value}"
                )
                return False
            
            # Pattern validation
            if schema.pattern and isinstance(value, str):
                if not re.match(schema.pattern, value):
                    self.validation_errors.append(
                        f"Field '{path}': value '{value}' does not match pattern '{schema.pattern}'"
                    )
                    return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Field '{path}': validation error - {str(e)}")
            return False
    
    def _path_exists(self, config: Dict[str, Any], path: str) -> bool:
        """Check if configuration path exists."""
        parts = path.split('.')
        current = config
        
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        
        return True
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self.validation_errors.copy()


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir or Path('./configs')
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.config: Dict[str, Any] = {}
        self.validator = ConfigValidator()
        self.env_prefix = "RLLLM_"
        
        # Register default schemas
        self._register_default_schemas()
        
        logger.info(f"Initialized ConfigManager with config_dir: {self.config_dir}")
    
    def _register_default_schemas(self):
        """Register default configuration schemas."""
        # Training configuration
        self.validator.register_schema(
            'training.learning_rate',
            ConfigSchema('learning_rate', float, 0.001, description='Learning rate for training')
        )
        
        self.validator.register_schema(
            'training.batch_size',
            ConfigSchema('batch_size', int, 32, min_value=1, description='Batch size for training')
        )
        
        self.validator.register_schema(
            'training.max_episodes',
            ConfigSchema('max_episodes', int, 1000, min_value=1, description='Maximum training episodes')
        )
        
        # Environment configuration
        self.validator.register_schema(
            'environment.name',
            ConfigSchema('name', str, required=True, description='Environment name')
        )
        
        # Agent configuration
        self.validator.register_schema(
            'agent.type',
            ConfigSchema('type', str, required=True, choices=['dqn', 'ppo', 'sac'], description='Agent type')
        )
        
        # Logging configuration
        self.validator.register_schema(
            'logging.level',
            ConfigSchema('level', str, 'INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], description='Logging level')
        )
    
    def load_config(self, config_path: Union[str, Path], merge: bool = True) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            merge: Whether to merge with existing configuration
            
        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    loaded_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Process environment variables
            loaded_config = self._process_env_vars(loaded_config)
            
            if merge:
                self.config = self._merge_configs(self.config, loaded_config)
            else:
                self.config = loaded_config
            
            logger.info(f"Loaded configuration from {config_path}")
            return loaded_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def save_config(self, config_path: Union[str, Path], config: Optional[Dict[str, Any]] = None):
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration
            config: Configuration to save (defaults to current config)
        """
        config_path = Path(config_path)
        config = config or self.config
        
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
                elif config_path.suffix.lower() == '.json':
                    json.dump(config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._get_nested_value(self.config, key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        self._set_nested_value(self.config, key, value)
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with dictionary.
        
        Args:
            updates: Dictionary of updates
        """
        self.config = self._merge_configs(self.config, updates)
    
    def validate(self) -> bool:
        """
        Validate current configuration.
        
        Returns:
            True if validation passes
        """
        is_valid = self.validator.validate(self.config)
        
        if not is_valid:
            errors = self.validator.get_validation_errors()
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
        
        return is_valid
    
    def load_from_args(self, args: argparse.Namespace):
        """
        Load configuration from command line arguments.
        
        Args:
            args: Parsed command line arguments
        """
        args_dict = vars(args)
        
        # Convert flat args to nested config
        nested_config = {}
        for key, value in args_dict.items():
            if value is not None:  # Only set non-None values
                self._set_nested_value(nested_config, key, value)
        
        # Merge with existing config
        self.config = self._merge_configs(self.config, nested_config)
        
        logger.info("Loaded configuration from command line arguments")
    
    def create_config_template(self, output_path: Union[str, Path]):
        """
        Create configuration template with all registered schemas.
        
        Args:
            output_path: Path to save template
        """
        template = {}
        
        for path, schema in self.validator.schemas.items():
            value = schema.default
            if schema.description:
                # Add comment (for YAML)
                value = {
                    '_comment': schema.description,
                    '_value': value
                }
            
            self._set_nested_value(template, path, value)
        
        self.save_config(output_path, template)
        logger.info(f"Created configuration template at {output_path}")
    
    def _process_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment variable substitutions."""
        def process_value(value):
            if isinstance(value, str):
                # Replace ${VAR} or ${VAR:default} patterns
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replace_env(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ''
                    
                    # Try with prefix first, then without
                    env_value = os.getenv(f"{self.env_prefix}{var_name}")
                    if env_value is None:
                        env_value = os.getenv(var_name, default_value)
                    
                    return env_value
                
                return re.sub(pattern, replace_env, value)
            
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            
            return value
        
        return process_value(config)
    
    def _merge_configs(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two configuration dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    def _get_nested_value(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get nested value using dot notation."""
        parts = key.split('.')
        current = config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """Set nested value using dot notation."""
        parts = key.split('.')
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration."""
        return copy.deepcopy(self.config)
    
    def print_config(self, show_defaults: bool = False):
        """Print current configuration."""
        print("Current Configuration:")
        print("=" * 50)
        
        def print_nested(config, indent=0):
            for key, value in config.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_nested(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")
        
        print_nested(self.config)
        
        if show_defaults:
            print("\nDefault Schema Values:")
            print("=" * 50)
            for path, schema in self.validator.schemas.items():
                print(f"{path}: {schema.default} ({schema.type.__name__})")
                if schema.description:
                    print(f"  Description: {schema.description}")
    
    def export_env_vars(self, output_path: Union[str, Path]):
        """
        Export configuration as environment variables.
        
        Args:
            output_path: Path to save environment variable file
        """
        env_vars = []
        
        def flatten_config(config, prefix=""):
            for key, value in config.items():
                env_key = f"{self.env_prefix}{prefix}{key}".upper()
                
                if isinstance(value, dict):
                    flatten_config(value, f"{prefix}{key}_")
                else:
                    env_vars.append(f"export {env_key}={value}")
        
        flatten_config(self.config)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(env_vars))
        
        logger.info(f"Exported environment variables to {output_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for configuration management."""
    parser = argparse.ArgumentParser(description='RL-LLM Configuration')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Configuration file path')
    
    # Training parameters
    parser.add_argument('--training.learning_rate', type=float, help='Learning rate')
    parser.add_argument('--training.batch_size', type=int, help='Batch size')
    parser.add_argument('--training.max_episodes', type=int, help='Maximum episodes')
    
    # Environment parameters
    parser.add_argument('--environment.name', type=str, help='Environment name')
    
    # Agent parameters
    parser.add_argument('--agent.type', type=str, choices=['dqn', 'ppo', 'sac'], help='Agent type')
    
    # Logging parameters
    parser.add_argument('--logging.level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, help='Output directory')
    
    return parser


def load_config_from_cli() -> ConfigManager:
    """Load configuration from command line interface."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    config_manager = ConfigManager()
    
    # Load from file if specified
    if args.config:
        config_manager.load_config(args.config)
    
    # Override with command line arguments
    config_manager.load_from_args(args)
    
    # Validate configuration
    if not config_manager.validate():
        raise ValueError("Configuration validation failed")
    
    return config_manager


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> Optional[ConfigManager]:
    """Get global configuration manager instance."""
    return _config_manager


def initialize_config_manager(config_dir: Optional[Path] = None) -> ConfigManager:
    """Initialize global configuration manager."""
    global _config_manager
    _config_manager = ConfigManager(config_dir)
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value from global manager."""
    if _config_manager is None:
        raise RuntimeError("Configuration manager not initialized")
    return _config_manager.get(key, default)


def set_config(key: str, value: Any):
    """Set configuration value in global manager."""
    if _config_manager is None:
        raise RuntimeError("Configuration manager not initialized")
    _config_manager.set(key, value)

