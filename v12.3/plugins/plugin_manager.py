"""
Plugin System Manager for RL-LLM

This module provides a comprehensive plugin system for extending the RL-LLM
functionality with custom components, environments, and algorithms.
"""

import importlib
import inspect
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional, Type, Callable, Union
import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
import threading
import time

logger = logging.getLogger(__name__)


class PluginType(Enum):
    """Plugin types supported by the system."""
    ENVIRONMENT = "environment"
    AGENT = "agent"
    REWARD_FUNCTION = "reward_function"
    VISUALIZATION = "visualization"
    DATA_PROCESSOR = "data_processor"
    CALLBACK = "callback"
    UTILITY = "utility"


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    entry_point: str = ""
    enabled: bool = True
    priority: int = 0


class PluginInterface(ABC):
    """Base interface for all plugins."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin.
        
        Args:
            config: Plugin configuration
        """
        self.config = config
        self.metadata = self.get_metadata()
        self.initialized = False
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Get plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize plugin.
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Cleanup plugin resources."""
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate plugin configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
        """
        # Default implementation - override in subclasses
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin status information."""
        return {
            'name': self.metadata.name,
            'initialized': self.initialized,
            'enabled': self.metadata.enabled,
            'type': self.metadata.plugin_type.value
        }


class EnvironmentPlugin(PluginInterface):
    """Base class for environment plugins."""
    
    @abstractmethod
    def create_environment(self, config: Dict[str, Any]):
        """Create environment instance."""
        pass


class AgentPlugin(PluginInterface):
    """Base class for agent plugins."""
    
    @abstractmethod
    def create_agent(self, config: Dict[str, Any]):
        """Create agent instance."""
        pass


class RewardFunctionPlugin(PluginInterface):
    """Base class for reward function plugins."""
    
    @abstractmethod
    def compute_reward(self, state: Any, action: Any, next_state: Any, info: Dict[str, Any]) -> float:
        """Compute reward for transition."""
        pass


class VisualizationPlugin(PluginInterface):
    """Base class for visualization plugins."""
    
    @abstractmethod
    def create_visualizer(self, config: Dict[str, Any]):
        """Create visualizer instance."""
        pass


class CallbackPlugin(PluginInterface):
    """Base class for callback plugins."""
    
    @abstractmethod
    def on_episode_start(self, episode: int, **kwargs):
        """Called at episode start."""
        pass
    
    @abstractmethod
    def on_episode_end(self, episode: int, result: Dict[str, Any], **kwargs):
        """Called at episode end."""
        pass
    
    @abstractmethod
    def on_step(self, step: int, **kwargs):
        """Called at each step."""
        pass


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self):
        """Initialize plugin registry."""
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_types: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def register_plugin(self, plugin: PluginInterface) -> bool:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if registration successful
        """
        with self._lock:
            try:
                plugin_name = plugin.metadata.name
                
                if plugin_name in self.plugins:
                    logger.warning(f"Plugin {plugin_name} already registered, replacing")
                
                # Validate plugin
                if not self._validate_plugin(plugin):
                    return False
                
                # Register plugin
                self.plugins[plugin_name] = plugin
                self.plugin_types[plugin.metadata.plugin_type].append(plugin_name)
                
                logger.info(f"Registered plugin: {plugin_name} ({plugin.metadata.plugin_type.value})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to register plugin: {e}")
                return False
    
    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            plugin_name: Name of plugin to unregister
            
        Returns:
            True if unregistration successful
        """
        with self._lock:
            if plugin_name not in self.plugins:
                logger.warning(f"Plugin {plugin_name} not found")
                return False
            
            try:
                plugin = self.plugins[plugin_name]
                
                # Cleanup plugin
                plugin.cleanup()
                
                # Remove from registry
                del self.plugins[plugin_name]
                self.plugin_types[plugin.metadata.plugin_type].remove(plugin_name)
                
                if plugin_name in self.plugin_configs:
                    del self.plugin_configs[plugin_name]
                
                logger.info(f"Unregistered plugin: {plugin_name}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
                return False
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get plugin by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of specified type."""
        plugin_names = self.plugin_types.get(plugin_type, [])
        return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [plugin.get_status() for plugin in self.plugins.values()]
    
    def _validate_plugin(self, plugin: PluginInterface) -> bool:
        """Validate plugin before registration."""
        try:
            # Check required methods
            required_methods = ['get_metadata', 'initialize', 'cleanup']
            for method in required_methods:
                if not hasattr(plugin, method):
                    logger.error(f"Plugin missing required method: {method}")
                    return False
            
            # Validate metadata
            metadata = plugin.metadata
            if not metadata.name or not metadata.version:
                logger.error("Plugin metadata missing name or version")
                return False
            
            # Check dependencies
            for dep in metadata.dependencies:
                if not self._check_dependency(dep):
                    logger.error(f"Plugin dependency not met: {dep}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Plugin validation failed: {e}")
            return False
    
    def _check_dependency(self, dependency: str) -> bool:
        """Check if dependency is available."""
        try:
            # Try to import the dependency
            importlib.import_module(dependency)
            return True
        except ImportError:
            # Check if it's a registered plugin
            return dependency in self.plugins


class PluginManager:
    """Main plugin manager for the RL-LLM system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize plugin manager.
        
        Args:
            config: Plugin manager configuration
        """
        self.config = config
        self.registry = PluginRegistry()
        self.plugin_directories = config.get('plugin_directories', ['./plugins'])
        self.auto_load = config.get('auto_load', True)
        self.enabled_plugins = config.get('enabled_plugins', [])
        
        # Plugin discovery cache
        self._discovery_cache = {}
        self._last_discovery_time = 0
        self._discovery_interval = config.get('discovery_interval', 300)  # 5 minutes
        
        logger.info("Initialized PluginManager")
        
        if self.auto_load:
            self.discover_and_load_plugins()
    
    def discover_plugins(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Discover available plugins.
        
        Args:
            force_refresh: Force refresh of discovery cache
            
        Returns:
            List of discovered plugin information
        """
        current_time = time.time()
        
        if (not force_refresh and 
            current_time - self._last_discovery_time < self._discovery_interval and 
            self._discovery_cache):
            return self._discovery_cache
        
        discovered_plugins = []
        
        for plugin_dir in self.plugin_directories:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                continue
            
            # Look for plugin manifests
            for manifest_file in plugin_path.rglob('plugin.json'):
                try:
                    plugin_info = self._load_plugin_manifest(manifest_file)
                    if plugin_info:
                        plugin_info['manifest_path'] = str(manifest_file)
                        plugin_info['plugin_dir'] = str(manifest_file.parent)
                        discovered_plugins.append(plugin_info)
                        
                except Exception as e:
                    logger.error(f"Error loading plugin manifest {manifest_file}: {e}")
        
        self._discovery_cache = discovered_plugins
        self._last_discovery_time = current_time
        
        logger.info(f"Discovered {len(discovered_plugins)} plugins")
        return discovered_plugins
    
    def _load_plugin_manifest(self, manifest_path: Path) -> Optional[Dict[str, Any]]:
        """Load plugin manifest file."""
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Validate required fields
            required_fields = ['name', 'version', 'entry_point', 'plugin_type']
            for field in required_fields:
                if field not in manifest:
                    logger.error(f"Plugin manifest missing required field: {field}")
                    return None
            
            return manifest
            
        except Exception as e:
            logger.error(f"Failed to load plugin manifest {manifest_path}: {e}")
            return None
    
    def load_plugin(self, plugin_info: Dict[str, Any]) -> bool:
        """
        Load a plugin from plugin information.
        
        Args:
            plugin_info: Plugin information dictionary
            
        Returns:
            True if loading successful
        """
        try:
            plugin_name = plugin_info['name']
            plugin_dir = Path(plugin_info['plugin_dir'])
            entry_point = plugin_info['entry_point']
            
            # Add plugin directory to Python path
            if str(plugin_dir) not in sys.path:
                sys.path.insert(0, str(plugin_dir))
            
            # Import plugin module
            module_name, class_name = entry_point.rsplit('.', 1)
            module = importlib.import_module(module_name)
            plugin_class = getattr(module, class_name)
            
            # Create plugin instance
            plugin_config = self.config.get('plugin_configs', {}).get(plugin_name, {})
            plugin_instance = plugin_class(plugin_config)
            
            # Initialize plugin
            if plugin_instance.initialize():
                # Register plugin
                if self.registry.register_plugin(plugin_instance):
                    logger.info(f"Successfully loaded plugin: {plugin_name}")
                    return True
                else:
                    logger.error(f"Failed to register plugin: {plugin_name}")
                    return False
            else:
                logger.error(f"Failed to initialize plugin: {plugin_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_info.get('name', 'unknown')}: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin.
        
        Args:
            plugin_name: Name of plugin to unload
            
        Returns:
            True if unloading successful
        """
        return self.registry.unregister_plugin(plugin_name)
    
    def discover_and_load_plugins(self):
        """Discover and load all available plugins."""
        discovered_plugins = self.discover_plugins()
        
        loaded_count = 0
        for plugin_info in discovered_plugins:
            plugin_name = plugin_info['name']
            
            # Check if plugin should be loaded
            if self.enabled_plugins and plugin_name not in self.enabled_plugins:
                continue
            
            # Skip if already loaded
            if self.registry.get_plugin(plugin_name):
                continue
            
            if self.load_plugin(plugin_info):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} plugins")
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """
        Reload a plugin.
        
        Args:
            plugin_name: Name of plugin to reload
            
        Returns:
            True if reloading successful
        """
        # Find plugin info
        discovered_plugins = self.discover_plugins(force_refresh=True)
        plugin_info = None
        
        for info in discovered_plugins:
            if info['name'] == plugin_name:
                plugin_info = info
                break
        
        if not plugin_info:
            logger.error(f"Plugin {plugin_name} not found for reload")
            return False
        
        # Unload existing plugin
        self.unload_plugin(plugin_name)
        
        # Load plugin again
        return self.load_plugin(plugin_info)
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get plugin by name."""
        return self.registry.get_plugin(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get plugins by type."""
        return self.registry.get_plugins_by_type(plugin_type)
    
    def create_environment(self, env_name: str, config: Dict[str, Any]):
        """Create environment using environment plugin."""
        env_plugins = self.get_plugins_by_type(PluginType.ENVIRONMENT)
        
        for plugin in env_plugins:
            if plugin.metadata.name == env_name:
                return plugin.create_environment(config)
        
        raise ValueError(f"Environment plugin '{env_name}' not found")
    
    def create_agent(self, agent_name: str, config: Dict[str, Any]):
        """Create agent using agent plugin."""
        agent_plugins = self.get_plugins_by_type(PluginType.AGENT)
        
        for plugin in agent_plugins:
            if plugin.metadata.name == agent_name:
                return plugin.create_agent(config)
        
        raise ValueError(f"Agent plugin '{agent_name}' not found")
    
    def get_callbacks(self) -> List[CallbackPlugin]:
        """Get all callback plugins."""
        return [plugin for plugin in self.get_plugins_by_type(PluginType.CALLBACK) 
                if isinstance(plugin, CallbackPlugin)]
    
    def get_status(self) -> Dict[str, Any]:
        """Get plugin manager status."""
        return {
            'total_plugins': len(self.registry.plugins),
            'plugins_by_type': {
                plugin_type.value: len(plugins) 
                for plugin_type, plugins in self.registry.plugin_types.items()
            },
            'plugin_list': self.registry.list_plugins(),
            'plugin_directories': self.plugin_directories
        }
    
    def cleanup(self):
        """Cleanup all plugins."""
        logger.info("Cleaning up plugin manager")
        
        for plugin_name in list(self.registry.plugins.keys()):
            self.unload_plugin(plugin_name)


# Global plugin manager instance
_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> Optional[PluginManager]:
    """Get global plugin manager instance."""
    return _plugin_manager


def initialize_plugin_manager(config: Dict[str, Any]) -> PluginManager:
    """Initialize global plugin manager."""
    global _plugin_manager
    _plugin_manager = PluginManager(config)
    return _plugin_manager


def create_example_plugin_manifest() -> Dict[str, Any]:
    """Create example plugin manifest for reference."""
    return {
        "name": "example_plugin",
        "version": "1.0.0",
        "description": "Example plugin for demonstration",
        "author": "RL-LLM Team",
        "plugin_type": "environment",
        "entry_point": "example_plugin.ExampleEnvironmentPlugin",
        "dependencies": ["gym", "numpy"],
        "config_schema": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "default": "default_value"},
                "param2": {"type": "integer", "minimum": 1, "default": 10}
            }
        },
        "enabled": True,
        "priority": 0
    }

