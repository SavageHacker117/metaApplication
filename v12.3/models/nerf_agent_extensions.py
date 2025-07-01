"""
Agent Action Space Extensions for NeRF Integration

This module extends the RL agent's action space to include NeRF asset
selection and application as part of the procedural generation process.

Key Features:
- NeRF action space integration
- Context-aware NeRF selection
- Performance-optimized NeRF actions
- Quality-based NeRF rewards
- Dynamic NeRF asset management
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time

# Import enhanced components
try:
    from nerf_integration_module import NeRFAssetType, NeRFQuality, NeRFRenderRequest
    from nerf_asset_management import EnhancedNeRFAssetManager, NeRFRenderer
    from transformer_agent_enhanced import EnhancedTransformerRLAgent
    from reward_system_enhanced import EnhancedRewardSystem
    from enhanced_config_manager import MasterConfig
except ImportError as e:
    logging.warning(f"Some enhanced components not available: {e}")
    # Define fallback enums if imports fail
    from enum import Enum
    
    class NeRFAssetType(Enum):
        MESH = "mesh"
        POINT_CLOUD = "point_cloud"
        TEXTURE_ATLAS = "texture_atlas"
        VOLUMETRIC = "volumetric"
        ENVIRONMENT = "environment"
    
    class NeRFQuality(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        ULTRA = "ultra"
    
    # Fallback classes
    class EnhancedTransformerRLAgent:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_available_actions(self, game_state):
            return []
        
        def execute_action(self, action, game_state):
            return True, {}
        
        def calculate_reward(self, game_state, action, next_state):
            return 0.0
    
    class EnhancedNeRFAssetManager:
        def __init__(self, *args, **kwargs):
            pass
    
    class NeRFRenderer:
        def __init__(self, *args, **kwargs):
            pass
    
    class MasterConfig:
        pass

class NeRFActionType(Enum):
    """Types of NeRF actions available to the agent."""
    APPLY_SKIN = "apply_skin"           # Apply NeRF skin to object
    CHANGE_QUALITY = "change_quality"   # Change NeRF quality level
    REMOVE_SKIN = "remove_skin"         # Remove NeRF skin from object
    OPTIMIZE_PERFORMANCE = "optimize"   # Optimize NeRF performance
    SELECT_ENVIRONMENT = "environment"  # Select NeRF environment

@dataclass
class NeRFAction:
    """Represents a NeRF action that can be taken by the agent."""
    action_type: NeRFActionType
    target_object: str
    asset_id: Optional[str] = None
    quality_level: Optional[NeRFQuality] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    expected_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'action_type': self.action_type.value,
            'target_object': self.target_object,
            'asset_id': self.asset_id,
            'quality_level': self.quality_level.value if self.quality_level else None,
            'parameters': self.parameters,
            'priority': self.priority,
            'expected_reward': self.expected_reward
        }

class NeRFActionSpace:
    """Defines the NeRF action space for the RL agent."""
    
    def __init__(self, 
                 asset_manager: EnhancedNeRFAssetManager,
                 max_concurrent_nerfs: int = 10,
                 enable_quality_actions: bool = True,
                 enable_environment_actions: bool = True):
        self.asset_manager = asset_manager
        self.max_concurrent_nerfs = max_concurrent_nerfs
        self.enable_quality_actions = enable_quality_actions
        self.enable_environment_actions = enable_environment_actions
        
        self.logger = logging.getLogger(__name__)
        
        # Track active NeRF applications
        self.active_nerfs = {}  # target_object -> NeRFAction
        
        # Cache available actions for performance
        self._action_cache = {}
        self._cache_timestamp = 0
        self._cache_ttl = 60  # Cache for 60 seconds
    
    def get_available_actions(self, game_state: Dict[str, Any]) -> List[NeRFAction]:
        """Get all available NeRF actions for the current game state."""
        current_time = time.time()
        
        # Check cache validity
        if (current_time - self._cache_timestamp) < self._cache_ttl:
            cached_actions = self._action_cache.get('available_actions')
            if cached_actions:
                return self._filter_actions_by_state(cached_actions, game_state)
        
        # Generate available actions
        available_actions = []
        
        # Get available objects that can have NeRF skins applied
        available_objects = self._get_available_objects(game_state)
        
        for obj_id in available_objects:
            # Apply skin actions
            skin_actions = self._generate_skin_actions(obj_id, game_state)
            available_actions.extend(skin_actions)
            
            # Quality change actions (if object already has NeRF)
            if self.enable_quality_actions and obj_id in self.active_nerfs:
                quality_actions = self._generate_quality_actions(obj_id, game_state)
                available_actions.extend(quality_actions)
            
            # Remove skin actions (if object has NeRF)
            if obj_id in self.active_nerfs:
                remove_action = NeRFAction(
                    action_type=NeRFActionType.REMOVE_SKIN,
                    target_object=obj_id,
                    expected_reward=self._estimate_remove_reward(obj_id, game_state)
                )
                available_actions.append(remove_action)
        
        # Environment actions
        if self.enable_environment_actions:
            env_actions = self._generate_environment_actions(game_state)
            available_actions.extend(env_actions)
        
        # Performance optimization actions
        if len(self.active_nerfs) > 0:
            optimize_action = NeRFAction(
                action_type=NeRFActionType.OPTIMIZE_PERFORMANCE,
                target_object="global",
                expected_reward=self._estimate_optimization_reward(game_state)
            )
            available_actions.append(optimize_action)
        
        # Cache the results
        self._action_cache['available_actions'] = available_actions
        self._cache_timestamp = current_time
        
        return available_actions
    
    def _get_available_objects(self, game_state: Dict[str, Any]) -> List[str]:
        """Get list of objects that can have NeRF skins applied."""
        available_objects = []
        
        # Extract objects from game state
        if 'objects' in game_state:
            for obj_id, obj_data in game_state['objects'].items():
                # Check if object is eligible for NeRF application
                if self._is_object_eligible(obj_id, obj_data):
                    available_objects.append(obj_id)
        
        # Add common game objects if not in state
        common_objects = ['walls', 'towers', 'ground', 'enemies', 'projectiles']
        for obj_type in common_objects:
            if obj_type not in available_objects:
                available_objects.append(obj_type)
        
        return available_objects
    
    def _is_object_eligible(self, obj_id: str, obj_data: Dict[str, Any]) -> bool:
        """Check if an object is eligible for NeRF application."""
        # Check if object is visible
        if not obj_data.get('visible', True):
            return False
        
        # Check if object is not already at max NeRF capacity
        if len(self.active_nerfs) >= self.max_concurrent_nerfs:
            return obj_id in self.active_nerfs  # Only allow modifications to existing
        
        # Check object type compatibility
        obj_type = obj_data.get('type', 'unknown')
        if obj_type in ['ui', 'debug', 'temporary']:
            return False
        
        return True
    
    def _generate_skin_actions(self, obj_id: str, game_state: Dict[str, Any]) -> List[NeRFAction]:
        """Generate NeRF skin application actions for an object."""
        actions = []
        
        # Get context for asset selection
        context = self._build_context(obj_id, game_state)
        
        # Get compatible assets
        compatible_assets = self.asset_manager.get_recommended_assets(
            target_object=obj_id,
            context=context,
            max_results=5  # Limit to top 5 to avoid action space explosion
        )
        
        for asset_id, confidence_score in compatible_assets:
            asset_metadata = self.asset_manager.get_asset_metadata(asset_id)
            if not asset_metadata:
                continue
            
            # Create action for each quality level
            for quality in [NeRFQuality.LOW, NeRFQuality.MEDIUM, NeRFQuality.HIGH]:
                action = NeRFAction(
                    action_type=NeRFActionType.APPLY_SKIN,
                    target_object=obj_id,
                    asset_id=asset_id,
                    quality_level=quality,
                    expected_reward=self._estimate_skin_reward(
                        obj_id, asset_metadata, quality, confidence_score, game_state
                    )
                )
                actions.append(action)
        
        return actions
    
    def _generate_quality_actions(self, obj_id: str, game_state: Dict[str, Any]) -> List[NeRFAction]:
        """Generate quality change actions for objects with existing NeRF skins."""
        actions = []
        
        current_action = self.active_nerfs.get(obj_id)
        if not current_action:
            return actions
        
        current_quality = current_action.quality_level
        
        # Generate actions for different quality levels
        for quality in [NeRFQuality.LOW, NeRFQuality.MEDIUM, NeRFQuality.HIGH, NeRFQuality.ULTRA]:
            if quality != current_quality:
                action = NeRFAction(
                    action_type=NeRFActionType.CHANGE_QUALITY,
                    target_object=obj_id,
                    asset_id=current_action.asset_id,
                    quality_level=quality,
                    expected_reward=self._estimate_quality_change_reward(
                        obj_id, current_quality, quality, game_state
                    )
                )
                actions.append(action)
        
        return actions
    
    def _generate_environment_actions(self, game_state: Dict[str, Any]) -> List[NeRFAction]:
        """Generate environment NeRF actions."""
        actions = []
        
        # Get environment assets
        env_assets = self.asset_manager.search_assets(
            asset_type=NeRFAssetType.ENVIRONMENT,
            min_quality_score=0.5
        )
        
        for asset in env_assets[:3]:  # Limit to top 3 environment options
            action = NeRFAction(
                action_type=NeRFActionType.SELECT_ENVIRONMENT,
                target_object="environment",
                asset_id=asset.asset_id,
                quality_level=NeRFQuality.MEDIUM,
                expected_reward=self._estimate_environment_reward(asset, game_state)
            )
            actions.append(action)
        
        return actions
    
    def _build_context(self, obj_id: str, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Build context for NeRF asset selection."""
        context = {
            'target_object': obj_id,
            'game_phase': game_state.get('phase', 'unknown'),
            'performance_budget': game_state.get('performance_budget', 1.0),
            'visual_importance': self._calculate_visual_importance(obj_id, game_state),
            'environment_type': game_state.get('environment_type', 'generic')
        }
        
        # Add performance constraints
        if 'performance_metrics' in game_state:
            metrics = game_state['performance_metrics']
            context['current_fps'] = metrics.get('fps', 60)
            context['memory_usage'] = metrics.get('memory_usage', 0.5)
            context['gpu_utilization'] = metrics.get('gpu_utilization', 0.5)
        
        return context
    
    def _calculate_visual_importance(self, obj_id: str, game_state: Dict[str, Any]) -> float:
        """Calculate the visual importance of an object."""
        # Base importance
        importance = 0.5
        
        # Object type importance
        type_importance = {
            'towers': 0.8,
            'walls': 0.7,
            'enemies': 0.9,
            'projectiles': 0.6,
            'ground': 0.4,
            'background': 0.3
        }
        
        obj_type = obj_id.split('_')[0] if '_' in obj_id else obj_id
        importance = type_importance.get(obj_type, 0.5)
        
        # Screen position importance (center is more important)
        if 'objects' in game_state and obj_id in game_state['objects']:
            obj_data = game_state['objects'][obj_id]
            if 'position' in obj_data:
                pos = obj_data['position']
                # Assume screen center is (0.5, 0.5)
                distance_from_center = np.sqrt((pos[0] - 0.5)**2 + (pos[1] - 0.5)**2)
                center_importance = 1.0 - min(1.0, distance_from_center * 2)
                importance = (importance + center_importance) / 2
        
        return importance
    
    def _estimate_skin_reward(self, obj_id: str, asset_metadata, quality: NeRFQuality, 
                             confidence: float, game_state: Dict[str, Any]) -> float:
        """Estimate reward for applying a NeRF skin."""
        # Base reward from asset quality and confidence
        base_reward = asset_metadata.visual_quality_score * confidence
        
        # Quality modifier
        quality_multiplier = {
            NeRFQuality.LOW: 0.7,
            NeRFQuality.MEDIUM: 1.0,
            NeRFQuality.HIGH: 1.3,
            NeRFQuality.ULTRA: 1.5
        }
        base_reward *= quality_multiplier.get(quality, 1.0)
        
        # Performance penalty
        performance_penalty = self._calculate_performance_penalty(quality, game_state)
        
        # Visual importance bonus
        visual_importance = self._calculate_visual_importance(obj_id, game_state)
        importance_bonus = visual_importance * 0.2
        
        # Diversity bonus (encourage using different assets)
        diversity_bonus = self._calculate_diversity_bonus(asset_metadata.asset_id)
        
        total_reward = base_reward - performance_penalty + importance_bonus + diversity_bonus
        
        return max(0.0, min(1.0, total_reward))
    
    def _estimate_quality_change_reward(self, obj_id: str, current_quality: NeRFQuality,
                                       new_quality: NeRFQuality, game_state: Dict[str, Any]) -> float:
        """Estimate reward for changing NeRF quality."""
        # Performance improvement/degradation
        current_cost = self._get_quality_performance_cost(current_quality)
        new_cost = self._get_quality_performance_cost(new_quality)
        performance_delta = current_cost - new_cost
        
        # Visual quality improvement/degradation
        quality_values = {
            NeRFQuality.LOW: 0.3,
            NeRFQuality.MEDIUM: 0.6,
            NeRFQuality.HIGH: 0.8,
            NeRFQuality.ULTRA: 1.0
        }
        visual_delta = quality_values[new_quality] - quality_values[current_quality]
        
        # Context-based weighting
        visual_importance = self._calculate_visual_importance(obj_id, game_state)
        performance_importance = 1.0 - visual_importance
        
        reward = (visual_delta * visual_importance + performance_delta * performance_importance)
        
        return max(-0.5, min(0.5, reward))
    
    def _estimate_remove_reward(self, obj_id: str, game_state: Dict[str, Any]) -> float:
        """Estimate reward for removing a NeRF skin."""
        current_action = self.active_nerfs.get(obj_id)
        if not current_action:
            return 0.0
        
        # Performance improvement from removal
        performance_gain = self._get_quality_performance_cost(current_action.quality_level)
        
        # Visual quality loss
        visual_loss = self._calculate_visual_importance(obj_id, game_state) * 0.3
        
        # Net reward (usually negative unless performance is critical)
        return performance_gain - visual_loss
    
    def _estimate_environment_reward(self, asset_metadata, game_state: Dict[str, Any]) -> float:
        """Estimate reward for selecting an environment NeRF."""
        base_reward = asset_metadata.visual_quality_score * 0.5  # Environments have lower impact
        
        # Theme matching bonus
        current_theme = game_state.get('theme', 'generic')
        if current_theme in asset_metadata.compatibility_tags:
            base_reward += 0.2
        
        return base_reward
    
    def _estimate_optimization_reward(self, game_state: Dict[str, Any]) -> float:
        """Estimate reward for performance optimization."""
        # Reward based on current performance issues
        performance_metrics = game_state.get('performance_metrics', {})
        
        fps = performance_metrics.get('fps', 60)
        memory_usage = performance_metrics.get('memory_usage', 0.5)
        
        # Higher reward if performance is poor
        fps_penalty = max(0, (60 - fps) / 60) * 0.3
        memory_penalty = max(0, memory_usage - 0.8) * 0.2
        
        return fps_penalty + memory_penalty
    
    def _calculate_performance_penalty(self, quality: NeRFQuality, game_state: Dict[str, Any]) -> float:
        """Calculate performance penalty for a given quality level."""
        base_cost = self._get_quality_performance_cost(quality)
        
        # Increase penalty if performance is already poor
        performance_metrics = game_state.get('performance_metrics', {})
        current_fps = performance_metrics.get('fps', 60)
        
        if current_fps < 30:
            return base_cost * 2.0
        elif current_fps < 45:
            return base_cost * 1.5
        else:
            return base_cost
    
    def _get_quality_performance_cost(self, quality: NeRFQuality) -> float:
        """Get performance cost for a quality level."""
        costs = {
            NeRFQuality.LOW: 0.1,
            NeRFQuality.MEDIUM: 0.2,
            NeRFQuality.HIGH: 0.4,
            NeRFQuality.ULTRA: 0.6
        }
        return costs.get(quality, 0.2)
    
    def _calculate_diversity_bonus(self, asset_id: str) -> float:
        """Calculate diversity bonus for using different assets."""
        # Count how many times this asset is currently used
        usage_count = sum(1 for action in self.active_nerfs.values() 
                         if action.asset_id == asset_id)
        
        # Diminishing returns for repeated use
        if usage_count == 0:
            return 0.1  # Bonus for new asset
        elif usage_count == 1:
            return 0.0  # Neutral
        else:
            return -0.05 * (usage_count - 1)  # Penalty for overuse
    
    def _filter_actions_by_state(self, actions: List[NeRFAction], 
                                game_state: Dict[str, Any]) -> List[NeRFAction]:
        """Filter cached actions based on current game state."""
        # For now, return all actions
        # In a more sophisticated implementation, this would filter based on
        # current game state changes
        return actions
    
    def execute_action(self, action: NeRFAction, game_state: Dict[str, Any]) -> bool:
        """Execute a NeRF action and update internal state."""
        try:
            if action.action_type == NeRFActionType.APPLY_SKIN:
                success = self._execute_apply_skin(action, game_state)
            elif action.action_type == NeRFActionType.CHANGE_QUALITY:
                success = self._execute_change_quality(action, game_state)
            elif action.action_type == NeRFActionType.REMOVE_SKIN:
                success = self._execute_remove_skin(action, game_state)
            elif action.action_type == NeRFActionType.OPTIMIZE_PERFORMANCE:
                success = self._execute_optimize_performance(action, game_state)
            elif action.action_type == NeRFActionType.SELECT_ENVIRONMENT:
                success = self._execute_select_environment(action, game_state)
            else:
                self.logger.warning(f"Unknown NeRF action type: {action.action_type}")
                return False
            
            if success:
                self.logger.info(f"Successfully executed NeRF action: {action.action_type.value}")
                # Invalidate action cache
                self._cache_timestamp = 0
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to execute NeRF action: {e}")
            return False
    
    def _execute_apply_skin(self, action: NeRFAction, game_state: Dict[str, Any]) -> bool:
        """Execute apply skin action."""
        # Remove existing skin if present
        if action.target_object in self.active_nerfs:
            self._execute_remove_skin(
                NeRFAction(NeRFActionType.REMOVE_SKIN, action.target_object),
                game_state
            )
        
        # Store the action
        self.active_nerfs[action.target_object] = action
        
        # The actual rendering will be handled by the NeRF renderer
        # This just tracks the action for the agent
        return True
    
    def _execute_change_quality(self, action: NeRFAction, game_state: Dict[str, Any]) -> bool:
        """Execute quality change action."""
        if action.target_object in self.active_nerfs:
            # Update the stored action
            self.active_nerfs[action.target_object].quality_level = action.quality_level
            return True
        return False
    
    def _execute_remove_skin(self, action: NeRFAction, game_state: Dict[str, Any]) -> bool:
        """Execute remove skin action."""
        if action.target_object in self.active_nerfs:
            del self.active_nerfs[action.target_object]
            return True
        return False
    
    def _execute_optimize_performance(self, action: NeRFAction, game_state: Dict[str, Any]) -> bool:
        """Execute performance optimization action."""
        # Reduce quality of least important NeRF assets
        if not self.active_nerfs:
            return False
        
        # Find least important active NeRF
        least_important = min(
            self.active_nerfs.items(),
            key=lambda x: self._calculate_visual_importance(x[0], game_state)
        )
        
        obj_id, current_action = least_important
        
        # Reduce quality by one level
        quality_order = [NeRFQuality.ULTRA, NeRFQuality.HIGH, NeRFQuality.MEDIUM, NeRFQuality.LOW]
        current_idx = quality_order.index(current_action.quality_level)
        
        if current_idx < len(quality_order) - 1:
            new_quality = quality_order[current_idx + 1]
            current_action.quality_level = new_quality
            return True
        
        return False
    
    def _execute_select_environment(self, action: NeRFAction, game_state: Dict[str, Any]) -> bool:
        """Execute environment selection action."""
        # Store environment selection
        self.active_nerfs["environment"] = action
        return True
    
    def get_action_state(self) -> Dict[str, Any]:
        """Get current state of NeRF actions."""
        return {
            'active_nerfs': {obj_id: action.to_dict() 
                           for obj_id, action in self.active_nerfs.items()},
            'total_active': len(self.active_nerfs),
            'max_concurrent': self.max_concurrent_nerfs
        }

class NeRFEnhancedAgent(EnhancedTransformerRLAgent):
    """Enhanced RL agent with NeRF action space integration."""
    
    def __init__(self, 
                 config,
                 asset_manager: EnhancedNeRFAssetManager,
                 nerf_renderer: NeRFRenderer,
                 **kwargs):
        super().__init__(config, **kwargs)
        
        self.asset_manager = asset_manager
        self.nerf_renderer = nerf_renderer
        
        # Initialize NeRF action space
        self.nerf_action_space = NeRFActionSpace(
            asset_manager=asset_manager,
            max_concurrent_nerfs=getattr(config, 'max_concurrent_nerfs', 10),
            enable_quality_actions=getattr(config, 'enable_nerf_quality_actions', True),
            enable_environment_actions=getattr(config, 'enable_nerf_environment_actions', True)
        )
        
        # Extend action space to include NeRF actions
        self._integrate_nerf_actions()
        
        self.logger.info("NeRF-enhanced agent initialized")
    
    def _integrate_nerf_actions(self):
        """Integrate NeRF actions into the agent's action space."""
        # This would extend the agent's action space
        # Implementation depends on the specific RL framework being used
        pass
    
    def get_available_actions(self, game_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get all available actions including NeRF actions."""
        # Get base actions from parent class
        base_actions = super().get_available_actions(game_state)
        
        # Get NeRF actions
        nerf_actions = self.nerf_action_space.get_available_actions(game_state)
        
        # Convert NeRF actions to agent action format
        nerf_action_dicts = [action.to_dict() for action in nerf_actions]
        
        # Combine actions
        all_actions = base_actions + nerf_action_dicts
        
        return all_actions
    
    def execute_action(self, action: Dict[str, Any], game_state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute action, handling both regular and NeRF actions."""
        # Check if this is a NeRF action
        if 'action_type' in action and action['action_type'] in [at.value for at in NeRFActionType]:
            # Convert to NeRF action object
            nerf_action = NeRFAction(
                action_type=NeRFActionType(action['action_type']),
                target_object=action['target_object'],
                asset_id=action.get('asset_id'),
                quality_level=NeRFQuality(action['quality_level']) if action.get('quality_level') else None,
                parameters=action.get('parameters', {}),
                priority=action.get('priority', 1)
            )
            
            # Execute NeRF action
            success = self.nerf_action_space.execute_action(nerf_action, game_state)
            
            # If successful, also execute the rendering
            if success and nerf_action.action_type == NeRFActionType.APPLY_SKIN:
                render_request = NeRFRenderRequest(
                    asset_id=nerf_action.asset_id,
                    target_object=nerf_action.target_object,
                    render_quality=nerf_action.quality_level
                )
                # Note: Actual rendering would be handled asynchronously
                
            return success, {'nerf_action_executed': True}
        else:
            # Execute regular action
            return super().execute_action(action, game_state)
    
    def calculate_reward(self, game_state: Dict[str, Any], action: Dict[str, Any], 
                        next_state: Dict[str, Any]) -> float:
        """Calculate reward including NeRF-specific components."""
        # Get base reward
        base_reward = super().calculate_reward(game_state, action, next_state)
        
        # Add NeRF-specific reward components
        nerf_reward = self._calculate_nerf_reward(game_state, action, next_state)
        
        # Combine rewards (NeRF reward is typically a smaller component)
        total_reward = base_reward + (nerf_reward * 0.2)  # 20% weight for NeRF rewards
        
        return total_reward
    
    def _calculate_nerf_reward(self, game_state: Dict[str, Any], action: Dict[str, Any],
                              next_state: Dict[str, Any]) -> float:
        """Calculate NeRF-specific reward components."""
        reward = 0.0
        
        # Visual quality improvement reward
        visual_improvement = self._calculate_visual_improvement(game_state, next_state)
        reward += visual_improvement * 0.5
        
        # Performance efficiency reward
        performance_efficiency = self._calculate_performance_efficiency(game_state, next_state)
        reward += performance_efficiency * 0.3
        
        # Diversity reward
        diversity_score = self._calculate_nerf_diversity()
        reward += diversity_score * 0.2
        
        return reward
    
    def _calculate_visual_improvement(self, game_state: Dict[str, Any], 
                                    next_state: Dict[str, Any]) -> float:
        """Calculate visual quality improvement from NeRF actions."""
        # This would integrate with the visual assessment system
        # For now, return a placeholder
        return 0.0
    
    def _calculate_performance_efficiency(self, game_state: Dict[str, Any],
                                        next_state: Dict[str, Any]) -> float:
        """Calculate performance efficiency of NeRF usage."""
        # Compare performance metrics
        prev_fps = game_state.get('performance_metrics', {}).get('fps', 60)
        curr_fps = next_state.get('performance_metrics', {}).get('fps', 60)
        
        fps_change = (curr_fps - prev_fps) / 60.0
        return max(-0.5, min(0.5, fps_change))
    
    def _calculate_nerf_diversity(self) -> float:
        """Calculate diversity score for NeRF asset usage."""
        active_assets = set()
        for action in self.nerf_action_space.active_nerfs.values():
            if action.asset_id:
                active_assets.add(action.asset_id)
        
        # Reward for using diverse assets
        diversity_score = len(active_assets) / max(1, len(self.nerf_action_space.active_nerfs))
        return diversity_score - 0.5  # Center around 0

def create_nerf_enhanced_agent(config: MasterConfig,
                              asset_manager: EnhancedNeRFAssetManager,
                              nerf_renderer: NeRFRenderer) -> NeRFEnhancedAgent:
    """Create a NeRF-enhanced RL agent."""
    agent = NeRFEnhancedAgent(
        config=config,
        asset_manager=asset_manager,
        nerf_renderer=nerf_renderer
    )
    
    logging.getLogger(__name__).info("NeRF-enhanced agent created successfully")
    
    return agent

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("NeRF Agent Action Space Extensions initialized!")
    print("Features:")
    print("- NeRF skin application actions")
    print("- Quality adjustment actions")
    print("- Environment selection actions")
    print("- Performance optimization actions")
    print("- Context-aware asset selection")
    print("- Reward integration for NeRF usage")

