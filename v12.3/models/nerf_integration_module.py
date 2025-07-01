"""
NeRF Integration Module for RL Training Script v3

This module provides comprehensive Neural Radiance Field (NeRF) integration
for the RL training system, enabling agents to use NeRF-generated assets
as skins, textures, and environmental elements.

Key Features:
- NeRF asset management and loading
- Real-time NeRF rendering integration with Three.js
- Agent action space extensions for NeRF skin selection
- Performance optimization and caching
- Quality assessment metrics for NeRF assets
"""

import os
import json
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import enhanced components
try:
    from enhanced_config_manager import MasterConfig
    from visual_assessment_gpu_enhanced import EnhancedVisualAssessmentGPU
    from async_rendering_pipeline_enhanced import EnhancedAsyncRenderingPipeline
except ImportError as e:
    logging.warning(f"Some enhanced components not available: {e}")

class NeRFAssetType(Enum):
    """Types of NeRF assets supported."""
    MESH = "mesh"                    # .glb/.gltf/.obj files
    POINT_CLOUD = "point_cloud"      # .ply files
    TEXTURE_ATLAS = "texture_atlas"  # Multi-view texture sets
    VOLUMETRIC = "volumetric"        # Volumetric representations
    ENVIRONMENT = "environment"      # 360-degree environments

class NeRFQuality(Enum):
    """Quality levels for NeRF assets."""
    LOW = "low"          # Fast rendering, lower quality
    MEDIUM = "medium"    # Balanced quality/performance
    HIGH = "high"        # High quality, slower rendering
    ULTRA = "ultra"      # Maximum quality for final renders

@dataclass
class NeRFAssetMetadata:
    """Metadata for a NeRF asset."""
    asset_id: str
    name: str
    asset_type: NeRFAssetType
    quality_level: NeRFQuality
    file_path: str
    file_size: int
    creation_date: str
    source_scene: Optional[str] = None
    training_views: int = 0
    resolution: Tuple[int, int] = (512, 512)
    performance_score: float = 0.0
    visual_quality_score: float = 0.0
    compatibility_tags: List[str] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'asset_id': self.asset_id,
            'name': self.name,
            'asset_type': self.asset_type.value,
            'quality_level': self.quality_level.value,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'creation_date': self.creation_date,
            'source_scene': self.source_scene,
            'training_views': self.training_views,
            'resolution': list(self.resolution),
            'performance_score': self.performance_score,
            'visual_quality_score': self.visual_quality_score,
            'compatibility_tags': self.compatibility_tags,
            'usage_count': self.usage_count,
            'last_used': self.last_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeRFAssetMetadata':
        """Create from dictionary."""
        return cls(
            asset_id=data['asset_id'],
            name=data['name'],
            asset_type=NeRFAssetType(data['asset_type']),
            quality_level=NeRFQuality(data['quality_level']),
            file_path=data['file_path'],
            file_size=data['file_size'],
            creation_date=data['creation_date'],
            source_scene=data.get('source_scene'),
            training_views=data.get('training_views', 0),
            resolution=tuple(data.get('resolution', [512, 512])),
            performance_score=data.get('performance_score', 0.0),
            visual_quality_score=data.get('visual_quality_score', 0.0),
            compatibility_tags=data.get('compatibility_tags', []),
            usage_count=data.get('usage_count', 0),
            last_used=data.get('last_used')
        )

@dataclass
class NeRFRenderRequest:
    """Request for NeRF asset rendering."""
    asset_id: str
    target_object: str
    render_quality: NeRFQuality
    view_angle: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    lighting_conditions: Dict[str, Any] = field(default_factory=dict)
    post_processing: List[str] = field(default_factory=list)
    priority: int = 1
    timeout: float = 30.0
    cache_result: bool = True

class NeRFAssetManager:
    """Manages NeRF assets and their metadata."""
    
    def __init__(self, asset_directory: str = "nerf_assets", cache_size: int = 100):
        self.asset_directory = Path(asset_directory)
        self.asset_directory.mkdir(exist_ok=True)
        
        self.metadata_file = self.asset_directory / "asset_metadata.json"
        self.cache_size = cache_size
        self.asset_cache = {}
        self.metadata_cache = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Load existing metadata
        self._load_metadata()
        
        # Create subdirectories for different asset types
        for asset_type in NeRFAssetType:
            (self.asset_directory / asset_type.value).mkdir(exist_ok=True)
    
    def _load_metadata(self):
        """Load asset metadata from file."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for asset_id, metadata_dict in data.items():
                        self.metadata_cache[asset_id] = NeRFAssetMetadata.from_dict(metadata_dict)
                self.logger.info(f"Loaded metadata for {len(self.metadata_cache)} NeRF assets")
            except Exception as e:
                self.logger.error(f"Failed to load NeRF metadata: {e}")
    
    def _save_metadata(self):
        """Save asset metadata to file."""
        try:
            data = {asset_id: metadata.to_dict() 
                   for asset_id, metadata in self.metadata_cache.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save NeRF metadata: {e}")
    
    def register_asset(self, 
                      file_path: str, 
                      name: str, 
                      asset_type: NeRFAssetType,
                      quality_level: NeRFQuality = NeRFQuality.MEDIUM,
                      **kwargs) -> str:
        """Register a new NeRF asset."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"NeRF asset file not found: {file_path}")
        
        # Generate asset ID
        asset_id = hashlib.md5(f"{name}_{file_path.name}_{time.time()}".encode()).hexdigest()[:16]
        
        # Create metadata
        metadata = NeRFAssetMetadata(
            asset_id=asset_id,
            name=name,
            asset_type=asset_type,
            quality_level=quality_level,
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
            creation_date=time.strftime("%Y-%m-%d %H:%M:%S"),
            **kwargs
        )
        
        # Store metadata
        self.metadata_cache[asset_id] = metadata
        self._save_metadata()
        
        self.logger.info(f"Registered NeRF asset: {name} ({asset_id})")
        return asset_id
    
    def get_asset_metadata(self, asset_id: str) -> Optional[NeRFAssetMetadata]:
        """Get metadata for a specific asset."""
        return self.metadata_cache.get(asset_id)
    
    def list_assets(self, 
                   asset_type: Optional[NeRFAssetType] = None,
                   quality_level: Optional[NeRFQuality] = None,
                   tags: Optional[List[str]] = None) -> List[NeRFAssetMetadata]:
        """List assets with optional filtering."""
        assets = list(self.metadata_cache.values())
        
        if asset_type:
            assets = [a for a in assets if a.asset_type == asset_type]
        
        if quality_level:
            assets = [a for a in assets if a.quality_level == quality_level]
        
        if tags:
            assets = [a for a in assets if any(tag in a.compatibility_tags for tag in tags)]
        
        return sorted(assets, key=lambda a: a.visual_quality_score, reverse=True)
    
    def load_asset(self, asset_id: str) -> Optional[Any]:
        """Load a NeRF asset into memory."""
        if asset_id in self.asset_cache:
            return self.asset_cache[asset_id]
        
        metadata = self.get_asset_metadata(asset_id)
        if not metadata:
            self.logger.error(f"Asset not found: {asset_id}")
            return None
        
        try:
            # Load based on asset type
            if metadata.asset_type == NeRFAssetType.MESH:
                asset_data = self._load_mesh_asset(metadata.file_path)
            elif metadata.asset_type == NeRFAssetType.POINT_CLOUD:
                asset_data = self._load_point_cloud_asset(metadata.file_path)
            elif metadata.asset_type == NeRFAssetType.TEXTURE_ATLAS:
                asset_data = self._load_texture_atlas_asset(metadata.file_path)
            else:
                self.logger.warning(f"Unsupported asset type: {metadata.asset_type}")
                return None
            
            # Cache the asset
            if len(self.asset_cache) >= self.cache_size:
                # Remove least recently used asset
                oldest_asset = min(self.asset_cache.keys(), 
                                 key=lambda k: self.metadata_cache[k].last_used or "")
                del self.asset_cache[oldest_asset]
            
            self.asset_cache[asset_id] = asset_data
            
            # Update usage statistics
            metadata.usage_count += 1
            metadata.last_used = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_metadata()
            
            self.logger.info(f"Loaded NeRF asset: {metadata.name}")
            return asset_data
            
        except Exception as e:
            self.logger.error(f"Failed to load NeRF asset {asset_id}: {e}")
            return None
    
    def _load_mesh_asset(self, file_path: str) -> Dict[str, Any]:
        """Load mesh-based NeRF asset."""
        # This would integrate with Three.js loaders
        return {
            'type': 'mesh',
            'file_path': file_path,
            'format': Path(file_path).suffix.lower(),
            'loaded_at': time.time()
        }
    
    def _load_point_cloud_asset(self, file_path: str) -> Dict[str, Any]:
        """Load point cloud NeRF asset."""
        return {
            'type': 'point_cloud',
            'file_path': file_path,
            'format': Path(file_path).suffix.lower(),
            'loaded_at': time.time()
        }
    
    def _load_texture_atlas_asset(self, file_path: str) -> Dict[str, Any]:
        """Load texture atlas NeRF asset."""
        return {
            'type': 'texture_atlas',
            'file_path': file_path,
            'format': Path(file_path).suffix.lower(),
            'loaded_at': time.time()
        }
    
    def get_compatible_assets(self, target_object: str, context: Dict[str, Any]) -> List[str]:
        """Get assets compatible with a target object and context."""
        compatible_assets = []
        
        for asset_id, metadata in self.metadata_cache.items():
            # Check compatibility based on tags and context
            if self._is_compatible(metadata, target_object, context):
                compatible_assets.append(asset_id)
        
        # Sort by quality and performance scores
        compatible_assets.sort(
            key=lambda aid: (
                self.metadata_cache[aid].visual_quality_score,
                self.metadata_cache[aid].performance_score
            ),
            reverse=True
        )
        
        return compatible_assets
    
    def _is_compatible(self, metadata: NeRFAssetMetadata, target_object: str, context: Dict[str, Any]) -> bool:
        """Check if an asset is compatible with the target and context."""
        # Basic compatibility checks
        if target_object in metadata.compatibility_tags:
            return True
        
        # Context-based compatibility
        if context.get('environment_type') in metadata.compatibility_tags:
            return True
        
        # Default compatibility for generic assets
        if 'generic' in metadata.compatibility_tags:
            return True
        
        return False

class NeRFRenderer:
    """Handles NeRF asset rendering integration with Three.js."""
    
    def __init__(self, asset_manager: NeRFAssetManager, rendering_pipeline: Optional[Any] = None):
        self.asset_manager = asset_manager
        self.rendering_pipeline = rendering_pipeline
        self.logger = logging.getLogger(__name__)
        
        # Rendering cache for frequently used assets
        self.render_cache = {}
        self.cache_size = 50
        
        # Performance tracking
        self.render_stats = {
            'total_renders': 0,
            'cache_hits': 0,
            'average_render_time': 0.0,
            'failed_renders': 0
        }
    
    async def render_nerf_asset(self, request: NeRFRenderRequest) -> Optional[Dict[str, Any]]:
        """Render a NeRF asset according to the request parameters."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if request.cache_result and cache_key in self.render_cache:
                self.render_stats['cache_hits'] += 1
                self.logger.debug(f"Cache hit for NeRF render: {request.asset_id}")
                return self.render_cache[cache_key]
            
            # Load asset
            asset_data = self.asset_manager.load_asset(request.asset_id)
            if not asset_data:
                self.render_stats['failed_renders'] += 1
                return None
            
            # Perform rendering based on asset type
            metadata = self.asset_manager.get_asset_metadata(request.asset_id)
            if metadata.asset_type == NeRFAssetType.MESH:
                result = await self._render_mesh_asset(asset_data, request)
            elif metadata.asset_type == NeRFAssetType.POINT_CLOUD:
                result = await self._render_point_cloud_asset(asset_data, request)
            elif metadata.asset_type == NeRFAssetType.TEXTURE_ATLAS:
                result = await self._render_texture_atlas_asset(asset_data, request)
            else:
                self.logger.error(f"Unsupported asset type for rendering: {metadata.asset_type}")
                self.render_stats['failed_renders'] += 1
                return None
            
            # Cache result if requested
            if request.cache_result and result:
                if len(self.render_cache) >= self.cache_size:
                    # Remove oldest cache entry
                    oldest_key = min(self.render_cache.keys(), 
                                   key=lambda k: self.render_cache[k].get('timestamp', 0))
                    del self.render_cache[oldest_key]
                
                result['timestamp'] = time.time()
                self.render_cache[cache_key] = result
            
            # Update statistics
            render_time = time.time() - start_time
            self.render_stats['total_renders'] += 1
            self.render_stats['average_render_time'] = (
                (self.render_stats['average_render_time'] * (self.render_stats['total_renders'] - 1) + render_time) /
                self.render_stats['total_renders']
            )
            
            self.logger.info(f"Rendered NeRF asset {request.asset_id} in {render_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to render NeRF asset {request.asset_id}: {e}")
            self.render_stats['failed_renders'] += 1
            return None
    
    def _generate_cache_key(self, request: NeRFRenderRequest) -> str:
        """Generate cache key for render request."""
        key_data = f"{request.asset_id}_{request.target_object}_{request.render_quality.value}_{request.view_angle}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    async def _render_mesh_asset(self, asset_data: Dict[str, Any], request: NeRFRenderRequest) -> Dict[str, Any]:
        """Render mesh-based NeRF asset."""
        # This would integrate with the Three.js rendering pipeline
        return {
            'type': 'mesh_render',
            'asset_id': request.asset_id,
            'target_object': request.target_object,
            'render_data': {
                'mesh_path': asset_data['file_path'],
                'format': asset_data['format'],
                'view_angle': request.view_angle,
                'quality': request.render_quality.value,
                'lighting': request.lighting_conditions
            },
            'three_js_config': {
                'loader_type': 'GLTFLoader' if asset_data['format'] in ['.glb', '.gltf'] else 'OBJLoader',
                'material_override': True,
                'shadow_casting': True,
                'receive_shadows': True
            }
        }
    
    async def _render_point_cloud_asset(self, asset_data: Dict[str, Any], request: NeRFRenderRequest) -> Dict[str, Any]:
        """Render point cloud NeRF asset."""
        return {
            'type': 'point_cloud_render',
            'asset_id': request.asset_id,
            'target_object': request.target_object,
            'render_data': {
                'point_cloud_path': asset_data['file_path'],
                'format': asset_data['format'],
                'view_angle': request.view_angle,
                'quality': request.render_quality.value,
                'point_size': self._get_point_size_for_quality(request.render_quality)
            },
            'three_js_config': {
                'loader_type': 'PLYLoader',
                'material_type': 'PointsMaterial',
                'vertex_colors': True,
                'size_attenuation': True
            }
        }
    
    async def _render_texture_atlas_asset(self, asset_data: Dict[str, Any], request: NeRFRenderRequest) -> Dict[str, Any]:
        """Render texture atlas NeRF asset."""
        return {
            'type': 'texture_atlas_render',
            'asset_id': request.asset_id,
            'target_object': request.target_object,
            'render_data': {
                'texture_path': asset_data['file_path'],
                'format': asset_data['format'],
                'view_angle': request.view_angle,
                'quality': request.render_quality.value,
                'uv_mapping': 'auto'
            },
            'three_js_config': {
                'loader_type': 'TextureLoader',
                'wrap_s': 'RepeatWrapping',
                'wrap_t': 'RepeatWrapping',
                'mag_filter': 'LinearFilter',
                'min_filter': 'LinearMipmapLinearFilter'
            }
        }
    
    def _get_point_size_for_quality(self, quality: NeRFQuality) -> float:
        """Get point size based on quality level."""
        quality_map = {
            NeRFQuality.LOW: 1.0,
            NeRFQuality.MEDIUM: 2.0,
            NeRFQuality.HIGH: 3.0,
            NeRFQuality.ULTRA: 4.0
        }
        return quality_map.get(quality, 2.0)
    
    def get_render_stats(self) -> Dict[str, Any]:
        """Get rendering performance statistics."""
        cache_hit_rate = (self.render_stats['cache_hits'] / max(1, self.render_stats['total_renders'])) * 100
        
        return {
            **self.render_stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.render_cache),
            'success_rate': ((self.render_stats['total_renders'] - self.render_stats['failed_renders']) / 
                           max(1, self.render_stats['total_renders'])) * 100
        }

def create_nerf_integration_system(config: Optional[MasterConfig] = None) -> Tuple[NeRFAssetManager, NeRFRenderer]:
    """Create a complete NeRF integration system."""
    # Use configuration if provided
    asset_dir = "nerf_assets"
    cache_size = 100
    
    if config and hasattr(config, 'nerf'):
        asset_dir = getattr(config.nerf, 'asset_directory', asset_dir)
        cache_size = getattr(config.nerf, 'cache_size', cache_size)
    
    # Create components
    asset_manager = NeRFAssetManager(asset_directory=asset_dir, cache_size=cache_size)
    renderer = NeRFRenderer(asset_manager=asset_manager)
    
    logging.getLogger(__name__).info("NeRF integration system created successfully")
    
    return asset_manager, renderer

# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create NeRF integration system
    asset_manager, renderer = create_nerf_integration_system()
    
    # Example: Register a NeRF asset
    # asset_id = asset_manager.register_asset(
    #     file_path="path/to/nerf_asset.glb",
    #     name="Fantasy Castle Wall",
    #     asset_type=NeRFAssetType.MESH,
    #     quality_level=NeRFQuality.HIGH,
    #     compatibility_tags=["wall", "castle", "medieval", "stone"]
    # )
    
    print("NeRF Integration Module initialized successfully!")
    print(f"Asset manager ready with {len(asset_manager.metadata_cache)} assets")
    print(f"Renderer ready with cache size {renderer.cache_size}")

