"""
NeRF Asset Management System

Advanced asset management system for NeRF assets that integrates with
the enhanced configuration system and provides intelligent asset
selection, caching, and optimization.
"""

import os
import json
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

# Import enhanced components
try:
    from enhanced_config_manager import MasterConfig, ConfigurationManager
    from nerf_integration_module import (
        NeRFAssetType, NeRFQuality, NeRFAssetMetadata, 
        NeRFAssetManager, NeRFRenderer
    )
    from visual_assessment_gpu_enhanced import EnhancedVisualAssessmentGPU
except ImportError as e:
    logging.warning(f"Some enhanced components not available: {e}")

@dataclass
class NeRFConfig:
    """Configuration for NeRF integration system."""
    asset_directory: str = "nerf_assets"
    cache_size: int = 100
    max_concurrent_renders: int = 4
    enable_quality_assessment: bool = True
    enable_performance_monitoring: bool = True
    auto_optimize_assets: bool = True
    asset_database_path: str = "nerf_assets.db"
    default_quality: str = "medium"
    enable_asset_streaming: bool = False
    streaming_chunk_size: int = 1024 * 1024  # 1MB chunks
    
    def validate(self):
        """Validate NeRF configuration."""
        if self.cache_size < 1:
            raise ValueError("Cache size must be at least 1")
        if self.max_concurrent_renders < 1:
            raise ValueError("Max concurrent renders must be at least 1")
        if self.default_quality not in ["low", "medium", "high", "ultra"]:
            raise ValueError("Default quality must be one of: low, medium, high, ultra")

class NeRFAssetDatabase:
    """SQLite database for NeRF asset management."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the asset database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nerf_assets (
                    asset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    quality_level TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    creation_date TEXT,
                    source_scene TEXT,
                    training_views INTEGER DEFAULT 0,
                    resolution_width INTEGER DEFAULT 512,
                    resolution_height INTEGER DEFAULT 512,
                    performance_score REAL DEFAULT 0.0,
                    visual_quality_score REAL DEFAULT 0.0,
                    usage_count INTEGER DEFAULT 0,
                    last_used TEXT,
                    metadata_json TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_tags (
                    asset_id TEXT,
                    tag TEXT,
                    FOREIGN KEY (asset_id) REFERENCES nerf_assets (asset_id),
                    PRIMARY KEY (asset_id, tag)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_performance (
                    asset_id TEXT,
                    render_time REAL,
                    memory_usage INTEGER,
                    gpu_utilization REAL,
                    timestamp TEXT,
                    FOREIGN KEY (asset_id) REFERENCES nerf_assets (asset_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_asset_type ON nerf_assets (asset_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_level ON nerf_assets (quality_level)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_performance_score ON nerf_assets (performance_score)
            """)
    
    def insert_asset(self, metadata: NeRFAssetMetadata):
        """Insert asset metadata into database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO nerf_assets 
                    (asset_id, name, asset_type, quality_level, file_path, file_size,
                     creation_date, source_scene, training_views, resolution_width,
                     resolution_height, performance_score, visual_quality_score,
                     usage_count, last_used, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.asset_id, metadata.name, metadata.asset_type.value,
                    metadata.quality_level.value, metadata.file_path, metadata.file_size,
                    metadata.creation_date, metadata.source_scene, metadata.training_views,
                    metadata.resolution[0], metadata.resolution[1],
                    metadata.performance_score, metadata.visual_quality_score,
                    metadata.usage_count, metadata.last_used,
                    json.dumps(metadata.to_dict())
                ))
                
                # Insert tags
                conn.execute("DELETE FROM asset_tags WHERE asset_id = ?", (metadata.asset_id,))
                for tag in metadata.compatibility_tags:
                    conn.execute("INSERT INTO asset_tags (asset_id, tag) VALUES (?, ?)",
                               (metadata.asset_id, tag))
    
    def get_asset(self, asset_id: str) -> Optional[NeRFAssetMetadata]:
        """Get asset metadata from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT metadata_json FROM nerf_assets WHERE asset_id = ?
            """, (asset_id,))
            row = cursor.fetchone()
            
            if row:
                return NeRFAssetMetadata.from_dict(json.loads(row[0]))
            return None
    
    def search_assets(self, 
                     asset_type: Optional[NeRFAssetType] = None,
                     quality_level: Optional[NeRFQuality] = None,
                     tags: Optional[List[str]] = None,
                     min_quality_score: float = 0.0,
                     limit: int = 100) -> List[NeRFAssetMetadata]:
        """Search for assets with filters."""
        query = "SELECT metadata_json FROM nerf_assets WHERE 1=1"
        params = []
        
        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type.value)
        
        if quality_level:
            query += " AND quality_level = ?"
            params.append(quality_level.value)
        
        if min_quality_score > 0:
            query += " AND visual_quality_score >= ?"
            params.append(min_quality_score)
        
        if tags:
            tag_placeholders = ",".join("?" * len(tags))
            query += f"""
                AND asset_id IN (
                    SELECT asset_id FROM asset_tags 
                    WHERE tag IN ({tag_placeholders})
                    GROUP BY asset_id 
                    HAVING COUNT(*) = ?
                )
            """
            params.extend(tags)
            params.append(len(tags))
        
        query += " ORDER BY visual_quality_score DESC, performance_score DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            results = []
            for row in cursor.fetchall():
                results.append(NeRFAssetMetadata.from_dict(json.loads(row[0])))
            return results
    
    def update_performance_stats(self, asset_id: str, render_time: float, 
                                memory_usage: int, gpu_utilization: float):
        """Update performance statistics for an asset."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO asset_performance 
                    (asset_id, render_time, memory_usage, gpu_utilization, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (asset_id, render_time, memory_usage, gpu_utilization,
                     time.strftime("%Y-%m-%d %H:%M:%S")))

class EnhancedNeRFAssetManager(NeRFAssetManager):
    """Enhanced NeRF asset manager with database integration and advanced features."""
    
    def __init__(self, config: NeRFConfig):
        self.config = config
        self.config.validate()
        
        # Initialize parent class
        super().__init__(
            asset_directory=config.asset_directory,
            cache_size=config.cache_size
        )
        
        # Initialize database
        self.database = NeRFAssetDatabase(config.asset_database_path)
        
        # Initialize visual assessment if enabled
        self.visual_assessor = None
        if config.enable_quality_assessment:
            try:
                self.visual_assessor = EnhancedVisualAssessmentGPU(
                    use_gpu=True,
                    enable_caching=True,
                    cache_size=config.cache_size
                )
            except Exception as e:
                self.logger.warning(f"Could not initialize visual assessor: {e}")
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor() if config.enable_performance_monitoring else None
        
        # Asset optimization
        self.asset_optimizer = AssetOptimizer() if config.auto_optimize_assets else None
        
        # Sync database with file system
        self._sync_database()
    
    def _sync_database(self):
        """Sync database with file system metadata."""
        # Load existing assets from database
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.execute("SELECT asset_id, metadata_json FROM nerf_assets")
                for row in cursor.fetchall():
                    asset_id, metadata_json = row
                    metadata = NeRFAssetMetadata.from_dict(json.loads(metadata_json))
                    self.metadata_cache[asset_id] = metadata
            
            self.logger.info(f"Synced {len(self.metadata_cache)} assets from database")
        except Exception as e:
            self.logger.error(f"Failed to sync database: {e}")
    
    def register_asset(self, 
                      file_path: str, 
                      name: str, 
                      asset_type: NeRFAssetType,
                      quality_level: NeRFQuality = None,
                      auto_assess_quality: bool = True,
                      **kwargs) -> str:
        """Register asset with enhanced features."""
        if quality_level is None:
            quality_level = NeRFQuality(self.config.default_quality)
        
        # Register with parent class
        asset_id = super().register_asset(file_path, name, asset_type, quality_level, **kwargs)
        
        # Store in database
        metadata = self.metadata_cache[asset_id]
        self.database.insert_asset(metadata)
        
        # Assess quality if enabled
        if auto_assess_quality and self.visual_assessor:
            try:
                quality_score = self._assess_asset_quality(file_path)
                metadata.visual_quality_score = quality_score
                self.database.insert_asset(metadata)
                self.logger.info(f"Asset quality assessed: {quality_score:.3f}")
            except Exception as e:
                self.logger.warning(f"Failed to assess asset quality: {e}")
        
        # Optimize asset if enabled
        if self.asset_optimizer:
            try:
                self.asset_optimizer.optimize_asset(asset_id, metadata)
            except Exception as e:
                self.logger.warning(f"Failed to optimize asset: {e}")
        
        return asset_id
    
    def _assess_asset_quality(self, file_path: str) -> float:
        """Assess visual quality of an asset."""
        if not self.visual_assessor:
            return 0.0
        
        # This would integrate with the visual assessment system
        # For now, return a placeholder score
        return 0.8  # Placeholder
    
    def search_assets(self, 
                     asset_type: Optional[NeRFAssetType] = None,
                     quality_level: Optional[NeRFQuality] = None,
                     tags: Optional[List[str]] = None,
                     min_quality_score: float = 0.0,
                     context: Optional[Dict[str, Any]] = None) -> List[NeRFAssetMetadata]:
        """Enhanced asset search with context awareness."""
        # Use database search for better performance
        assets = self.database.search_assets(
            asset_type=asset_type,
            quality_level=quality_level,
            tags=tags,
            min_quality_score=min_quality_score
        )
        
        # Apply context-based filtering if provided
        if context:
            assets = self._filter_by_context(assets, context)
        
        return assets
    
    def _filter_by_context(self, assets: List[NeRFAssetMetadata], context: Dict[str, Any]) -> List[NeRFAssetMetadata]:
        """Filter assets based on context."""
        filtered_assets = []
        
        for asset in assets:
            if self._matches_context(asset, context):
                filtered_assets.append(asset)
        
        return filtered_assets
    
    def _matches_context(self, asset: NeRFAssetMetadata, context: Dict[str, Any]) -> bool:
        """Check if asset matches the given context."""
        # Environment type matching
        if 'environment_type' in context:
            if context['environment_type'] not in asset.compatibility_tags:
                return False
        
        # Performance requirements
        if 'min_performance_score' in context:
            if asset.performance_score < context['min_performance_score']:
                return False
        
        # Resolution requirements
        if 'min_resolution' in context:
            min_res = context['min_resolution']
            if asset.resolution[0] < min_res[0] or asset.resolution[1] < min_res[1]:
                return False
        
        return True
    
    def get_recommended_assets(self, 
                              target_object: str, 
                              context: Dict[str, Any],
                              max_results: int = 5) -> List[Tuple[str, float]]:
        """Get recommended assets with confidence scores."""
        # Search for compatible assets
        compatible_assets = self.search_assets(
            tags=[target_object],
            context=context
        )
        
        # Calculate recommendation scores
        recommendations = []
        for asset in compatible_assets:
            score = self._calculate_recommendation_score(asset, target_object, context)
            recommendations.append((asset.asset_id, score))
        
        # Sort by score and return top results
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:max_results]
    
    def _calculate_recommendation_score(self, 
                                      asset: NeRFAssetMetadata, 
                                      target_object: str, 
                                      context: Dict[str, Any]) -> float:
        """Calculate recommendation score for an asset."""
        score = 0.0
        
        # Base quality score (40% weight)
        score += asset.visual_quality_score * 0.4
        
        # Performance score (30% weight)
        score += asset.performance_score * 0.3
        
        # Compatibility score (20% weight)
        compatibility_score = 0.0
        if target_object in asset.compatibility_tags:
            compatibility_score += 0.5
        
        # Context matching (10% weight)
        context_score = 0.0
        if self._matches_context(asset, context):
            context_score = 1.0
        
        score += compatibility_score * 0.2
        score += context_score * 0.1
        
        # Usage popularity bonus (small boost for frequently used assets)
        if asset.usage_count > 0:
            popularity_bonus = min(0.1, asset.usage_count / 100.0)
            score += popularity_bonus
        
        return min(1.0, score)  # Cap at 1.0

class PerformanceMonitor:
    """Monitors NeRF asset performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_data = {}
    
    def record_performance(self, asset_id: str, render_time: float, 
                          memory_usage: int, gpu_utilization: float):
        """Record performance metrics for an asset."""
        if asset_id not in self.performance_data:
            self.performance_data[asset_id] = []
        
        self.performance_data[asset_id].append({
            'render_time': render_time,
            'memory_usage': memory_usage,
            'gpu_utilization': gpu_utilization,
            'timestamp': time.time()
        })
        
        # Keep only recent data (last 100 records)
        if len(self.performance_data[asset_id]) > 100:
            self.performance_data[asset_id] = self.performance_data[asset_id][-100:]
    
    def get_average_performance(self, asset_id: str) -> Dict[str, float]:
        """Get average performance metrics for an asset."""
        if asset_id not in self.performance_data:
            return {'render_time': 0.0, 'memory_usage': 0.0, 'gpu_utilization': 0.0}
        
        data = self.performance_data[asset_id]
        if not data:
            return {'render_time': 0.0, 'memory_usage': 0.0, 'gpu_utilization': 0.0}
        
        return {
            'render_time': sum(d['render_time'] for d in data) / len(data),
            'memory_usage': sum(d['memory_usage'] for d in data) / len(data),
            'gpu_utilization': sum(d['gpu_utilization'] for d in data) / len(data)
        }

class AssetOptimizer:
    """Optimizes NeRF assets for better performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_asset(self, asset_id: str, metadata: NeRFAssetMetadata):
        """Optimize an asset for better performance."""
        try:
            # Placeholder for asset optimization logic
            # This could include:
            # - Mesh simplification
            # - Texture compression
            # - LOD generation
            # - Format conversion
            
            self.logger.info(f"Optimizing asset: {asset_id}")
            
            # Update performance score based on optimization
            metadata.performance_score = min(1.0, metadata.performance_score + 0.1)
            
        except Exception as e:
            self.logger.error(f"Failed to optimize asset {asset_id}: {e}")

def create_enhanced_nerf_system(config: Optional[MasterConfig] = None) -> Tuple[EnhancedNeRFAssetManager, NeRFRenderer]:
    """Create enhanced NeRF system with configuration integration."""
    # Create NeRF configuration
    nerf_config = NeRFConfig()
    
    # Override with master config if provided
    if config and hasattr(config, 'nerf'):
        for field_name in nerf_config.__dataclass_fields__:
            if hasattr(config.nerf, field_name):
                setattr(nerf_config, field_name, getattr(config.nerf, field_name))
    
    # Create enhanced components
    asset_manager = EnhancedNeRFAssetManager(nerf_config)
    renderer = NeRFRenderer(asset_manager=asset_manager)
    
    logging.getLogger(__name__).info("Enhanced NeRF system created successfully")
    
    return asset_manager, renderer

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create enhanced NeRF system
    asset_manager, renderer = create_enhanced_nerf_system()
    
    print("Enhanced NeRF Asset Management System initialized!")
    print(f"Database: {asset_manager.database.db_path}")
    print(f"Asset directory: {asset_manager.asset_directory}")
    print(f"Cache size: {asset_manager.cache_size}")

