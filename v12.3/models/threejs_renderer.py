"""
Three.js Visual Themes and Rendering System

This module provides Three.js-based visual themes and rendering stubs
for the procedurally generated tower archetypes.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import math

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from systems.tower_generation.tower_generator import TowerArchetype, TowerTheme, ParticleType
from utils.logging import setup_logger


class GeometryType(Enum):
    """Types of 3D geometries for towers."""
    CYLINDER = "cylinder"
    BOX = "box"
    CONE = "cone"
    SPHERE = "sphere"
    OCTAHEDRON = "octahedron"
    DODECAHEDRON = "dodecahedron"
    TORUS = "torus"
    CUSTOM = "custom"


class MaterialType(Enum):
    """Types of materials for rendering."""
    BASIC = "basic"
    LAMBERT = "lambert"
    PHONG = "phong"
    STANDARD = "standard"
    PHYSICAL = "physical"
    SHADER = "shader"


@dataclass
class GeometryConfig:
    """Configuration for 3D geometry."""
    geometry_type: GeometryType
    width: float
    height: float
    depth: float
    radius_top: float
    radius_bottom: float
    radial_segments: int
    height_segments: int
    
    def to_threejs_params(self) -> Dict[str, Any]:
        """Convert to Three.js geometry parameters."""
        if self.geometry_type == GeometryType.CYLINDER:
            return {
                'radiusTop': self.radius_top,
                'radiusBottom': self.radius_bottom,
                'height': self.height,
                'radialSegments': self.radial_segments,
                'heightSegments': self.height_segments
            }
        elif self.geometry_type == GeometryType.BOX:
            return {
                'width': self.width,
                'height': self.height,
                'depth': self.depth
            }
        elif self.geometry_type == GeometryType.CONE:
            return {
                'radius': self.radius_bottom,
                'height': self.height,
                'radialSegments': self.radial_segments
            }
        elif self.geometry_type == GeometryType.SPHERE:
            return {
                'radius': self.radius_bottom,
                'widthSegments': self.radial_segments,
                'heightSegments': self.height_segments
            }
        else:
            return {
                'width': self.width,
                'height': self.height,
                'depth': self.depth
            }


@dataclass
class MaterialConfig:
    """Configuration for 3D materials."""
    material_type: MaterialType
    color: str
    emissive: str
    metalness: float
    roughness: float
    opacity: float
    transparent: bool
    texture_map: Optional[str]
    normal_map: Optional[str]
    emission_map: Optional[str]
    
    def to_threejs_params(self) -> Dict[str, Any]:
        """Convert to Three.js material parameters."""
        params = {
            'color': self.color,
            'opacity': self.opacity,
            'transparent': self.transparent
        }
        
        if self.material_type in [MaterialType.STANDARD, MaterialType.PHYSICAL]:
            params.update({
                'metalness': self.metalness,
                'roughness': self.roughness,
                'emissive': self.emissive
            })
        elif self.material_type == MaterialType.PHONG:
            params.update({
                'emissive': self.emissive,
                'shininess': 100 - (self.roughness * 100)
            })
        
        if self.texture_map:
            params['map'] = self.texture_map
        if self.normal_map:
            params['normalMap'] = self.normal_map
        if self.emission_map:
            params['emissiveMap'] = self.emission_map
        
        return params


@dataclass
class LightConfig:
    """Configuration for lighting."""
    light_type: str
    color: str
    intensity: float
    position: Tuple[float, float, float]
    target: Optional[Tuple[float, float, float]]
    distance: float
    decay: float
    
    def to_threejs_params(self) -> Dict[str, Any]:
        """Convert to Three.js light parameters."""
        return {
            'color': self.color,
            'intensity': self.intensity,
            'position': self.position,
            'target': self.target,
            'distance': self.distance,
            'decay': self.decay
        }


class ThreeJSRenderer:
    """
    Three.js rendering system for tower visualization.
    """
    
    def __init__(self):
        self.logger = setup_logger("threejs_renderer")
        self.scene_objects = {}
        self.materials_cache = {}
        self.geometries_cache = {}
        
    def generate_tower_mesh_config(self, tower: TowerArchetype) -> Dict[str, Any]:
        """
        Generate Three.js mesh configuration for a tower.
        
        Args:
            tower: Tower archetype to render
            
        Returns:
            Three.js mesh configuration
        """
        # Generate geometry based on theme and tier
        geometry = self._generate_tower_geometry(tower)
        
        # Generate material based on visual properties
        material = self._generate_tower_material(tower)
        
        # Generate additional components (base, details, etc.)
        components = self._generate_tower_components(tower)
        
        # Generate lighting setup
        lighting = self._generate_tower_lighting(tower)
        
        # Generate particle systems
        particles = self._generate_particle_systems(tower)
        
        return {
            'id': f"tower_{tower.id}",
            'name': tower.name,
            'geometry': geometry.to_threejs_params(),
            'geometryType': geometry.geometry_type.value,
            'material': material.to_threejs_params(),
            'materialType': material.material_type.value,
            'components': components,
            'lighting': [light.to_threejs_params() for light in lighting],
            'particles': particles,
            'scale': tower.visual.size_scale,
            'animations': self._generate_animations(tower),
            'boundingBox': self._calculate_bounding_box(geometry),
            'renderOrder': self._calculate_render_order(tower)
        }
    
    def _generate_tower_geometry(self, tower: TowerArchetype) -> GeometryConfig:
        """Generate geometry configuration for tower."""
        theme = tower.visual.theme
        tier = (tower.id % 10) + 1
        
        # Theme-specific geometry preferences
        if theme == TowerTheme.MEDIEVAL:
            geometry_types = [GeometryType.CYLINDER, GeometryType.BOX, GeometryType.CONE]
        elif theme == TowerTheme.SCI_FI:
            geometry_types = [GeometryType.BOX, GeometryType.OCTAHEDRON, GeometryType.CYLINDER]
        else:  # COSMIC
            geometry_types = [GeometryType.SPHERE, GeometryType.DODECAHEDRON, GeometryType.TORUS]
        
        # Select geometry type based on tier
        geometry_type = geometry_types[min(tier // 4, len(geometry_types) - 1)]
        
        # Calculate dimensions based on tier and theme
        base_size = 1.0 + (tier - 1) * 0.1
        height = 1.5 + (tier - 1) * 0.2
        
        return GeometryConfig(
            geometry_type=geometry_type,
            width=base_size,
            height=height,
            depth=base_size,
            radius_top=base_size * 0.8 if geometry_type == GeometryType.CYLINDER else base_size,
            radius_bottom=base_size,
            radial_segments=max(8, 4 + tier),
            height_segments=max(4, 2 + tier // 2)
        )
    
    def _generate_tower_material(self, tower: TowerArchetype) -> MaterialConfig:
        """Generate material configuration for tower."""
        theme = tower.visual.theme
        tier = (tower.id % 10) + 1
        
        # Theme-specific material preferences
        if theme == TowerTheme.MEDIEVAL:
            material_type = MaterialType.LAMBERT if tier < 5 else MaterialType.PHONG
            metalness = 0.3 + tier * 0.05
            roughness = 0.8 - tier * 0.05
        elif theme == TowerTheme.SCI_FI:
            material_type = MaterialType.STANDARD if tier < 7 else MaterialType.PHYSICAL
            metalness = 0.7 + tier * 0.03
            roughness = 0.2 + tier * 0.02
        else:  # COSMIC
            material_type = MaterialType.STANDARD if tier < 8 else MaterialType.SHADER
            metalness = 0.1 + tier * 0.02
            roughness = 0.1 + tier * 0.01
        
        # Generate texture maps based on theme
        texture_map = self._get_texture_map(theme, tower.visual.texture_pattern)
        normal_map = self._get_normal_map(theme, tier)
        emission_map = self._get_emission_map(theme, tier) if tier >= 6 else None
        
        return MaterialConfig(
            material_type=material_type,
            color=tower.visual.primary_color,
            emissive=tower.visual.accent_color if tower.visual.glow_intensity > 0.5 else '#000000',
            metalness=metalness,
            roughness=roughness,
            opacity=1.0,
            transparent=False,
            texture_map=texture_map,
            normal_map=normal_map,
            emission_map=emission_map
        )
    
    def _generate_tower_components(self, tower: TowerArchetype) -> List[Dict[str, Any]]:
        """Generate additional tower components (base, turret, details)."""
        components = []
        theme = tower.visual.theme
        tier = (tower.id % 10) + 1
        
        # Base component
        base_config = self._generate_base_component(tower)
        components.append(base_config)
        
        # Turret/weapon component for higher tiers
        if tier >= 3:
            turret_config = self._generate_turret_component(tower)
            components.append(turret_config)
        
        # Detail components for high-tier towers
        if tier >= 6:
            detail_configs = self._generate_detail_components(tower)
            components.extend(detail_configs)
        
        # Theme-specific components
        if theme == TowerTheme.MEDIEVAL and tier >= 4:
            components.append(self._generate_medieval_details(tower))
        elif theme == TowerTheme.SCI_FI and tier >= 5:
            components.append(self._generate_scifi_details(tower))
        elif theme == TowerTheme.COSMIC and tier >= 7:
            components.append(self._generate_cosmic_details(tower))
        
        return components
    
    def _generate_base_component(self, tower: TowerArchetype) -> Dict[str, Any]:
        """Generate base component for tower."""
        return {
            'name': 'base',
            'geometry': {
                'type': 'cylinder',
                'radiusTop': 1.2,
                'radiusBottom': 1.4,
                'height': 0.3,
                'radialSegments': 16
            },
            'material': {
                'type': 'lambert',
                'color': tower.visual.secondary_color,
                'opacity': 1.0
            },
            'position': [0, -0.15, 0],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1]
        }
    
    def _generate_turret_component(self, tower: TowerArchetype) -> Dict[str, Any]:
        """Generate turret component for tower."""
        return {
            'name': 'turret',
            'geometry': {
                'type': 'box',
                'width': 0.6,
                'height': 0.4,
                'depth': 1.0
            },
            'material': {
                'type': 'phong',
                'color': tower.visual.primary_color,
                'emissive': tower.visual.accent_color,
                'opacity': 1.0
            },
            'position': [0, 0.8, 0],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1],
            'animations': ['rotate_y']
        }
    
    def _generate_detail_components(self, tower: TowerArchetype) -> List[Dict[str, Any]]:
        """Generate detail components for high-tier towers."""
        details = []
        
        # Energy rings for sci-fi/cosmic themes
        if tower.visual.theme in [TowerTheme.SCI_FI, TowerTheme.COSMIC]:
            for i in range(2):
                ring = {
                    'name': f'energy_ring_{i}',
                    'geometry': {
                        'type': 'torus',
                        'radius': 1.0 + i * 0.3,
                        'tube': 0.05,
                        'radialSegments': 16,
                        'tubularSegments': 32
                    },
                    'material': {
                        'type': 'standard',
                        'color': tower.visual.accent_color,
                        'emissive': tower.visual.accent_color,
                        'opacity': 0.7,
                        'transparent': True
                    },
                    'position': [0, 0.5 + i * 0.3, 0],
                    'rotation': [math.pi / 2, 0, 0],
                    'scale': [1, 1, 1],
                    'animations': ['rotate_z', 'pulse']
                }
                details.append(ring)
        
        return details
    
    def _generate_medieval_details(self, tower: TowerArchetype) -> Dict[str, Any]:
        """Generate medieval-specific details."""
        return {
            'name': 'battlements',
            'geometry': {
                'type': 'box',
                'width': 1.6,
                'height': 0.2,
                'depth': 1.6
            },
            'material': {
                'type': 'lambert',
                'color': tower.visual.secondary_color,
                'opacity': 1.0
            },
            'position': [0, 1.2, 0],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1]
        }
    
    def _generate_scifi_details(self, tower: TowerArchetype) -> Dict[str, Any]:
        """Generate sci-fi specific details."""
        return {
            'name': 'energy_core',
            'geometry': {
                'type': 'sphere',
                'radius': 0.3,
                'widthSegments': 16,
                'heightSegments': 12
            },
            'material': {
                'type': 'standard',
                'color': tower.visual.accent_color,
                'emissive': tower.visual.accent_color,
                'metalness': 0.0,
                'roughness': 0.1,
                'opacity': 0.8,
                'transparent': True
            },
            'position': [0, 0.5, 0],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1],
            'animations': ['pulse', 'rotate_y']
        }
    
    def _generate_cosmic_details(self, tower: TowerArchetype) -> Dict[str, Any]:
        """Generate cosmic-specific details."""
        return {
            'name': 'void_portal',
            'geometry': {
                'type': 'torus',
                'radius': 0.5,
                'tube': 0.1,
                'radialSegments': 8,
                'tubularSegments': 24
            },
            'material': {
                'type': 'shader',
                'uniforms': {
                    'time': 0.0,
                    'color': tower.visual.accent_color,
                    'opacity': 0.9
                },
                'vertexShader': 'cosmic_vertex',
                'fragmentShader': 'cosmic_fragment',
                'transparent': True
            },
            'position': [0, 1.0, 0],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1],
            'animations': ['rotate_x', 'rotate_y', 'cosmic_distortion']
        }
    
    def _generate_tower_lighting(self, tower: TowerArchetype) -> List[LightConfig]:
        """Generate lighting setup for tower."""
        lights = []
        theme = tower.visual.theme
        tier = (tower.id % 10) + 1
        
        # Ambient light
        ambient = LightConfig(
            light_type='ambient',
            color='#404040',
            intensity=0.4,
            position=(0, 0, 0),
            target=None,
            distance=0,
            decay=0
        )
        lights.append(ambient)
        
        # Directional light
        directional = LightConfig(
            light_type='directional',
            color='#ffffff',
            intensity=0.8,
            position=(5, 10, 5),
            target=(0, 0, 0),
            distance=0,
            decay=0
        )
        lights.append(directional)
        
        # Point light for glow effect (high-tier towers)
        if tier >= 5 and tower.visual.glow_intensity > 0.3:
            point = LightConfig(
                light_type='point',
                color=tower.visual.accent_color,
                intensity=tower.visual.glow_intensity * 2.0,
                position=(0, 1.0, 0),
                target=None,
                distance=5.0,
                decay=2.0
            )
            lights.append(point)
        
        # Theme-specific lighting
        if theme == TowerTheme.COSMIC and tier >= 7:
            # Add mystical lighting
            mystical = LightConfig(
                light_type='spot',
                color=tower.visual.accent_color,
                intensity=1.5,
                position=(0, 3.0, 0),
                target=(0, 0, 0),
                distance=8.0,
                decay=1.5
            )
            lights.append(mystical)
        
        return lights
    
    def _generate_particle_systems(self, tower: TowerArchetype) -> List[Dict[str, Any]]:
        """Generate particle system configurations."""
        particle_systems = []
        
        for particle_effect in tower.particles:
            system_config = {
                'name': f"{particle_effect.particle_type.value}_system",
                'type': particle_effect.particle_type.value,
                'count': particle_effect.count,
                'lifetime': particle_effect.lifetime,
                'size': particle_effect.size,
                'color': particle_effect.color,
                'velocity': {
                    'x': particle_effect.velocity[0],
                    'y': particle_effect.velocity[1],
                    'z': particle_effect.velocity[2]
                },
                'gravity': particle_effect.gravity,
                'fadeRate': particle_effect.fade_rate,
                'emissionRate': particle_effect.count / particle_effect.lifetime,
                'position': [0, 1.0, 0],  # Emit from top of tower
                'spread': 0.5,
                'texture': self._get_particle_texture(particle_effect.particle_type),
                'blending': 'additive' if particle_effect.particle_type in [
                    ParticleType.FIRE, ParticleType.LIGHTNING, ParticleType.ENERGY
                ] else 'normal'
            }
            particle_systems.append(system_config)
        
        return particle_systems
    
    def _generate_animations(self, tower: TowerArchetype) -> List[Dict[str, Any]]:
        """Generate animation configurations."""
        animations = []
        theme = tower.visual.theme
        tier = (tower.id % 10) + 1
        
        # Base idle animation
        idle_animation = {
            'name': 'idle',
            'type': 'rotation',
            'axis': 'y',
            'speed': tower.visual.animation_speed * 0.1,
            'amplitude': 0.1,
            'loop': True
        }
        animations.append(idle_animation)
        
        # Glow animation for high-tier towers
        if tier >= 5 and tower.visual.glow_intensity > 0.3:
            glow_animation = {
                'name': 'glow_pulse',
                'type': 'material_property',
                'property': 'emissiveIntensity',
                'speed': tower.visual.animation_speed * 0.5,
                'min_value': 0.5,
                'max_value': 1.5,
                'loop': True
            }
            animations.append(glow_animation)
        
        # Theme-specific animations
        if theme == TowerTheme.SCI_FI:
            tech_animation = {
                'name': 'tech_scan',
                'type': 'shader_uniform',
                'uniform': 'scanLine',
                'speed': tower.visual.animation_speed * 2.0,
                'min_value': 0.0,
                'max_value': 1.0,
                'loop': True
            }
            animations.append(tech_animation)
        elif theme == TowerTheme.COSMIC:
            cosmic_animation = {
                'name': 'cosmic_distortion',
                'type': 'vertex_displacement',
                'speed': tower.visual.animation_speed * 0.3,
                'amplitude': 0.05,
                'frequency': 2.0,
                'loop': True
            }
            animations.append(cosmic_animation)
        
        return animations
    
    def _get_texture_map(self, theme: TowerTheme, pattern: str) -> str:
        """Get texture map path for theme and pattern."""
        texture_maps = {
            TowerTheme.MEDIEVAL: {
                'stone': 'textures/medieval/stone_diffuse.jpg',
                'metal': 'textures/medieval/metal_diffuse.jpg',
                'wood': 'textures/medieval/wood_diffuse.jpg',
                'brick': 'textures/medieval/brick_diffuse.jpg',
                'carved': 'textures/medieval/carved_diffuse.jpg'
            },
            TowerTheme.SCI_FI: {
                'metallic': 'textures/scifi/metallic_diffuse.jpg',
                'carbon_fiber': 'textures/scifi/carbon_fiber_diffuse.jpg',
                'plasma': 'textures/scifi/plasma_diffuse.jpg',
                'holographic': 'textures/scifi/holographic_diffuse.jpg',
                'crystalline': 'textures/scifi/crystalline_diffuse.jpg'
            },
            TowerTheme.COSMIC: {
                'ethereal': 'textures/cosmic/ethereal_diffuse.jpg',
                'starfield': 'textures/cosmic/starfield_diffuse.jpg',
                'nebula': 'textures/cosmic/nebula_diffuse.jpg',
                'void': 'textures/cosmic/void_diffuse.jpg',
                'energy': 'textures/cosmic/energy_diffuse.jpg'
            }
        }
        
        return texture_maps.get(theme, {}).get(pattern, 'textures/default_diffuse.jpg')
    
    def _get_normal_map(self, theme: TowerTheme, tier: int) -> Optional[str]:
        """Get normal map path for theme and tier."""
        if tier < 3:
            return None
        
        normal_maps = {
            TowerTheme.MEDIEVAL: 'textures/medieval/stone_normal.jpg',
            TowerTheme.SCI_FI: 'textures/scifi/metallic_normal.jpg',
            TowerTheme.COSMIC: 'textures/cosmic/ethereal_normal.jpg'
        }
        
        return normal_maps.get(theme)
    
    def _get_emission_map(self, theme: TowerTheme, tier: int) -> Optional[str]:
        """Get emission map path for theme and tier."""
        if tier < 6:
            return None
        
        emission_maps = {
            TowerTheme.MEDIEVAL: 'textures/medieval/runes_emission.jpg',
            TowerTheme.SCI_FI: 'textures/scifi/circuits_emission.jpg',
            TowerTheme.COSMIC: 'textures/cosmic/energy_emission.jpg'
        }
        
        return emission_maps.get(theme)
    
    def _get_particle_texture(self, particle_type: ParticleType) -> str:
        """Get particle texture path."""
        particle_textures = {
            ParticleType.FIRE: 'textures/particles/fire.png',
            ParticleType.ICE: 'textures/particles/ice.png',
            ParticleType.LIGHTNING: 'textures/particles/lightning.png',
            ParticleType.MAGIC: 'textures/particles/magic.png',
            ParticleType.PLASMA: 'textures/particles/plasma.png',
            ParticleType.ENERGY: 'textures/particles/energy.png',
            ParticleType.SMOKE: 'textures/particles/smoke.png',
            ParticleType.SPARKS: 'textures/particles/sparks.png',
            ParticleType.EXPLOSION: 'textures/particles/explosion.png',
            ParticleType.TRAIL: 'textures/particles/trail.png'
        }
        
        return particle_textures.get(particle_type, 'textures/particles/default.png')
    
    def _calculate_bounding_box(self, geometry: GeometryConfig) -> Dict[str, float]:
        """Calculate bounding box for geometry."""
        if geometry.geometry_type == GeometryType.CYLINDER:
            radius = max(geometry.radius_top, geometry.radius_bottom)
            return {
                'min_x': -radius, 'max_x': radius,
                'min_y': 0, 'max_y': geometry.height,
                'min_z': -radius, 'max_z': radius
            }
        elif geometry.geometry_type == GeometryType.BOX:
            return {
                'min_x': -geometry.width/2, 'max_x': geometry.width/2,
                'min_y': 0, 'max_y': geometry.height,
                'min_z': -geometry.depth/2, 'max_z': geometry.depth/2
            }
        elif geometry.geometry_type == GeometryType.SPHERE:
            radius = geometry.radius_bottom
            return {
                'min_x': -radius, 'max_x': radius,
                'min_y': -radius, 'max_y': radius,
                'min_z': -radius, 'max_z': radius
            }
        else:
            # Default bounding box
            return {
                'min_x': -1, 'max_x': 1,
                'min_y': 0, 'max_y': 2,
                'min_z': -1, 'max_z': 1
            }
    
    def _calculate_render_order(self, tower: TowerArchetype) -> int:
        """Calculate render order for tower."""
        # Higher tier towers render later (on top)
        tier = (tower.id % 10) + 1
        base_order = tier * 10
        
        # Transparent/glowing towers render later
        if tower.visual.glow_intensity > 0.5:
            base_order += 100
        
        return base_order
    
    def generate_scene_config(self, towers: List[TowerArchetype]) -> Dict[str, Any]:
        """Generate complete Three.js scene configuration."""
        scene_config = {
            'scene': {
                'background': '#001122',
                'fog': {
                    'type': 'exponential',
                    'color': '#001122',
                    'density': 0.01
                }
            },
            'camera': {
                'type': 'perspective',
                'fov': 75,
                'aspect': 16/9,
                'near': 0.1,
                'far': 1000,
                'position': [10, 10, 10],
                'lookAt': [0, 0, 0]
            },
            'renderer': {
                'antialias': True,
                'shadowMap': {
                    'enabled': True,
                    'type': 'PCFSoftShadowMap'
                },
                'toneMapping': 'ACESFilmicToneMapping',
                'toneMappingExposure': 1.0
            },
            'towers': [self.generate_tower_mesh_config(tower) for tower in towers],
            'environment': {
                'ground': {
                    'geometry': {
                        'type': 'plane',
                        'width': 50,
                        'height': 50
                    },
                    'material': {
                        'type': 'lambert',
                        'color': '#2a4d3a',
                        'opacity': 1.0
                    },
                    'rotation': [-math.pi/2, 0, 0],
                    'receiveShadow': True
                },
                'skybox': {
                    'type': 'cube',
                    'textures': [
                        'textures/skybox/px.jpg',
                        'textures/skybox/nx.jpg',
                        'textures/skybox/py.jpg',
                        'textures/skybox/ny.jpg',
                        'textures/skybox/pz.jpg',
                        'textures/skybox/nz.jpg'
                    ]
                }
            },
            'postProcessing': {
                'bloom': {
                    'enabled': True,
                    'strength': 0.5,
                    'radius': 0.8,
                    'threshold': 0.9
                },
                'fxaa': {
                    'enabled': True
                }
            }
        }
        
        return scene_config
    
    def export_to_json(self, towers: List[TowerArchetype], filename: str):
        """Export tower configurations to JSON file."""
        scene_config = self.generate_scene_config(towers)
        
        with open(filename, 'w') as f:
            json.dump(scene_config, f, indent=2)
        
        self.logger.info(f"Exported {len(towers)} tower configurations to {filename}")


class ShaderLibrary:
    """
    Library of custom shaders for tower rendering.
    """
    
    @staticmethod
    def get_cosmic_vertex_shader() -> str:
        """Get cosmic theme vertex shader."""
        return """
        uniform float time;
        varying vec2 vUv;
        varying vec3 vPosition;
        
        void main() {
            vUv = uv;
            vPosition = position;
            
            vec3 pos = position;
            pos.x += sin(time + position.y * 2.0) * 0.1;
            pos.z += cos(time + position.y * 2.0) * 0.1;
            
            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
        """
    
    @staticmethod
    def get_cosmic_fragment_shader() -> str:
        """Get cosmic theme fragment shader."""
        return """
        uniform float time;
        uniform vec3 color;
        uniform float opacity;
        varying vec2 vUv;
        varying vec3 vPosition;
        
        void main() {
            vec2 uv = vUv;
            
            // Create swirling pattern
            float angle = atan(uv.y - 0.5, uv.x - 0.5);
            float radius = length(uv - 0.5);
            
            float swirl = sin(angle * 3.0 + time + radius * 10.0) * 0.5 + 0.5;
            float pulse = sin(time * 2.0) * 0.3 + 0.7;
            
            vec3 finalColor = color * swirl * pulse;
            float alpha = opacity * (1.0 - radius * 2.0);
            
            gl_FragColor = vec4(finalColor, alpha);
        }
        """
    
    @staticmethod
    def get_energy_vertex_shader() -> str:
        """Get energy effect vertex shader."""
        return """
        uniform float time;
        varying vec2 vUv;
        
        void main() {
            vUv = uv;
            
            vec3 pos = position;
            pos.y += sin(time * 3.0 + position.x * 5.0) * 0.05;
            
            gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
        }
        """
    
    @staticmethod
    def get_energy_fragment_shader() -> str:
        """Get energy effect fragment shader."""
        return """
        uniform float time;
        uniform vec3 color;
        varying vec2 vUv;
        
        void main() {
            vec2 uv = vUv;
            
            float energy = sin(uv.y * 10.0 + time * 5.0) * 0.5 + 0.5;
            energy *= sin(uv.x * 8.0 + time * 3.0) * 0.5 + 0.5;
            
            vec3 finalColor = color * energy;
            float alpha = energy * 0.8;
            
            gl_FragColor = vec4(finalColor, alpha);
        }
        """

