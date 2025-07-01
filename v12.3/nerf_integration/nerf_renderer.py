"""
NeRF Integration Module for RL-LLM System

This module provides Neural Radiance Fields integration for 3D scene rendering
and procedural content generation in the RL training environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import json
from pathlib import Path
import cv2
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class RayBatch:
    """Batch of rays for NeRF rendering."""
    origins: torch.Tensor      # [N, 3] ray origins
    directions: torch.Tensor   # [N, 3] ray directions
    near: torch.Tensor        # [N] near bounds
    far: torch.Tensor         # [N] far bounds
    viewdirs: Optional[torch.Tensor] = None  # [N, 3] view directions


class PositionalEncoding(nn.Module):
    """Positional encoding for NeRF inputs."""
    
    def __init__(self, input_dim: int, max_freq_log2: int = 10, num_freqs: int = None):
        """
        Initialize positional encoding.
        
        Args:
            input_dim: Input dimension (3 for positions, 3 for directions)
            max_freq_log2: Maximum frequency (log2)
            num_freqs: Number of frequency bands
        """
        super().__init__()
        
        if num_freqs is None:
            num_freqs = max_freq_log2
        
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.max_freq = max_freq_log2
        
        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(0.0, max_freq_log2, num_freqs)
        self.register_buffer('freq_bands', freq_bands)
        
        # Output dimension
        self.output_dim = input_dim * (1 + 2 * num_freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply positional encoding.
        
        Args:
            x: Input tensor [..., input_dim]
            
        Returns:
            Encoded tensor [..., output_dim]
        """
        # Original coordinates
        encoded = [x]
        
        # Sinusoidal encodings
        for freq in self.freq_bands:
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        
        return torch.cat(encoded, dim=-1)


class NeRFMLP(nn.Module):
    """NeRF MLP network for density and color prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NeRF MLP.
        
        Args:
            config: Network configuration
        """
        super().__init__()
        
        self.config = config
        self.pos_encoding_dim = config.get('pos_encoding_dim', 60)  # 3 * (1 + 2*10)
        self.dir_encoding_dim = config.get('dir_encoding_dim', 24)  # 3 * (1 + 2*4)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 8)
        self.skip_layer = config.get('skip_layer', 4)
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(3, max_freq_log2=10)
        self.dir_encoder = PositionalEncoding(3, max_freq_log2=4)
        
        # Density network
        density_layers = []
        input_dim = self.pos_encoding_dim
        
        for i in range(self.num_layers):
            if i == self.skip_layer:
                # Skip connection
                input_dim += self.pos_encoding_dim
            
            if i == self.num_layers - 1:
                # Output layer for density
                density_layers.append(nn.Linear(input_dim, self.hidden_dim))
            else:
                density_layers.append(nn.Linear(input_dim, self.hidden_dim))
                density_layers.append(nn.ReLU(inplace=True))
            
            input_dim = self.hidden_dim
        
        self.density_net = nn.ModuleList(density_layers)
        
        # Density output
        self.density_head = nn.Linear(self.hidden_dim, 1)
        
        # Color network
        self.color_net = nn.Sequential(
            nn.Linear(self.hidden_dim + self.dir_encoding_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Sigmoid()
        )
    
    def forward(self, positions: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through NeRF MLP.
        
        Args:
            positions: 3D positions [..., 3]
            directions: View directions [..., 3]
            
        Returns:
            Tuple of (density, color)
        """
        # Encode positions
        pos_encoded = self.pos_encoder(positions)
        dir_encoded = self.dir_encoder(directions)
        
        # Density network
        x = pos_encoded
        skip_input = pos_encoded
        
        for i, layer in enumerate(self.density_net):
            if i == self.skip_layer * 2:  # Account for ReLU layers
                x = torch.cat([x, skip_input], dim=-1)
            x = layer(x)
        
        # Density output
        density = self.density_head(x)
        density = F.relu(density)
        
        # Color network
        color_input = torch.cat([x, dir_encoded], dim=-1)
        color = self.color_net(color_input)
        
        return density, color


class VolumeRenderer:
    """Volume rendering for NeRF."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize volume renderer.
        
        Args:
            config: Rendering configuration
        """
        self.config = config
        self.num_samples = config.get('num_samples', 64)
        self.num_importance_samples = config.get('num_importance_samples', 128)
        self.perturb = config.get('perturb', True)
        self.raw_noise_std = config.get('raw_noise_std', 0.0)
    
    def sample_points_along_rays(self, ray_batch: RayBatch, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample points along rays.
        
        Args:
            ray_batch: Batch of rays
            num_samples: Number of samples per ray
            
        Returns:
            Tuple of (sample_points, sample_distances)
        """
        batch_size = ray_batch.origins.shape[0]
        device = ray_batch.origins.device
        
        # Linear sampling in depth
        t_vals = torch.linspace(0.0, 1.0, num_samples, device=device)
        z_vals = ray_batch.near[..., None] * (1.0 - t_vals) + ray_batch.far[..., None] * t_vals
        
        # Perturb sampling locations
        if self.perturb and self.training:
            # Get intervals between samples
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)
            
            # Uniform samples in those intervals
            t_rand = torch.rand_like(z_vals)
            z_vals = lower + (upper - lower) * t_rand
        
        # Sample points
        sample_points = ray_batch.origins[..., None, :] + ray_batch.directions[..., None, :] * z_vals[..., :, None]
        
        return sample_points, z_vals
    
    def render_rays(self, nerf_model: NeRFMLP, ray_batch: RayBatch) -> Dict[str, torch.Tensor]:
        """
        Render rays using NeRF model.
        
        Args:
            nerf_model: NeRF model
            ray_batch: Batch of rays
            
        Returns:
            Rendering results dictionary
        """
        batch_size = ray_batch.origins.shape[0]
        
        # Coarse sampling
        sample_points, z_vals = self.sample_points_along_rays(ray_batch, self.num_samples)
        
        # Flatten for network evaluation
        points_flat = sample_points.reshape(-1, 3)
        dirs_flat = ray_batch.directions[..., None, :].expand_as(sample_points).reshape(-1, 3)
        
        # Evaluate NeRF
        density_flat, color_flat = nerf_model(points_flat, dirs_flat)
        
        # Reshape back
        density = density_flat.reshape(batch_size, self.num_samples, 1)
        color = color_flat.reshape(batch_size, self.num_samples, 3)
        
        # Volume rendering
        rgb, weights, depth = self.volume_integrate(density, color, z_vals)
        
        results = {
            'rgb': rgb,
            'depth': depth,
            'weights': weights,
            'z_vals': z_vals
        }
        
        # Hierarchical sampling (fine network)
        if self.num_importance_samples > 0:
            fine_results = self._hierarchical_sampling(nerf_model, ray_batch, weights, z_vals)
            results.update(fine_results)
        
        return results
    
    def volume_integrate(self, density: torch.Tensor, color: torch.Tensor, z_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform volume integration.
        
        Args:
            density: Density values [..., num_samples, 1]
            color: Color values [..., num_samples, 3]
            z_vals: Depth values [..., num_samples]
            
        Returns:
            Tuple of (rgb, weights, depth)
        """
        # Add noise to density during training
        if self.training and self.raw_noise_std > 0:
            noise = torch.randn_like(density) * self.raw_noise_std
            density = density + noise
        
        # Compute distances between samples
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Alpha compositing
        alpha = 1.0 - torch.exp(-density[..., 0] * dists)
        
        # Compute weights
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1)
        weights = alpha * transmittance
        
        # Composite color
        rgb = torch.sum(weights[..., None] * color, dim=-2)
        
        # Composite depth
        depth = torch.sum(weights * z_vals, dim=-1)
        
        return rgb, weights, depth
    
    def _hierarchical_sampling(self, nerf_model: NeRFMLP, ray_batch: RayBatch, 
                             weights: torch.Tensor, z_vals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform hierarchical sampling for fine network."""
        # Sample more points based on weights
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = self._sample_pdf(z_vals_mid, weights[..., 1:-1], self.num_importance_samples)
        
        # Combine coarse and fine samples
        z_vals_combined = torch.cat([z_vals, z_samples], dim=-1)
        z_vals_combined, _ = torch.sort(z_vals_combined, dim=-1)
        
        # Sample points at fine locations
        sample_points = ray_batch.origins[..., None, :] + ray_batch.directions[..., None, :] * z_vals_combined[..., :, None]
        
        # Evaluate fine network
        points_flat = sample_points.reshape(-1, 3)
        dirs_flat = ray_batch.directions[..., None, :].expand_as(sample_points).reshape(-1, 3)
        
        density_flat, color_flat = nerf_model(points_flat, dirs_flat)
        
        batch_size = ray_batch.origins.shape[0]
        total_samples = self.num_samples + self.num_importance_samples
        
        density_fine = density_flat.reshape(batch_size, total_samples, 1)
        color_fine = color_flat.reshape(batch_size, total_samples, 3)
        
        # Volume rendering with fine samples
        rgb_fine, weights_fine, depth_fine = self.volume_integrate(density_fine, color_fine, z_vals_combined)
        
        return {
            'rgb_fine': rgb_fine,
            'depth_fine': depth_fine,
            'weights_fine': weights_fine,
            'z_vals_fine': z_vals_combined
        }
    
    def _sample_pdf(self, bins: torch.Tensor, weights: torch.Tensor, num_samples: int) -> torch.Tensor:
        """Sample from PDF defined by weights."""
        # Normalize weights
        weights = weights + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        
        # Uniform samples
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
        
        # Invert CDF
        indices = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(indices - 1, 0, cdf.shape[-1] - 1)
        above = torch.clamp(indices, 0, cdf.shape[-1] - 1)
        
        # Linear interpolation
        bins_below = torch.gather(bins, -1, below)
        bins_above = torch.gather(bins, -1, above)
        cdf_below = torch.gather(cdf, -1, below)
        cdf_above = torch.gather(cdf, -1, above)
        
        denom = cdf_above - cdf_below
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_below) / denom
        
        samples = bins_below + t * (bins_above - bins_below)
        
        return samples


class NeRFRenderer:
    """High-level NeRF renderer for RL environment integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize NeRF renderer.
        
        Args:
            config: Renderer configuration
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize NeRF model
        self.nerf_model = NeRFMLP(config.get('nerf_config', {})).to(self.device)
        
        # Initialize volume renderer
        self.volume_renderer = VolumeRenderer(config.get('render_config', {}))
        
        # Camera parameters
        self.image_width = config.get('image_width', 256)
        self.image_height = config.get('image_height', 256)
        self.focal_length = config.get('focal_length', 200.0)
        
        # Scene bounds
        self.scene_bounds = config.get('scene_bounds', [-2.0, 2.0])
        
        logger.info("Initialized NeRF renderer")
    
    def generate_camera_rays(self, camera_pose: np.ndarray) -> RayBatch:
        """
        Generate camera rays for rendering.
        
        Args:
            camera_pose: Camera pose matrix [4, 4]
            
        Returns:
            RayBatch for rendering
        """
        # Create pixel coordinates
        i, j = np.meshgrid(
            np.arange(self.image_width, dtype=np.float32),
            np.arange(self.image_height, dtype=np.float32),
            indexing='xy'
        )
        
        # Convert to camera coordinates
        dirs = np.stack([
            (i - self.image_width * 0.5) / self.focal_length,
            -(j - self.image_height * 0.5) / self.focal_length,
            -np.ones_like(i)
        ], axis=-1)
        
        # Transform ray directions to world coordinates
        rays_d = np.sum(dirs[..., None, :] * camera_pose[:3, :3], axis=-1)
        rays_o = np.broadcast_to(camera_pose[:3, -1], rays_d.shape)
        
        # Flatten
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        # Normalize directions
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        
        # Convert to tensors
        origins = torch.from_numpy(rays_o).float().to(self.device)
        directions = torch.from_numpy(rays_d).float().to(self.device)
        
        # Set bounds
        near = torch.full((origins.shape[0],), self.scene_bounds[0], device=self.device)
        far = torch.full((origins.shape[0],), self.scene_bounds[1], device=self.device)
        
        return RayBatch(origins=origins, directions=directions, near=near, far=far)
    
    def render_image(self, camera_pose: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
        """
        Render image from camera pose.
        
        Args:
            camera_pose: Camera pose matrix [4, 4]
            chunk_size: Chunk size for batched rendering
            
        Returns:
            Rendered RGB image [H, W, 3]
        """
        ray_batch = self.generate_camera_rays(camera_pose)
        
        # Render in chunks
        all_rgb = []
        num_rays = ray_batch.origins.shape[0]
        
        self.nerf_model.eval()
        with torch.no_grad():
            for i in range(0, num_rays, chunk_size):
                end_i = min(i + chunk_size, num_rays)
                
                chunk_batch = RayBatch(
                    origins=ray_batch.origins[i:end_i],
                    directions=ray_batch.directions[i:end_i],
                    near=ray_batch.near[i:end_i],
                    far=ray_batch.far[i:end_i]
                )
                
                results = self.volume_renderer.render_rays(self.nerf_model, chunk_batch)
                rgb = results.get('rgb_fine', results['rgb'])
                all_rgb.append(rgb)
        
        # Combine chunks
        rgb_image = torch.cat(all_rgb, dim=0)
        rgb_image = rgb_image.reshape(self.image_height, self.image_width, 3)
        
        # Convert to numpy
        rgb_image = rgb_image.cpu().numpy()
        rgb_image = np.clip(rgb_image, 0, 1)
        
        return rgb_image
    
    def render_tower_defense_scene(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Render tower defense scene from game state.
        
        Args:
            game_state: Current game state
            
        Returns:
            Rendered scene image
        """
        # Extract camera parameters from game state
        camera_height = game_state.get('camera_height', 5.0)
        camera_angle = game_state.get('camera_angle', 45.0)
        
        # Create camera pose
        camera_pose = self._create_overhead_camera_pose(camera_height, camera_angle)
        
        # Render scene
        image = self.render_image(camera_pose)
        
        return image
    
    def _create_overhead_camera_pose(self, height: float, angle: float) -> np.ndarray:
        """Create overhead camera pose for tower defense view."""
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Camera position
        camera_pos = np.array([0, height * np.sin(angle_rad), height * np.cos(angle_rad)])
        
        # Look at origin
        target = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Create view matrix
        forward = target - camera_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Camera pose matrix
        pose = np.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward
        pose[:3, 3] = camera_pos
        
        return pose
    
    def save_model(self, filepath: Path):
        """Save NeRF model."""
        torch.save({
            'model_state_dict': self.nerf_model.state_dict(),
            'config': self.config
        }, filepath)
        
        logger.info(f"Saved NeRF model to {filepath}")
    
    def load_model(self, filepath: Path):
        """Load NeRF model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nerf_model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded NeRF model from {filepath}")


def create_nerf_renderer(config: Dict[str, Any]) -> NeRFRenderer:
    """
    Factory function to create NeRF renderer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        NeRFRenderer instance
    """
    return NeRFRenderer(config)

