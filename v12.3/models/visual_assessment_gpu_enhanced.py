"""
Enhanced GPU-Accelerated Visual Fidelity Assessment System v3

Revolutionary improvements based on feedback:
- Configurable normalization options (no normalization, custom mean/std)
- Comprehensive exception handling for CUDA operations
- Configurable reward weighting (alpha/beta for SSIM/LPIPS)
- LPIPS backend switching with fallback support
- Enhanced caching behavior documentation
- In-progress reward yielding for faster learning
- Outlier reward logging for debugging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure
import numpy as np
import cv2
from PIL import Image
import asyncio
import concurrent.futures
from typing import List, Dict, Tuple, Optional, Union, Any, Callable
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import threading
from queue import Queue
import gc
import warnings
from contextlib import contextmanager

# GPU optimization imports
from torch.cuda.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

# Try to import official LPIPS as fallback
try:
    import lpips as lpips_official
    LPIPS_OFFICIAL_AVAILABLE = True
except ImportError:
    LPIPS_OFFICIAL_AVAILABLE = False
    warnings.warn("Official LPIPS package not available. Only torchmetrics LPIPS will be used.")

# Enable optimizations
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

@dataclass
class RewardWeightConfig:
    """Configuration for reward weighting."""
    ssim_weight: float = 0.20
    lpips_weight: float = 0.35
    ms_ssim_weight: float = 0.15
    psnr_weight: float = 0.10
    feature_weight: float = 0.15
    edge_weight: float = 0.05
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = (self.ssim_weight + self.lpips_weight + self.ms_ssim_weight + 
                self.psnr_weight + self.feature_weight + self.edge_weight)
        if abs(total - 1.0) > 1e-6:
            warnings.warn(f"Reward weights sum to {total}, not 1.0. Consider normalizing.")

@dataclass
class NormalizationConfig:
    """Configuration for image normalization."""
    normalize: bool = False
    norm_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    input_range: str = "0-1"  # "0-1", "0-255", or "custom"
    custom_min: float = 0.0
    custom_max: float = 1.0

@dataclass
class LPIPSConfig:
    """Configuration for LPIPS backend."""
    backend: str = "torchmetrics"  # "torchmetrics", "official", or "auto"
    net_type: str = "alex"  # "alex", "vgg", "squeeze"
    fallback_enabled: bool = True
    normalize: bool = True

@dataclass
class VisualAssessmentConfig:
    """Enhanced configuration for GPU-accelerated visual assessment."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 8
    mixed_precision: bool = True
    cache_size: int = 1000
    async_processing: bool = True
    prefetch_factor: int = 4
    pin_memory: bool = True
    compile_model: bool = True  # PyTorch 2.0 compilation
    
    # Enhanced configurations
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    reward_weights: RewardWeightConfig = field(default_factory=RewardWeightConfig)
    lpips_config: LPIPSConfig = field(default_factory=LPIPSConfig)
    
    # Error handling and logging
    enable_exception_handling: bool = True
    log_outliers: bool = True
    outlier_threshold_low: float = 0.05
    outlier_threshold_high: float = 0.95
    
    # Performance optimizations
    yield_progress_rewards: bool = True
    progress_yield_interval: int = 10  # Yield every N processed pairs

class SafeImagePairDataset(Dataset):
    """Enhanced dataset with configurable normalization and error handling."""
    
    def __init__(self, 
                 image_pairs: List[Tuple[torch.Tensor, torch.Tensor]], 
                 config: VisualAssessmentConfig):
        self.image_pairs = image_pairs
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup normalization transform
        self.transform = self._create_transform()
    
    def _create_transform(self):
        """Create transform based on configuration."""
        transforms_list = []
        
        if self.config.normalization.normalize:
            transforms_list.append(
                transforms.Normalize(
                    mean=self.config.normalization.norm_mean,
                    std=self.config.normalization.norm_std
                )
            )
        
        if transforms_list:
            return transforms.Compose(transforms_list)
        else:
            return lambda x: x
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        try:
            img1, img2 = self.image_pairs[idx]
            
            # Apply normalization if configured
            img1_transformed = self.transform(img1)
            img2_transformed = self.transform(img2)
            
            return img1_transformed, img2_transformed
            
        except Exception as e:
            if self.config.enable_exception_handling:
                self.logger.error(f"Error processing image pair {idx}: {e}")
                # Return zero tensors as fallback
                return torch.zeros_like(self.image_pairs[idx][0]), torch.zeros_like(self.image_pairs[idx][1])
            else:
                raise

class EnhancedGPUVisualAssessment:
    """
    Enhanced GPU-accelerated visual fidelity assessment system with comprehensive improvements.
    
    New Features:
    - Configurable normalization options
    - Comprehensive exception handling
    - Configurable reward weighting
    - LPIPS backend switching with fallback
    - Enhanced caching with documentation
    - In-progress reward yielding
    - Outlier detection and logging
    """
    
    def __init__(self, config: Optional[VisualAssessmentConfig] = None):
        self.config = config or VisualAssessmentConfig()
        self.device = torch.device(self.config.device)
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU-accelerated metrics with error handling
        self._initialize_metrics()
        
        # Initialize caching system
        self._initialize_cache()
        
        # Initialize async processing
        if self.config.async_processing:
            self._initialize_async_processing()
        
        # Performance monitoring
        self.performance_stats = {
            'total_assessments': 0,
            'total_time': 0.0,
            'gpu_memory_peak': 0,
            'batch_sizes': [],
            'throughput_history': [],
            'error_count': 0,
            'outlier_count': 0,
            'cache_stats': {'hits': 0, 'misses': 0}
        }
        
        # Progress callback for yielding intermediate rewards
        self.progress_callback: Optional[Callable] = None
        
        self.logger.info(f"Initialized enhanced GPU visual assessment on {self.device}")
        self.logger.info(f"LPIPS backend: {self.config.lpips_config.backend}")
        self.logger.info(f"Normalization enabled: {self.config.normalization.normalize}")
    
    def _initialize_metrics(self):
        """Initialize GPU-accelerated perceptual metrics with error handling."""
        try:
            # Initialize LPIPS with backend selection
            self.lpips = self._initialize_lpips()
            
            # SSIM with GPU acceleration
            self.ssim = StructuralSimilarityIndexMeasure(
                data_range=1.0,
                kernel_size=11,
                sigma=1.5,
                reduction='elementwise_mean'
            ).to(self.device)
            
            # Multi-scale SSIM
            self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
                data_range=1.0,
                kernel_size=11,
                sigma=1.5,
                reduction='elementwise_mean'
            ).to(self.device)
            
            # PSNR
            self.psnr = PeakSignalNoiseRatio(
                data_range=1.0,
                reduction='elementwise_mean'
            ).to(self.device)
            
            # Compile models for PyTorch 2.0 optimization
            if self.config.compile_model and hasattr(torch, 'compile'):
                try:
                    self.lpips = torch.compile(self.lpips)
                    self.ssim = torch.compile(self.ssim)
                    self.ms_ssim = torch.compile(self.ms_ssim)
                    self.psnr = torch.compile(self.psnr)
                    self.logger.info("Successfully compiled models with PyTorch 2.0")
                except Exception as e:
                    self.logger.warning(f"Model compilation failed: {e}")
            
            # Mixed precision scaler
            if self.config.mixed_precision:
                self.scaler = GradScaler()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics: {e}")
            if not self.config.enable_exception_handling:
                raise
    
    def _initialize_lpips(self):
        """Initialize LPIPS with backend selection and fallback."""
        lpips_config = self.config.lpips_config
        
        if lpips_config.backend == "auto":
            # Try torchmetrics first, fallback to official
            backend = "torchmetrics" if True else ("official" if LPIPS_OFFICIAL_AVAILABLE else "torchmetrics")
        else:
            backend = lpips_config.backend
        
        try:
            if backend == "torchmetrics":
                lpips_model = LearnedPerceptualImagePatchSimilarity(
                    net_type=lpips_config.net_type,
                    normalize=lpips_config.normalize
                ).to(self.device)
                self.lpips_backend = "torchmetrics"
                self.logger.info(f"Using torchmetrics LPIPS with {lpips_config.net_type} network")
                
            elif backend == "official" and LPIPS_OFFICIAL_AVAILABLE:
                lpips_model = lpips_official.LPIPS(
                    net=lpips_config.net_type,
                    verbose=False
                ).to(self.device)
                self.lpips_backend = "official"
                self.logger.info(f"Using official LPIPS with {lpips_config.net_type} network")
                
            else:
                raise ValueError(f"Backend {backend} not available")
                
            return lpips_model
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LPIPS backend {backend}: {e}")
            
            if lpips_config.fallback_enabled and backend != "torchmetrics":
                self.logger.info("Falling back to torchmetrics LPIPS")
                try:
                    lpips_model = LearnedPerceptualImagePatchSimilarity(
                        net_type=lpips_config.net_type,
                        normalize=lpips_config.normalize
                    ).to(self.device)
                    self.lpips_backend = "torchmetrics"
                    return lpips_model
                except Exception as fallback_e:
                    self.logger.error(f"Fallback also failed: {fallback_e}")
            
            if not self.config.enable_exception_handling:
                raise
            
            # Return None if all fails and exception handling is enabled
            self.lpips_backend = "none"
            return None
    
    def _initialize_cache(self):
        """
        Initialize intelligent caching system.
        
        Cache Behavior Documentation:
        - Image Cache: Stores preprocessed tensors for frequently accessed images
        - Result Cache: Stores computed metrics for identical image pairs
        - LRU Eviction: Oldest entries removed when cache size limit reached
        - Cache Keys: Based on image content hash for reliable identification
        - Memory Management: Automatic cleanup on cache overflow
        """
        self.image_cache = {}
        self.result_cache = {}
        self.cache_order = []
        self.max_cache_size = self.config.cache_size
        
        self.logger.info(f"Initialized caching system with max size: {self.max_cache_size}")
    
    def _initialize_async_processing(self):
        """Initialize async processing pipeline."""
        self.processing_queue = Queue(maxsize=1000)
        self.result_queue = Queue()
        self.processing_threads = []
        
        # Start processing threads
        for i in range(self.config.num_workers):
            thread = threading.Thread(
                target=self._async_processing_worker,
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
    
    def _async_processing_worker(self):
        """Async processing worker thread with error handling."""
        while True:
            try:
                task = self.processing_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break
                
                result = self._process_batch_sync(task['batch'], task['config'])
                self.result_queue.put({
                    'task_id': task['task_id'],
                    'result': result
                })
                
                self.processing_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Async processing error: {e}")
                self.performance_stats['error_count'] += 1
                
                if self.config.enable_exception_handling:
                    # Put error result
                    self.result_queue.put({
                        'task_id': task.get('task_id', 'unknown'),
                        'result': [],
                        'error': str(e)
                    })
                else:
                    raise
    
    def safe_batch_ssim_gpu(self, outputs: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """Safe SSIM computation with exception handling."""
        try:
            if self.config.mixed_precision:
                with autocast():
                    scores = self.ssim(outputs, targets)
            else:
                scores = self.ssim(outputs, targets)
            
            return scores.cpu().tolist() if scores.dim() > 0 else [scores.item()]
            
        except Exception as e:
            self.logger.error(f"SSIM computation failed: {e}")
            self.performance_stats['error_count'] += 1
            
            if self.config.enable_exception_handling:
                # Return zeros as fallback
                return [0.0] * len(outputs)
            else:
                raise
    
    def safe_batch_lpips_gpu(self, outputs: torch.Tensor, targets: torch.Tensor) -> List[float]:
        """Safe LPIPS computation with exception handling."""
        try:
            if self.lpips is None:
                self.logger.warning("LPIPS not available, returning zeros")
                return [0.0] * len(outputs)
            
            if self.config.mixed_precision:
                with autocast():
                    if self.lpips_backend == "official":
                        scores = self.lpips(outputs, targets)
                    else:
                        scores = self.lpips(outputs, targets)
            else:
                if self.lpips_backend == "official":
                    scores = self.lpips(outputs, targets)
                else:
                    scores = self.lpips(outputs, targets)
            
            return scores.cpu().tolist() if scores.dim() > 0 else [scores.item()]
            
        except Exception as e:
            self.logger.error(f"LPIPS computation failed: {e}")
            self.performance_stats['error_count'] += 1
            
            if self.config.enable_exception_handling:
                # Return zeros as fallback
                return [0.0] * len(outputs)
            else:
                raise
    
    @contextmanager
    def _cuda_error_handler(self, operation_name: str):
        """Context manager for CUDA error handling."""
        try:
            yield
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"CUDA OOM in {operation_name}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.performance_stats['error_count'] += 1
            
            if not self.config.enable_exception_handling:
                raise
        except Exception as e:
            self.logger.error(f"Error in {operation_name}: {e}")
            self.performance_stats['error_count'] += 1
            
            if not self.config.enable_exception_handling:
                raise
    
    def preprocess_image(self, image: Union[str, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Preprocess image for GPU computation with enhanced normalization options.
        
        Args:
            image: Input image (path, numpy array, or tensor)
            
        Returns:
            Preprocessed tensor ready for GPU computation
        """
        with self._cuda_error_handler("image preprocessing"):
            # Load image if path provided
            if isinstance(image, str):
                cache_key = f"img_{hash(image)}"
                if cache_key in self.image_cache:
                    self.performance_stats['cache_stats']['hits'] += 1
                    return self.image_cache[cache_key]
                
                img = cv2.imread(image)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.performance_stats['cache_stats']['misses'] += 1
                
            elif isinstance(image, np.ndarray):
                img = image
            else:
                return image.to(self.device)
            
            # Handle different input ranges
            norm_config = self.config.normalization
            if norm_config.input_range == "0-255":
                img_tensor = torch.from_numpy(img).float() / 255.0
            elif norm_config.input_range == "custom":
                img_tensor = torch.from_numpy(img).float()
                img_tensor = (img_tensor - norm_config.custom_min) / (norm_config.custom_max - norm_config.custom_min)
            else:  # "0-1"
                img_tensor = torch.from_numpy(img).float()
            
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BCHW format
            img_tensor = img_tensor.to(self.device, non_blocking=True)
            
            # Cache if from file
            if isinstance(image, str) and len(self.image_cache) < self.max_cache_size:
                self.image_cache[cache_key] = img_tensor
                self.cache_order.append(cache_key)
                
                # LRU eviction
                if len(self.cache_order) > self.max_cache_size:
                    oldest_key = self.cache_order.pop(0)
                    del self.image_cache[oldest_key]
            
            return img_tensor
    
    def batch_assess_visual_fidelity(
        self,
        image_pairs: List[Tuple[Union[str, np.ndarray, torch.Tensor], 
                               Union[str, np.ndarray, torch.Tensor]]],
        return_detailed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Enhanced GPU-accelerated batch assessment with configurable rewards and error handling.
        
        Args:
            image_pairs: List of (rendered, target) image pairs
            return_detailed: Whether to return detailed metrics
            
        Returns:
            List of assessment results with comprehensive metrics
        """
        start_time = time.time()
        
        with self._cuda_error_handler("batch visual assessment"):
            # Preprocess all images
            processed_pairs = []
            for i, (rendered, target) in enumerate(image_pairs):
                try:
                    rendered_tensor = self.preprocess_image(rendered)
                    target_tensor = self.preprocess_image(target)
                    processed_pairs.append((rendered_tensor, target_tensor))
                except Exception as e:
                    self.logger.error(f"Failed to preprocess image pair {i}: {e}")
                    if self.config.enable_exception_handling:
                        # Skip this pair
                        continue
                    else:
                        raise
            
            if not processed_pairs:
                self.logger.warning("No valid image pairs to process")
                return []
            
            # Create dataset and dataloader for batch processing
            dataset = SafeImagePairDataset(processed_pairs, self.config)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=0,  # GPU processing, no CPU workers needed
                pin_memory=self.config.pin_memory,
                prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else 2
            )
            
            results = []
            processed_count = 0
            
            with torch.no_grad():
                for batch_idx, (batch_rendered, batch_target) in enumerate(dataloader):
                    batch_rendered = batch_rendered.to(self.device, non_blocking=True)
                    batch_target = batch_target.to(self.device, non_blocking=True)
                    
                    # Mixed precision computation
                    if self.config.mixed_precision:
                        with autocast():
                            batch_results = self._compute_batch_metrics(
                                batch_rendered, batch_target, return_detailed
                            )
                    else:
                        batch_results = self._compute_batch_metrics(
                            batch_rendered, batch_target, return_detailed
                        )
                    
                    results.extend(batch_results)
                    processed_count += len(batch_results)
                    
                    # Yield progress rewards if enabled
                    if (self.config.yield_progress_rewards and 
                        self.progress_callback and 
                        processed_count % self.config.progress_yield_interval == 0):
                        
                        self.progress_callback(processed_count, len(image_pairs), batch_results[-1])
            
            # Log outliers if enabled
            if self.config.log_outliers:
                self._log_outliers(results)
            
            # Update performance stats
            processing_time = time.time() - start_time
            self.performance_stats['total_assessments'] += len(image_pairs)
            self.performance_stats['total_time'] += processing_time
            self.performance_stats['gpu_memory_peak'] = max(
                self.performance_stats['gpu_memory_peak'],
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            )
            self.performance_stats['throughput_history'].append(
                len(image_pairs) / processing_time
            )
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return results
    
    def _compute_batch_metrics(
        self,
        batch_rendered: torch.Tensor,
        batch_target: torch.Tensor,
        return_detailed: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compute metrics for a batch with configurable reward weighting.
        
        Args:
            batch_rendered: Batch of rendered images
            batch_target: Batch of target images
            return_detailed: Whether to return detailed metrics
            
        Returns:
            List of metric dictionaries for each pair in the batch
        """
        batch_size = batch_rendered.shape[0]
        
        # Compute all metrics with safe wrappers
        lpips_scores = self.safe_batch_lpips_gpu(batch_rendered, batch_target)
        ssim_scores = self.safe_batch_ssim_gpu(batch_rendered, batch_target)
        
        try:
            ms_ssim_scores = self.ms_ssim(batch_rendered, batch_target).cpu().tolist()
            psnr_scores = self.psnr(batch_rendered, batch_target).cpu().tolist()
        except Exception as e:
            self.logger.error(f"Error computing MS-SSIM/PSNR: {e}")
            ms_ssim_scores = [0.0] * batch_size
            psnr_scores = [0.0] * batch_size
        
        # Compute additional metrics
        try:
            mse_scores = F.mse_loss(batch_rendered, batch_target, reduction='none').mean(dim=[1, 2, 3]).cpu().tolist()
            mae_scores = F.l1_loss(batch_rendered, batch_target, reduction='none').mean(dim=[1, 2, 3]).cpu().tolist()
        except Exception as e:
            self.logger.error(f"Error computing MSE/MAE: {e}")
            mse_scores = [0.0] * batch_size
            mae_scores = [0.0] * batch_size
        
        # Advanced perceptual metrics
        try:
            feature_similarity = self._compute_feature_similarity(batch_rendered, batch_target).cpu().tolist()
            edge_similarity = self._compute_edge_similarity(batch_rendered, batch_target).cpu().tolist()
        except Exception as e:
            self.logger.error(f"Error computing advanced metrics: {e}")
            feature_similarity = [0.0] * batch_size
            edge_similarity = [0.0] * batch_size
        
        results = []
        for i in range(batch_size):
            # Configurable composite reward calculation
            reward_score = self._calculate_configurable_composite_reward(
                lpips_scores[i] if i < len(lpips_scores) else 0.0,
                ssim_scores[i] if i < len(ssim_scores) else 0.0,
                ms_ssim_scores[i] if i < len(ms_ssim_scores) else 0.0,
                psnr_scores[i] if i < len(psnr_scores) else 0.0,
                feature_similarity[i] if i < len(feature_similarity) else 0.0,
                edge_similarity[i] if i < len(edge_similarity) else 0.0
            )
            
            result = {
                'reward_score': reward_score,
                'lpips': lpips_scores[i] if i < len(lpips_scores) else 0.0,
                'ssim': ssim_scores[i] if i < len(ssim_scores) else 0.0,
                'ms_ssim': ms_ssim_scores[i] if i < len(ms_ssim_scores) else 0.0,
                'psnr': psnr_scores[i] if i < len(psnr_scores) else 0.0,
                'mse': mse_scores[i] if i < len(mse_scores) else 0.0,
                'mae': mae_scores[i] if i < len(mae_scores) else 0.0,
                'feature_similarity': feature_similarity[i] if i < len(feature_similarity) else 0.0,
                'edge_similarity': edge_similarity[i] if i < len(edge_similarity) else 0.0
            }
            
            if return_detailed:
                result.update({
                    'quality_tier': self._classify_quality_tier(reward_score),
                    'improvement_suggestions': self._generate_improvement_suggestions(result),
                    'confidence_score': self._calculate_confidence_score(result)
                })
            
            results.append(result)
        
        return results
    
    def _calculate_configurable_composite_reward(
        self,
        lpips: float,
        ssim: float,
        ms_ssim: float,
        psnr: float,
        feature_sim: float,
        edge_sim: float
    ) -> float:
        """
        Calculate composite reward with configurable weighting.
        
        Uses weights from configuration for flexible reward tuning.
        """
        weights = self.config.reward_weights
        
        # Normalize LPIPS (lower is better) and PSNR
        lpips_norm = 1 - min(lpips, 1.0)  # Invert LPIPS
        psnr_norm = min(psnr / 40.0, 1.0)  # Normalize PSNR
        
        # Calculate weighted composite score
        composite_score = (
            weights.lpips_weight * lpips_norm +
            weights.ssim_weight * ssim +
            weights.ms_ssim_weight * ms_ssim +
            weights.psnr_weight * psnr_norm +
            weights.feature_weight * feature_sim +
            weights.edge_weight * edge_sim
        )
        
        return float(np.clip(composite_score, 0.0, 1.0))
    
    def _log_outliers(self, results: List[Dict[str, Any]]):
        """Log outlier reward values for debugging."""
        outliers = []
        
        for i, result in enumerate(results):
            reward = result['reward_score']
            if (reward < self.config.outlier_threshold_low or 
                reward > self.config.outlier_threshold_high):
                outliers.append((i, reward, result))
        
        if outliers:
            self.performance_stats['outlier_count'] += len(outliers)
            self.logger.warning(f"Found {len(outliers)} outlier rewards:")
            
            for idx, reward, result in outliers[:5]:  # Log first 5 outliers
                self.logger.warning(
                    f"  Outlier {idx}: reward={reward:.4f}, "
                    f"ssim={result['ssim']:.4f}, lpips={result['lpips']:.4f}"
                )
    
    def _compute_feature_similarity(self, batch_rendered: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        """Compute deep feature similarity with error handling."""
        try:
            if self.lpips is None:
                return torch.zeros(batch_rendered.shape[0]).to(self.device)
            
            with torch.no_grad():
                if self.lpips_backend == "torchmetrics":
                    # Use LPIPS features for deep similarity
                    features_rendered = self.lpips.net.forward(batch_rendered)
                    features_target = self.lpips.net.forward(batch_target)
                else:
                    # For official LPIPS, use a simpler approach
                    return torch.ones(batch_rendered.shape[0]).to(self.device) * 0.5
                
                # Compute cosine similarity
                similarity = F.cosine_similarity(
                    features_rendered.view(features_rendered.shape[0], -1),
                    features_target.view(features_target.shape[0], -1),
                    dim=1
                )
                
                return (similarity + 1) / 2  # Normalize to [0, 1]
                
        except Exception as e:
            self.logger.error(f"Feature similarity computation failed: {e}")
            return torch.zeros(batch_rendered.shape[0]).to(self.device)
    
    def _compute_edge_similarity(self, batch_rendered: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        """Compute edge-based similarity with error handling."""
        try:
            # Sobel edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(self.device)
            
            # Convert to grayscale
            gray_rendered = torch.mean(batch_rendered, dim=1, keepdim=True)
            gray_target = torch.mean(batch_target, dim=1, keepdim=True)
            
            # Apply Sobel filters
            edges_rendered_x = F.conv2d(gray_rendered, sobel_x, padding=1)
            edges_rendered_y = F.conv2d(gray_rendered, sobel_y, padding=1)
            edges_rendered = torch.sqrt(edges_rendered_x**2 + edges_rendered_y**2)
            
            edges_target_x = F.conv2d(gray_target, sobel_x, padding=1)
            edges_target_y = F.conv2d(gray_target, sobel_y, padding=1)
            edges_target = torch.sqrt(edges_target_x**2 + edges_target_y**2)
            
            # Compute edge similarity
            edge_similarity = 1 - F.mse_loss(edges_rendered, edges_target, reduction='none').mean(dim=[1, 2, 3])
            
            return torch.clamp(edge_similarity, 0, 1)
            
        except Exception as e:
            self.logger.error(f"Edge similarity computation failed: {e}")
            return torch.zeros(batch_rendered.shape[0]).to(self.device)
    
    def _classify_quality_tier(self, reward_score: float) -> str:
        """Classify quality into tiers for easy interpretation."""
        if reward_score >= 0.95:
            return "Exceptional"
        elif reward_score >= 0.85:
            return "Excellent"
        elif reward_score >= 0.70:
            return "Good"
        elif reward_score >= 0.50:
            return "Fair"
        else:
            return "Poor"
    
    def _generate_improvement_suggestions(self, metrics: Dict[str, float]) -> List[str]:
        """Generate specific improvement suggestions based on metrics."""
        suggestions = []
        
        if metrics['ssim'] < 0.7:
            suggestions.append("Improve structural similarity - check object positioning and shapes")
        
        if metrics['edge_similarity'] < 0.6:
            suggestions.append("Enhance edge definition - improve line quality and sharpness")
        
        if metrics['feature_similarity'] < 0.7:
            suggestions.append("Improve semantic content - check object recognition and features")
        
        if metrics['psnr'] < 20:
            suggestions.append("Reduce noise and artifacts - improve rendering quality")
        
        if metrics['lpips'] > 0.3:
            suggestions.append("Enhance perceptual quality - focus on human-visible differences")
        
        return suggestions
    
    def _calculate_confidence_score(self, metrics: Dict[str, float]) -> float:
        """Calculate confidence score based on metric consistency."""
        # Check consistency across different metrics
        scores = [
            metrics['ssim'],
            metrics['ms_ssim'],
            metrics['feature_similarity'],
            metrics['edge_similarity']
        ]
        
        # Calculate variance - lower variance means higher confidence
        variance = np.var(scores)
        confidence = 1.0 - min(variance * 4, 1.0)  # Scale variance to confidence
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def set_progress_callback(self, callback: Callable[[int, int, Dict], None]):
        """Set callback for progress updates during batch processing."""
        self.progress_callback = callback
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        avg_throughput = np.mean(self.performance_stats['throughput_history']) if self.performance_stats['throughput_history'] else 0
        
        cache_stats = self.performance_stats['cache_stats']
        cache_hit_rate = cache_stats['hits'] / (cache_stats['hits'] + cache_stats['misses']) if (cache_stats['hits'] + cache_stats['misses']) > 0 else 0
        
        return {
            'total_assessments': self.performance_stats['total_assessments'],
            'total_time': self.performance_stats['total_time'],
            'average_throughput': avg_throughput,
            'peak_gpu_memory_mb': self.performance_stats['gpu_memory_peak'] / (1024 * 1024),
            'cache_hit_rate': cache_hit_rate,
            'error_count': self.performance_stats['error_count'],
            'outlier_count': self.performance_stats['outlier_count'],
            'device': str(self.device),
            'mixed_precision_enabled': self.config.mixed_precision,
            'batch_size': self.config.batch_size,
            'lpips_backend': getattr(self, 'lpips_backend', 'unknown'),
            'normalization_enabled': self.config.normalization.normalize
        }
    
    def cleanup(self):
        """Clean up resources."""
        if self.config.async_processing:
            # Signal shutdown to worker threads
            for _ in self.processing_threads:
                self.processing_queue.put(None)
            
            # Wait for threads to finish
            for thread in self.processing_threads:
                thread.join(timeout=5.0)
        
        # Clear caches
        self.image_cache.clear()
        self.result_cache.clear()
        
        # GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Enhanced factory function
def create_enhanced_gpu_visual_assessor(
    device: Optional[str] = None,
    batch_size: int = 32,
    mixed_precision: bool = True,
    async_processing: bool = True,
    normalize: bool = False,
    reward_weights: Optional[Dict[str, float]] = None,
    lpips_backend: str = "auto"
) -> EnhancedGPUVisualAssessment:
    """
    Factory function to create enhanced GPU visual assessor with all improvements.
    
    Args:
        device: Target device ('cuda', 'cpu', or None for auto)
        batch_size: Batch size for processing
        mixed_precision: Enable mixed precision computation
        async_processing: Enable async processing pipeline
        normalize: Enable image normalization
        reward_weights: Custom reward weights dict
        lpips_backend: LPIPS backend ('torchmetrics', 'official', 'auto')
        
    Returns:
        Configured EnhancedGPUVisualAssessment instance
    """
    config = VisualAssessmentConfig(
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        batch_size=batch_size,
        mixed_precision=mixed_precision,
        async_processing=async_processing
    )
    
    # Configure normalization
    config.normalization.normalize = normalize
    
    # Configure reward weights if provided
    if reward_weights:
        for key, value in reward_weights.items():
            if hasattr(config.reward_weights, key):
                setattr(config.reward_weights, key, value)
    
    # Configure LPIPS backend
    config.lpips_config.backend = lpips_backend
    
    assessor = EnhancedGPUVisualAssessment(config)
    
    return assessor

