/**
 * Three.js NeRF Rendering Integration
 * 
 * This module provides Three.js integration for rendering NeRF assets
 * in real-time within the tower defense game environment.
 * 
 * Features:
 * - NeRF mesh loading and rendering
 * - Point cloud NeRF support
 * - Texture atlas integration
 * - Performance optimization
 * - Dynamic quality adjustment
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';

/**
 * NeRF Asset Types supported by the renderer
 */
export const NeRFAssetType = {
    MESH: 'mesh',
    POINT_CLOUD: 'point_cloud',
    TEXTURE_ATLAS: 'texture_atlas',
    VOLUMETRIC: 'volumetric',
    ENVIRONMENT: 'environment'
};

/**
 * Quality levels for NeRF rendering
 */
export const NeRFQuality = {
    LOW: 'low',
    MEDIUM: 'medium',
    HIGH: 'high',
    ULTRA: 'ultra'
};

/**
 * NeRF Renderer Configuration
 */
export class NeRFRendererConfig {
    constructor(options = {}) {
        this.enableCaching = options.enableCaching !== false;
        this.cacheSize = options.cacheSize || 50;
        this.enableLOD = options.enableLOD !== false;
        this.maxConcurrentLoads = options.maxConcurrentLoads || 4;
        this.enablePerformanceMonitoring = options.enablePerformanceMonitoring !== false;
        this.defaultQuality = options.defaultQuality || NeRFQuality.MEDIUM;
        this.enableDynamicQuality = options.enableDynamicQuality !== false;
        this.targetFPS = options.targetFPS || 60;
        this.enableCompression = options.enableCompression !== false;
    }
}

/**
 * NeRF Asset Cache for efficient loading and reuse
 */
export class NeRFAssetCache {
    constructor(maxSize = 50) {
        this.cache = new Map();
        this.maxSize = maxSize;
        this.accessOrder = [];
    }

    get(key) {
        if (this.cache.has(key)) {
            // Update access order (LRU)
            this.accessOrder = this.accessOrder.filter(k => k !== key);
            this.accessOrder.push(key);
            return this.cache.get(key);
        }
        return null;
    }

    set(key, value) {
        if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
            // Remove least recently used item
            const lruKey = this.accessOrder.shift();
            const lruValue = this.cache.get(lruKey);
            
            // Dispose of Three.js resources
            if (lruValue && lruValue.dispose) {
                lruValue.dispose();
            }
            
            this.cache.delete(lruKey);
        }

        this.cache.set(key, value);
        this.accessOrder = this.accessOrder.filter(k => k !== key);
        this.accessOrder.push(key);
    }

    clear() {
        // Dispose of all cached resources
        for (const [key, value] of this.cache) {
            if (value && value.dispose) {
                value.dispose();
            }
        }
        this.cache.clear();
        this.accessOrder = [];
    }

    getStats() {
        return {
            size: this.cache.size,
            maxSize: this.maxSize,
            hitRate: this.hitCount / Math.max(1, this.hitCount + this.missCount),
            memoryUsage: this._estimateMemoryUsage()
        };
    }

    _estimateMemoryUsage() {
        // Rough estimation of memory usage
        let totalSize = 0;
        for (const [key, value] of this.cache) {
            if (value && value.geometry) {
                totalSize += value.geometry.attributes.position.count * 12; // 3 floats * 4 bytes
            }
        }
        return totalSize;
    }
}

/**
 * Performance Monitor for NeRF rendering
 */
export class NeRFPerformanceMonitor {
    constructor() {
        this.metrics = {
            loadTimes: [],
            renderTimes: [],
            memoryUsage: [],
            frameRates: []
        };
        this.isMonitoring = false;
        this.lastFrameTime = performance.now();
    }

    startMonitoring() {
        this.isMonitoring = true;
        this._monitorLoop();
    }

    stopMonitoring() {
        this.isMonitoring = false;
    }

    recordLoadTime(assetId, loadTime) {
        this.metrics.loadTimes.push({
            assetId,
            loadTime,
            timestamp: Date.now()
        });
        
        // Keep only recent data
        if (this.metrics.loadTimes.length > 100) {
            this.metrics.loadTimes = this.metrics.loadTimes.slice(-100);
        }
    }

    recordRenderTime(renderTime) {
        this.metrics.renderTimes.push({
            renderTime,
            timestamp: Date.now()
        });
        
        if (this.metrics.renderTimes.length > 100) {
            this.metrics.renderTimes = this.metrics.renderTimes.slice(-100);
        }
    }

    _monitorLoop() {
        if (!this.isMonitoring) return;

        const now = performance.now();
        const frameTime = now - this.lastFrameTime;
        const fps = 1000 / frameTime;
        
        this.metrics.frameRates.push({
            fps,
            timestamp: Date.now()
        });
        
        if (this.metrics.frameRates.length > 100) {
            this.metrics.frameRates = this.metrics.frameRates.slice(-100);
        }
        
        this.lastFrameTime = now;
        
        // Monitor memory usage if available
        if (performance.memory) {
            this.metrics.memoryUsage.push({
                used: performance.memory.usedJSHeapSize,
                total: performance.memory.totalJSHeapSize,
                timestamp: Date.now()
            });
            
            if (this.metrics.memoryUsage.length > 100) {
                this.metrics.memoryUsage = this.metrics.memoryUsage.slice(-100);
            }
        }

        requestAnimationFrame(() => this._monitorLoop());
    }

    getAverageMetrics() {
        return {
            averageLoadTime: this._average(this.metrics.loadTimes.map(m => m.loadTime)),
            averageRenderTime: this._average(this.metrics.renderTimes.map(m => m.renderTime)),
            averageFPS: this._average(this.metrics.frameRates.map(m => m.fps)),
            currentMemoryUsage: this.metrics.memoryUsage.length > 0 ? 
                this.metrics.memoryUsage[this.metrics.memoryUsage.length - 1].used : 0
        };
    }

    _average(array) {
        return array.length > 0 ? array.reduce((a, b) => a + b, 0) / array.length : 0;
    }
}

/**
 * Main NeRF Renderer for Three.js integration
 */
export class NeRFRenderer {
    constructor(scene, config = new NeRFRendererConfig()) {
        this.scene = scene;
        this.config = config;
        
        // Initialize loaders
        this.loaders = this._initializeLoaders();
        
        // Initialize cache
        this.cache = new NeRFAssetCache(config.cacheSize);
        
        // Initialize performance monitor
        this.performanceMonitor = config.enablePerformanceMonitoring ? 
            new NeRFPerformanceMonitor() : null;
        
        // Loading queue
        this.loadingQueue = [];
        this.activeLoads = 0;
        
        // Quality management
        this.currentQuality = config.defaultQuality;
        this.qualityAdjustmentEnabled = config.enableDynamicQuality;
        
        // NeRF objects in scene
        this.nerfObjects = new Map();
        
        console.log('NeRF Renderer initialized with config:', config);
    }

    _initializeLoaders() {
        const loaders = {};
        
        // GLTF/GLB Loader with Draco compression support
        loaders.gltf = new GLTFLoader();
        if (this.config.enableCompression) {
            const dracoLoader = new DRACOLoader();
            dracoLoader.setDecoderPath('/draco/');
            loaders.gltf.setDRACOLoader(dracoLoader);
        }
        
        // OBJ Loader
        loaders.obj = new OBJLoader();
        
        // PLY Loader for point clouds
        loaders.ply = new PLYLoader();
        
        // Texture Loader
        loaders.texture = new THREE.TextureLoader();
        
        return loaders;
    }

    /**
     * Load and apply NeRF asset to a target object
     */
    async loadNeRFAsset(assetConfig) {
        const startTime = performance.now();
        
        try {
            // Check cache first
            const cacheKey = this._generateCacheKey(assetConfig);
            let nerfAsset = this.cache.get(cacheKey);
            
            if (!nerfAsset) {
                // Load asset
                nerfAsset = await this._loadAssetByType(assetConfig);
                
                if (nerfAsset && this.config.enableCaching) {
                    this.cache.set(cacheKey, nerfAsset);
                }
            }
            
            if (nerfAsset) {
                // Apply to target object
                await this._applyNeRFAsset(assetConfig.targetObject, nerfAsset, assetConfig);
                
                // Record performance
                if (this.performanceMonitor) {
                    const loadTime = performance.now() - startTime;
                    this.performanceMonitor.recordLoadTime(assetConfig.assetId, loadTime);
                }
                
                return true;
            }
            
            return false;
            
        } catch (error) {
            console.error('Failed to load NeRF asset:', error);
            return false;
        }
    }

    async _loadAssetByType(assetConfig) {
        const { assetType, filePath, renderData } = assetConfig;
        
        switch (assetType) {
            case NeRFAssetType.MESH:
                return await this._loadMeshAsset(filePath, renderData);
            
            case NeRFAssetType.POINT_CLOUD:
                return await this._loadPointCloudAsset(filePath, renderData);
            
            case NeRFAssetType.TEXTURE_ATLAS:
                return await this._loadTextureAtlasAsset(filePath, renderData);
            
            case NeRFAssetType.ENVIRONMENT:
                return await this._loadEnvironmentAsset(filePath, renderData);
            
            default:
                throw new Error(`Unsupported NeRF asset type: ${assetType}`);
        }
    }

    async _loadMeshAsset(filePath, renderData) {
        const fileExtension = filePath.split('.').pop().toLowerCase();
        let loader;
        
        switch (fileExtension) {
            case 'glb':
            case 'gltf':
                loader = this.loaders.gltf;
                break;
            case 'obj':
                loader = this.loaders.obj;
                break;
            default:
                throw new Error(`Unsupported mesh format: ${fileExtension}`);
        }
        
        return new Promise((resolve, reject) => {
            loader.load(
                filePath,
                (result) => {
                    let mesh;
                    
                    if (result.scene) {
                        // GLTF result
                        mesh = result.scene;
                    } else {
                        // OBJ result
                        mesh = result;
                    }
                    
                    // Apply quality settings
                    this._applyQualitySettings(mesh, renderData.quality);
                    
                    // Apply lighting settings
                    if (renderData.lighting) {
                        this._applyLightingSettings(mesh, renderData.lighting);
                    }
                    
                    resolve(mesh);
                },
                (progress) => {
                    // Loading progress
                    console.log('Loading progress:', progress);
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }

    async _loadPointCloudAsset(filePath, renderData) {
        return new Promise((resolve, reject) => {
            this.loaders.ply.load(
                filePath,
                (geometry) => {
                    // Create point cloud material
                    const material = new THREE.PointsMaterial({
                        size: renderData.pointSize || 2.0,
                        vertexColors: true,
                        sizeAttenuation: true
                    });
                    
                    // Apply quality settings
                    this._applyPointCloudQuality(material, renderData.quality);
                    
                    // Create points object
                    const points = new THREE.Points(geometry, material);
                    
                    resolve(points);
                },
                (progress) => {
                    console.log('Loading progress:', progress);
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }

    async _loadTextureAtlasAsset(filePath, renderData) {
        return new Promise((resolve, reject) => {
            this.loaders.texture.load(
                filePath,
                (texture) => {
                    // Configure texture settings
                    texture.wrapS = THREE.RepeatWrapping;
                    texture.wrapT = THREE.RepeatWrapping;
                    texture.magFilter = THREE.LinearFilter;
                    texture.minFilter = THREE.LinearMipmapLinearFilter;
                    
                    // Apply quality settings
                    this._applyTextureQuality(texture, renderData.quality);
                    
                    resolve(texture);
                },
                (progress) => {
                    console.log('Loading progress:', progress);
                },
                (error) => {
                    reject(error);
                }
            );
        });
    }

    async _loadEnvironmentAsset(filePath, renderData) {
        // Load 360-degree environment texture
        return new Promise((resolve, reject) => {
            this.loaders.texture.load(
                filePath,
                (texture) => {
                    texture.mapping = THREE.EquirectangularReflectionMapping;
                    
                    // Create environment sphere
                    const geometry = new THREE.SphereGeometry(1000, 64, 32);
                    const material = new THREE.MeshBasicMaterial({
                        map: texture,
                        side: THREE.BackSide
                    });
                    
                    const environmentSphere = new THREE.Mesh(geometry, material);
                    
                    resolve(environmentSphere);
                },
                undefined,
                reject
            );
        });
    }

    async _applyNeRFAsset(targetObjectId, nerfAsset, assetConfig) {
        // Find target object in scene
        const targetObject = this.scene.getObjectByName(targetObjectId);
        
        if (!targetObject) {
            console.warn(`Target object not found: ${targetObjectId}`);
            return;
        }
        
        // Apply asset based on type
        switch (assetConfig.assetType) {
            case NeRFAssetType.MESH:
                this._applyMeshToObject(targetObject, nerfAsset, assetConfig);
                break;
                
            case NeRFAssetType.POINT_CLOUD:
                this._applyPointCloudToObject(targetObject, nerfAsset, assetConfig);
                break;
                
            case NeRFAssetType.TEXTURE_ATLAS:
                this._applyTextureToObject(targetObject, nerfAsset, assetConfig);
                break;
                
            case NeRFAssetType.ENVIRONMENT:
                this._applyEnvironmentToScene(nerfAsset, assetConfig);
                break;
        }
        
        // Store reference for management
        this.nerfObjects.set(targetObjectId, {
            asset: nerfAsset,
            config: assetConfig,
            appliedAt: Date.now()
        });
    }

    _applyMeshToObject(targetObject, nerfMesh, config) {
        // Clone the NeRF mesh to avoid modifying the cached version
        const meshClone = nerfMesh.clone();
        
        // Scale and position to match target object
        const targetBox = new THREE.Box3().setFromObject(targetObject);
        const meshBox = new THREE.Box3().setFromObject(meshClone);
        
        const targetSize = targetBox.getSize(new THREE.Vector3());
        const meshSize = meshBox.getSize(new THREE.Vector3());
        
        const scale = Math.min(
            targetSize.x / meshSize.x,
            targetSize.y / meshSize.y,
            targetSize.z / meshSize.z
        );
        
        meshClone.scale.setScalar(scale);
        meshClone.position.copy(targetObject.position);
        
        // Add to scene
        this.scene.add(meshClone);
        
        // Optionally hide original object
        if (config.hideOriginal) {
            targetObject.visible = false;
        }
    }

    _applyPointCloudToObject(targetObject, pointCloud, config) {
        // Position point cloud at target object location
        pointCloud.position.copy(targetObject.position);
        
        // Scale to match target object
        const targetBox = new THREE.Box3().setFromObject(targetObject);
        const targetSize = targetBox.getSize(new THREE.Vector3());
        const avgSize = (targetSize.x + targetSize.y + targetSize.z) / 3;
        
        pointCloud.scale.setScalar(avgSize / 10); // Adjust scale factor as needed
        
        this.scene.add(pointCloud);
        
        if (config.hideOriginal) {
            targetObject.visible = false;
        }
    }

    _applyTextureToObject(targetObject, texture, config) {
        // Apply texture to target object's material
        targetObject.traverse((child) => {
            if (child.isMesh && child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(material => {
                        material.map = texture;
                        material.needsUpdate = true;
                    });
                } else {
                    child.material.map = texture;
                    child.material.needsUpdate = true;
                }
            }
        });
    }

    _applyEnvironmentToScene(environmentSphere, config) {
        // Add environment sphere to scene
        this.scene.add(environmentSphere);
        
        // Optionally set as scene background
        if (config.setAsBackground) {
            this.scene.background = environmentSphere.material.map;
        }
    }

    _applyQualitySettings(object, quality) {
        const qualitySettings = this._getQualitySettings(quality);
        
        object.traverse((child) => {
            if (child.isMesh) {
                // Apply LOD if enabled
                if (this.config.enableLOD && qualitySettings.enableLOD) {
                    this._applyLOD(child, qualitySettings);
                }
                
                // Apply material quality settings
                if (child.material) {
                    this._applyMaterialQuality(child.material, qualitySettings);
                }
            }
        });
    }

    _applyPointCloudQuality(material, quality) {
        const qualitySettings = this._getQualitySettings(quality);
        
        material.size = qualitySettings.pointSize;
        material.sizeAttenuation = qualitySettings.sizeAttenuation;
    }

    _applyTextureQuality(texture, quality) {
        const qualitySettings = this._getQualitySettings(quality);
        
        // Set texture resolution based on quality
        if (qualitySettings.maxTextureSize) {
            texture.image.width = Math.min(texture.image.width, qualitySettings.maxTextureSize);
            texture.image.height = Math.min(texture.image.height, qualitySettings.maxTextureSize);
        }
        
        // Set filtering based on quality
        texture.magFilter = qualitySettings.magFilter;
        texture.minFilter = qualitySettings.minFilter;
        texture.generateMipmaps = qualitySettings.generateMipmaps;
    }

    _getQualitySettings(quality) {
        const settings = {
            [NeRFQuality.LOW]: {
                enableLOD: true,
                lodLevels: 3,
                maxTextureSize: 512,
                pointSize: 1.0,
                sizeAttenuation: false,
                magFilter: THREE.LinearFilter,
                minFilter: THREE.LinearFilter,
                generateMipmaps: false
            },
            [NeRFQuality.MEDIUM]: {
                enableLOD: true,
                lodLevels: 2,
                maxTextureSize: 1024,
                pointSize: 2.0,
                sizeAttenuation: true,
                magFilter: THREE.LinearFilter,
                minFilter: THREE.LinearMipmapLinearFilter,
                generateMipmaps: true
            },
            [NeRFQuality.HIGH]: {
                enableLOD: false,
                lodLevels: 1,
                maxTextureSize: 2048,
                pointSize: 3.0,
                sizeAttenuation: true,
                magFilter: THREE.LinearFilter,
                minFilter: THREE.LinearMipmapLinearFilter,
                generateMipmaps: true
            },
            [NeRFQuality.ULTRA]: {
                enableLOD: false,
                lodLevels: 1,
                maxTextureSize: 4096,
                pointSize: 4.0,
                sizeAttenuation: true,
                magFilter: THREE.LinearFilter,
                minFilter: THREE.LinearMipmapLinearFilter,
                generateMipmaps: true
            }
        };
        
        return settings[quality] || settings[NeRFQuality.MEDIUM];
    }

    _generateCacheKey(assetConfig) {
        return `${assetConfig.assetId}_${assetConfig.quality}_${assetConfig.targetObject}`;
    }

    /**
     * Remove NeRF asset from target object
     */
    removeNeRFAsset(targetObjectId) {
        const nerfData = this.nerfObjects.get(targetObjectId);
        
        if (nerfData) {
            // Remove from scene
            this.scene.remove(nerfData.asset);
            
            // Dispose of resources
            if (nerfData.asset.dispose) {
                nerfData.asset.dispose();
            }
            
            // Show original object if it was hidden
            const targetObject = this.scene.getObjectByName(targetObjectId);
            if (targetObject && nerfData.config.hideOriginal) {
                targetObject.visible = true;
            }
            
            this.nerfObjects.delete(targetObjectId);
            
            return true;
        }
        
        return false;
    }

    /**
     * Update quality for all NeRF assets
     */
    updateQuality(newQuality) {
        this.currentQuality = newQuality;
        
        // Update all active NeRF objects
        for (const [targetObjectId, nerfData] of this.nerfObjects) {
            if (nerfData.config.quality !== newQuality) {
                // Reload with new quality
                const newConfig = { ...nerfData.config, quality: newQuality };
                this.loadNeRFAsset(newConfig);
            }
        }
    }

    /**
     * Get performance statistics
     */
    getPerformanceStats() {
        const stats = {
            cacheStats: this.cache.getStats(),
            activeAssets: this.nerfObjects.size,
            currentQuality: this.currentQuality
        };
        
        if (this.performanceMonitor) {
            stats.performanceMetrics = this.performanceMonitor.getAverageMetrics();
        }
        
        return stats;
    }

    /**
     * Cleanup resources
     */
    dispose() {
        // Clear cache
        this.cache.clear();
        
        // Remove all NeRF objects
        for (const targetObjectId of this.nerfObjects.keys()) {
            this.removeNeRFAsset(targetObjectId);
        }
        
        // Stop performance monitoring
        if (this.performanceMonitor) {
            this.performanceMonitor.stopMonitoring();
        }
        
        console.log('NeRF Renderer disposed');
    }
}

/**
 * Utility function to create NeRF renderer with default configuration
 */
export function createNeRFRenderer(scene, options = {}) {
    const config = new NeRFRendererConfig(options);
    return new NeRFRenderer(scene, config);
}

/**
 * Example usage:
 * 
 * const nerfRenderer = createNeRFRenderer(scene, {
 *     enableCaching: true,
 *     cacheSize: 50,
 *     defaultQuality: NeRFQuality.HIGH
 * });
 * 
 * await nerfRenderer.loadNeRFAsset({
 *     assetId: 'castle_wall_001',
 *     assetType: NeRFAssetType.MESH,
 *     filePath: '/assets/nerf/castle_wall.glb',
 *     targetObject: 'wall_01',
 *     quality: NeRFQuality.HIGH,
 *     hideOriginal: true
 * });
 */

