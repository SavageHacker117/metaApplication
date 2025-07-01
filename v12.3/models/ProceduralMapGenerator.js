import * as THREE from 'three';
import { NeRFRenderer, NeRFAssetType } from './NeRFRenderer_enhanced.js';
import { ProceduralGeometryGenerator } from './ProceduralGeometryGenerator.js';

export class ProceduralMapGenerator {
    /**
     * @param {NeRFRenderer} nerfRenderer - The NeRFRenderer instance to use for loading assets.
     */
    constructor(nerfRenderer) {
        this.nerfRenderer = nerfRenderer;
        this.noise = this._createNoiseFunction(); // Placeholder for a noise function
        console.log('ProceduralMapGenerator initialized.');
    }

    _createNoiseFunction() {
        // In a real application, you'd use a library like 'simplex-noise' or 'perlin-noise'
        // For this example, we'll use a simple Math.random based noise for demonstration.
        // You would typically import and use a proper noise library here.
        // Example: import { createNoise2D } from 'simplex-noise'; this.noise = createNoise2D();
        return (x, y) => Math.random(); // Simple placeholder
    }

    /**
     * Generates a procedural map.
     * @param {object} options - Configuration options for map generation.
     * @param {number} [options.seed] - Seed for reproducible map generation.
     * @param {number} [options.size=100] - Size of the map (width and depth).
     * @param {number} [options.resolution=64] - Number of segments for the terrain plane.
     * @param {number} [options.heightScale=10] - Maximum height variation for terrain.
     * @param {number} [options.featureDensity=0.1] - Density of features (e.g., trees, rocks).
     * @returns {object} An object containing generated terrain and features.
     */
    generateMap(options = {}) {
        const { seed, size = 100, resolution = 64, heightScale = 10, featureDensity = 0.1 } = options;

        // If using a proper noise library, you'd initialize it with the seed here.
        // For Math.random, seed doesn't directly apply without a custom seeded random function.
        if (seed) {
            console.log(`Generating map with seed: ${seed}`);
            // Implement seeded random for reproducibility if using Math.random
        }

        const terrain = this._generateTerrain(size, resolution, heightScale);
        const features = this._generateFeatures(size, featureDensity, terrain.geometry);

        // Determine a plausible start position for the agent
        const startPosition = new THREE.Vector3(0, heightScale / 2 + 1, 0); // Roughly center of map, slightly above terrain

        return {
            terrain: [terrain], // Return as array for consistency with features
            features: features,
            startPosition: startPosition
        };
    }

    _generateTerrain(size, resolution, heightScale) {
        const geometry = new THREE.PlaneGeometry(size, size, resolution, resolution);
        geometry.rotateX(-Math.PI / 2); // Rotate to be flat on XZ plane

        const positionAttribute = geometry.attributes.position;
        for (let i = 0; i < positionAttribute.count; i++) {
            const x = positionAttribute.getX(i);
            const z = positionAttribute.getZ(i);

            // Apply noise to Y coordinate (height)
            // Scale x and z for noise input to control terrain frequency
            const y = this.noise(x * 0.1, z * 0.1) * heightScale; 
            positionAttribute.setY(i, y);
        }
        geometry.computeVertexNormals(); // Recalculate normals after modifying vertices

        const material = new THREE.MeshStandardMaterial({ color: 0x88aa88, flatShading: true });
        const terrainMesh = new THREE.Mesh(geometry, material);
        terrainMesh.receiveShadow = true;
        terrainMesh.castShadow = true; // Terrain can cast shadows
        terrainMesh.name = 'Terrain';

        return terrainMesh;
    }

    _generateFeatures(mapSize, featureDensity, terrainGeometry) {
        const features = [];
        const numFeatures = Math.floor(mapSize * mapSize * featureDensity);

        for (let i = 0; i < numFeatures; i++) {
            const x = (Math.random() - 0.5) * mapSize;
            const z = (Math.random() - 0.5) * mapSize;

            // Find terrain height at this (x, z) position
            // This is a simplified approach; a more robust solution would involve raycasting
            // or sampling the heightmap directly.
            let y = 0;
            // Simple approximation: find the closest vertex height
            let minDistance = Infinity;
            const positionAttribute = terrainGeometry.attributes.position;
            for (let j = 0; j < positionAttribute.count; j++) {
                const vx = positionAttribute.getX(j);
                const vz = positionAttribute.getZ(j);
                const dist = Math.sqrt(Math.pow(x - vx, 2) + Math.pow(z - vz, 2));
                if (dist < minDistance) {
                    minDistance = dist;
                    y = positionAttribute.getY(j);
                }
            }

            // Add a simple tree (cylinder + sphere) or a rock (sphere)
            if (Math.random() < 0.7) { // Tree
                const trunkHeight = Math.random() * 3 + 2;
                const trunkRadius = Math.random() * 0.2 + 0.1;
                const trunk = new THREE.Mesh(
                    new THREE.CylinderGeometry(trunkRadius, trunkRadius, trunkHeight, 8),
                    new THREE.MeshStandardMaterial({ color: 0x8B4513 })
                );
                trunk.position.set(x, y + trunkHeight / 2, z);
                trunk.castShadow = true;
                trunk.receiveShadow = true;
                features.push(trunk);

                const leavesRadius = Math.random() * 1.5 + 0.5;
                const leaves = new THREE.Mesh(
                    new THREE.SphereGeometry(leavesRadius, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0x228B22 })
                );
                leaves.position.set(x, y + trunkHeight + leavesRadius * 0.5, z);
                leaves.castShadow = true;
                leaves.receiveShadow = true;
                features.push(leaves);
            } else { // Rock
                const rockRadius = Math.random() * 1 + 0.5;
                const rock = new THREE.Mesh(
                    new THREE.SphereGeometry(rockRadius, 8, 8),
                    new THREE.MeshStandardMaterial({ color: 0x777777 })
                );
                rock.position.set(x, y + rockRadius * 0.5, z);
                rock.castShadow = true;
                rock.receiveShadow = true;
                features.push(rock);
            }
        }
        return features;
    }

    /**
     * Example of loading a NeRF asset as a map feature.
     * This would typically be called internally by generateFeatures.
     * @param {string} assetPath - Path to the NeRF asset.
     * @param {THREE.Vector3} position - Position to place the asset.
     * @param {number} scale - Scale of the asset.
     * @returns {Promise<THREE.Object3D>}
     */
    async loadNeRFMapFeature(assetPath, position, scale = 1.0) {
        const assetConfig = {
            assetId: `map_feature_${Date.now()}`,
            assetType: NeRFAssetType.MESH, // Or POINT_CLOUD, etc.
            filePath: assetPath,
            targetObject: 'scene_root', // A dummy target, as we'll position it manually
            renderData: {},
            hideOriginal: false
        };

        const success = await this.nerfRenderer.loadNeRFAsset(assetConfig);
        if (success) {
            const loadedObject = this.nerfRenderer.nerfObjects.get(assetConfig.assetId).asset;
            loadedObject.position.copy(position);
            loadedObject.scale.setScalar(scale);
            return loadedObject;
        } else {
            console.error(`Failed to load NeRF map feature from ${assetPath}`);
            return null;
        }
    }
}


