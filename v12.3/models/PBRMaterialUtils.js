import * as THREE from 'three';

export class PBRMaterialUtils {
    /**
     * Creates a Three.js MeshStandardMaterial (PBR material) based on provided configuration.
     * @param {object} config - Configuration object for the material.
     * @param {number} [config.color=0xffffff] - Base color of the material.
     * @param {number} [config.roughness=0.5] - Roughness of the material (0.0 to 1.0).
     * @param {number} [config.metalness=0.0] - Metalness of the material (0.0 to 1.0).
     * @param {string} [config.map] - Path to the albedo (base color) texture.
     * @param {string} [config.normalMap] - Path to the normal map texture.
     * @param {string} [config.aoMap] - Path to the ambient occlusion map texture.
     * @param {number} [config.emissive=0x000000] - Emissive color of the material.
     * @param {number} [config.emissiveIntensity=0.0] - Intensity of the emissive color.
     * @param {THREE.TextureLoader} textureLoader - Three.js TextureLoader instance.
     * @returns {THREE.MeshStandardMaterial}
     */
    static createPBRMaterial(config, textureLoader) {
        const material = new THREE.MeshStandardMaterial({
            color: config.color || 0xffffff,
            roughness: config.roughness !== undefined ? config.roughness : 0.5,
            metalness: config.metalness !== undefined ? config.metalness : 0.0,
            emissive: config.emissive || 0x000000,
            emissiveIntensity: config.emissiveIntensity !== undefined ? config.emissiveIntensity : 0.0,
        });

        if (config.map) {
            material.map = textureLoader.load(config.map);
        }
        if (config.normalMap) {
            material.normalMap = textureLoader.load(config.normalMap);
        }
        if (config.aoMap) {
            material.aoMap = textureLoader.load(config.aoMap);
        }
        // Add other PBR maps (e.g., roughnessMap, metalnessMap, lightMap) as needed

        return material;
    }

    /**
     * Applies a PBR material to an object and its children if they are meshes.
     * @param {THREE.Object3D} object - The object to apply the material to.
     * @param {THREE.MeshStandardMaterial} material - The PBR material to apply.
     */
    static applyPBRMaterialToObject(object, material) {
        object.traverse((child) => {
            if (child.isMesh) {
                child.material = material;
            }
        });
    }

    // Future enhancements could include functions for:
    // - Disentangling diffuse/specular properties based on Disney BRDF principles
    // - Applying spectral data concepts (e.g., converting spectral data to RGB for PBR)
}


