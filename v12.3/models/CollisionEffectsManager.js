import * as THREE from 'three';
import { NeRFParticleShaderEffects } from './NeRFParticleShaderEffects.js';
import { PBRMaterialUtils } from './PBRMaterialUtils.js';

export class CollisionEffectsManager {
    /**
     * @param {THREE.Scene} scene - The Three.js scene to add effects to.
     * @param {THREE.TextureLoader} textureLoader - Three.js TextureLoader instance for material changes.
     */
    constructor(scene, textureLoader) {
        this.scene = scene;
        this.textureLoader = textureLoader;
        this.activeEffects = [];
        console.log('CollisionEffectsManager initialized.');
    }

    /**
     * Triggers visual effects for a collision.
     * @param {object} collisionData - Data about the collision.
     * @param {THREE.Vector3} collisionData.point - The world position of the collision.
     * @param {THREE.Object3D} collisionData.objectA - The first colliding object.
     * @param {THREE.Object3D} collisionData.objectB - The second colliding object.
     * @param {number} [collisionData.impulse=1.0] - The impulse of the collision, influencing effect intensity.
     * @param {string} [collisionData.type='default'] - Type of collision (e.g., 'metal', 'wood', 'energy').
     */
    triggerCollisionEffect(collisionData) {
        const { point, objectA, objectB, impulse = 1.0, type = 'default' } = collisionData;

        // 1. Generate sparks/particles
        this._createSparks(point, impulse, type);

        // 2. Apply temporary glow/material change to colliding objects
        this._applyTemporaryMaterialEffect(objectA, impulse, type);
        this._applyTemporaryMaterialEffect(objectB, impulse, type);

        // Future: Add sound effects, camera shake, etc.
    }

    _createSparks(position, impulse, type) {
        const sparkCount = Math.floor(impulse * 50) + 10; // More sparks for higher impulse
        const sparkGeometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];

        const sparkColor = new THREE.Color();
        if (type === 'metal') sparkColor.set(0xffa500); // Orange for metal sparks
        else if (type === 'energy') sparkColor.set(0x00ffff); // Cyan for energy sparks
        else sparkColor.set(0xffffff); // White for default

        for (let i = 0; i < sparkCount; i++) {
            // Random position around the collision point
            positions.push(position.x + (Math.random() - 0.5) * 0.5);
            positions.push(position.y + (Math.random() - 0.5) * 0.5);
            positions.push(position.z + (Math.random() - 0.5) * 0.5);

            // Random color variation
            colors.push(sparkColor.r + (Math.random() - 0.5) * 0.2);
            colors.push(sparkColor.g + (Math.random() - 0.5) * 0.2);
            colors.push(sparkColor.b + (Math.random() - 0.5) * 0.2);

            sizes.push(Math.random() * 0.5 + 0.1); // Random size
        }

        sparkGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        sparkGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        sparkGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        const sparkMaterial = NeRFParticleShaderEffects.createGlowingParticleMaterial(sparkColor, impulse * 2); // Reuse glowing particle shader
        const sparks = new THREE.Points(sparkGeometry, sparkMaterial);
        sparks.userData.lifetime = 0; // Custom property for managing effect lifetime
        sparks.userData.maxLifetime = 0.5 + impulse * 0.5; // Sparks last longer with higher impulse

        this.scene.add(sparks);
        this.activeEffects.push(sparks);
    }

    _applyTemporaryMaterialEffect(object, impulse, type) {
        if (!object || !object.isMesh) return;

        const originalMaterial = object.material;
        let tempMaterial;

        if (type === 'energy') {
            tempMaterial = new THREE.MeshStandardMaterial({
                color: 0x00ffff, // Cyan glow
                emissive: 0x00ffff,
                emissiveIntensity: impulse * 2,
                roughness: 0.5,
                metalness: 0.0
            });
        } else {
            tempMaterial = new THREE.MeshStandardMaterial({
                color: 0xff0000, // Red flash
                emissive: 0xff0000,
                emissiveIntensity: impulse * 1.5,
                roughness: 0.5,
                metalness: 0.0
            });
        }

        object.material = tempMaterial;

        // Revert material after a short delay
        const revertDelay = 0.1 + impulse * 0.1; // Shorter delay for weaker impulses
        setTimeout(() => {
            if (object.material === tempMaterial) {
                object.material = originalMaterial;
                tempMaterial.dispose(); // Clean up temporary material
            }
        }, revertDelay * 1000);
    }

    /**
     * Updates active collision effects (e.g., particle lifetimes).
     * This method should be called in the animation loop.
     * @param {number} deltaTime - The time elapsed since the last frame.
     */
    update(deltaTime) {
        for (let i = this.activeEffects.length - 1; i >= 0; i--) {
            const effect = this.activeEffects[i];
            effect.userData.lifetime += deltaTime;

            if (effect.userData.lifetime >= effect.userData.maxLifetime) {
                this.scene.remove(effect);
                if (effect.geometry) effect.geometry.dispose();
                if (effect.material) effect.material.dispose();
                this.activeEffects.splice(i, 1);
            } else {
                // Update particle shader uniforms if needed (e.g., time for pulsating)
                if (effect.material.uniforms && effect.material.uniforms.time) {
                    effect.material.uniforms.time.value += deltaTime;
                }
                // Fade out effect
                effect.material.opacity = 1.0 - (effect.userData.lifetime / effect.userData.maxLifetime);
                effect.material.needsUpdate = true;
            }
        }
    }
}


