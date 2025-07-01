import * as THREE from 'three';
import { DynamicLightManager } from './DynamicLightManager.js';
import { NeRFParticleShaderEffects } from './NeRFParticleShaderEffects.js';

export class ExplosionManager {
    /**
     * @param {THREE.Scene} scene - The Three.js scene to add effects to.
     * @param {DynamicLightManager} dynamicLightManager - Instance of DynamicLightManager.
     */
    constructor(scene, dynamicLightManager) {
        this.scene = scene;
        this.dynamicLightManager = dynamicLightManager;
        this.activeExplosions = [];
        console.log('ExplosionManager initialized.');
    }

    /**
     * Triggers an explosion effect at a given position.
     * @param {object} explosionData - Data about the explosion.
     * @param {THREE.Vector3} explosionData.point - The world position of the explosion.
     * @param {number} [explosionData.magnitude=1.0] - The magnitude of the explosion, influencing effect intensity and scale.
     * @param {string} [explosionData.type='default'] - Type of explosion (e.g., 'fiery', 'electrical', 'smoke').
     */
    triggerExplosion(explosionData) {
        const { point, magnitude = 1.0, type = 'default' } = explosionData;

        // 1. Initial Flash (temporary light)
        this._createFlash(point, magnitude);

        // 2. Fireball (expanding mesh/particle system)
        this._createFireball(point, magnitude, type);

        // 3. Smoke Plume (particle system)
        this._createSmokePlume(point, magnitude, type);

        // 4. Debris (simple particles for now, could be meshes)
        this._createDebris(point, magnitude, type);

        // Future: Shockwave, camera shake, sound effects
    }

    _createFlash(position, magnitude) {
        const flashLight = new THREE.PointLight(0xffffff, magnitude * 10, 100);
        flashLight.position.copy(position);
        this.dynamicLightManager.addLight(flashLight);

        // Remove flash after a short duration
        setTimeout(() => {
            this.dynamicLightManager.removeLight(flashLight);
            flashLight.dispose();
        }, 200);
    }

    _createFireball(position, magnitude, type) {
        const fireballRadius = magnitude * 5;
        const fireballGeometry = new THREE.SphereGeometry(fireballRadius, 32, 32);
        const fireballMaterial = NeRFParticleShaderEffects.createFireballMaterial();
        const fireball = new THREE.Mesh(fireballGeometry, fireballMaterial);
        fireball.position.copy(position);
        fireball.userData.lifetime = 0;
        fireball.userData.maxLifetime = 1.0 + magnitude * 0.5;
        fireball.userData.initialScale = fireball.scale.clone();

        this.scene.add(fireball);
        this.activeExplosions.push(fireball);
    }

    _createSmokePlume(position, magnitude, type) {
        const smokeCount = Math.floor(magnitude * 200);
        const smokeGeometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];

        const smokeColor = new THREE.Color(0x888888); // Grey smoke

        for (let i = 0; i < smokeCount; i++) {
            positions.push(position.x + (Math.random() - 0.5) * magnitude);
            positions.push(position.y + (Math.random() - 0.5) * magnitude);
            positions.push(position.z + (Math.random() - 0.5) * magnitude);

            colors.push(smokeColor.r + (Math.random() - 0.5) * 0.1);
            colors.push(smokeColor.g + (Math.random() - 0.5) * 0.1);
            colors.push(smokeColor.b + (Math.random() - 0.5) * 0.1);

            sizes.push(Math.random() * 2 + 1);
        }

        smokeGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        smokeGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        smokeGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        const smokeMaterial = NeRFParticleShaderEffects.createSmokeMaterial();
        const smoke = new THREE.Points(smokeGeometry, smokeMaterial);
        smoke.position.copy(position);
        smoke.userData.lifetime = 0;
        smoke.userData.maxLifetime = 3.0 + magnitude * 1.0;

        this.scene.add(smoke);
        this.activeExplosions.push(smoke);
    }

    _createDebris(position, magnitude, type) {
        const debrisCount = Math.floor(magnitude * 50);
        const debrisGeometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];

        const debrisColor = new THREE.Color(0x555555); // Dark grey debris

        for (let i = 0; i < debrisCount; i++) {
            positions.push(position.x + (Math.random() - 0.5) * magnitude * 2);
            positions.push(position.y + (Math.random() - 0.5) * magnitude * 2);
            positions.push(position.z + (Math.random() - 0.5) * magnitude * 2);

            colors.push(debrisColor.r + (Math.random() - 0.5) * 0.1);
            colors.push(debrisColor.g + (Math.random() - 0.5) * 0.1);
            colors.push(debrisColor.b + (Math.random() - 0.5) * 0.1);

            sizes.push(Math.random() * 1 + 0.5);
        }

        debrisGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        debrisGeometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        debrisGeometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        const debrisMaterial = NeRFParticleShaderEffects.createDebrisMaterial();
        const debris = new THREE.Points(debrisGeometry, debrisMaterial);
        debris.position.copy(position);
        debris.userData.lifetime = 0;
        debris.userData.maxLifetime = 2.0 + magnitude * 0.5;

        this.scene.add(debris);
        this.activeExplosions.push(debris);
    }

    /**
     * Updates active explosion effects (e.g., particle lifetimes, fireball expansion).
     * This method should be called in the animation loop.
     * @param {number} deltaTime - The time elapsed since the last frame.
     */
    update(deltaTime) {
        for (let i = this.activeExplosions.length - 1; i >= 0; i--) {
            const effect = this.activeExplosions[i];
            effect.userData.lifetime += deltaTime;

            const progress = effect.userData.lifetime / effect.userData.maxLifetime;

            if (progress >= 1.0) {
                this.scene.remove(effect);
                if (effect.geometry) effect.geometry.dispose();
                if (effect.material) effect.material.dispose();
                this.activeExplosions.splice(i, 1);
            } else {
                // Update fireball scale and opacity
                if (effect.material.uniforms && effect.material.uniforms.time) {
                    effect.material.uniforms.time.value += deltaTime;
                }

                // Fireball expansion and fade
                if (effect.isMesh) {
                    const scaleFactor = 1.0 + progress * 2.0; // Expand to 3x original size
                    effect.scale.copy(effect.userData.initialScale).multiplyScalar(scaleFactor);
                    effect.material.opacity = 1.0 - progress; // Fade out
                    effect.material.needsUpdate = true;
                }
                // Smoke plume rising and fading
                else if (effect.isPoints && effect.material.uniforms && effect.material.uniforms.time) {
                    effect.position.y += deltaTime * 2.0; // Rise
                    effect.material.opacity = 1.0 - progress; // Fade out
                    effect.material.needsUpdate = true;
                }
            }
        }
    }
}


