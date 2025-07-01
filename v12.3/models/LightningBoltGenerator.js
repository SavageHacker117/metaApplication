import * as THREE from 'three';
import { NeRFParticleShaderEffects } from './NeRFParticleShaderEffects.js';

export class LightningBoltGenerator {
    /**
     * @param {THREE.Scene} scene - The Three.js scene to add the lightning bolt to.
     * @param {object} [options] - Configuration options.
     * @param {number} [options.length=10] - Overall length of the main bolt segment.
     * @param {number} [options.thickness=1] - Base thickness of the main bolt.
     * @param {number} [options.stubbyFactor=0.5] - Controls how much the bolt deviates from a straight line (0-1).
     * @param {number} [options.branchingProbability=0.3] - Probability of new branches forming (0-1).
     * @param {number} [options.maxDepth=3] - Maximum recursion depth for branching.
     * @param {THREE.Color} [options.boltColor=0x00FFFF] - Color of the lightning bolt.
     * @param {number} [options.particleSize=0.5] - Size of individual particles.
     * @param {number} [options.particleCountMultiplier=100] - Multiplier for particle count based on bolt length.
     */
    constructor(scene, options = {}) {
        this.scene = scene;
        this.options = { ...{
            length: 10,
            thickness: 1,
            stubbyFactor: 0.5,
            branchingProbability: 0.3,
            maxDepth: 3,
            boltColor: new THREE.Color(0x00FFFF), // Cyan
            particleSize: 0.5,
            particleCountMultiplier: 100
        }, ...options };

        this.boltPoints = []; // Stores the vertices of the lightning bolt
        this.particleSystem = null;
        this.targetPositions = []; // Target positions for particles

        this._generateLightningBolt();
        console.log('LightningBoltGenerator initialized.');
    }

    _generateLightningBolt() {
        this.boltPoints = [];
        this.targetPositions = [];

        const startPoint = new THREE.Vector3(0, this.options.length / 2, 0);
        const endPoint = new THREE.Vector3(0, -this.options.length / 2, 0);

        this._generateBranch(startPoint, endPoint, this.options.thickness, 0);

        // Create a BufferGeometry for the particles
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(this.targetPositions.length * 3);
        const colors = new Float32Array(this.targetPositions.length * 3);
        const sizes = new Float32Array(this.targetPositions.length);

        for (let i = 0; i < this.targetPositions.length; i++) {
            const p = this.targetPositions[i];
            positions[i * 3] = p.x;
            positions[i * 3 + 1] = p.y;
            positions[i * 3 + 2] = p.z;

            // Assign color and size
            colors[i * 3] = this.options.boltColor.r;
            colors[i * 3 + 1] = this.options.boltColor.g;
            colors[i * 3 + 2] = this.options.boltColor.b;
            sizes[i] = this.options.particleSize;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('customColor', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

        const material = NeRFParticleShaderEffects.createGlowingParticleMaterial(this.options.boltColor, 2.0);
        this.particleSystem = new THREE.Points(geometry, material);
        this.particleSystem.frustumCulled = false; // Ensure particles are always rendered
        this.scene.add(this.particleSystem);
    }

    _generateBranch(start, end, thickness, depth) {
        if (depth > this.options.maxDepth) return;

        const midPoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
        const direction = new THREE.Vector3().subVectors(end, start);
        const length = direction.length();

        if (length < 0.5) { // Base case for recursion
            this.boltPoints.push(start.clone(), end.clone());
            this.targetPositions.push(start.clone(), end.clone());
            return;
        }

        // Introduce randomness for jaggedness (stubby effect)
        const perpendicular = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5)).normalize();
        const offset = perpendicular.multiplyScalar(length * this.options.stubbyFactor * (Math.random() - 0.5));
        midPoint.add(offset);

        this.boltPoints.push(start.clone(), midPoint.clone());
        this.targetPositions.push(start.clone(), midPoint.clone());

        // Recursively generate sub-branches
        this._generateBranch(start, midPoint, thickness * 0.8, depth + 1);
        this._generateBranch(midPoint, end, thickness * 0.8, depth + 1);

        // Randomly create new branches
        if (Math.random() < this.options.branchingProbability) {
            const branchStart = midPoint.clone();
            const branchEnd = new THREE.Vector3().addVectors(midPoint, new THREE.Vector3(
                (Math.random() - 0.5) * length * 0.5,
                (Math.random() - 0.5) * length * 0.5,
                (Math.random() - 0.5) * length * 0.5
            ));
            this._generateBranch(branchStart, branchEnd, thickness * 0.5, depth + 1);
        }
    }

    /**
     * Returns the particle system representing the lightning bolt.
     * @returns {THREE.Points}
     */
    getParticleSystem() {
        return this.particleSystem;
    }

    /**
     * Updates the lightning bolt animation (e.g., for pulsating effect).
     * @param {number} deltaTime - Time elapsed since last frame.
     * @param {number} elapsedTime - Total time elapsed.
     */
    update(deltaTime, elapsedTime) {
        if (this.particleSystem && this.particleSystem.material.uniforms.time) {
            this.particleSystem.material.uniforms.time.value = elapsedTime;
        }
    }

    /**
     * Disposes of the lightning bolt resources.
     */
    dispose() {
        if (this.particleSystem) {
            this.scene.remove(this.particleSystem);
            if (this.particleSystem.geometry) this.particleSystem.geometry.dispose();
            if (this.particleSystem.material) this.particleSystem.material.dispose();
        }
    }
}


