import * as THREE from 'three';

export class DynamicLightManager {
    constructor(scene) {
        this.scene = scene;
        this.lights = [];
    }

    /**
     * Adds a dynamic light to the scene and manages it.
     * @param {THREE.Light} light - The Three.js Light object to add.
     */
    addLight(light) {
        this.scene.add(light);
        this.lights.push(light);
        console.log(`Dynamic light added: ${light.type}`);
    }

    /**
     * Removes a dynamic light from the scene.
     * @param {THREE.Light} light - The Three.js Light object to remove.
     */
    removeLight(light) {
        this.scene.remove(light);
        this.lights = this.lights.filter(l => l !== light);
        console.log(`Dynamic light removed: ${light.type}`);
    }

    /**
     * Updates the state of dynamic lights, e.g., for orbiting effects.
     * This method should be called in the animation loop.
     * @param {number} deltaTime - The time elapsed since the last frame.
     * @param {number} totalTime - The total time elapsed since the start of the animation.
     */
    update(deltaTime, totalTime) {
        // Example: Make the first light orbit a central point
        if (this.lights.length > 0) {
            const orbitingLight = this.lights[0];
            const orbitRadius = 10; // Example radius
            const orbitSpeed = 0.5; // Example speed (radians per second)

            orbitingLight.position.x = Math.cos(totalTime * orbitSpeed) * orbitRadius;
            orbitingLight.position.z = Math.sin(totalTime * orbitSpeed) * orbitRadius;
            orbitingLight.position.y = Math.sin(totalTime * orbitSpeed * 0.5) * (orbitRadius / 2); // Add some vertical movement
        }

        // Future enhancements could include:
        // - Managing multiple orbiting lights with different parameters
        // - Implementing light intensity fluctuations
        // - Handling light colors based on spectral concepts
    }

    /**
     * Returns all currently managed dynamic lights.
     * @returns {Array<THREE.Light>}
     */
    getAllLights() {
        return this.lights;
    }
}


