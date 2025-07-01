import * as THREE from 'three';
import { DynamicLightManager } from './DynamicLightManager.js';

export class TimeOfDayManager {
    /**
     * @param {THREE.Scene} scene - The Three.js scene to manage.
     * @param {DynamicLightManager} dynamicLightManager - Instance of DynamicLightManager.
     * @param {object} options - Configuration options.
     * @param {number} [options.cycleDuration=60] - Duration of a full day/night cycle in seconds.
     * @param {number} [options.initialTime=0] - Initial time of day (0.0 to 1.0, where 0.0/1.0 is midnight, 0.25 is sunrise, 0.5 is noon, 0.75 is sunset).
     */
    constructor(scene, dynamicLightManager, options = {}) {
        this.scene = scene;
        this.dynamicLightManager = dynamicLightManager;
        this.cycleDuration = options.cycleDuration || 60; // seconds for a full cycle
        this.currentTime = options.initialTime || 0; // Normalized time (0.0 to 1.0)
        this.clock = new THREE.Clock();

        // Create a directional light for sun/moon
        this.sunMoonLight = new THREE.DirectionalLight(0xffffff, 1.0);
        this.sunMoonLight.castShadow = true;
        this.sunMoonLight.shadow.mapSize.width = 2048;
        this.sunMoonLight.shadow.mapSize.height = 2048;
        this.sunMoonLight.shadow.camera.near = 0.5;
        this.sunMoonLight.shadow.camera.far = 500;
        this.sunMoonLight.shadow.camera.left = -50;
        this.sunMoonLight.shadow.camera.right = 50;
        this.sunMoonLight.shadow.camera.top = 50;
        this.sunMoonLight.shadow.camera.bottom = -50;
        this.scene.add(this.sunMoonLight);
        this.scene.add(this.sunMoonLight.target); // Target for directional light

        // Create an ambient light
        this.ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(this.ambientLight);

        // Add the sun/moon light to the dynamic light manager for updates
        this.dynamicLightManager.addLight(this.sunMoonLight);

        console.log('TimeOfDayManager initialized.');
    }

    /**
     * Updates the time of day and adjusts lighting accordingly.
     * This method should be called in the animation loop.
     */
    update() {
        const deltaTime = this.clock.getDelta();
        this.currentTime = (this.currentTime + (deltaTime / this.cycleDuration)) % 1.0;

        // Calculate sun/moon position based on current time
        // 0.0/1.0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset
        const angle = this.currentTime * Math.PI * 2; // Full circle
        const radius = 100; // Distance from origin

        this.sunMoonLight.position.set(
            Math.sin(angle) * radius,
            Math.cos(angle) * radius,
            Math.sin(angle * 0.5) * radius * 0.5 // Slight variation in Z
        );
        this.sunMoonLight.target.position.set(0, 0, 0); // Always point to origin

        // Adjust light intensity and color based on time of day
        let lightIntensity = 0;
        let lightColor = new THREE.Color();
        let ambientIntensity = 0;
        let ambientColor = new THREE.Color();

        if (this.currentTime >= 0.0 && this.currentTime < 0.25) { // Night to Dawn
            const progress = this.currentTime / 0.25;
            lightIntensity = THREE.MathUtils.lerp(0.1, 1.0, progress); // From dim to bright
            lightColor.setRGB(
                THREE.MathUtils.lerp(0.1, 1.0, progress), // Red
                THREE.MathUtils.lerp(0.2, 1.0, progress), // Green
                THREE.MathUtils.lerp(0.4, 1.0, progress)  // Blue
            ); // From blueish to white
            ambientIntensity = THREE.MathUtils.lerp(0.1, 0.5, progress);
            ambientColor.setRGB(
                THREE.MathUtils.lerp(0.1, 1.0, progress), 
                THREE.MathUtils.lerp(0.2, 1.0, progress), 
                THREE.MathUtils.lerp(0.4, 1.0, progress)
            );
        } else if (this.currentTime >= 0.25 && this.currentTime < 0.5) { // Dawn to Noon
            const progress = (this.currentTime - 0.25) / 0.25;
            lightIntensity = THREE.MathUtils.lerp(1.0, 1.5, progress); // From bright to brighter
            lightColor.setRGB(
                THREE.MathUtils.lerp(1.0, 1.0, progress), 
                THREE.MathUtils.lerp(1.0, 1.0, progress), 
                THREE.MathUtils.lerp(1.0, 1.0, progress)
            ); // White
            ambientIntensity = THREE.MathUtils.lerp(0.5, 0.8, progress);
            ambientColor.setRGB(1.0, 1.0, 1.0);
        } else if (this.currentTime >= 0.5 && this.currentTime < 0.75) { // Noon to Dusk
            const progress = (this.currentTime - 0.5) / 0.25;
            lightIntensity = THREE.MathUtils.lerp(1.5, 1.0, progress); // From brighter to bright
            lightColor.setRGB(
                THREE.MathUtils.lerp(1.0, 1.0, progress), 
                THREE.MathUtils.lerp(1.0, 0.8, progress), 
                THREE.MathUtils.lerp(1.0, 0.6, progress)
            ); // From white to warm
            ambientIntensity = THREE.MathUtils.lerp(0.8, 0.5, progress);
            ambientColor.setRGB(
                THREE.MathUtils.lerp(1.0, 1.0, progress), 
                THREE.MathUtils.lerp(1.0, 0.8, progress), 
                THREE.MathUtils.lerp(1.0, 0.6, progress)
            );
        } else { // Dusk to Night
            const progress = (this.currentTime - 0.75) / 0.25;
            lightIntensity = THREE.MathUtils.lerp(1.0, 0.1, progress); // From bright to dim
            lightColor.setRGB(
                THREE.MathUtils.lerp(1.0, 0.1, progress), 
                THREE.MathUtils.lerp(0.8, 0.2, progress), 
                THREE.MathUtils.lerp(0.6, 0.4, progress)
            ); // From warm to blueish
            ambientIntensity = THREE.MathUtils.lerp(0.5, 0.1, progress);
            ambientColor.setRGB(
                THREE.MathUtils.lerp(1.0, 0.1, progress), 
                THREE.MathUtils.lerp(0.8, 0.2, progress), 
                THREE.MathUtils.lerp(0.6, 0.4, progress)
            );
        }

        this.sunMoonLight.intensity = lightIntensity;
        this.sunMoonLight.color.copy(lightColor);
        this.ambientLight.intensity = ambientIntensity;
        this.ambientLight.color.copy(ambientColor);

        // Update scene background/environment (simple color for now)
        this.scene.background = lightColor.clone().multiplyScalar(ambientIntensity * 0.5);

        // You can add more complex skybox/environment map blending here
    }

    /**
     * Sets the current time of day.
     * @param {number} time - Normalized time (0.0 to 1.0).
     */
    setTime(time) {
        this.currentTime = THREE.MathUtils.clamp(time, 0.0, 1.0);
        this.update(); // Force update lights immediately
    }

    /**
     * Gets the current normalized time of day.
     * @returns {number}
     */
    getCurrentTime() {
        return this.currentTime;
    }
}


