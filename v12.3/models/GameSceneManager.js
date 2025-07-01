import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { NeRFRenderer, NeRFRendererConfig } from './NeRFRenderer_enhanced.js';

export class GameSceneManager {
    /**
     * @param {HTMLElement} container - The DOM element to append the renderer to.
     * @param {object} [options] - Configuration options.
     * @param {NeRFRendererConfig} [options.nerfConfig] - Configuration for the NeRFRenderer.
     * @param {boolean} [options.enableOrbitControls=true] - Whether to enable OrbitControls for debugging.
     */
    constructor(container, options = {}) {
        this.container = container;
        this.options = { ...{
            nerfConfig: {},
            enableOrbitControls: true
        }, ...options };

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 2000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.clock = new THREE.Clock();

        this._setupRenderer();
        this._setupCamera();
        this._setupLights(); // Basic lights, TimeOfDayManager will handle dynamic ones
        this._setupEventListeners();

        // Initialize NeRF Renderer with its sub-managers
        this.nerfRenderer = new NeRFRenderer(this.scene, new NeRFRendererConfig(this.options.nerfConfig));

        if (this.options.enableOrbitControls) {
            this.controls = new OrbitControls(this.camera, this.renderer.domElement);
            this.controls.enableDamping = true; // an animation loop is required when damping is enabled
            this.controls.dampingFactor = 0.05;
        }

        console.log('GameSceneManager initialized.');
    }

    _setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap; // default THREE.PCFShadowMap
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.container.appendChild(this.renderer.domElement);
    }

    _setupCamera() {
        this.camera.position.set(0, 20, 50); // Default camera position
        this.camera.lookAt(0, 0, 0);
    }

    _setupLights() {
        // Add a basic ambient light. TimeOfDayManager will override/enhance this.
        const ambientLight = new THREE.AmbientLight(0x404040, 0.5); // soft white light
        this.scene.add(ambientLight);

        // Add a basic directional light. TimeOfDayManager will manage the main sun/moon light.
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(0, 50, 0); // Directly above
        directionalLight.castShadow = true;
        this.scene.add(directionalLight);
    }

    _setupEventListeners() {
        window.addEventListener('resize', this._onWindowResize.bind(this), false);
    }

    _onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    /**
     * Starts the main animation loop.
     */
    start() {
        this.renderer.setAnimationLoop(this._animate.bind(this));
        console.log('GameSceneManager animation loop started.');
    }

    /**
     * The main animation loop.
     */
    _animate() {
        const deltaTime = this.clock.getDelta();
        const elapsedTime = this.clock.getElapsedTime();

        // Update OrbitControls if enabled
        if (this.controls) {
            this.controls.update();
        }

        // Update NeRF Renderer and its sub-managers
        // The NeRFRenderer's animate method will handle TimeOfDayManager, CollisionEffectsManager, ExplosionManager updates
        this.nerfRenderer.animate(deltaTime, elapsedTime); 

        // Render the scene
        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Returns the Three.js scene.
     * @returns {THREE.Scene}
     */
    getScene() {
        return this.scene;
    }

    /**
     * Returns the Three.js camera.
     * @returns {THREE.Camera}
     */
    getCamera() {
        return this.camera;
    }

    /**
     * Returns the NeRFRenderer instance.
     * @returns {NeRFRenderer}
     */
    getNeRFRenderer() {
        return this.nerfRenderer;
    }

    /**
     * Disposes of all resources.
     */
    dispose() {
        this.renderer.setAnimationLoop(null);
        this.nerfRenderer.dispose();
        this.scene.traverse((object) => {
            if (object.geometry) object.geometry.dispose();
            if (object.material) {
                if (Array.isArray(object.material)) {
                    object.material.forEach(material => material.dispose());
                } else {
                    object.material.dispose();
                }
            }
            if (object.texture) object.texture.dispose();
        });
        this.renderer.dispose();
        if (this.controls) this.controls.dispose();
        window.removeEventListener('resize', this._onWindowResize.bind(this));
        this.container.removeChild(this.renderer.domElement);
        console.log('GameSceneManager disposed.');
    }
}


