import * as THREE from 'three';
import { LightningBoltGenerator } from './LightningBoltGenerator.js';

export class LoadingScreenManager {
    /**
     * @param {HTMLElement} parentElement - The DOM element to append the loading screen to.
     */
    constructor(parentElement) {
        this.parentElement = parentElement;
        this.loadingScreenElement = this._createLoadingScreenElement();
        this.parentElement.appendChild(this.loadingScreenElement);

        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        this.clock = new THREE.Clock();

        this._setupRenderer();
        this._setupCamera();
        this._setupLights();

        this.lightningBoltGenerator = new LightningBoltGenerator(this.scene, {
            length: 15, // Make it a bit larger for the loading screen
            thickness: 1.5,
            stubbyFactor: 0.6,
            branchingProbability: 0.4,
            maxDepth: 4,
            boltColor: new THREE.Color(0xFFD700), // Gold color for lightning
            particleSize: 0.8
        });

        this.progressText = document.createElement('div');
        this.progressText.style.cssText = `
            position: absolute;
            bottom: 20%;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            font-family: 'Arial', sans-serif;
            font-size: 1.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        `;
        this.loadingScreenElement.appendChild(this.progressText);
        this.updateProgress(0);

        this.isVisible = false;
        this.loadingScreenElement.style.display = 'none';

        this._setupEventListeners();
        console.log('LoadingScreenManager initialized.');
    }

    _createLoadingScreenElement() {
        const element = document.createElement('div');
        element.id = 'loading-screen';
        element.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.9);
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            z-index: 1000;
            opacity: 1;
            transition: opacity 1s ease-in-out;
        `;

        const title = document.createElement('h1');
        title.textContent = 'Loading Quantum Reality...';
        title.style.cssText = `
            color: white;
            font-family: 'Arial', sans-serif;
            font-size: 3em;
            margin-bottom: 50px;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.7);
        `;
        element.appendChild(title);

        return element;
    }

    _setupRenderer() {
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.loadingScreenElement.appendChild(this.renderer.domElement);
    }

    _setupCamera() {
        this.camera.position.z = 20; // Position camera to view the lightning bolt
        this.camera.lookAt(0, 0, 0);
    }

    _setupLights() {
        const ambientLight = new THREE.AmbientLight(0x404040, 0.8);
        this.scene.add(ambientLight);

        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(0, 1, 1).normalize();
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
     * Shows the loading screen and starts its animation.
     */
    show() {
        this.loadingScreenElement.style.display = 'flex';
        this.isVisible = true;
        this.renderer.setAnimationLoop(this._animate.bind(this));
        console.log('Loading screen shown.');
    }

    /**
     * Hides the loading screen with a fade-out effect.
     */
    hide() {
        this.loadingScreenElement.style.opacity = '0';
        setTimeout(() => {
            this.loadingScreenElement.style.display = 'none';
            this.isVisible = false;
            this.renderer.setAnimationLoop(null); // Stop animation loop
            this.dispose(); // Clean up resources
            console.log('Loading screen hidden.');
        }, 1000); // Match CSS transition duration
    }

    /**
     * Updates the loading progress text.
     * @param {number} progress - A value between 0 and 1 representing the loading progress.
     */
    updateProgress(progress) {
        const percentage = Math.floor(progress * 100);
        this.progressText.textContent = `Loading: ${percentage}%`;
        // You could also use progress to influence the lightning bolt animation
        // For example, make it more complete as progress increases.
    }

    _animate() {
        const deltaTime = this.clock.getDelta();
        const elapsedTime = this.clock.getElapsedTime();

        // Update lightning bolt animation
        this.lightningBoltGenerator.update(deltaTime, elapsedTime);

        this.renderer.render(this.scene, this.camera);
    }

    /**
     * Disposes of all resources used by the loading screen.
     */
    dispose() {
        this.lightningBoltGenerator.dispose();
        this.renderer.dispose();
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
        this.parentElement.removeChild(this.loadingScreenElement);
        window.removeEventListener('resize', this._onWindowResize.bind(this));
        console.log('LoadingScreenManager disposed.');
    }
}


