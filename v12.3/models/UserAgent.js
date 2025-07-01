import * as THREE from 'three';

export class UserAgent {
    /**
     * @param {THREE.Scene} scene - The Three.js scene to add the agent to.
     * @param {THREE.Camera} [camera] - Optional camera to attach to the agent.
     * @param {object} [options] - Configuration options.
     * @param {string} [options.modelPath] - Path to the agent's 3D model (e.g., GLB, OBJ).
     * @param {number} [options.movementSpeed=5] - Speed of agent movement.
     * @param {number} [options.rotationSpeed=0.05] - Speed of agent rotation.
     * @param {THREE.Vector3} [options.initialPosition] - Initial position of the agent.
     * @param {THREE.Vector3} [options.cameraOffset] - Offset for the attached camera.
     */
    constructor(scene, camera, options = {}) {
        this.scene = scene;
        this.camera = camera;
        this.options = { ...{
            modelPath: null,
            movementSpeed: 5,
            rotationSpeed: 0.05,
            initialPosition: new THREE.Vector3(0, 0, 0),
            cameraOffset: new THREE.Vector3(0, 5, -10) // Default third-person offset
        }, ...options };

        this.model = null; // Three.js object representing the agent
        this.velocity = new THREE.Vector3();
        this.angularVelocity = 0;

        this._loadModel();
        this.setPosition(this.options.initialPosition);

        // Input state
        this.keys = {};
        this._setupInputListeners();

        console.log('UserAgent initialized.');
    }

    async _loadModel() {
        if (this.options.modelPath) {
            // For simplicity, using a basic BoxGeometry if no modelPath is provided
            // In a real scenario, you'd use GLTFLoader, OBJLoader etc.
            const geometry = new THREE.BoxGeometry(1, 2, 1);
            const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
            this.model = new THREE.Mesh(geometry, material);
            this.model.name = 'UserAgentModel';
            this.scene.add(this.model);
            console.log(`UserAgent model loaded: ${this.options.modelPath || 'Default Box'}`);
        } else {
            const geometry = new THREE.BoxGeometry(1, 2, 1);
            const material = new THREE.MeshStandardMaterial({ color: 0x00ff00 });
            this.model = new THREE.Mesh(geometry, material);
            this.model.name = 'UserAgentModel';
            this.scene.add(this.model);
            console.log('UserAgent using default box model.');
        }
    }

    _setupInputListeners() {
        document.addEventListener('keydown', (event) => {
            this.keys[event.code] = true;
        });
        document.addEventListener('keyup', (event) => {
            this.keys[event.code] = false;
        });
    }

    /**
     * Updates the agent's position, rotation, and camera.
     * @param {number} deltaTime - Time elapsed since last frame in seconds.
     */
    update(deltaTime) {
        if (!this.model) return;

        // Reset velocities
        this.velocity.set(0, 0, 0);
        this.angularVelocity = 0;

        // Handle movement input
        if (this.keys['KeyW']) { // Forward
            this.velocity.z = -this.options.movementSpeed;
        }
        if (this.keys['KeyS']) { // Backward
            this.velocity.z = this.options.movementSpeed;
        }
        if (this.keys['KeyA']) { // Strafe Left
            this.velocity.x = -this.options.movementSpeed;
        }
        if (this.keys['KeyD']) { // Strafe Right
            this.velocity.x = this.options.movementSpeed;
        }

        // Handle rotation input
        if (this.keys['ArrowLeft']) { // Rotate Left
            this.angularVelocity = this.options.rotationSpeed;
        }
        if (this.keys['ArrowRight']) { // Rotate Right
            this.angularVelocity = -this.options.rotationSpeed;
        }

        // Apply rotation
        this.model.rotation.y += this.angularVelocity;

        // Apply movement relative to agent's orientation
        const forward = new THREE.Vector3(0, 0, 0);
        this.model.getWorldDirection(forward);
        forward.y = 0; // Keep movement on horizontal plane
        forward.normalize();

        const right = new THREE.Vector3();
        right.crossVectors(this.model.up, forward);
        right.normalize();

        this.model.position.add(forward.multiplyScalar(this.velocity.z * deltaTime));
        this.model.position.add(right.multiplyScalar(this.velocity.x * deltaTime));

        // Update attached camera
        if (this.camera) {
            const tempV = new THREE.Vector3();
            this.model.getWorldPosition(tempV);
            this.camera.position.copy(tempV).add(this.options.cameraOffset);
            this.camera.lookAt(tempV);
        }
    }

    /**
     * Sets the agent's position.
     * @param {THREE.Vector3} position - The new position.
     */
    setPosition(position) {
        if (this.model) {
            this.model.position.copy(position);
        }
    }

    /**
     * Gets the agent's current position.
     * @returns {THREE.Vector3}
     */
    getPosition() {
        return this.model ? this.model.position.clone() : new THREE.Vector3();
    }

    /**
     * Gets the agent's 3D model.
     * @returns {THREE.Object3D}
     */
    getModel() {
        return this.model;
    }

    /**
     * Example method for LLM control: move agent in a direction.
     * @param {string} direction - 'forward', 'backward', 'left', 'right'.
     */
    move(direction) {
        // This would typically set internal flags or apply forces
        // For this example, we'll just log it.
        console.log(`LLM commanded agent to move: ${direction}`);
        // In a real system, this would trigger the appropriate key state or velocity.
        // e.g., this.keys['KeyW'] = true; for a short duration.
    }

    /**
     * Example method for LLM control: fire a projectile.
     */
    fireProjectile() {
        console.log('LLM commanded agent to fire projectile.');
        // Implement projectile spawning logic here.
        // This would likely involve creating a new mesh/object and giving it an initial velocity.
    }
}


