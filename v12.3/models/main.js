import * as THREE from 'three';
import { GameSceneManager } from './GameSceneManager.js';
import { UserAgent } from './UserAgent.js';
import { ProceduralMapGenerator } from './ProceduralMapGenerator.js';
import { GameContainer } from './GameContainer.js';
import { LoadingScreenManager } from './LoadingScreenManager.js';

// Main game initialization function
async function initApp() {
    // 1. Initialize GameContainer
    const gameContainer = new GameContainer();
    const gameRootElement = gameContainer.getElement();

    // 2. Initialize LoadingScreenManager and show it
    const loadingScreenManager = new LoadingScreenManager(document.body); // Use document.body for the loading screen overlay
    loadingScreenManager.show();

    // 3. Initialize GameSceneManager
    // This will set up the Three.js scene, camera, renderer, and also initialize NeRFRenderer
    // with all its sub-managers (TimeOfDayManager, CollisionEffectsManager, ExplosionManager).
    const gameSceneManager = new GameSceneManager(gameRootElement, {
        nerfConfig: {
            enablePBR: true,
            enableProceduralGeometry: true,
            enableDynamicLighting: true,
            enableCustomShaderEffects: true,
            enableTimeOfDay: true, // Enable day/night cycle
            enableCollisionEffects: true, // Enable collision visual feedback
            enableExplosionEffects: true // Enable explosion effects
        },
        enableOrbitControls: true // Enable orbit controls for easy navigation during development
    });

    const scene = gameSceneManager.getScene();
    const camera = gameSceneManager.getCamera();
    const nerfRenderer = gameSceneManager.getNeRFRenderer();

    // --- Simulate Asset Loading Progress ---
    // In a real application, you would hook into your asset loading pipeline
    // (e.g., GLTFLoader, TextureLoader, NeRFRenderer's own loading queue)
    // to update the loadingScreenManager.updateProgress() method.
    // For this example, we'll use a mock loading process.

    let totalAssetsToLoad = 5;
    let loadedAssetsCount = 0;

    const mockLoadAsset = async (assetName, delay) => {
        return new Promise(resolve => {
            setTimeout(() => {
                loadedAssetsCount++;
                loadingScreenManager.updateProgress(loadedAssetsCount / totalAssetsToLoad);
                console.log(`Loaded ${assetName}`);
                resolve();
            }, delay);
        });
    };

    console.log('Simulating loading of critical game assets...');
    await Promise.all([
        mockLoadAsset('Map Terrain Data', 1500),
        mockLoadAsset('Agent 3D Model', 1000),
        mockLoadAsset('Core Shader Programs', 800),
        mockLoadAsset('Initial NeRF Data', 2000),
        mockLoadAsset('UI Textures', 500)
    ]);
    console.log('All critical assets mocked as loaded.');

    // 4. Initialize ProceduralMapGenerator
    const mapGenerator = new ProceduralMapGenerator(nerfRenderer);

    // 5. Generate Map
    console.log('Generating map...');
    const mapData = mapGenerator.generateMap({
        seed: 123, // Use a seed for reproducible maps
        size: 200, // Map size (e.g., 200x200 units)
        resolution: 128, // Detail of the terrain mesh
        heightScale: 20, // Max height variation
        featureDensity: 0.005 // Density of trees/rocks
    });

    // 6. Add Map to Scene
    mapData.terrain.forEach(mesh => scene.add(mesh));
    mapData.features.forEach(feature => scene.add(feature));
    console.log('Map generated and added to scene.');

    // 7. Initialize User Agent
    const userAgent = new UserAgent(scene, camera, {
        initialPosition: mapData.startPosition, // Place agent at generated start position
        movementSpeed: 10,
        rotationSpeed: 0.08
    });
    // The UserAgent model is added to the scene internally by its constructor
    console.log('UserAgent initialized.');

    // 8. Hide loading screen and start game loop
    loadingScreenManager.hide();
    gameSceneManager.start();
    console.log('Game loop started.');

    // --- Example Usage of New Features (for testing) ---

    // Day/Night Cycle:
    // The TimeOfDayManager is automatically updated by NeRFRenderer's animate loop.
    // You can manually set time for testing:
    // nerfRenderer.timeOfDayManager.setTime(0.7); // Set to evening

    // Collision Effects:
    // Simulate a collision after a few seconds
    setTimeout(() => {
        console.log('Simulating a collision effect...');
        const collisionPoint = new THREE.Vector3(0, 5, 0);
        const dummyObjectA = new THREE.Mesh(new THREE.BoxGeometry(1,1,1), new THREE.MeshStandardMaterial({color: 0xff0000}));
        dummyObjectA.position.copy(collisionPoint);
        scene.add(dummyObjectA);

        nerfRenderer.collisionEffectsManager.triggerCollisionEffect({
            point: collisionPoint,
            objectA: dummyObjectA,
            objectB: new THREE.Object3D(), // Dummy object
            impulse: 0.7,
            type: 'metal'
        });
        // Remove dummy object after a short delay
        setTimeout(() => scene.remove(dummyObjectA), 1000);
    }, 5000);

    // Explosion Effects:
    // Simulate an explosion after some more time
    setTimeout(() => {
        console.log('Simulating an explosion effect...');
        nerfRenderer.explosionManager.triggerExplosion({
            point: new THREE.Vector3(10, 10, -10),
            magnitude: 1.2,
            type: 'fiery'
        });
    }, 10000);

    // --- LLM Integration (Conceptual) ---
    // This is where your RL-LLM trainer would interact with the game.
    // The LLM would receive observations from the game state (e.g., agent position, visible objects)
    // and then issue commands to the UserAgent or other game entities.

    // Example: Expose game state for LLM observation
    window.getGameState = () => {
        return {
            agentPosition: userAgent.getPosition(),
            currentTimeOfDay: nerfRenderer.timeOfDayManager ? nerfRenderer.timeOfDayManager.getCurrentTime() : null,
            // Add more relevant game state here
        };
    };

    // Example: LLM command interface
    window.llmCommand = (commandType, ...args) => {
        switch (commandType) {
            case 'moveAgent':
                userAgent.move(args[0]); // args[0] would be 'forward', 'backward', etc.
                break;
            case 'fireProjectile':
                userAgent.fireProjectile();
                break;
            case 'triggerExplosion':
                nerfRenderer.explosionManager.triggerExplosion(args[0]); // args[0] would be explosionData
                break;
            // Add more commands as needed
        }
    };
}

// Run the game initialization when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initApp);


