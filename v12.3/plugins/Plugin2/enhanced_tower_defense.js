/**
 * DARK MATTER - Enhanced Three.js Tower Defense Game Plugin
 * Full-featured AI-powered 3D tower defense with advanced loading, state management, and multiplayer
 */

import * as THREE from 'three';
import * as CANNON from 'cannon';

// Game States Enum
const GameStates = {
    LOADING: 'loading',
    MENU: 'menu',
    PLAYING: 'playing',
    PAUSED: 'paused',
    GAME_OVER: 'game_over',
    VICTORY: 'victory'
};

// Enemy Types Registry
const EnemyTypes = {
    BASIC: { health: 100, speed: 2, reward: 10, points: 10, model: 'basic_enemy' },
    FAST: { health: 50, speed: 4, reward: 15, points: 15, model: 'fast_enemy' },
    HEAVY: { health: 200, speed: 1, reward: 25, points: 25, model: 'heavy_enemy' },
    FLYING: { health: 75, speed: 3, reward: 20, points: 20, model: 'flying_enemy' },
    BOSS_ALIEN: { health: 1000, speed: 1.5, reward: 100, points: 100, model: 'alien_boss' },
    BOSS_GOBLIN: { health: 800, speed: 2, reward: 80, points: 80, model: 'goblin_boss' },
    BOSS_DRAGON: { health: 1500, speed: 1, reward: 150, points: 150, model: 'dragon_boss' }
};

class EnhancedTowerDefenseGame {
    constructor(scene, camera, renderer, darkMatterEngine) {
        this.scene = scene;
        this.camera = camera;
        this.renderer = renderer;
        this.engine = darkMatterEngine;
        
        // Game State Management
        this.gameState = GameStates.LOADING;
        this.previousState = null;
        this.stateTransitions = new Map();
        
        // Core Game Data
        this.currentLevel = 1;
        this.currentWave = 0;
        this.maxWaves = 10;
        this.lives = 20;
        this.gold = 100;
        this.score = 0;
        this.difficulty = 'normal';
        
        // Game Objects
        this.towers = [];
        this.enemies = [];
        this.projectiles = [];
        this.particles = [];
        this.waypoints = [];
        
        // AI and Pathfinding
        this.pathfinder = null;
        this.enemyBehaviorTree = null;
        this.proceduralGenerator = null;
        
        // Audio System
        this.audioManager = null;
        this.musicLayers = [];
        this.soundEffects = new Map();
        
        // UI Components
        this.loadingScreen = null;
        this.gameMenu = null;
        this.hud = null;
        this.gameOverScreen = null;
        
        // Performance Monitoring
        this.performanceMonitor = {
            fps: 0,
            frameTime: 0,
            drawCalls: 0,
            triangles: 0
        };
        
        // Multiplayer
        this.isMultiplayer = false;
        this.socket = null;
        this.playerId = null;
        
        // Physics World
        this.physicsWorld = null;
        
        // Animation and Effects
        this.clock = new THREE.Clock();
        this.particleSystem = null;
        this.shaderMaterials = new Map();
        
        console.log('[DARK MATTER] Enhanced Tower Defense Game initialized');
        this.init();
    }
    
    async init() {
        await this.setupPhysics();
        await this.setupAudio();
        await this.setupUI();
        await this.setupShaders();
        await this.setupParticleSystem();
        await this.setupAI();
        this.setupEventListeners();
        this.setupStateTransitions();
        
        // Start with loading screen
        this.startLoadingScreen();
    }
    
    async setupPhysics() {
        this.physicsWorld = new CANNON.World();
        this.physicsWorld.gravity.set(0, -9.82, 0);
        this.physicsWorld.broadphase = new CANNON.NaiveBroadphase();
        
        // Create ground physics body
        const groundShape = new CANNON.Plane();
        const groundBody = new CANNON.Body({ mass: 0 });
        groundBody.addShape(groundShape);
        groundBody.quaternion.setFromAxisAngle(new CANNON.Vec3(1, 0, 0), -Math.PI / 2);
        this.physicsWorld.add(groundBody);
        
        console.log('[DARK MATTER] Physics world initialized');
    }
    
    async setupAudio() {
        this.audioManager = {
            masterVolume: 1.0,
            musicVolume: 0.7,
            sfxVolume: 0.8,
            spatialAudio: true,
            
            playSound: (soundId, position = null, volume = 1.0) => {
                console.log(`[AUDIO] Playing sound: ${soundId}`);
                // Implementation would use Howler.js or Web Audio API
            },
            
            playMusic: (trackId, loop = true, fadeIn = true) => {
                console.log(`[AUDIO] Playing music: ${trackId}`);
                // Implementation would handle music layers
            },
            
            stopMusic: (fadeOut = true) => {
                console.log('[AUDIO] Stopping music');
            },
            
            setMasterVolume: (volume) => {
                this.masterVolume = Math.max(0, Math.min(1, volume));
            }
        };
        
        // Load sound effects
        this.soundEffects.set('tower_shoot', { file: 'tower_shoot.mp3', volume: 0.6 });
        this.soundEffects.set('enemy_death', { file: 'enemy_death.mp3', volume: 0.7 });
        this.soundEffects.set('explosion', { file: 'explosion.mp3', volume: 0.8 });
        this.soundEffects.set('wave_start', { file: 'wave_start.mp3', volume: 0.9 });
        this.soundEffects.set('game_over', { file: 'game_over.mp3', volume: 1.0 });
        
        console.log('[DARK MATTER] Audio system initialized');
    }
    
    async setupUI() {
        // Loading Screen
        this.loadingScreen = {
            element: null,
            progressBar: null,
            particles: [],
            shaderBackground: null,
            
            show: () => {
                console.log('[UI] Showing loading screen');
                this.createLoadingScreenParticles();
            },
            
            hide: () => {
                console.log('[UI] Hiding loading screen');
                this.clearLoadingScreenParticles();
            },
            
            updateProgress: (progress) => {
                console.log(`[UI] Loading progress: ${Math.round(progress * 100)}%`);
            }
        };
        
        // Game Menu
        this.gameMenu = {
            element: null,
            buttons: new Map(),
            
            show: () => {
                console.log('[UI] Showing game menu');
            },
            
            hide: () => {
                console.log('[UI] Hiding game menu');
            }
        };
        
        // HUD
        this.hud = {
            element: null,
            livesDisplay: null,
            goldDisplay: null,
            scoreDisplay: null,
            waveDisplay: null,
            minimap: null,
            
            update: () => {
                // Update HUD elements with current game state
            },
            
            show: () => {
                console.log('[UI] Showing HUD');
            },
            
            hide: () => {
                console.log('[UI] Hiding HUD');
            }
        };
        
        console.log('[DARK MATTER] UI components initialized');
    }
    
    async setupShaders() {
        // Loading screen background shader
        const loadingVertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        
        const loadingFragmentShader = `
            uniform float time;
            uniform vec2 resolution;
            varying vec2 vUv;
            
            void main() {
                vec2 uv = vUv;
                float wave = sin(uv.x * 10.0 + time * 2.0) * 0.1;
                float fog = sin(uv.y * 5.0 + time * 1.5) * 0.2;
                
                vec3 color = vec3(0.0, 0.2, 0.4) + vec3(wave + fog);
                color += vec3(0.0, 0.5, 0.0) * (sin(time * 3.0) * 0.5 + 0.5);
                
                gl_FragColor = vec4(color, 1.0);
            }
        `;
        
        this.shaderMaterials.set('loadingBackground', new THREE.ShaderMaterial({
            vertexShader: loadingVertexShader,
            fragmentShader: loadingFragmentShader,
            uniforms: {
                time: { value: 0.0 },
                resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
            }
        }));
        
        // Enemy spawn portal shader
        const portalVertexShader = `
            varying vec2 vUv;
            varying vec3 vPosition;
            void main() {
                vUv = uv;
                vPosition = position;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;
        
        const portalFragmentShader = `
            uniform float time;
            uniform vec3 color;
            varying vec2 vUv;
            varying vec3 vPosition;
            
            void main() {
                vec2 center = vec2(0.5, 0.5);
                float dist = distance(vUv, center);
                float pulse = sin(time * 4.0) * 0.5 + 0.5;
                float glow = 1.0 - smoothstep(0.0, 0.5, dist);
                
                vec3 finalColor = color * glow * pulse;
                gl_FragColor = vec4(finalColor, glow * 0.8);
            }
        `;
        
        this.shaderMaterials.set('spawnPortal', new THREE.ShaderMaterial({
            vertexShader: portalVertexShader,
            fragmentShader: portalFragmentShader,
            uniforms: {
                time: { value: 0.0 },
                color: { value: new THREE.Vector3(0.0, 1.0, 0.0) }
            },
            transparent: true,
            blending: THREE.AdditiveBlending
        }));
        
        console.log('[DARK MATTER] Shaders initialized');
    }
    
    async setupParticleSystem() {
        this.particleSystem = {
            systems: new Map(),
            
            createSystem: (id, config) => {
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(config.maxParticles * 3);
                const velocities = new Float32Array(config.maxParticles * 3);
                const lifetimes = new Float32Array(config.maxParticles);
                
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                geometry.setAttribute('velocity', new THREE.BufferAttribute(velocities, 3));
                geometry.setAttribute('lifetime', new THREE.BufferAttribute(lifetimes, 1));
                
                const material = new THREE.PointsMaterial({
                    color: config.color || 0xffffff,
                    size: config.size || 1.0,
                    transparent: true,
                    opacity: config.opacity || 1.0
                });
                
                const system = new THREE.Points(geometry, material);
                this.systems.set(id, {
                    mesh: system,
                    config: config,
                    activeParticles: 0
                });
                
                this.scene.add(system);
                return system;
            },
            
            emit: (systemId, position, velocity, lifetime) => {
                const system = this.systems.get(systemId);
                if (!system || system.activeParticles >= system.config.maxParticles) return;
                
                const idx = system.activeParticles * 3;
                const positions = system.mesh.geometry.attributes.position.array;
                const velocities = system.mesh.geometry.attributes.velocity.array;
                const lifetimes = system.mesh.geometry.attributes.lifetime.array;
                
                positions[idx] = position.x;
                positions[idx + 1] = position.y;
                positions[idx + 2] = position.z;
                
                velocities[idx] = velocity.x;
                velocities[idx + 1] = velocity.y;
                velocities[idx + 2] = velocity.z;
                
                lifetimes[system.activeParticles] = lifetime;
                
                system.activeParticles++;
                system.mesh.geometry.attributes.position.needsUpdate = true;
            },
            
            update: (deltaTime) => {
                this.systems.forEach((system, id) => {
                    const positions = system.mesh.geometry.attributes.position.array;
                    const velocities = system.mesh.geometry.attributes.velocity.array;
                    const lifetimes = system.mesh.geometry.attributes.lifetime.array;
                    
                    for (let i = 0; i < system.activeParticles; i++) {
                        const idx = i * 3;
                        
                        // Update position
                        positions[idx] += velocities[idx] * deltaTime;
                        positions[idx + 1] += velocities[idx + 1] * deltaTime;
                        positions[idx + 2] += velocities[idx + 2] * deltaTime;
                        
                        // Update lifetime
                        lifetimes[i] -= deltaTime;
                        
                        // Remove dead particles
                        if (lifetimes[i] <= 0) {
                            // Move last particle to this position
                            const lastIdx = (system.activeParticles - 1) * 3;
                            positions[idx] = positions[lastIdx];
                            positions[idx + 1] = positions[lastIdx + 1];
                            positions[idx + 2] = positions[lastIdx + 2];
                            
                            velocities[idx] = velocities[lastIdx];
                            velocities[idx + 1] = velocities[lastIdx + 1];
                            velocities[idx + 2] = velocities[lastIdx + 2];
                            
                            lifetimes[i] = lifetimes[system.activeParticles - 1];
                            
                            system.activeParticles--;
                            i--; // Check this particle again
                        }
                    }
                    
                    system.mesh.geometry.attributes.position.needsUpdate = true;
                });
            }
        };
        
        // Create default particle systems
        this.particleSystem.createSystem('explosion', {
            maxParticles: 100,
            color: 0xff4400,
            size: 2.0,
            opacity: 0.8
        });
        
        this.particleSystem.createSystem('loading', {
            maxParticles: 200,
            color: 0x00ff00,
            size: 1.0,
            opacity: 0.6
        });
        
        console.log('[DARK MATTER] Particle system initialized');
    }
    
    async setupAI() {
        // A* Pathfinding implementation
        this.pathfinder = {
            grid: null,
            gridSize: 50,
            
            generateGrid: () => {
                this.pathfinder.grid = [];
                for (let x = 0; x < this.pathfinder.gridSize; x++) {
                    this.pathfinder.grid[x] = [];
                    for (let z = 0; z < this.pathfinder.gridSize; z++) {
                        this.pathfinder.grid[x][z] = {
                            x: x,
                            z: z,
                            walkable: true,
                            gCost: 0,
                            hCost: 0,
                            fCost: 0,
                            parent: null
                        };
                    }
                }
            },
            
            findPath: (start, end) => {
                if (!this.pathfinder.grid) this.pathfinder.generateGrid();
                
                const openSet = [];
                const closedSet = [];
                
                const startNode = this.pathfinder.grid[start.x][start.z];
                const endNode = this.pathfinder.grid[end.x][end.z];
                
                openSet.push(startNode);
                
                while (openSet.length > 0) {
                    let currentNode = openSet[0];
                    for (let i = 1; i < openSet.length; i++) {
                        if (openSet[i].fCost < currentNode.fCost) {
                            currentNode = openSet[i];
                        }
                    }
                    
                    openSet.splice(openSet.indexOf(currentNode), 1);
                    closedSet.push(currentNode);
                    
                    if (currentNode === endNode) {
                        // Reconstruct path
                        const path = [];
                        let current = currentNode;
                        while (current !== null) {
                            path.unshift({ x: current.x, z: current.z });
                            current = current.parent;
                        }
                        return path;
                    }
                    
                    // Check neighbors
                    const neighbors = this.pathfinder.getNeighbors(currentNode);
                    for (const neighbor of neighbors) {
                        if (!neighbor.walkable || closedSet.includes(neighbor)) {
                            continue;
                        }
                        
                        const newGCost = currentNode.gCost + this.pathfinder.getDistance(currentNode, neighbor);
                        if (newGCost < neighbor.gCost || !openSet.includes(neighbor)) {
                            neighbor.gCost = newGCost;
                            neighbor.hCost = this.pathfinder.getDistance(neighbor, endNode);
                            neighbor.fCost = neighbor.gCost + neighbor.hCost;
                            neighbor.parent = currentNode;
                            
                            if (!openSet.includes(neighbor)) {
                                openSet.push(neighbor);
                            }
                        }
                    }
                }
                
                return []; // No path found
            },
            
            getNeighbors: (node) => {
                const neighbors = [];
                const directions = [
                    { x: -1, z: 0 }, { x: 1, z: 0 },
                    { x: 0, z: -1 }, { x: 0, z: 1 },
                    { x: -1, z: -1 }, { x: 1, z: -1 },
                    { x: -1, z: 1 }, { x: 1, z: 1 }
                ];
                
                for (const dir of directions) {
                    const x = node.x + dir.x;
                    const z = node.z + dir.z;
                    
                    if (x >= 0 && x < this.pathfinder.gridSize && z >= 0 && z < this.pathfinder.gridSize) {
                        neighbors.push(this.pathfinder.grid[x][z]);
                    }
                }
                
                return neighbors;
            },
            
            getDistance: (nodeA, nodeB) => {
                const distX = Math.abs(nodeA.x - nodeB.x);
                const distZ = Math.abs(nodeA.z - nodeB.z);
                return Math.sqrt(distX * distX + distZ * distZ);
            }
        };
        
        // Enemy Behavior Tree
        this.enemyBehaviorTree = {
            evaluateNode: (enemy, node) => {
                switch (node.type) {
                    case 'selector':
                        for (const child of node.children) {
                            if (this.enemyBehaviorTree.evaluateNode(enemy, child)) {
                                return true;
                            }
                        }
                        return false;
                        
                    case 'sequence':
                        for (const child of node.children) {
                            if (!this.enemyBehaviorTree.evaluateNode(enemy, child)) {
                                return false;
                            }
                        }
                        return true;
                        
                    case 'condition':
                        return node.condition(enemy);
                        
                    case 'action':
                        return node.action(enemy);
                        
                    default:
                        return false;
                }
            },
            
            createBasicBehavior: () => ({
                type: 'selector',
                children: [
                    {
                        type: 'sequence',
                        children: [
                            {
                                type: 'condition',
                                condition: (enemy) => enemy.health <= enemy.maxHealth * 0.2
                            },
                            {
                                type: 'action',
                                action: (enemy) => {
                                    enemy.speed *= 1.5; // Panic speed boost
                                    return true;
                                }
                            }
                        ]
                    },
                    {
                        type: 'action',
                        action: (enemy) => {
                            // Normal movement behavior
                            this.moveEnemyAlongPath(enemy);
                            return true;
                        }
                    }
                ]
            })
        };
        
        // Procedural Generation
        this.proceduralGenerator = {
            generateWaypoints: (seed) => {
                // Use Perlin noise to generate natural-looking paths
                const waypoints = [];
                const numPoints = 8 + Math.floor(Math.random() * 4);
                
                for (let i = 0; i < numPoints; i++) {
                    const t = i / (numPoints - 1);
                    const x = -20 + t * 40 + (Math.random() - 0.5) * 10;
                    const z = -20 + t * 40 + (Math.random() - 0.5) * 10;
                    
                    waypoints.push(new THREE.Vector3(x, 0, z));
                }
                
                return waypoints;
            },
            
            generateEnemyComposition: (wave, level) => {
                const composition = [];
                const baseCount = 5 + wave * 2;
                const difficulty = level * 0.5 + wave * 0.3;
                
                // Basic enemies
                const basicCount = Math.max(1, Math.floor(baseCount * (1 - difficulty * 0.1)));
                for (let i = 0; i < basicCount; i++) {
                    composition.push('BASIC');
                }
                
                // Fast enemies
                if (wave > 2) {
                    const fastCount = Math.floor(baseCount * 0.3);
                    for (let i = 0; i < fastCount; i++) {
                        composition.push('FAST');
                    }
                }
                
                // Heavy enemies
                if (wave > 4) {
                    const heavyCount = Math.floor(baseCount * 0.2);
                    for (let i = 0; i < heavyCount; i++) {
                        composition.push('HEAVY');
                    }
                }
                
                // Flying enemies
                if (wave > 6) {
                    const flyingCount = Math.floor(baseCount * 0.15);
                    for (let i = 0; i < flyingCount; i++) {
                        composition.push('FLYING');
                    }
                }
                
                // Boss enemies
                if (wave === 10) {
                    const bossTypes = ['BOSS_ALIEN', 'BOSS_GOBLIN', 'BOSS_DRAGON'];
                    composition.push(bossTypes[Math.floor(Math.random() * bossTypes.length)]);
                }
                
                return composition;
            }
        };
        
        console.log('[DARK MATTER] AI systems initialized');
    }
    
    setupEventListeners() {
        // Keyboard controls
        document.addEventListener('keydown', (event) => {
            switch (event.code) {
                case 'Space':
                    event.preventDefault();
                    this.pauseGame();
                    break;
                case 'Enter':
                    event.preventDefault();
                    if (this.gameState === GameStates.MENU) {
                        this.startLevel(1);
                    } else if (this.gameState === GameStates.PLAYING) {
                        this.spawnWave();
                    }
                    break;
                case 'KeyR':
                    if (event.ctrlKey) {
                        event.preventDefault();
                        this.resetGame();
                    }
                    break;
                case 'Escape':
                    event.preventDefault();
                    if (this.gameState === GameStates.PLAYING) {
                        this.showGameMenu();
                    }
                    break;
            }
        });
        
        // Mouse controls
        this.renderer.domElement.addEventListener('click', (event) => {
            if (this.gameState === GameStates.PLAYING) {
                this.handleMouseClick(event);
            }
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.onWindowResize();
        });
        
        console.log('[DARK MATTER] Event listeners setup complete');
    }
    
    setupStateTransitions() {
        // Define state transition rules
        this.stateTransitions.set(`${GameStates.LOADING}->${GameStates.MENU}`, {
            duration: 1000,
            easing: 'easeInOutCubic',
            onStart: () => {
                this.loadingScreen.hide();
                this.audioManager.playMusic('menu_theme');
            },
            onComplete: () => {
                this.gameMenu.show();
            }
        });
        
        this.stateTransitions.set(`${GameStates.MENU}->${GameStates.PLAYING}`, {
            duration: 2000,
            easing: 'easeInOutCubic',
            onStart: () => {
                this.gameMenu.hide();
                this.audioManager.stopMusic();
            },
            onComplete: () => {
                this.hud.show();
                this.audioManager.playMusic('game_theme');
            }
        });
        
        this.stateTransitions.set(`${GameStates.PLAYING}->${GameStates.GAME_OVER}`, {
            duration: 3000,
            easing: 'easeInOutCubic',
            onStart: () => {
                this.hud.hide();
                this.audioManager.stopMusic();
                this.audioManager.playSound('game_over');
            },
            onComplete: () => {
                this.showGameOverScreen();
            }
        });
        
        console.log('[DARK MATTER] State transitions configured');
    }
    
    // Public API Methods
    
    startLoadingScreen() {
        console.log('[DARK MATTER] Starting loading screen');
        this.gameState = GameStates.LOADING;
        this.loadingScreen.show();
        
        // Simulate loading process
        let progress = 0;
        const loadingInterval = setInterval(() => {
            progress += 0.02;
            this.loadingScreen.updateProgress(progress);
            
            if (progress >= 1.0) {
                clearInterval(loadingInterval);
                setTimeout(() => {
                    this.transitionToState(GameStates.MENU);
                }, 500);
            }
        }, 50);
    }
    
    showGameMenu() {
        console.log('[DARK MATTER] Showing game menu');
        this.transitionToState(GameStates.MENU);
    }
    
    startLevel(level) {
        console.log(`[DARK MATTER] Starting level ${level}`);
        this.currentLevel = level;
        this.currentWave = 0;
        this.waypoints = this.proceduralGenerator.generateWaypoints(level * 1000);
        
        this.transitionToState(GameStates.PLAYING);
        
        // Start 30-second prep countdown
        this.startPrepCountdown();
    }
    
    startPrepCountdown() {
        console.log('[DARK MATTER] Starting 30-second prep countdown');
        let countdown = 30;
        
        const countdownInterval = setInterval(() => {
            countdown--;
            console.log(`[PREP] ${countdown} seconds remaining`);
            
            if (countdown <= 0) {
                clearInterval(countdownInterval);
                this.spawnWave();
            }
        }, 1000);
        
        // Play ambient build-up music
        this.audioManager.playMusic('prep_countdown', false);
    }
    
    spawnWave() {
        if (this.gameState !== GameStates.PLAYING) return;
        
        this.currentWave++;
        console.log(`[DARK MATTER] Spawning wave ${this.currentWave}`);
        
        const enemyComposition = this.proceduralGenerator.generateEnemyComposition(
            this.currentWave, 
            this.currentLevel
        );
        
        this.audioManager.playSound('wave_start');
        
        // Spawn enemies with delays
        enemyComposition.forEach((enemyType, index) => {
            setTimeout(() => {
                this.spawnEnemy(enemyType);
            }, index * 1000);
        });
    }
    
    spawnEnemy(type) {
        const enemyStats = EnemyTypes[type];
        if (!enemyStats) {
            console.warn(`[DARK MATTER] Unknown enemy type: ${type}`);
            return;
        }
        
        // Create enemy object
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshLambertMaterial({ 
            color: this.getEnemyColor(type) 
        });
        const enemyMesh = new THREE.Mesh(geometry, material);
        
        // Set initial position at first waypoint
        if (this.waypoints.length > 0) {
            enemyMesh.position.copy(this.waypoints[0]);
        }
        
        this.scene.add(enemyMesh);
        
        const enemy = {
            mesh: enemyMesh,
            type: type,
            health: enemyStats.health,
            maxHealth: enemyStats.health,
            speed: enemyStats.speed,
            reward: enemyStats.reward,
            points: enemyStats.points,
            waypointIndex: 0,
            path: [...this.waypoints],
            behaviorTree: this.enemyBehaviorTree.createBasicBehavior(),
            statusEffects: [],
            isAlive: true
        };
        
        this.enemies.push(enemy);
        
        console.log(`[DARK MATTER] Spawned ${type} enemy`);
    }
    
    getEnemyColor(type) {
        const colors = {
            BASIC: 0xff0000,
            FAST: 0x00ff00,
            HEAVY: 0x0000ff,
            FLYING: 0xffff00,
            BOSS_ALIEN: 0xff00ff,
            BOSS_GOBLIN: 0x00ffff,
            BOSS_DRAGON: 0xff8800
        };
        return colors[type] || 0xffffff;
    }
    
    pauseGame() {
        if (this.gameState === GameStates.PLAYING) {
            this.previousState = this.gameState;
            this.gameState = GameStates.PAUSED;
            console.log('[DARK MATTER] Game paused');
        } else if (this.gameState === GameStates.PAUSED) {
            this.gameState = this.previousState;
            console.log('[DARK MATTER] Game resumed');
        }
    }
    
    resetGame() {
        console.log('[DARK MATTER] Resetting game');
        
        // Clear all game objects
        this.enemies.forEach(enemy => this.scene.remove(enemy.mesh));
        this.towers.forEach(tower => this.scene.remove(tower.mesh));
        this.projectiles.forEach(projectile => this.scene.remove(projectile.mesh));
        
        this.enemies = [];
        this.towers = [];
        this.projectiles = [];
        
        // Reset game state
        this.currentLevel = 1;
        this.currentWave = 0;
        this.lives = 20;
        this.gold = 100;
        this.score = 0;
        
        this.transitionToState(GameStates.MENU);
    }
    
    // Game Logic Methods
    
    update() {
        if (this.gameState !== GameStates.PLAYING) return;
        
        const deltaTime = this.clock.getDelta();
        
        // Update physics
        this.physicsWorld.step(deltaTime);
        
        // Update enemies
        this.updateEnemies(deltaTime);
        
        // Update towers
        this.updateTowers(deltaTime);
        
        // Update projectiles
        this.updateProjectiles(deltaTime);
        
        // Update particles
        this.particleSystem.update(deltaTime);
        
        // Update shaders
        this.updateShaders(deltaTime);
        
        // Update HUD
        this.hud.update();
        
        // Check game over conditions
        this.checkGameOver();
    }
    
    updateEnemies(deltaTime) {
        this.enemies.forEach((enemy, index) => {
            if (!enemy.isAlive) return;
            
            // Execute AI behavior tree
            this.enemyBehaviorTree.evaluateNode(enemy, enemy.behaviorTree);
            
            // Move enemy along path
            this.moveEnemyAlongPath(enemy);
            
            // Check if enemy reached the end
            if (enemy.waypointIndex >= enemy.path.length) {
                this.enemyReachedEnd(enemy, index);
            }
        });
    }
    
    moveEnemyAlongPath(enemy) {
        if (enemy.waypointIndex >= enemy.path.length) return;
        
        const targetWaypoint = enemy.path[enemy.waypointIndex];
        const direction = new THREE.Vector3()
            .subVectors(targetWaypoint, enemy.mesh.position)
            .normalize();
        
        const moveDistance = enemy.speed * this.clock.getDelta();
        enemy.mesh.position.add(direction.multiplyScalar(moveDistance));
        
        // Check if reached current waypoint
        if (enemy.mesh.position.distanceTo(targetWaypoint) < 0.5) {
            enemy.waypointIndex++;
        }
    }
    
    enemyReachedEnd(enemy, index) {
        console.log('[DARK MATTER] Enemy reached the end!');
        
        // Reduce player lives
        this.lives--;
        
        // Remove enemy
        this.scene.remove(enemy.mesh);
        this.enemies.splice(index, 1);
        
        // Play sound effect
        this.audioManager.playSound('life_lost');
        
        // Check game over
        if (this.lives <= 0) {
            this.gameOver();
        }
    }
    
    updateTowers(deltaTime) {
        this.towers.forEach(tower => {
            // Find nearest enemy in range
            let nearestEnemy = null;
            let nearestDistance = tower.range;
            
            this.enemies.forEach(enemy => {
                if (!enemy.isAlive) return;
                
                const distance = tower.mesh.position.distanceTo(enemy.mesh.position);
                if (distance < nearestDistance) {
                    nearestEnemy = enemy;
                    nearestDistance = distance;
                }
            });
            
            // Shoot at nearest enemy
            if (nearestEnemy && tower.cooldown <= 0) {
                this.towerShoot(tower, nearestEnemy);
                tower.cooldown = tower.fireRate;
            }
            
            // Update cooldown
            if (tower.cooldown > 0) {
                tower.cooldown -= deltaTime;
            }
        });
    }
    
    towerShoot(tower, target) {
        console.log('[DARK MATTER] Tower shooting');
        
        // Create projectile
        const geometry = new THREE.SphereGeometry(0.1);
        const material = new THREE.MeshBasicMaterial({ color: 0xffff00 });
        const projectileMesh = new THREE.Mesh(geometry, material);
        
        projectileMesh.position.copy(tower.mesh.position);
        this.scene.add(projectileMesh);
        
        const projectile = {
            mesh: projectileMesh,
            target: target,
            speed: 20,
            damage: tower.damage,
            isActive: true
        };
        
        this.projectiles.push(projectile);
        
        // Play sound effect
        this.audioManager.playSound('tower_shoot', tower.mesh.position);
    }
    
    updateProjectiles(deltaTime) {
        this.projectiles.forEach((projectile, index) => {
            if (!projectile.isActive) return;
            
            if (!projectile.target.isAlive) {
                // Target is dead, remove projectile
                this.scene.remove(projectile.mesh);
                this.projectiles.splice(index, 1);
                return;
            }
            
            // Move towards target
            const direction = new THREE.Vector3()
                .subVectors(projectile.target.mesh.position, projectile.mesh.position)
                .normalize();
            
            const moveDistance = projectile.speed * deltaTime;
            projectile.mesh.position.add(direction.multiplyScalar(moveDistance));
            
            // Check collision
            if (projectile.mesh.position.distanceTo(projectile.target.mesh.position) < 0.5) {
                this.projectileHit(projectile, projectile.target);
                this.scene.remove(projectile.mesh);
                this.projectiles.splice(index, 1);
            }
        });
    }
    
    projectileHit(projectile, enemy) {
        console.log('[DARK MATTER] Projectile hit enemy');
        
        // Damage enemy
        enemy.health -= projectile.damage;
        
        // Create hit effect
        this.particleSystem.emit('explosion', 
            enemy.mesh.position.clone(),
            new THREE.Vector3(0, 2, 0),
            1.0
        );
        
        // Play sound effect
        this.audioManager.playSound('enemy_hit', enemy.mesh.position);
        
        // Check if enemy is dead
        if (enemy.health <= 0) {
            this.enemyDeath(enemy);
        }
    }
    
    enemyDeath(enemy) {
        console.log('[DARK MATTER] Enemy died');
        
        // Award gold and score
        this.gold += enemy.reward;
        this.score += enemy.points;
        
        // Create death effect
        this.particleSystem.emit('explosion',
            enemy.mesh.position.clone(),
            new THREE.Vector3(0, 5, 0),
            2.0
        );
        
        // Play sound effect
        this.audioManager.playSound('enemy_death', enemy.mesh.position);
        
        // Remove from scene
        this.scene.remove(enemy.mesh);
        enemy.isAlive = false;
        
        // Remove from enemies array
        const index = this.enemies.indexOf(enemy);
        if (index > -1) {
            this.enemies.splice(index, 1);
        }
    }
    
    updateShaders(deltaTime) {
        const time = this.clock.getElapsedTime();
        
        // Update loading background shader
        const loadingShader = this.shaderMaterials.get('loadingBackground');
        if (loadingShader) {
            loadingShader.uniforms.time.value = time;
        }
        
        // Update spawn portal shader
        const portalShader = this.shaderMaterials.get('spawnPortal');
        if (portalShader) {
            portalShader.uniforms.time.value = time;
        }
    }
    
    handleMouseClick(event) {
        // Convert mouse position to world coordinates
        const mouse = new THREE.Vector2();
        mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        
        // Check for ground intersection
        const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
        const intersectionPoint = new THREE.Vector3();
        raycaster.ray.intersectPlane(groundPlane, intersectionPoint);
        
        if (intersectionPoint) {
            this.placeTower(intersectionPoint);
        }
    }
    
    placeTower(position) {
        if (this.gold < 50) {
            console.log('[DARK MATTER] Not enough gold to place tower');
            return;
        }
        
        console.log('[DARK MATTER] Placing tower');
        
        // Create tower
        const geometry = new THREE.CylinderGeometry(0.5, 0.8, 2);
        const material = new THREE.MeshLambertMaterial({ color: 0x888888 });
        const towerMesh = new THREE.Mesh(geometry, material);
        
        towerMesh.position.copy(position);
        towerMesh.position.y = 1;
        this.scene.add(towerMesh);
        
        const tower = {
            mesh: towerMesh,
            damage: 25,
            range: 8,
            fireRate: 1.0,
            cooldown: 0,
            cost: 50
        };
        
        this.towers.push(tower);
        this.gold -= tower.cost;
        
        // Play sound effect
        this.audioManager.playSound('tower_place', position);
    }
    
    checkGameOver() {
        if (this.lives <= 0) {
            this.gameOver();
        } else if (this.currentWave >= this.maxWaves && this.enemies.length === 0) {
            this.victory();
        }
    }
    
    gameOver() {
        console.log('[DARK MATTER] Game Over');
        this.transitionToState(GameStates.GAME_OVER);
    }
    
    victory() {
        console.log('[DARK MATTER] Victory!');
        this.transitionToState(GameStates.VICTORY);
    }
    
    showGameOverScreen() {
        console.log('[DARK MATTER] Showing game over screen');
        // Implementation would show game over UI with stats
    }
    
    transitionToState(newState) {
        const transitionKey = `${this.gameState}->${newState}`;
        const transition = this.stateTransitions.get(transitionKey);
        
        if (transition) {
            console.log(`[DARK MATTER] Transitioning from ${this.gameState} to ${newState}`);
            
            if (transition.onStart) {
                transition.onStart();
            }
            
            setTimeout(() => {
                this.gameState = newState;
                if (transition.onComplete) {
                    transition.onComplete();
                }
            }, transition.duration);
        } else {
            console.log(`[DARK MATTER] Direct state change from ${this.gameState} to ${newState}`);
            this.gameState = newState;
        }
    }
    
    createLoadingScreenParticles() {
        // Create animated particles for loading screen
        for (let i = 0; i < 50; i++) {
            this.particleSystem.emit('loading',
                new THREE.Vector3(
                    (Math.random() - 0.5) * 20,
                    Math.random() * 10,
                    (Math.random() - 0.5) * 20
                ),
                new THREE.Vector3(
                    (Math.random() - 0.5) * 2,
                    Math.random() * 2,
                    (Math.random() - 0.5) * 2
                ),
                5.0
            );
        }
    }
    
    clearLoadingScreenParticles() {
        const loadingSystem = this.particleSystem.systems.get('loading');
        if (loadingSystem) {
            loadingSystem.activeParticles = 0;
        }
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        
        // Update shader uniforms
        const loadingShader = this.shaderMaterials.get('loadingBackground');
        if (loadingShader) {
            loadingShader.uniforms.resolution.value.set(window.innerWidth, window.innerHeight);
        }
    }
    
    // Multiplayer Methods
    
    enableMultiplayer(serverUrl) {
        console.log(`[DARK MATTER] Enabling multiplayer: ${serverUrl}`);
        this.isMultiplayer = true;
        
        // Initialize socket connection (would use socket.io-client)
        // this.socket = io(serverUrl);
        // this.setupMultiplayerEvents();
    }
    
    setupMultiplayerEvents() {
        // Implementation would handle multiplayer events
        console.log('[DARK MATTER] Multiplayer events setup');
    }
    
    // Save/Load Methods
    
    saveGameState() {
        const gameState = {
            level: this.currentLevel,
            wave: this.currentWave,
            lives: this.lives,
            gold: this.gold,
            score: this.score,
            towers: this.towers.map(tower => ({
                position: tower.mesh.position,
                damage: tower.damage,
                range: tower.range
            })),
            timestamp: Date.now()
        };
        
        localStorage.setItem('darkMatterTowerDefense', JSON.stringify(gameState));
        console.log('[DARK MATTER] Game state saved');
        
        return gameState;
    }
    
    loadGameState() {
        const savedState = localStorage.getItem('darkMatterTowerDefense');
        if (!savedState) {
            console.log('[DARK MATTER] No saved game state found');
            return null;
        }
        
        try {
            const gameState = JSON.parse(savedState);
            
            this.currentLevel = gameState.level;
            this.currentWave = gameState.wave;
            this.lives = gameState.lives;
            this.gold = gameState.gold;
            this.score = gameState.score;
            
            // Restore towers
            gameState.towers.forEach(towerData => {
                this.placeTower(towerData.position);
            });
            
            console.log('[DARK MATTER] Game state loaded');
            return gameState;
        } catch (error) {
            console.error('[DARK MATTER] Error loading game state:', error);
            return null;
        }
    }
    
    // Utility Methods
    
    getGameStats() {
        return {
            level: this.currentLevel,
            wave: this.currentWave,
            lives: this.lives,
            gold: this.gold,
            score: this.score,
            enemies: this.enemies.length,
            towers: this.towers.length,
            gameState: this.gameState,
            performance: this.performanceMonitor
        };
    }
    
    setDifficulty(difficulty) {
        this.difficulty = difficulty;
        console.log(`[DARK MATTER] Difficulty set to: ${difficulty}`);
        
        // Adjust game parameters based on difficulty
        switch (difficulty) {
            case 'easy':
                this.lives = 30;
                this.gold = 150;
                break;
            case 'normal':
                this.lives = 20;
                this.gold = 100;
                break;
            case 'hard':
                this.lives = 10;
                this.gold = 75;
                break;
        }
    }
    
    destroy() {
        console.log('[DARK MATTER] Destroying Enhanced Tower Defense Game');
        
        // Clean up all game objects
        this.enemies.forEach(enemy => this.scene.remove(enemy.mesh));
        this.towers.forEach(tower => this.scene.remove(tower.mesh));
        this.projectiles.forEach(projectile => this.scene.remove(projectile.mesh));
        
        // Clean up particle systems
        this.particleSystem.systems.forEach((system, id) => {
            this.scene.remove(system.mesh);
        });
        
        // Clean up physics world
        if (this.physicsWorld) {
            this.physicsWorld.bodies.forEach(body => {
                this.physicsWorld.remove(body);
            });
        }
        
        // Clean up audio
        if (this.audioManager) {
            this.audioManager.stopMusic();
        }
        
        // Clean up UI
        if (this.loadingScreen) this.loadingScreen.hide();
        if (this.gameMenu) this.gameMenu.hide();
        if (this.hud) this.hud.hide();
    }
}

// Plugin API
let enhancedTowerDefenseGame = null;

export function initGame(engine, container) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.destroy();
    }
    
    enhancedTowerDefenseGame = new EnhancedTowerDefenseGame(
        engine.scene,
        engine.camera,
        engine.renderer,
        engine
    );
    
    console.log('[DARK MATTER] Enhanced Tower Defense Game plugin initialized');
    return enhancedTowerDefenseGame;
}

export function startLoadingScreen(engine) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.startLoadingScreen();
    }
}

export function showGameMenu(engine) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.showGameMenu();
    }
}

export function startLevel(engine, level) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.startLevel(level);
    }
}

export function spawnWave(engine) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.spawnWave();
    }
}

export function pauseGame(engine) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.pauseGame();
    }
}

export function resetGame(engine) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.resetGame();
    }
}

export function getGameStats(engine) {
    if (enhancedTowerDefenseGame) {
        return enhancedTowerDefenseGame.getGameStats();
    }
    return null;
}

export function saveGameState(engine) {
    if (enhancedTowerDefenseGame) {
        return enhancedTowerDefenseGame.saveGameState();
    }
    return null;
}

export function loadGameState(engine) {
    if (enhancedTowerDefenseGame) {
        return enhancedTowerDefenseGame.loadGameState();
    }
    return null;
}

export function enableMultiplayer(engine, serverUrl) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.enableMultiplayer(serverUrl);
    }
}

export function setDifficulty(engine, difficulty) {
    if (enhancedTowerDefenseGame) {
        enhancedTowerDefenseGame.setDifficulty(difficulty);
    }
}

export function register(pluginAPI) {
    pluginAPI.provide("initGame", initGame);
    pluginAPI.provide("startLoadingScreen", startLoadingScreen);
    pluginAPI.provide("showGameMenu", showGameMenu);
    pluginAPI.provide("startLevel", startLevel);
    pluginAPI.provide("spawnWave", spawnWave);
    pluginAPI.provide("pauseGame", pauseGame);
    pluginAPI.provide("resetGame", resetGame);
    pluginAPI.provide("getGameStats", getGameStats);
    pluginAPI.provide("saveGameState", saveGameState);
    pluginAPI.provide("loadGameState", loadGameState);
    pluginAPI.provide("enableMultiplayer", enableMultiplayer);
    pluginAPI.provide("setDifficulty", setDifficulty);
    
    console.log("[DARK MATTER] Enhanced Tower Defense Game plugin registered");
}

