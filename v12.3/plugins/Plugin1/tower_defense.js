/**
 * DARK MATTER - Three.js Tower Defense Plugin
 * A comprehensive tower defense game implementation using Three.js
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

class TowerDefenseGame {
    constructor(container, darkMatterEngine) {
        this.container = container;
        this.engine = darkMatterEngine;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.clock = new THREE.Clock();
        
        // Game state
        this.gameState = {
            isPlaying: false,
            isPaused: false,
            currentWave: 0,
            lives: 20,
            gold: 100,
            score: 0
        };
        
        // Game objects
        this.towers = [];
        this.enemies = [];
        this.projectiles = [];
        this.path = [];
        this.terrain = null;
        
        // Managers
        this.waveManager = null;
        this.pathfinding = null;
        this.effectsManager = null;
        
        this.init();
    }
    
    init() {
        this.setupScene();
        this.setupLighting();
        this.createTerrain();
        this.createPath();
        this.setupManagers();
        this.setupEventListeners();
        this.animate();
        
        console.log('[DARK MATTER] Tower Defense initialized');
    }
    
    setupScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x87CEEB); // Sky blue
        this.scene.fog = new THREE.Fog(0x87CEEB, 50, 200);
        
        // Camera
        this.camera = new THREE.PerspectiveCamera(
            75, 
            this.container.clientWidth / this.container.clientHeight, 
            0.1, 
            1000
        );
        this.camera.position.set(0, 25, 25);
        this.camera.lookAt(0, 0, 0);
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.container.appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.maxPolarAngle = Math.PI / 2.2;
        this.controls.minDistance = 10;
        this.controls.maxDistance = 100;
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.4);
        this.scene.add(ambientLight);
        
        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 25);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 200;
        directionalLight.shadow.camera.left = -50;
        directionalLight.shadow.camera.right = 50;
        directionalLight.shadow.camera.top = 50;
        directionalLight.shadow.camera.bottom = -50;
        this.scene.add(directionalLight);
        
        // Point light for dramatic effect
        const pointLight = new THREE.PointLight(0xff6600, 0.5, 30);
        pointLight.position.set(0, 10, 0);
        this.scene.add(pointLight);
    }
    
    createTerrain() {
        const geometry = new THREE.PlaneGeometry(60, 60, 32, 32);
        
        // Add some height variation
        const vertices = geometry.attributes.position.array;
        for (let i = 0; i < vertices.length; i += 3) {
            vertices[i + 2] = Math.random() * 2 - 1; // Z coordinate (height)
        }
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
        
        const material = new THREE.MeshLambertMaterial({ 
            color: 0x4a7c59,
            wireframe: false
        });
        
        this.terrain = new THREE.Mesh(geometry, material);
        this.terrain.rotation.x = -Math.PI / 2;
        this.terrain.receiveShadow = true;
        this.scene.add(this.terrain);
    }
    
    createPath() {
        // Define waypoints for enemy path
        const waypoints = [
            new THREE.Vector3(-25, 1, 0),
            new THREE.Vector3(-15, 1, -10),
            new THREE.Vector3(0, 1, -15),
            new THREE.Vector3(15, 1, -10),
            new THREE.Vector3(20, 1, 5),
            new THREE.Vector3(10, 1, 15),
            new THREE.Vector3(-5, 1, 20),
            new THREE.Vector3(-20, 1, 15),
            new THREE.Vector3(-25, 1, 0)
        ];
        
        this.path = waypoints;
        
        // Visualize path
        const pathGeometry = new THREE.BufferGeometry().setFromPoints(waypoints);
        const pathMaterial = new THREE.LineBasicMaterial({ 
            color: 0xff0000, 
            linewidth: 3 
        });
        const pathLine = new THREE.Line(pathGeometry, pathMaterial);
        this.scene.add(pathLine);
        
        // Add path markers
        waypoints.forEach((point, index) => {
            const markerGeometry = new THREE.SphereGeometry(0.5, 8, 6);
            const markerMaterial = new THREE.MeshBasicMaterial({ 
                color: index === 0 ? 0x00ff00 : (index === waypoints.length - 1 ? 0xff0000 : 0xffff00)
            });
            const marker = new THREE.Mesh(markerGeometry, markerMaterial);
            marker.position.copy(point);
            this.scene.add(marker);
        });
    }
    
    setupManagers() {
        this.waveManager = new WaveManager(this);
        this.pathfinding = new Pathfinding(this.path);
        this.effectsManager = new EffectsManager(this.scene);
    }
    
    setupEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Mouse events for tower placement
        this.renderer.domElement.addEventListener('click', (event) => {
            this.onMouseClick(event);
        });
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    onMouseClick(event) {
        if (!this.gameState.isPlaying || this.gameState.isPaused) return;
        
        const mouse = new THREE.Vector2();
        const rect = this.renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        
        const intersects = raycaster.intersectObject(this.terrain);
        if (intersects.length > 0) {
            const point = intersects[0].point;
            this.placeTower(point);
        }
    }
    
    placeTower(position) {
        if (this.gameState.gold >= 50) {
            const tower = new Tower(position, this.scene, this);
            this.towers.push(tower);
            this.gameState.gold -= 50;
            this.updateUI();
            
            console.log('[DARK MATTER] Tower placed at:', position);
        }
    }
    
    spawnEnemy(type = 'basic') {
        const enemy = new Enemy(type, this.path, this.scene, this);
        this.enemies.push(enemy);
        console.log('[DARK MATTER] Enemy spawned:', type);
    }
    
    startWave() {
        if (!this.gameState.isPlaying) {
            this.gameState.isPlaying = true;
        }
        this.waveManager.startWave();
    }
    
    pauseGame() {
        this.gameState.isPaused = !this.gameState.isPaused;
        console.log('[DARK MATTER] Game paused:', this.gameState.isPaused);
    }
    
    resetGame() {
        // Clear all game objects
        this.towers.forEach(tower => tower.destroy());
        this.enemies.forEach(enemy => enemy.destroy());
        this.projectiles.forEach(projectile => projectile.destroy());
        
        this.towers = [];
        this.enemies = [];
        this.projectiles = [];
        
        // Reset game state
        this.gameState = {
            isPlaying: false,
            isPaused: false,
            currentWave: 0,
            lives: 20,
            gold: 100,
            score: 0
        };
        
        this.updateUI();
        console.log('[DARK MATTER] Game reset');
    }
    
    update(deltaTime) {
        if (!this.gameState.isPlaying || this.gameState.isPaused) return;
        
        // Update towers
        this.towers.forEach(tower => tower.update(deltaTime));
        
        // Update enemies
        this.enemies = this.enemies.filter(enemy => {
            enemy.update(deltaTime);
            if (enemy.isDead) {
                this.gameState.gold += enemy.reward;
                this.gameState.score += enemy.points;
                enemy.destroy();
                return false;
            }
            if (enemy.reachedEnd) {
                this.gameState.lives--;
                enemy.destroy();
                return false;
            }
            return true;
        });
        
        // Update projectiles
        this.projectiles = this.projectiles.filter(projectile => {
            projectile.update(deltaTime);
            if (projectile.shouldDestroy) {
                projectile.destroy();
                return false;
            }
            return true;
        });
        
        // Update wave manager
        this.waveManager.update(deltaTime);
        
        // Update effects
        this.effectsManager.update(deltaTime);
        
        // Check game over conditions
        if (this.gameState.lives <= 0) {
            this.gameOver();
        }
        
        this.updateUI();
    }
    
    gameOver() {
        this.gameState.isPlaying = false;
        console.log('[DARK MATTER] Game Over! Final Score:', this.gameState.score);
        // Trigger game over UI
        if (this.engine && this.engine.events) {
            this.engine.events.emit('gameOver', { score: this.gameState.score });
        }
    }
    
    updateUI() {
        // Update UI elements
        const event = new CustomEvent('towerDefenseUpdate', {
            detail: { ...this.gameState }
        });
        window.dispatchEvent(event);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        
        const deltaTime = this.clock.getDelta();
        
        this.update(deltaTime);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    destroy() {
        // Clean up resources
        this.resetGame();
        if (this.renderer) {
            this.container.removeChild(this.renderer.domElement);
            this.renderer.dispose();
        }
        console.log('[DARK MATTER] Tower Defense destroyed');
    }
}

// Tower class
class Tower {
    constructor(position, scene, game) {
        this.position = position.clone();
        this.scene = scene;
        this.game = game;
        this.range = 8;
        this.damage = 25;
        this.fireRate = 1; // shots per second
        this.lastFireTime = 0;
        this.target = null;
        
        this.createMesh();
    }
    
    createMesh() {
        // Tower base
        const baseGeometry = new THREE.CylinderGeometry(1, 1.5, 2, 8);
        const baseMaterial = new THREE.MeshLambertMaterial({ color: 0x666666 });
        this.base = new THREE.Mesh(baseGeometry, baseMaterial);
        this.base.position.copy(this.position);
        this.base.castShadow = true;
        this.scene.add(this.base);
        
        // Tower turret
        const turretGeometry = new THREE.BoxGeometry(0.8, 0.5, 1.5);
        const turretMaterial = new THREE.MeshLambertMaterial({ color: 0x444444 });
        this.turret = new THREE.Mesh(turretGeometry, turretMaterial);
        this.turret.position.copy(this.position);
        this.turret.position.y += 1.25;
        this.turret.castShadow = true;
        this.scene.add(this.turret);
        
        // Range indicator (initially hidden)
        const rangeGeometry = new THREE.RingGeometry(this.range - 0.1, this.range, 32);
        const rangeMaterial = new THREE.MeshBasicMaterial({ 
            color: 0x00ff00, 
            transparent: true, 
            opacity: 0.3,
            side: THREE.DoubleSide
        });
        this.rangeIndicator = new THREE.Mesh(rangeGeometry, rangeMaterial);
        this.rangeIndicator.position.copy(this.position);
        this.rangeIndicator.rotation.x = -Math.PI / 2;
        this.rangeIndicator.visible = false;
        this.scene.add(this.rangeIndicator);
    }
    
    update(deltaTime) {
        const currentTime = performance.now() / 1000;
        
        // Find target
        this.findTarget();
        
        // Aim at target
        if (this.target) {
            this.aimAt(this.target.position);
            
            // Fire if ready
            if (currentTime - this.lastFireTime >= 1 / this.fireRate) {
                this.fire();
                this.lastFireTime = currentTime;
            }
        }
    }
    
    findTarget() {
        this.target = null;
        let closestDistance = this.range;
        
        this.game.enemies.forEach(enemy => {
            const distance = this.position.distanceTo(enemy.position);
            if (distance <= this.range && distance < closestDistance) {
                this.target = enemy;
                closestDistance = distance;
            }
        });
    }
    
    aimAt(targetPosition) {
        const direction = new THREE.Vector3()
            .subVectors(targetPosition, this.position)
            .normalize();
        
        const angle = Math.atan2(direction.x, direction.z);
        this.turret.rotation.y = angle;
    }
    
    fire() {
        if (!this.target) return;
        
        const projectile = new Projectile(
            this.position.clone().add(new THREE.Vector3(0, 1.5, 0)),
            this.target,
            this.damage,
            this.scene,
            this.game
        );
        
        this.game.projectiles.push(projectile);
        
        // Muzzle flash effect
        this.game.effectsManager.createMuzzleFlash(this.turret.position);
    }
    
    showRange() {
        this.rangeIndicator.visible = true;
    }
    
    hideRange() {
        this.rangeIndicator.visible = false;
    }
    
    destroy() {
        this.scene.remove(this.base);
        this.scene.remove(this.turret);
        this.scene.remove(this.rangeIndicator);
    }
}

// Enemy class
class Enemy {
    constructor(type, path, scene, game) {
        this.type = type;
        this.path = path;
        this.scene = scene;
        this.game = game;
        this.pathIndex = 0;
        this.position = path[0].clone();
        this.health = this.getTypeStats().health;
        this.maxHealth = this.health;
        this.speed = this.getTypeStats().speed;
        this.reward = this.getTypeStats().reward;
        this.points = this.getTypeStats().points;
        this.isDead = false;
        this.reachedEnd = false;
        
        this.createMesh();
        this.createHealthBar();
    }
    
    getTypeStats() {
        const stats = {
            basic: { health: 100, speed: 3, reward: 10, points: 10 },
            fast: { health: 50, speed: 6, reward: 15, points: 15 },
            heavy: { health: 200, speed: 1.5, reward: 25, points: 25 },
            boss: { health: 500, speed: 2, reward: 100, points: 100 }
        };
        return stats[this.type] || stats.basic;
    }
    
    createMesh() {
        const colors = {
            basic: 0xff0000,
            fast: 0x00ff00,
            heavy: 0x0000ff,
            boss: 0xff00ff
        };
        
        const geometry = new THREE.BoxGeometry(1, 1, 1);
        const material = new THREE.MeshLambertMaterial({ 
            color: colors[this.type] || colors.basic 
        });
        
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.position.copy(this.position);
        this.mesh.castShadow = true;
        this.scene.add(this.mesh);
    }
    
    createHealthBar() {
        // Health bar background
        const bgGeometry = new THREE.PlaneGeometry(1.5, 0.2);
        const bgMaterial = new THREE.MeshBasicMaterial({ color: 0x000000 });
        this.healthBarBg = new THREE.Mesh(bgGeometry, bgMaterial);
        this.healthBarBg.position.copy(this.position);
        this.healthBarBg.position.y += 1.5;
        this.scene.add(this.healthBarBg);
        
        // Health bar foreground
        const fgGeometry = new THREE.PlaneGeometry(1.4, 0.15);
        const fgMaterial = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
        this.healthBar = new THREE.Mesh(fgGeometry, fgMaterial);
        this.healthBar.position.copy(this.position);
        this.healthBar.position.y += 1.51;
        this.scene.add(this.healthBar);
    }
    
    update(deltaTime) {
        if (this.isDead || this.reachedEnd) return;
        
        this.moveAlongPath(deltaTime);
        this.updateHealthBar();
        this.faceMovementDirection();
    }
    
    moveAlongPath(deltaTime) {
        if (this.pathIndex >= this.path.length - 1) {
            this.reachedEnd = true;
            return;
        }
        
        const target = this.path[this.pathIndex + 1];
        const direction = new THREE.Vector3()
            .subVectors(target, this.position)
            .normalize();
        
        const moveDistance = this.speed * deltaTime;
        const newPosition = this.position.clone()
            .add(direction.multiplyScalar(moveDistance));
        
        // Check if we've reached the next waypoint
        if (this.position.distanceTo(target) <= moveDistance) {
            this.pathIndex++;
            this.position.copy(target);
        } else {
            this.position.copy(newPosition);
        }
        
        this.mesh.position.copy(this.position);
        this.healthBarBg.position.copy(this.position);
        this.healthBarBg.position.y += 1.5;
        this.healthBar.position.copy(this.position);
        this.healthBar.position.y += 1.51;
    }
    
    updateHealthBar() {
        const healthPercent = this.health / this.maxHealth;
        this.healthBar.scale.x = healthPercent;
        
        // Change color based on health
        if (healthPercent > 0.6) {
            this.healthBar.material.color.setHex(0x00ff00);
        } else if (healthPercent > 0.3) {
            this.healthBar.material.color.setHex(0xffff00);
        } else {
            this.healthBar.material.color.setHex(0xff0000);
        }
    }
    
    faceMovementDirection() {
        if (this.pathIndex < this.path.length - 1) {
            const target = this.path[this.pathIndex + 1];
            const direction = new THREE.Vector3()
                .subVectors(target, this.position)
                .normalize();
            
            const angle = Math.atan2(direction.x, direction.z);
            this.mesh.rotation.y = angle;
        }
    }
    
    takeDamage(damage) {
        this.health -= damage;
        if (this.health <= 0) {
            this.isDead = true;
            this.game.effectsManager.createExplosion(this.position);
        }
        
        // Damage number effect
        this.game.effectsManager.createDamageNumber(this.position, damage);
    }
    
    destroy() {
        this.scene.remove(this.mesh);
        this.scene.remove(this.healthBar);
        this.scene.remove(this.healthBarBg);
    }
}

// Projectile class
class Projectile {
    constructor(startPosition, target, damage, scene, game) {
        this.startPosition = startPosition.clone();
        this.target = target;
        this.damage = damage;
        this.scene = scene;
        this.game = game;
        this.speed = 20;
        this.position = startPosition.clone();
        this.shouldDestroy = false;
        
        this.createMesh();
    }
    
    createMesh() {
        const geometry = new THREE.SphereGeometry(0.1, 8, 6);
        const material = new THREE.MeshBasicMaterial({ color: 0xffff00 });
        this.mesh = new THREE.Mesh(geometry, material);
        this.mesh.position.copy(this.position);
        this.scene.add(this.mesh);
    }
    
    update(deltaTime) {
        if (this.shouldDestroy) return;
        
        // Check if target still exists
        if (!this.target || this.target.isDead) {
            this.shouldDestroy = true;
            return;
        }
        
        const targetPosition = this.target.position.clone();
        const direction = new THREE.Vector3()
            .subVectors(targetPosition, this.position)
            .normalize();
        
        const moveDistance = this.speed * deltaTime;
        this.position.add(direction.multiplyScalar(moveDistance));
        this.mesh.position.copy(this.position);
        
        // Check for hit
        if (this.position.distanceTo(targetPosition) < 0.5) {
            this.hit();
        }
        
        // Destroy if too far from start
        if (this.position.distanceTo(this.startPosition) > 50) {
            this.shouldDestroy = true;
        }
    }
    
    hit() {
        if (this.target && !this.target.isDead) {
            this.target.takeDamage(this.damage);
        }
        this.shouldDestroy = true;
    }
    
    destroy() {
        this.scene.remove(this.mesh);
    }
}

// Wave Manager
class WaveManager {
    constructor(game) {
        this.game = game;
        this.currentWave = 0;
        this.enemiesSpawned = 0;
        this.enemiesInWave = 0;
        this.spawnTimer = 0;
        this.spawnInterval = 1; // seconds between spawns
        this.waveInProgress = false;
    }
    
    startWave() {
        if (this.waveInProgress) return;
        
        this.currentWave++;
        this.game.gameState.currentWave = this.currentWave;
        this.enemiesSpawned = 0;
        this.enemiesInWave = this.getWaveEnemyCount();
        this.spawnTimer = 0;
        this.waveInProgress = true;
        
        console.log(`[DARK MATTER] Starting wave ${this.currentWave} with ${this.enemiesInWave} enemies`);
    }
    
    getWaveEnemyCount() {
        return Math.floor(5 + this.currentWave * 2.5);
    }
    
    getEnemyTypeForWave() {
        const rand = Math.random();
        const wave = this.currentWave;
        
        if (wave >= 10 && rand < 0.1) return 'boss';
        if (wave >= 5 && rand < 0.3) return 'heavy';
        if (wave >= 3 && rand < 0.4) return 'fast';
        return 'basic';
    }
    
    update(deltaTime) {
        if (!this.waveInProgress) return;
        
        this.spawnTimer += deltaTime;
        
        // Spawn enemies
        if (this.enemiesSpawned < this.enemiesInWave && 
            this.spawnTimer >= this.spawnInterval) {
            
            const enemyType = this.getEnemyTypeForWave();
            this.game.spawnEnemy(enemyType);
            this.enemiesSpawned++;
            this.spawnTimer = 0;
        }
        
        // Check if wave is complete
        if (this.enemiesSpawned >= this.enemiesInWave && 
            this.game.enemies.length === 0) {
            this.completeWave();
        }
    }
    
    completeWave() {
        this.waveInProgress = false;
        this.game.gameState.gold += 25 + this.currentWave * 5;
        
        console.log(`[DARK MATTER] Wave ${this.currentWave} completed!`);
        
        // Auto-start next wave after delay
        setTimeout(() => {
            if (this.game.gameState.isPlaying) {
                this.startWave();
            }
        }, 3000);
    }
}

// Pathfinding utility
class Pathfinding {
    constructor(waypoints) {
        this.waypoints = waypoints;
    }
    
    getPath() {
        return this.waypoints;
    }
    
    getNextWaypoint(currentIndex) {
        return currentIndex < this.waypoints.length - 1 ? 
               this.waypoints[currentIndex + 1] : null;
    }
}

// Effects Manager
class EffectsManager {
    constructor(scene) {
        this.scene = scene;
        this.effects = [];
    }
    
    createMuzzleFlash(position) {
        const geometry = new THREE.SphereGeometry(0.3, 8, 6);
        const material = new THREE.MeshBasicMaterial({ 
            color: 0xffff00,
            transparent: true,
            opacity: 0.8
        });
        
        const flash = new THREE.Mesh(geometry, material);
        flash.position.copy(position);
        this.scene.add(flash);
        
        // Animate and remove
        setTimeout(() => {
            this.scene.remove(flash);
        }, 100);
    }
    
    createExplosion(position) {
        const geometry = new THREE.SphereGeometry(1, 8, 6);
        const material = new THREE.MeshBasicMaterial({ 
            color: 0xff4400,
            transparent: true,
            opacity: 0.7
        });
        
        const explosion = new THREE.Mesh(geometry, material);
        explosion.position.copy(position);
        this.scene.add(explosion);
        
        // Animate explosion
        let scale = 0.1;
        const animate = () => {
            scale += 0.1;
            explosion.scale.setScalar(scale);
            explosion.material.opacity -= 0.05;
            
            if (explosion.material.opacity > 0) {
                requestAnimationFrame(animate);
            } else {
                this.scene.remove(explosion);
            }
        };
        animate();
    }
    
    createDamageNumber(position, damage) {
        // This would typically use a text sprite or HTML overlay
        console.log(`Damage: ${damage} at`, position);
    }
    
    update(deltaTime) {
        // Update any ongoing effects
        this.effects = this.effects.filter(effect => {
            effect.update(deltaTime);
            return !effect.shouldDestroy;
        });
    }
}

// Plugin registration and API
let gameInstance = null;

export function initTowerDefense(engine, container) {
    if (gameInstance) {
        gameInstance.destroy();
    }
    
    gameInstance = new TowerDefenseGame(container, engine);
    console.log('[DARK MATTER] Tower Defense plugin initialized');
    return gameInstance;
}

export function spawnTower(position) {
    if (gameInstance) {
        gameInstance.placeTower(position);
    }
}

export function spawnEnemy(type = 'basic') {
    if (gameInstance) {
        gameInstance.spawnEnemy(type);
    }
}

export function startWave() {
    if (gameInstance) {
        gameInstance.startWave();
    }
}

export function pauseGame() {
    if (gameInstance) {
        gameInstance.pauseGame();
    }
}

export function resetGame() {
    if (gameInstance) {
        gameInstance.resetGame();
    }
}

export function register(pluginAPI) {
    pluginAPI.provide("initTowerDefense", initTowerDefense);
    pluginAPI.provide("spawnTower", spawnTower);
    pluginAPI.provide("spawnEnemy", spawnEnemy);
    pluginAPI.provide("startWave", startWave);
    pluginAPI.provide("pauseGame", pauseGame);
    pluginAPI.provide("resetGame", resetGame);
    
    console.log("[DARK MATTER] Tower Defense plugin registered");
}

