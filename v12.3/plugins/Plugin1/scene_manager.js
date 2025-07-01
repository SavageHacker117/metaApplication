/**
 * DARK MATTER - Three.js Scene Manager Plugin
 * Comprehensive scene management with camera controls, lighting, and optimization
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { FlyControls } from 'three/examples/jsm/controls/FlyControls.js';
import { FirstPersonControls } from 'three/examples/jsm/controls/FirstPersonControls.js';

class SceneManager {
    constructor(container, darkMatterEngine) {
        this.container = container;
        this.engine = darkMatterEngine;
        
        // Core Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.clock = new THREE.Clock();
        
        // Scene management
        this.objects = new Map();
        this.lights = new Map();
        this.objectId = 0;
        
        // Camera modes
        this.cameraMode = 'orbit'; // orbit, fly, first-person, cinematic
        this.cameraSettings = {
            orbit: {
                minDistance: 1,
                maxDistance: 100,
                enableDamping: true,
                dampingFactor: 0.05
            },
            fly: {
                movementSpeed: 10,
                rollSpeed: Math.PI / 24,
                autoForward: false,
                dragToLook: false
            },
            firstPerson: {
                movementSpeed: 10,
                lookSpeed: 0.1,
                lookVertical: true,
                autoForward: false
            }
        };
        
        // Performance optimization
        this.performanceMode = 'balanced'; // low, balanced, high
        this.frustumCulling = true;
        this.levelOfDetail = true;
        this.occlusionCulling = false;
        
        // Lighting presets
        this.lightingPresets = {
            studio: this.createStudioLighting.bind(this),
            outdoor: this.createOutdoorLighting.bind(this),
            indoor: this.createIndoorLighting.bind(this),
            dramatic: this.createDramaticLighting.bind(this),
            night: this.createNightLighting.bind(this)
        };
        
        this.init();
    }
    
    init() {
        this.setupRenderer();
        this.setupScene();
        this.setupCamera();
        this.setupControls();
        this.setupLighting();
        this.setupEventListeners();
        this.startRenderLoop();
        
        console.log('[DARK MATTER] Scene Manager initialized');
    }
    
    setupRenderer() {
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true,
            powerPreference: 'high-performance'
        });
        
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        
        // Enable shadows
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        // Color management
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1;
        
        // Performance settings
        this.renderer.info.autoReset = false;
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x202020);
        
        // Add scene to engine if available
        if (this.engine) {
            this.engine.scene = this.scene;
        }
    }
    
    setupCamera() {
        this.camera = new THREE.PerspectiveCamera(
            75,
            this.container.clientWidth / this.container.clientHeight,
            0.1,
            1000
        );
        
        this.camera.position.set(10, 10, 10);
        this.camera.lookAt(0, 0, 0);
        
        // Add camera to engine if available
        if (this.engine) {
            this.engine.camera = this.camera;
        }
    }
    
    setupControls() {
        this.setCameraMode(this.cameraMode);
    }
    
    setCameraMode(mode) {
        // Dispose existing controls
        if (this.controls) {
            this.controls.dispose();
        }
        
        this.cameraMode = mode;
        
        switch (mode) {
            case 'orbit':
                this.controls = new OrbitControls(this.camera, this.renderer.domElement);
                Object.assign(this.controls, this.cameraSettings.orbit);
                break;
                
            case 'fly':
                this.controls = new FlyControls(this.camera, this.renderer.domElement);
                Object.assign(this.controls, this.cameraSettings.fly);
                break;
                
            case 'first-person':
                this.controls = new FirstPersonControls(this.camera, this.renderer.domElement);
                Object.assign(this.controls, this.cameraSettings.firstPerson);
                break;
                
            case 'cinematic':
                // No controls for cinematic mode
                this.controls = null;
                break;
                
            default:
                console.warn(`[DARK MATTER] Unknown camera mode: ${mode}`);
                return;
        }
        
        console.log(`[DARK MATTER] Camera mode set to: ${mode}`);
    }
    
    setupLighting(preset = 'studio') {
        this.clearLights();
        this.lightingPresets[preset]();
        console.log(`[DARK MATTER] Lighting preset applied: ${preset}`);
    }
    
    createStudioLighting() {
        // Key light
        const keyLight = new THREE.DirectionalLight(0xffffff, 1);
        keyLight.position.set(10, 10, 5);
        keyLight.castShadow = true;
        keyLight.shadow.mapSize.width = 2048;
        keyLight.shadow.mapSize.height = 2048;
        this.addLight('key', keyLight);
        
        // Fill light
        const fillLight = new THREE.DirectionalLight(0xffffff, 0.3);
        fillLight.position.set(-5, 5, 5);
        this.addLight('fill', fillLight);
        
        // Rim light
        const rimLight = new THREE.DirectionalLight(0x4488ff, 0.2);
        rimLight.position.set(0, 5, -10);
        this.addLight('rim', rimLight);
        
        // Ambient
        const ambient = new THREE.AmbientLight(0x404040, 0.2);
        this.addLight('ambient', ambient);
    }
    
    createOutdoorLighting() {
        // Sun
        const sun = new THREE.DirectionalLight(0xffffff, 0.8);
        sun.position.set(50, 50, 25);
        sun.castShadow = true;
        sun.shadow.mapSize.width = 4096;
        sun.shadow.mapSize.height = 4096;
        sun.shadow.camera.left = -50;
        sun.shadow.camera.right = 50;
        sun.shadow.camera.top = 50;
        sun.shadow.camera.bottom = -50;
        this.addLight('sun', sun);
        
        // Sky
        const sky = new THREE.AmbientLight(0x87ceeb, 0.4);
        this.addLight('sky', sky);
        
        // Hemisphere light for ground bounce
        const hemisphere = new THREE.HemisphereLight(0x87ceeb, 0x8b7355, 0.3);
        this.addLight('hemisphere', hemisphere);
    }
    
    createIndoorLighting() {
        // Ceiling lights
        for (let i = 0; i < 4; i++) {
            const light = new THREE.PointLight(0xffffff, 0.5, 20);
            light.position.set(
                (i % 2) * 10 - 5,
                8,
                Math.floor(i / 2) * 10 - 5
            );
            light.castShadow = true;
            this.addLight(`ceiling_${i}`, light);
        }
        
        // Ambient
        const ambient = new THREE.AmbientLight(0x404040, 0.3);
        this.addLight('ambient', ambient);
    }
    
    createDramaticLighting() {
        // Strong key light
        const keyLight = new THREE.SpotLight(0xffffff, 2, 30, Math.PI / 6, 0.5);
        keyLight.position.set(5, 15, 5);
        keyLight.castShadow = true;
        this.addLight('dramatic_key', keyLight);
        
        // Colored rim lights
        const rimLight1 = new THREE.SpotLight(0xff0044, 1, 20, Math.PI / 4, 0.3);
        rimLight1.position.set(-10, 5, -5);
        this.addLight('rim1', rimLight1);
        
        const rimLight2 = new THREE.SpotLight(0x0044ff, 1, 20, Math.PI / 4, 0.3);
        rimLight2.position.set(10, 5, -5);
        this.addLight('rim2', rimLight2);
        
        // Low ambient
        const ambient = new THREE.AmbientLight(0x202040, 0.1);
        this.addLight('ambient', ambient);
    }
    
    createNightLighting() {
        // Moon
        const moon = new THREE.DirectionalLight(0x9999ff, 0.2);
        moon.position.set(-30, 30, 10);
        moon.castShadow = true;
        this.addLight('moon', moon);
        
        // Street lights
        for (let i = 0; i < 6; i++) {
            const streetLight = new THREE.PointLight(0xffaa44, 1, 15);
            streetLight.position.set(
                (i % 3) * 15 - 15,
                5,
                Math.floor(i / 3) * 15 - 7.5
            );
            streetLight.castShadow = true;
            this.addLight(`street_${i}`, streetLight);
        }
        
        // Dark ambient
        const ambient = new THREE.AmbientLight(0x001122, 0.1);
        this.addLight('ambient', ambient);
    }
    
    addLight(id, light) {
        this.lights.set(id, light);
        this.scene.add(light);
        
        // Add helper for debugging (hidden by default)
        if (light instanceof THREE.DirectionalLight) {
            const helper = new THREE.DirectionalLightHelper(light, 1);
            helper.visible = false;
            light.helper = helper;
            this.scene.add(helper);
        } else if (light instanceof THREE.SpotLight) {
            const helper = new THREE.SpotLightHelper(light);
            helper.visible = false;
            light.helper = helper;
            this.scene.add(helper);
        } else if (light instanceof THREE.PointLight) {
            const helper = new THREE.PointLightHelper(light, 0.5);
            helper.visible = false;
            light.helper = helper;
            this.scene.add(helper);
        }
    }
    
    clearLights() {
        this.lights.forEach((light, id) => {
            this.scene.remove(light);
            if (light.helper) {
                this.scene.remove(light.helper);
            }
        });
        this.lights.clear();
    }
    
    addObject(object, id = null) {
        if (!id) {
            id = `object_${this.objectId++}`;
        }
        
        this.objects.set(id, object);
        this.scene.add(object);
        
        // Apply performance optimizations
        this.optimizeObject(object);
        
        console.log(`[DARK MATTER] Object added: ${id}`);
        return id;
    }
    
    removeObject(id) {
        const object = this.objects.get(id);
        if (object) {
            this.scene.remove(object);
            this.objects.delete(id);
            console.log(`[DARK MATTER] Object removed: ${id}`);
            return true;
        }
        return false;
    }
    
    getObject(id) {
        return this.objects.get(id);
    }
    
    optimizeObject(object) {
        object.traverse((child) => {
            if (child instanceof THREE.Mesh) {
                // Enable frustum culling
                child.frustumCulled = this.frustumCulling;
                
                // Optimize materials
                if (child.material) {
                    this.optimizeMaterial(child.material);
                }
                
                // Optimize geometry
                if (child.geometry) {
                    this.optimizeGeometry(child.geometry);
                }
            }
        });
    }
    
    optimizeMaterial(material) {
        // Set appropriate precision
        if (this.performanceMode === 'low') {
            material.precision = 'lowp';
        } else if (this.performanceMode === 'balanced') {
            material.precision = 'mediump';
        }
        
        // Disable unnecessary features for performance
        if (this.performanceMode === 'low') {
            material.fog = false;
        }
    }
    
    optimizeGeometry(geometry) {
        // Merge vertices if possible
        if (geometry.attributes.position && !geometry.index) {
            geometry = geometry.toNonIndexed();
        }
        
        // Compute bounding sphere for frustum culling
        geometry.computeBoundingSphere();
    }
    
    optimizeScene() {
        console.log('[DARK MATTER] Optimizing scene...');
        
        // Update all objects
        this.objects.forEach((object) => {
            this.optimizeObject(object);
        });
        
        // Optimize renderer settings based on performance mode
        switch (this.performanceMode) {
            case 'low':
                this.renderer.setPixelRatio(1);
                this.renderer.shadowMap.enabled = false;
                break;
            case 'balanced':
                this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
                this.renderer.shadowMap.enabled = true;
                this.renderer.shadowMap.type = THREE.BasicShadowMap;
                break;
            case 'high':
                this.renderer.setPixelRatio(window.devicePixelRatio);
                this.renderer.shadowMap.enabled = true;
                this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                break;
        }
        
        console.log(`[DARK MATTER] Scene optimized for ${this.performanceMode} performance`);
    }
    
    setPerformanceMode(mode) {
        this.performanceMode = mode;
        this.optimizeScene();
    }
    
    setupEventListeners() {
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Performance monitoring
        this.renderer.info.autoReset = false;
        setInterval(() => {
            if (this.engine && this.engine.events) {
                this.engine.events.emit('performanceUpdate', {
                    triangles: this.renderer.info.render.triangles,
                    calls: this.renderer.info.render.calls,
                    frame: this.renderer.info.render.frame
                });
            }
            this.renderer.info.reset();
        }, 1000);
    }
    
    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }
    
    startRenderLoop() {
        const animate = () => {
            requestAnimationFrame(animate);
            this.update();
            this.render();
        };
        animate();
    }
    
    update() {
        const deltaTime = this.clock.getDelta();
        
        // Update controls
        if (this.controls && this.controls.update) {
            this.controls.update(deltaTime);
        }
        
        // Update objects with update methods
        this.objects.forEach((object) => {
            if (object.update && typeof object.update === 'function') {
                object.update(deltaTime);
            }
        });
        
        // Emit update event
        if (this.engine && this.engine.events) {
            this.engine.events.emit('sceneUpdate', { deltaTime });
        }
    }
    
    render() {
        this.renderer.render(this.scene, this.camera);
    }
    
    // Camera animation methods
    animateCameraTo(position, target, duration = 2000) {
        const startPosition = this.camera.position.clone();
        const startTarget = this.controls ? this.controls.target.clone() : new THREE.Vector3();
        const startTime = performance.now();
        
        const animate = () => {
            const elapsed = performance.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = this.easeInOutCubic(progress);
            
            // Interpolate position
            this.camera.position.lerpVectors(startPosition, position, eased);
            
            // Interpolate target
            if (this.controls && this.controls.target) {
                this.controls.target.lerpVectors(startTarget, target, eased);
            } else {
                this.camera.lookAt(target);
            }
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
    }
    
    // Utility methods
    screenToWorld(x, y, z = 0) {
        const vector = new THREE.Vector3(x, y, z);
        vector.unproject(this.camera);
        return vector;
    }
    
    worldToScreen(worldPosition) {
        const vector = worldPosition.clone();
        vector.project(this.camera);
        
        const x = (vector.x + 1) / 2 * this.container.clientWidth;
        const y = -(vector.y - 1) / 2 * this.container.clientHeight;
        
        return { x, y, z: vector.z };
    }
    
    raycast(x, y, objects = null) {
        const mouse = new THREE.Vector2();
        mouse.x = (x / this.container.clientWidth) * 2 - 1;
        mouse.y = -(y / this.container.clientHeight) * 2 + 1;
        
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);
        
        const targets = objects || Array.from(this.objects.values());
        return raycaster.intersectObjects(targets, true);
    }
    
    getStats() {
        return {
            objects: this.objects.size,
            lights: this.lights.size,
            triangles: this.renderer.info.render.triangles,
            calls: this.renderer.info.render.calls,
            memory: this.renderer.info.memory
        };
    }
    
    destroy() {
        // Dispose controls
        if (this.controls) {
            this.controls.dispose();
        }
        
        // Clear scene
        this.objects.clear();
        this.clearLights();
        
        // Dispose renderer
        if (this.renderer) {
            this.container.removeChild(this.renderer.domElement);
            this.renderer.dispose();
        }
        
        console.log('[DARK MATTER] Scene Manager destroyed');
    }
}

// Plugin API
let sceneManager = null;

export function initScene(engine, container) {
    if (sceneManager) {
        sceneManager.destroy();
    }
    
    sceneManager = new SceneManager(container, engine);
    console.log('[DARK MATTER] Scene Manager plugin initialized');
    return sceneManager;
}

export function addObject(engine, object, id = null) {
    if (sceneManager) {
        return sceneManager.addObject(object, id);
    }
    return null;
}

export function removeObject(engine, id) {
    if (sceneManager) {
        return sceneManager.removeObject(id);
    }
    return false;
}

export function setCameraMode(engine, mode) {
    if (sceneManager) {
        sceneManager.setCameraMode(mode);
    }
}

export function updateLighting(engine, preset) {
    if (sceneManager) {
        sceneManager.setupLighting(preset);
    }
}

export function optimizeScene(engine) {
    if (sceneManager) {
        sceneManager.optimizeScene();
    }
}

export function register(pluginAPI) {
    pluginAPI.provide("initScene", initScene);
    pluginAPI.provide("addObject", addObject);
    pluginAPI.provide("removeObject", removeObject);
    pluginAPI.provide("setCameraMode", setCameraMode);
    pluginAPI.provide("updateLighting", updateLighting);
    pluginAPI.provide("optimizeScene", optimizeScene);
    
    console.log("[DARK MATTER] Scene Manager plugin registered");
}

