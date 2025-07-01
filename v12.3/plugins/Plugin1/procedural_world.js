/**
 * DARK MATTER - Three.js Procedural World Generator Plugin
 * Generates dynamic, procedural 3D worlds with terrain, vegetation, and weather
 */

import * as THREE from 'three';
import { ImprovedNoise } from 'three/examples/jsm/math/ImprovedNoise.js';

class ProceduralWorldGenerator {
    constructor(scene, darkMatterEngine) {
        this.scene = scene;
        this.engine = darkMatterEngine;
        this.noise = new ImprovedNoise();
        
        // World parameters
        this.worldSize = 200;
        this.heightScale = 20;
        this.resolution = 128;
        
        // Generated objects
        this.terrain = null;
        this.vegetation = [];
        this.weather = null;
        this.water = null;
        this.sky = null;
        
        // Environmental state
        this.timeOfDay = 0.5; // 0 = midnight, 0.5 = noon, 1 = midnight
        this.weatherType = 'clear'; // clear, rain, snow, fog
        this.season = 'spring'; // spring, summer, autumn, winter
        
        console.log('[DARK MATTER] Procedural World Generator initialized');
    }
    
    generateTerrain(seed = Math.random() * 1000, biome = 'mixed') {
        console.log(`[DARK MATTER] Generating terrain with seed: ${seed}, biome: ${biome}`);
        
        // Clear existing terrain
        if (this.terrain) {
            this.scene.remove(this.terrain);
        }
        
        const geometry = new THREE.PlaneGeometry(
            this.worldSize, 
            this.worldSize, 
            this.resolution - 1, 
            this.resolution - 1
        );
        
        const vertices = geometry.attributes.position.array;
        const colors = [];
        
        // Generate height map using noise
        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const z = vertices[i + 2];
            
            // Multi-octave noise for realistic terrain
            let height = 0;
            let amplitude = 1;
            let frequency = 0.01;
            
            for (let octave = 0; octave < 4; octave++) {
                height += this.noise.noise(
                    (x + seed) * frequency,
                    (z + seed) * frequency,
                    seed * 0.01
                ) * amplitude;
                
                amplitude *= 0.5;
                frequency *= 2;
            }
            
            vertices[i + 1] = height * this.heightScale;
            
            // Generate colors based on height and biome
            const color = this.getTerrainColor(height * this.heightScale, biome);
            colors.push(color.r, color.g, color.b);
        }
        
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.computeVertexNormals();
        
        const material = new THREE.MeshLambertMaterial({
            vertexColors: true,
            wireframe: false
        });
        
        this.terrain = new THREE.Mesh(geometry, material);
        this.terrain.rotation.x = -Math.PI / 2;
        this.terrain.receiveShadow = true;
        this.terrain.castShadow = false;
        
        this.scene.add(this.terrain);
        
        // Generate water level
        this.generateWater();
        
        return this.terrain;
    }
    
    getTerrainColor(height, biome) {
        const colors = {
            mixed: [
                { height: -5, color: new THREE.Color(0x1e3a8a) },  // Deep water
                { height: 0, color: new THREE.Color(0x3b82f6) },   // Water
                { height: 1, color: new THREE.Color(0xfbbf24) },   // Sand
                { height: 3, color: new THREE.Color(0x22c55e) },   // Grass
                { height: 8, color: new THREE.Color(0x16a34a) },   // Forest
                { height: 12, color: new THREE.Color(0x78716c) },  // Rock
                { height: 16, color: new THREE.Color(0xf3f4f6) }   // Snow
            ],
            desert: [
                { height: -2, color: new THREE.Color(0x92400e) },  // Sand dunes
                { height: 2, color: new THREE.Color(0xfbbf24) },   // Light sand
                { height: 8, color: new THREE.Color(0xd97706) },   // Orange sand
                { height: 15, color: new THREE.Color(0x78716c) }   // Rock
            ],
            arctic: [
                { height: -2, color: new THREE.Color(0x1e40af) },  // Ice water
                { height: 0, color: new THREE.Color(0xe5e7eb) },   // Ice
                { height: 5, color: new THREE.Color(0xf3f4f6) },   // Snow
                { height: 15, color: new THREE.Color(0x9ca3af) }   // Rock
            ]
        };
        
        const biomeColors = colors[biome] || colors.mixed;
        
        for (let i = biomeColors.length - 1; i >= 0; i--) {
            if (height >= biomeColors[i].height) {
                return biomeColors[i].color;
            }
        }
        
        return biomeColors[0].color;
    }
    
    generateWater() {
        if (this.water) {
            this.scene.remove(this.water);
        }
        
        const waterGeometry = new THREE.PlaneGeometry(this.worldSize, this.worldSize);
        const waterMaterial = new THREE.MeshLambertMaterial({
            color: 0x006994,
            transparent: true,
            opacity: 0.6
        });
        
        this.water = new THREE.Mesh(waterGeometry, waterMaterial);
        this.water.rotation.x = -Math.PI / 2;
        this.water.position.y = 0.5;
        this.scene.add(this.water);
    }
    
    generateVegetation(density = 0.02, types = ['tree', 'bush', 'grass']) {
        console.log(`[DARK MATTER] Generating vegetation with density: ${density}`);
        
        // Clear existing vegetation
        this.clearVegetation();
        
        if (!this.terrain) {
            console.warn('[DARK MATTER] No terrain found, generating default terrain first');
            this.generateTerrain();
        }
        
        const terrainGeometry = this.terrain.geometry;
        const vertices = terrainGeometry.attributes.position.array;
        
        // Sample points from terrain for vegetation placement
        for (let i = 0; i < vertices.length; i += 9) { // Every 3rd vertex
            if (Math.random() < density) {
                const x = vertices[i];
                const y = vertices[i + 1];
                const z = vertices[i + 2];
                
                // Only place vegetation above water level and below snow line
                if (y > 1 && y < 12) {
                    const vegType = types[Math.floor(Math.random() * types.length)];
                    this.createVegetation(x, y, z, vegType);
                }
            }
        }
        
        console.log(`[DARK MATTER] Generated ${this.vegetation.length} vegetation objects`);
    }
    
    createVegetation(x, y, z, type) {
        let vegetation;
        
        switch (type) {
            case 'tree':
                vegetation = this.createTree(x, y, z);
                break;
            case 'bush':
                vegetation = this.createBush(x, y, z);
                break;
            case 'grass':
                vegetation = this.createGrass(x, y, z);
                break;
            default:
                vegetation = this.createTree(x, y, z);
        }
        
        if (vegetation) {
            this.vegetation.push(vegetation);
            this.scene.add(vegetation);
        }
    }
    
    createTree(x, y, z) {
        const tree = new THREE.Group();
        
        // Trunk
        const trunkGeometry = new THREE.CylinderGeometry(0.2, 0.3, 3 + Math.random() * 2);
        const trunkMaterial = new THREE.MeshLambertMaterial({ color: 0x8b4513 });
        const trunk = new THREE.Mesh(trunkGeometry, trunkMaterial);
        trunk.position.y = trunkGeometry.parameters.height / 2;
        trunk.castShadow = true;
        tree.add(trunk);
        
        // Foliage
        const foliageGeometry = new THREE.SphereGeometry(1.5 + Math.random() * 0.5, 8, 6);
        const foliageColor = this.season === 'autumn' ? 0xff8c00 : 
                           this.season === 'winter' ? 0x228b22 : 0x32cd32;
        const foliageMaterial = new THREE.MeshLambertMaterial({ color: foliageColor });
        const foliage = new THREE.Mesh(foliageGeometry, foliageMaterial);
        foliage.position.y = trunk.geometry.parameters.height + 1;
        foliage.castShadow = true;
        tree.add(foliage);
        
        tree.position.set(x, y, z);
        tree.rotation.y = Math.random() * Math.PI * 2;
        
        return tree;
    }
    
    createBush(x, y, z) {
        const bush = new THREE.Group();
        
        // Multiple small spheres for bush shape
        for (let i = 0; i < 3 + Math.random() * 3; i++) {
            const geometry = new THREE.SphereGeometry(0.3 + Math.random() * 0.2, 6, 4);
            const material = new THREE.MeshLambertMaterial({ color: 0x228b22 });
            const sphere = new THREE.Mesh(geometry, material);
            
            sphere.position.set(
                (Math.random() - 0.5) * 1.5,
                Math.random() * 0.5,
                (Math.random() - 0.5) * 1.5
            );
            sphere.castShadow = true;
            bush.add(sphere);
        }
        
        bush.position.set(x, y, z);
        return bush;
    }
    
    createGrass(x, y, z) {
        const grass = new THREE.Group();
        
        // Multiple grass blades
        for (let i = 0; i < 5 + Math.random() * 5; i++) {
            const geometry = new THREE.CylinderGeometry(0.01, 0.02, 0.5 + Math.random() * 0.3);
            const material = new THREE.MeshLambertMaterial({ color: 0x90ee90 });
            const blade = new THREE.Mesh(geometry, material);
            
            blade.position.set(
                (Math.random() - 0.5) * 0.5,
                geometry.parameters.height / 2,
                (Math.random() - 0.5) * 0.5
            );
            blade.rotation.z = (Math.random() - 0.5) * 0.2;
            grass.add(blade);
        }
        
        grass.position.set(x, y, z);
        return grass;
    }
    
    clearVegetation() {
        this.vegetation.forEach(veg => {
            this.scene.remove(veg);
        });
        this.vegetation = [];
    }
    
    generateWeather(type = 'clear', intensity = 0.5) {
        console.log(`[DARK MATTER] Generating weather: ${type}, intensity: ${intensity}`);
        
        this.weatherType = type;
        
        // Clear existing weather effects
        if (this.weather) {
            this.scene.remove(this.weather);
        }
        
        switch (type) {
            case 'rain':
                this.weather = this.createRain(intensity);
                break;
            case 'snow':
                this.weather = this.createSnow(intensity);
                break;
            case 'fog':
                this.createFog(intensity);
                break;
            default:
                this.clearWeather();
                return;
        }
        
        if (this.weather) {
            this.scene.add(this.weather);
        }
    }
    
    createRain(intensity) {
        const rainGroup = new THREE.Group();
        const particleCount = Math.floor(1000 * intensity);
        
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const velocities = [];
        
        for (let i = 0; i < particleCount; i++) {
            positions.push(
                (Math.random() - 0.5) * this.worldSize,
                Math.random() * 50 + 20,
                (Math.random() - 0.5) * this.worldSize
            );
            velocities.push(0, -10 - Math.random() * 5, 0);
        }
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('velocity', new THREE.Float32BufferAttribute(velocities, 3));
        
        const material = new THREE.PointsMaterial({
            color: 0x87ceeb,
            size: 0.1,
            transparent: true,
            opacity: 0.6
        });
        
        const rain = new THREE.Points(geometry, material);
        rainGroup.add(rain);
        
        return rainGroup;
    }
    
    createSnow(intensity) {
        const snowGroup = new THREE.Group();
        const particleCount = Math.floor(500 * intensity);
        
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        
        for (let i = 0; i < particleCount; i++) {
            positions.push(
                (Math.random() - 0.5) * this.worldSize,
                Math.random() * 50 + 20,
                (Math.random() - 0.5) * this.worldSize
            );
        }
        
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        
        const material = new THREE.PointsMaterial({
            color: 0xffffff,
            size: 0.3,
            transparent: true,
            opacity: 0.8
        });
        
        const snow = new THREE.Points(geometry, material);
        snowGroup.add(snow);
        
        return snowGroup;
    }
    
    createFog(intensity) {
        this.scene.fog = new THREE.Fog(0xcccccc, 10, 50 * (1 - intensity));
    }
    
    clearWeather() {
        if (this.weather) {
            this.scene.remove(this.weather);
            this.weather = null;
        }
        this.scene.fog = null;
    }
    
    setTimeOfDay(time) {
        this.timeOfDay = Math.max(0, Math.min(1, time));
        this.updateLighting();
        console.log(`[DARK MATTER] Time of day set to: ${this.timeOfDay}`);
    }
    
    updateLighting() {
        // Find or create directional light
        let directionalLight = this.scene.children.find(child => 
            child instanceof THREE.DirectionalLight
        );
        
        if (!directionalLight) {
            directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            this.scene.add(directionalLight);
        }
        
        // Calculate sun position based on time of day
        const sunAngle = (this.timeOfDay - 0.5) * Math.PI;
        directionalLight.position.set(
            Math.sin(sunAngle) * 50,
            Math.cos(sunAngle) * 50,
            25
        );
        
        // Adjust light color and intensity based on time
        if (this.timeOfDay < 0.2 || this.timeOfDay > 0.8) {
            // Night
            directionalLight.color.setHex(0x404080);
            directionalLight.intensity = 0.2;
        } else if (this.timeOfDay < 0.3 || this.timeOfDay > 0.7) {
            // Dawn/Dusk
            directionalLight.color.setHex(0xff8844);
            directionalLight.intensity = 0.5;
        } else {
            // Day
            directionalLight.color.setHex(0xffffff);
            directionalLight.intensity = 0.8;
        }
        
        // Update ambient light
        let ambientLight = this.scene.children.find(child => 
            child instanceof THREE.AmbientLight
        );
        
        if (!ambientLight) {
            ambientLight = new THREE.AmbientLight(0x404040, 0.4);
            this.scene.add(ambientLight);
        }
        
        ambientLight.intensity = 0.2 + (1 - Math.abs(this.timeOfDay - 0.5) * 2) * 0.3;
    }
    
    morphTerrain(targetSeed, duration = 2000) {
        if (!this.terrain) return;
        
        console.log(`[DARK MATTER] Morphing terrain to seed: ${targetSeed}`);
        
        const geometry = this.terrain.geometry;
        const vertices = geometry.attributes.position.array;
        const originalVertices = vertices.slice();
        
        // Generate target vertices
        const targetVertices = [];
        for (let i = 0; i < vertices.length; i += 3) {
            const x = vertices[i];
            const z = vertices[i + 2];
            
            let height = 0;
            let amplitude = 1;
            let frequency = 0.01;
            
            for (let octave = 0; octave < 4; octave++) {
                height += this.noise.noise(
                    (x + targetSeed) * frequency,
                    (z + targetSeed) * frequency,
                    targetSeed * 0.01
                ) * amplitude;
                
                amplitude *= 0.5;
                frequency *= 2;
            }
            
            targetVertices.push(x, height * this.heightScale, z);
        }
        
        // Animate morphing
        const startTime = performance.now();
        const animate = () => {
            const elapsed = performance.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = this.easeInOutCubic(progress);
            
            for (let i = 0; i < vertices.length; i += 3) {
                vertices[i + 1] = originalVertices[i + 1] + 
                    (targetVertices[i + 1] - originalVertices[i + 1]) * eased;
            }
            
            geometry.attributes.position.needsUpdate = true;
            geometry.computeVertexNormals();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1;
    }
    
    setSeason(season) {
        this.season = season;
        
        // Update vegetation colors
        this.vegetation.forEach(veg => {
            if (veg.children) {
                veg.children.forEach(child => {
                    if (child.material && child.material.color) {
                        switch (season) {
                            case 'spring':
                                if (child.geometry instanceof THREE.SphereGeometry) {
                                    child.material.color.setHex(0x90ee90);
                                }
                                break;
                            case 'summer':
                                if (child.geometry instanceof THREE.SphereGeometry) {
                                    child.material.color.setHex(0x32cd32);
                                }
                                break;
                            case 'autumn':
                                if (child.geometry instanceof THREE.SphereGeometry) {
                                    child.material.color.setHex(0xff8c00);
                                }
                                break;
                            case 'winter':
                                if (child.geometry instanceof THREE.SphereGeometry) {
                                    child.material.color.setHex(0x228b22);
                                }
                                break;
                        }
                    }
                });
            }
        });
        
        console.log(`[DARK MATTER] Season changed to: ${season}`);
    }
    
    update(deltaTime) {
        // Update weather effects
        if (this.weather && this.weatherType === 'rain') {
            const positions = this.weather.children[0].geometry.attributes.position.array;
            const velocities = this.weather.children[0].geometry.attributes.velocity.array;
            
            for (let i = 0; i < positions.length; i += 3) {
                positions[i + 1] += velocities[i + 1] * deltaTime;
                
                // Reset particles that fall below ground
                if (positions[i + 1] < 0) {
                    positions[i + 1] = 50 + Math.random() * 20;
                }
            }
            
            this.weather.children[0].geometry.attributes.position.needsUpdate = true;
        }
        
        if (this.weather && this.weatherType === 'snow') {
            const positions = this.weather.children[0].geometry.attributes.position.array;
            
            for (let i = 0; i < positions.length; i += 3) {
                positions[i + 1] -= 2 * deltaTime;
                positions[i] += Math.sin(performance.now() * 0.001 + i) * 0.1 * deltaTime;
                
                // Reset particles that fall below ground
                if (positions[i + 1] < 0) {
                    positions[i + 1] = 50 + Math.random() * 20;
                }
            }
            
            this.weather.children[0].geometry.attributes.position.needsUpdate = true;
        }
    }
    
    destroy() {
        this.clearVegetation();
        this.clearWeather();
        
        if (this.terrain) {
            this.scene.remove(this.terrain);
        }
        if (this.water) {
            this.scene.remove(this.water);
        }
        
        console.log('[DARK MATTER] Procedural World Generator destroyed');
    }
}

// Plugin API
let worldGenerator = null;

export function generateTerrain(engine, seed, biome = 'mixed') {
    if (!worldGenerator) {
        worldGenerator = new ProceduralWorldGenerator(engine.scene, engine);
    }
    return worldGenerator.generateTerrain(seed, biome);
}

export function generateVegetation(engine, density = 0.02, types = ['tree', 'bush', 'grass']) {
    if (!worldGenerator) {
        worldGenerator = new ProceduralWorldGenerator(engine.scene, engine);
    }
    worldGenerator.generateVegetation(density, types);
}

export function generateWeather(engine, type = 'clear', intensity = 0.5) {
    if (!worldGenerator) {
        worldGenerator = new ProceduralWorldGenerator(engine.scene, engine);
    }
    worldGenerator.generateWeather(type, intensity);
}

export function morphTerrain(engine, targetSeed, duration = 2000) {
    if (worldGenerator) {
        worldGenerator.morphTerrain(targetSeed, duration);
    }
}

export function setTimeOfDay(engine, time) {
    if (!worldGenerator) {
        worldGenerator = new ProceduralWorldGenerator(engine.scene, engine);
    }
    worldGenerator.setTimeOfDay(time);
}

export function register(pluginAPI) {
    pluginAPI.provide("generateTerrain", generateTerrain);
    pluginAPI.provide("generateVegetation", generateVegetation);
    pluginAPI.provide("generateWeather", generateWeather);
    pluginAPI.provide("morphTerrain", morphTerrain);
    pluginAPI.provide("setTimeOfDay", setTimeOfDay);
    
    console.log("[DARK MATTER] Procedural World Generator plugin registered");
}

