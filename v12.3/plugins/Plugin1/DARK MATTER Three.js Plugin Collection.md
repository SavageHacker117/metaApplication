# DARK MATTER Three.js Plugin Collection

## Overview

This collection provides ready-to-use Three.js plugins for the DARK MATTER toolkit, inspired by the AI-powered 3D tower defense game architecture. These plugins are designed to be modular, performant, and easily integrated into the DARK MATTER meta-environmental layer.

## Plugin Architecture

Each plugin follows the DARK MATTER plugin standard:
- **Manifest file** (`plugin.json`): Defines metadata, dependencies, and exposed functions
- **Main JavaScript file**: Contains the core plugin logic and Three.js implementation
- **UI files**: HTML and JavaScript for user interface components
- **Asset directories**: Textures, models, and other resources

## Plugin Collection

### 1. Tower Defense Game Plugin (`threejs_tower_defense`)

**Purpose**: A complete tower defense game implementation using Three.js with AI-powered enemies and strategic gameplay.

**Key Features**:
- Real-time 3D tower defense gameplay
- Multiple enemy types with different behaviors
- Tower placement and upgrade system
- Wave management with progressive difficulty
- Visual effects and particle systems
- Comprehensive UI with stats and controls

**Exposed Functions**:
- `initTowerDefense(engine, container)`: Initialize the game
- `spawnTower(position)`: Place a tower at specified position
- `spawnEnemy(type)`: Spawn enemy of specified type
- `startWave()`: Begin the next enemy wave
- `pauseGame()`: Pause/resume gameplay
- `resetGame()`: Reset game to initial state

**Usage Example**:
```javascript
// Initialize the tower defense game
const game = window.rubyEngine.plugins.threejs_tower_defense.initTowerDefense(engine, container);

// Start the first wave
window.rubyEngine.plugins.threejs_tower_defense.startWave();

// Spawn a test enemy
window.rubyEngine.plugins.threejs_tower_defense.spawnEnemy('heavy');
```

**Game Mechanics**:
- **Towers**: Automatically target and fire at enemies within range
- **Enemies**: Follow predefined paths with different health, speed, and rewards
- **Projectiles**: Homing projectiles with realistic physics
- **Economy**: Gold-based tower purchasing and upgrade system
- **Progression**: Increasingly difficult waves with boss enemies

### 2. Procedural World Generator Plugin (`threejs_procedural_world`)

**Purpose**: Generate dynamic, procedural 3D worlds with terrain, vegetation, weather effects, and environmental systems.

**Key Features**:
- Multi-octave noise-based terrain generation
- Biome-specific terrain coloring and features
- Procedural vegetation placement (trees, bushes, grass)
- Dynamic weather systems (rain, snow, fog)
- Day/night cycle with dynamic lighting
- Terrain morphing and real-time modifications
- Seasonal changes affecting vegetation

**Exposed Functions**:
- `generateTerrain(engine, seed, biome)`: Generate terrain with specified parameters
- `generateVegetation(engine, density, types)`: Add vegetation to the world
- `generateWeather(engine, type, intensity)`: Create weather effects
- `morphTerrain(engine, targetSeed, duration)`: Smoothly morph terrain
- `setTimeOfDay(engine, time)`: Control lighting and time of day

**Usage Example**:
```javascript
// Generate a mixed biome terrain
window.rubyEngine.plugins.threejs_procedural_world.generateTerrain(engine, 12345, 'mixed');

// Add vegetation
window.rubyEngine.plugins.threejs_procedural_world.generateVegetation(engine, 0.03, ['tree', 'bush']);

// Create rain weather
window.rubyEngine.plugins.threejs_procedural_world.generateWeather(engine, 'rain', 0.7);

// Set to sunset
window.rubyEngine.plugins.threejs_procedural_world.setTimeOfDay(engine, 0.8);
```

**Biome Types**:
- **Mixed**: Varied terrain with water, forests, and mountains
- **Desert**: Sand dunes and rocky outcroppings
- **Arctic**: Snow and ice with minimal vegetation

### 3. Scene Manager Plugin (`threejs_scene_manager`)

**Purpose**: Comprehensive scene management with camera controls, lighting systems, and performance optimization.

**Key Features**:
- Multiple camera control modes (orbit, fly, first-person, cinematic)
- Professional lighting presets (studio, outdoor, indoor, dramatic, night)
- Performance optimization with LOD and frustum culling
- Object management with automatic optimization
- Real-time performance monitoring
- Camera animation and smooth transitions
- Raycasting utilities for interaction

**Exposed Functions**:
- `initScene(engine, container)`: Initialize the scene manager
- `addObject(engine, object, id)`: Add and optimize objects
- `removeObject(engine, id)`: Remove objects from scene
- `setCameraMode(engine, mode)`: Change camera control mode
- `updateLighting(engine, preset)`: Apply lighting presets
- `optimizeScene(engine)`: Perform scene optimization

**Usage Example**:
```javascript
// Initialize scene manager
const sceneManager = window.rubyEngine.plugins.threejs_scene_manager.initScene(engine, container);

// Set fly camera mode
window.rubyEngine.plugins.threejs_scene_manager.setCameraMode(engine, 'fly');

// Apply outdoor lighting
window.rubyEngine.plugins.threejs_scene_manager.updateLighting(engine, 'outdoor');

// Add an object
const cubeId = window.rubyEngine.plugins.threejs_scene_manager.addObject(engine, cube, 'test_cube');
```

**Camera Modes**:
- **Orbit**: Mouse-controlled orbital camera with zoom and pan
- **Fly**: Free-flying camera with WASD movement
- **First-Person**: FPS-style camera controls
- **Cinematic**: Scripted camera movements for presentations

**Lighting Presets**:
- **Studio**: Professional 3-point lighting setup
- **Outdoor**: Sun and sky lighting with shadows
- **Indoor**: Multiple ceiling lights with ambient
- **Dramatic**: High-contrast lighting with colored rims
- **Night**: Moonlight and street lamp simulation

## Integration Guide

### Step 1: Plugin Installation

1. Copy plugin directories to your DARK MATTER `plugins/` folder:
   ```
   DARKMATTER/plugins/
   ├── threejs_tower_defense/
   ├── threejs_procedural_world/
   └── threejs_scene_manager/
   ```

2. Ensure Three.js dependencies are available in your project.

### Step 2: Plugin Registration

The DARK MATTER plugin system will automatically discover and load plugins based on their `plugin.json` manifests. Each plugin exports a `register` function that provides its API to the engine.

### Step 3: Basic Usage

```javascript
// Access plugins through the engine
const engine = window.rubyEngine; // or your DARK MATTER engine instance

// Initialize a scene
const sceneManager = engine.plugins.threejs_scene_manager.initScene(engine, document.getElementById('container'));

// Generate a world
engine.plugins.threejs_procedural_world.generateTerrain(engine, 42, 'mixed');
engine.plugins.threejs_procedural_world.generateVegetation(engine, 0.02);

// Start a tower defense game
const game = engine.plugins.threejs_tower_defense.initTowerDefense(engine, document.getElementById('game-container'));
```

### Step 4: Advanced Integration

For more complex scenarios, plugins can interact with each other:

```javascript
// Create a procedural world for the tower defense game
engine.plugins.threejs_procedural_world.generateTerrain(engine, 123, 'mixed');

// Set appropriate lighting
engine.plugins.threejs_scene_manager.updateLighting(engine, 'outdoor');

// Initialize tower defense on the generated terrain
const game = engine.plugins.threejs_tower_defense.initTowerDefense(engine, container);
```

## Performance Considerations

### Optimization Features

1. **Automatic Object Optimization**: Scene manager automatically optimizes added objects
2. **Performance Modes**: Adjustable quality settings (low, balanced, high)
3. **Frustum Culling**: Objects outside camera view are not rendered
4. **Level of Detail**: Simplified models at distance (planned feature)
5. **Shadow Map Optimization**: Configurable shadow quality

### Best Practices

1. **Use Scene Manager**: Always initialize the scene manager first for optimal performance
2. **Batch Operations**: Group multiple object additions for better performance
3. **Monitor Performance**: Use the built-in performance monitoring events
4. **Optimize Textures**: Use appropriate texture sizes for your target hardware
5. **Limit Particle Effects**: Weather and effect systems can be performance-intensive

## Customization and Extension

### Adding New Enemy Types

```javascript
// Extend the tower defense plugin
const customEnemyStats = {
    flying: { health: 75, speed: 4, reward: 20, points: 20 }
};

// Add to the enemy factory (requires plugin modification)
```

### Creating Custom Biomes

```javascript
// Extend the procedural world plugin
const customBiome = {
    volcanic: [
        { height: -2, color: new THREE.Color(0x8b0000) },  // Lava
        { height: 5, color: new THREE.Color(0x2f2f2f) },   // Ash
        { height: 15, color: new THREE.Color(0x696969) }   // Rock
    ]
};
```

### Custom Lighting Setups

```javascript
// Create custom lighting presets
const customLighting = {
    underwater: () => {
        const light = new THREE.DirectionalLight(0x0077be, 0.5);
        light.position.set(0, 10, 0);
        // Add to scene...
    }
};
```

## Troubleshooting

### Common Issues

1. **Plugin Not Loading**: Check `plugin.json` syntax and file paths
2. **Performance Issues**: Reduce particle counts and enable performance mode
3. **Camera Controls Not Working**: Ensure container element is properly sized
4. **Lighting Too Dark/Bright**: Adjust lighting preset or create custom setup
5. **Objects Not Visible**: Check object positioning and camera location

### Debug Features

- Enable light helpers for debugging lighting setup
- Use browser developer tools to monitor Three.js performance
- Check console for plugin loading and error messages
- Use the performance monitoring events for optimization

## Future Enhancements

### Planned Features

1. **AI Entity Plugin**: Intelligent NPCs with behavior trees
2. **Dynamic Effects Plugin**: Advanced particle systems and post-processing
3. **Audio Integration**: 3D positional audio system
4. **Networking Plugin**: Multiplayer and real-time collaboration
5. **VR/AR Support**: WebXR integration for immersive experiences

### Extension Points

- Custom shader materials for advanced visual effects
- Physics integration with Cannon.js or Ammo.js
- Advanced AI behaviors using machine learning
- Procedural animation systems
- Real-time global illumination

This plugin collection provides a solid foundation for building sophisticated 3D applications within the DARK MATTER ecosystem, with room for extensive customization and expansion based on specific project needs.

