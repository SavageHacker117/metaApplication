# DARK MATTER Three.js Plugins - Quick Start Guide

## Installation

1. **Copy Plugin Files**
   ```bash
   # Copy all plugin directories to your DARK MATTER plugins folder
   cp -r threejs_* /path/to/DARKMATTER/plugins/
   ```

2. **Verify Structure**
   ```
   DARKMATTER/plugins/
   ├── threejs_tower_defense/
   │   ├── plugin.json
   │   ├── tower_defense.js
   │   ├── tower_defense_ui.html
   │   └── tower_defense_ui.js
   ├── threejs_procedural_world/
   │   ├── plugin.json
   │   └── procedural_world.js
   └── threejs_scene_manager/
       ├── plugin.json
       └── scene_manager.js
   ```

## Basic Usage

### 1. Initialize Scene Manager (Required First)

```javascript
// Get your container element
const container = document.getElementById('darkmatter-container');

// Initialize scene manager
const sceneManager = engine.plugins.threejs_scene_manager.initScene(engine, container);

// Set camera mode
engine.plugins.threejs_scene_manager.setCameraMode(engine, 'orbit');

// Apply lighting
engine.plugins.threejs_scene_manager.updateLighting(engine, 'studio');
```

### 2. Generate a Procedural World

```javascript
// Generate terrain
engine.plugins.threejs_procedural_world.generateTerrain(engine, 12345, 'mixed');

// Add vegetation
engine.plugins.threejs_procedural_world.generateVegetation(engine, 0.02, ['tree', 'bush']);

// Add weather effects
engine.plugins.threejs_procedural_world.generateWeather(engine, 'clear', 0.5);

// Set time of day (0 = midnight, 0.5 = noon, 1 = midnight)
engine.plugins.threejs_procedural_world.setTimeOfDay(engine, 0.6);
```

### 3. Start Tower Defense Game

```javascript
// Initialize tower defense
const game = engine.plugins.threejs_tower_defense.initTowerDefense(engine, container);

// Start the first wave
engine.plugins.threejs_tower_defense.startWave();

// Spawn test enemies
engine.plugins.threejs_tower_defense.spawnEnemy('basic');
engine.plugins.threejs_tower_defense.spawnEnemy('fast');
```

## Complete Example

```html
<!DOCTYPE html>
<html>
<head>
    <title>DARK MATTER Three.js Demo</title>
    <style>
        body { margin: 0; padding: 0; overflow: hidden; }
        #container { width: 100vw; height: 100vh; }
    </style>
</head>
<body>
    <div id="container"></div>
    
    <script type="module">
        // Assuming DARK MATTER engine is available
        const engine = window.darkMatterEngine;
        const container = document.getElementById('container');
        
        // 1. Initialize scene
        const sceneManager = engine.plugins.threejs_scene_manager.initScene(engine, container);
        
        // 2. Set up environment
        engine.plugins.threejs_scene_manager.setCameraMode(engine, 'orbit');
        engine.plugins.threejs_scene_manager.updateLighting(engine, 'outdoor');
        
        // 3. Generate world
        engine.plugins.threejs_procedural_world.generateTerrain(engine, Math.random() * 1000, 'mixed');
        engine.plugins.threejs_procedural_world.generateVegetation(engine, 0.015);
        engine.plugins.threejs_procedural_world.setTimeOfDay(engine, 0.7); // Evening
        
        // 4. Optional: Start tower defense game
        // const game = engine.plugins.threejs_tower_defense.initTowerDefense(engine, container);
        // engine.plugins.threejs_tower_defense.startWave();
    </script>
</body>
</html>
```

## Keyboard Controls

### Scene Manager
- **Mouse**: Orbit camera (in orbit mode)
- **WASD**: Move camera (in fly/first-person mode)
- **Mouse Look**: Look around (in first-person mode)

### Tower Defense
- **Space**: Pause/Resume game
- **Enter**: Start next wave
- **Ctrl+R**: Reset game
- **E**: Spawn test enemy
- **Click**: Place tower (costs 50 gold)

## Quick Tips

1. **Always initialize Scene Manager first** - it sets up the basic Three.js infrastructure
2. **Use appropriate lighting** - outdoor for worlds, studio for objects, dramatic for presentations
3. **Monitor performance** - check browser console for performance stats
4. **Experiment with seeds** - different terrain seeds create unique worlds
5. **Combine plugins** - procedural worlds work great as tower defense battlefields

## Troubleshooting

- **Black screen**: Check if container element exists and has dimensions
- **No controls**: Verify scene manager is initialized first
- **Poor performance**: Try lower vegetation density or switch to 'low' performance mode
- **Plugins not found**: Ensure plugin files are in correct directory structure

## Next Steps

- Read the full [Plugin Documentation](PLUGIN_DOCUMENTATION.md) for advanced features
- Customize enemy types, biomes, and lighting setups
- Integrate with your existing DARK MATTER environments
- Extend plugins with your own game mechanics

