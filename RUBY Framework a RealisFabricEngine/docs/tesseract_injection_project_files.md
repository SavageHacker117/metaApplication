# tesseract_injection Project Files

---

## 1. README.md
```
# Tesseract Injection
A supercharged framework for in-game universe generation, editing, and event creation—driven by pure math, natural language, and AI. The tesseract and portal gun operate in tandem, as both tools and UI for “reality injection.” Use this as your dev suite for next-gen sandbox games, AI-driven experiments, or cosmic story simulators.
```

---

## 2. dev_philosophy.md
```
It’s just math.  
- Every world is a mathematical space.  
- Every transformation, every portal, every new universe is a function.  
- This suite gives you (and your AI agents) the power to operate on the universe’s geometry, logic, and rules—live, via prompt, code, or visual tool.
```

---

## 3. index.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tesseract Injection</title>
    <style>body{margin:0;overflow:hidden;background:#1a1a1a;}</style>
</head>
<body>
    <div id="dev-ui"></div>
    <script type="module" src="main.js"></script>
</body>
</html>
```

---

## 4. main.js
```javascript
import { Tesseract } from './tesseract/tesseract.js';
import { PortalGun } from './portal_gun/portal_gun.js';
import { setupDevSuiteUI } from './ui/dev_suite_ui.js';
import { hookAICommands } from './ai_hooks.js';

const tesseract = new Tesseract();
const portalGun = new PortalGun({ tesseract });
setupDevSuiteUI({ tesseract, portalGun });

hookAICommands({ tesseract, portalGun }); // Enables AIs (or players) to script, inject, or prompt new universes

// ...Insert Three.js scene/renderer/camera setup here
// ...Add tesseract and portalGun mesh/controls to scene
```

---

## 5. ai_hooks.js
```javascript
export function hookAICommands({ tesseract, portalGun }) {
    window.tesseractInject = (code) => tesseract.injectCode(code);
    window.tesseractPrompt = (prompt) => tesseract.promptAI(prompt);
    window.portalFire = (prompt) => portalGun.fire(prompt);
    // ...more hooks for RL agent or LLM (can be REST/websocket/event bus)
}
```

---

## 6. tesseract/tesseract.js
```javascript
import { generateTesseractVertices, rotate4D, project4Dto3D } from './tesseract_math.js';

export class Tesseract {
    constructor(size = 1) {
        this.size = size;
        this.vertices4D = generateTesseractVertices(size);
        // ...setup Three.js mesh/lines
        // ...optionally setup physics/energy FX
    }

    update(rotationParams) {
        // rotate in 4D, project to 3D, update mesh
    }

    injectCode(code) {
        // Evaluate/execute user- or AI-supplied code to alter the tesseract or current universe
    }

    promptAI(prompt) {
        // Send prompt to RL-LLM/AI for world or event generation
    }
}
```

---

## 7. tesseract/tesseract_math.js
```javascript
export function generateTesseractVertices(size=1) {
    // 16 4D vertices
    const v = [];
    for (let i=0;i<16;i++) {
        v.push([
            (i&1?1:-1)*size, (i&2?1:-1)*size, (i&4?1:-1)*size, (i&8?1:-1)*size
        ]);
    }
    return v;
}

export function rotate4D(v, a, b, theta) {
    // Rotate vector v in (a,b) 4D plane
    const c = Math.cos(theta), s = Math.sin(theta);
    let nv = v.slice();
    nv[a] = v[a]*c - v[b]*s;
    nv[b] = v[a]*s + v[b]*c;
    return nv;
}

export function project4Dto3D(v, d=3) {
    // Perspective projection (W to 3D)
    let w = d / (d - v[3]);
    return [v[0]*w, v[1]*w, v[2]*w];
}
```

---

## 8. portal_gun/portal_gun.js
```javascript
import { PortalFX } from './portal_fx.js';
import { GunAnimation } from './gun_animation.js';

export class PortalGun {
    constructor({ tesseract }) {
        this.tesseract = tesseract;
        this.fx = new PortalFX();
        this.anim = new GunAnimation();
        // ...setup gun mesh and state
    }

    loadTesseract() {
        this.anim.animateLoad(this.tesseract);
    }

    fire(prompt) {
        this.tesseract.promptAI(prompt);
        this.fx.createPortal(); // Spawns the portal and new universe link
    }
}
```

---

## 9. portal_gun/portal_fx.js
```javascript
export class PortalFX {
    createPortal() {
        // Animate swirling portal in 3D, using math-based shaders
        // On completion, triggers world/universe transition
    }
}
```

---

## 10. portal_gun/gun_animation.js
```javascript
export class GunAnimation {
    animateLoad(tesseract) {
        // Animate tesseract loading into gun, glow, electric arcs, etc.
    }
    animateFire() {
        // Animate gun firing, energy burst, camera shake, etc.
    }
}
```

---

## 11. ui/dev_suite_ui.js
```javascript
export function setupDevSuiteUI({ tesseract, portalGun }) {
    // Attach HTML controls to #dev-ui
    // - Prompt box for text/image/code
    // - Buttons for fire, load, preview
    // - Portal history, world merge/collide, gatekeeper settings
    // Display tesseract and gun states, visual feedback
}
```

---

## 12. docs/how_it_works.md
```
- Tesseract and portal gun system, grounded in 4D geometry and operator math.
- Any player or AI can alter, inject, or create universes via prompt/code inside the sim.
- The game world is a mutable function—AI and humans are equal creators.
```

---

## 13. docs/sci_fi_math.md
```
- Tesseract = universal address/folding mechanism.
- Portal gun = field generator and projector.
- “It’s just math”—all transformations are functions or code; all universes are parameterized mathematical objects.
```

---

## 14. docs/example_prompts.md
```
Take me to a new Earth, but with intelligent dinosaurs.
Generate a noir city where time moves backwards.
Inject this shader code into the new world’s sky:
[ ...user supplied GLSL... ]
```

---

## 15. docs/integration_guide.md
```
How to import/use the tesseract and portal gun in your Three.js/AI game or sandbox engine.
How to let RL/LLM agents inject code, prompts, or world events.
```

