# Advanced Plugin Templates for Dark Matter

This document outlines a set of advanced plugin templates, inspired by the "RUBY engine" concepts, adapted for integration into the Dark Matter meta-environmental layer. These templates provide blueprints for extending Dark Matter's capabilities in areas such as 4D rendering, AI-driven content generation, in-world UI, physics simulation, agent behavior, RL debugging, and procedural content generation.

## Core Plugin Structure

Each plugin generally consists of:
- `plugin.json`: A manifest file describing the plugin's metadata, entry point, type, and exposed functionalities.
- Python (`.py`) files: Backend logic, integration with Dark Matter core, and complex computations.
- JavaScript (`.js`) files: Frontend logic, UI interactions, and communication with the Python backend.
- HTML (`.html`) files: User interface panels for plugin configuration and control.

## Plugin Templates

### 1. 4D Mesh Generator Plugin

**Description:** A procedural 4D (tesseract/hypercube) mesh generator and timeline animator. This plugin is crucial for visualizing the space-time concepts introduced in the 4D Enrichment.

**Type:** `generator`

**Exposed Functions:**
- `generate_mesh4d(engine, size=1.0)`: Generates a 4D mesh and projects it to 3D for rendering.
- `animate_mesh4d(engine, rotation_plane=(0,3), steps=60)`: Animates the 4D mesh rotation through a specified plane.

**Integration Notes:**
- Requires `numpy` for mathematical operations.
- Interacts with the `engine.renderer` to create and update meshes, implying a rendering abstraction layer within Dark Matter.
- UI provides controls for mesh generation and animation.

**Example Use Case:** Visualizing quantum superposition states or temporal distortions within an environment.

### 2. AI Style Mutation Plugin

**Description:** An AI-driven material and world style mutation plugin using LLM prompts. This allows for dynamic, AI-generated visual transformations of environments.

**Type:** `ai+style`

**Exposed Functions:**
- `mutate_style(engine, target="scene", style_prompt="cyberpunk city at night")`: Applies an AI-generated style to a target (e.g., scene, object) based on a text prompt.
- `preview_style(engine, target="scene", style_prompt="cyberpunk city at night")`: Provides a preview of the style mutation without committing the changes.

**Integration Notes:**
- Assumes an `engine.llm` and `engine.renderer` interface for interacting with LLMs and applying style changes.
- The UI allows users to input style prompts and trigger mutations or previews.

**Example Use Case:** Rapidly prototyping different aesthetic themes for an environment or allowing an LLM to dynamically alter the visual mood of a simulation.

### 3. In-World UI Menu Plugin

**Description:** A minimalist in-world context menu for prompt injection, plugin triggers, and AI suggestions. This provides a direct, immersive way for users to interact with the Dark Matter environment.

**Type:** `ui`

**Exposed Functions:**
- `openMenu(engine)`: Opens the in-world UI menu.

**Integration Notes:**
- This plugin is primarily JavaScript-based, interacting directly with the browser DOM.
- It provides buttons to trigger other plugins (e.g., AI Style Mutation, 4D Mesh Generator) and inject prompts into the `rubyEngine` (which would map to Dark Matter's LLM integration).

**Example Use Case:** Allowing users to issue commands or change environment parameters directly within the 3D rendered view.

### 4. Physics Engine Plugin

**Description:** A pluggable physics engine for the world, supporting gravity, collision, and potentially soft body dynamics.

**Type:** `physics`

**Exposed Functions:**
- `apply_physics(timestep=0.016)`: Applies physics calculations for a given timestep.
- `set_gravity(g)`: Sets the gravity vector for the environment.
- `add_force(obj, force)`: Applies a force to a specified object.

**Integration Notes:**
- Requires access to the `engine.world.objects` to apply physics updates.
- The UI provides controls for setting gravity.

**Example Use Case:** Simulating realistic object interactions, agent movement, and environmental forces within Dark Matter environments.

### 5. Agent NPCs Plugin

**Description:** An AI-driven NPC agent plugin that supports RL/LLM logic and world actions. This enables the creation of intelligent, interactive non-player characters.

**Type:** `agent`

**Exposed Functions:**
- `spawn_npc(engine, name="RUBYNPC", position=[0,0,0], brain="llm")`: Spawns a new NPC with a specified name, position, and AI brain type (RL or LLM).
- `set_behavior(behavior)`: Changes the behavior of an NPC.
- `npc_speak(phrase)`: Makes an NPC speak a given phrase.

**Integration Notes:**
- Requires `engine.world.add_npc` to add NPCs to the environment.
- The UI allows for spawning NPCs and selecting their brain type.

**Example Use Case:** Populating environments with intelligent agents for training, simulation, or interactive storytelling.

### 6. Live RL Debugging Plugin

**Description:** A real-time RL agent monitor, action/reward/state inspector, with a dedicated UI. This is critical for understanding and debugging reinforcement learning processes within Dark Matter.

**Type:** `debugger`

**Exposed Functions:**
- `log_step(obs, action, reward, done)`: Logs a single step of an RL agent's interaction with the environment.
- `view_state(step_idx=-1)`: Retrieves the state of the agent at a specific step.
- `reset_agent()`: Clears the agent's log.

**Integration Notes:**
- Requires the RL framework within Dark Matter to call `log_step` at each interaction.
- The UI provides a way to view logged steps and reset the log.

**Example Use Case:** Analyzing agent behavior, identifying reward function issues, or visualizing state transitions during RL training.

### 7. Procedural City Builder Plugin

**Description:** A procedural city generator with timeline and style mutation support. This allows for the rapid creation of complex urban environments.

**Type:** `generator`

**Exposed Functions:**
- `build_city(engine, size=10, seed=None)`: Generates a city with a specified size and optional seed.
- `morph_city(engine, style="futurist", timeline_point=None)`: Morphs the city's style and potentially its structure based on a style prompt and timeline point.
- `branch_city(engine, from_point)`: Creates a new branch of the city's timeline from a specific point.

**Integration Notes:**
- Requires `engine.world.add_city` to add the generated city to the environment.
- Integrates with the AI Style Mutation plugin for dynamic styling.
- The UI provides controls for city generation and manipulation.

**Example Use Case:** Rapidly generating diverse urban environments for RL agent training, or simulating the evolution of a city over time and across different stylistic branches.

## Adaptation for Dark Matter

These templates, originally for the "RUBY engine," are highly relevant to Dark Matter's vision. The `engine` object referenced in these plugins will map directly to the `DarkMatterManager` or a similar core interface within Dark Matter, allowing these functionalities to extend the meta-environmental layer. The `compatibleWith` field `"RUBY>=8.0.0"` indicates a versioning system that Dark Matter should adopt for its plugin API. The `type` field provides a useful categorization for the plugin marketplace.

Integrating these plugins will involve:
- **Standardizing the `engine` API:** Ensuring `DarkMatterManager` exposes the necessary methods (e.g., `renderer`, `world`, `llm`, `sleep`, `injectPrompt`) that these plugins expect.
- **Plugin Loading Mechanism:** Implementing a system to discover, load, and register these plugins based on their `plugin.json` manifests.
- **UI Integration:** Adapting the HTML/JavaScript UI components to fit within Dark Matter's React frontend, potentially by embedding them as iframes or translating them into React components.
- **Backend Integration:** Ensuring the Python components can interact seamlessly with Dark Matter's core logic and data models.

These templates provide a robust starting point for building a rich and extensible plugin ecosystem for the Dark Matter project.

