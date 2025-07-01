# In-World Dev Suite UI Guide

## Purpose

Enable both developers and AI agents to interactively create, edit, and debug worlds, assets, events, and logic *from inside* the simulation—no engine restart, no external tools. The Dev Suite UI is always accessible, extensible, and AI-aware.

---

## Core UI Features

### 1. Universal Prompt Box
- Accepts text, code (JS/GLSL/Python), images, and audio
- Handles world creation, timeline events, asset injection, AI chat, and code eval
- “Feed the world” interface: drag-and-drop data or scripts

### 2. Timeline Navigator & Scrubber
- Lets user (or AI) jump to, branch, or merge any point in time
- Visual event injection—drag & drop to create or edit
- See live preview of world state at any timestamp

### 3. Causality/Event Editor
- Graphical event browser: inject, edit, connect, or delete events
- Visualize ripple effects, causal chains, and alternative histories

### 4. Code & Logic Console
- Live code execution (JS, Python, GLSL) in a sandboxed environment
- Script triggers, auto-run macros, custom inspector hooks
- AI-powered code suggestions and “repair” options

### 5. Visual Overlay & Debug Tools
- Toggle mesh/physics/collision overlays, state inspectors, and metrics
- Select any object to view its timeline, state, and causal links
- Export or snapshot any state or animation directly

### 6. AI Sidebar & Agent Tools
- LLM/AI prompt suggestions, event autocompletion, “what if?” scenario builder
- Click-to-apply AI-generated ideas, inject logic, or auto-script new rules

---

## UI/UX Design Principles

- Overlay should be minimalist, transparent, and non-intrusive
- All tools available via hotkey, radial menu, or context-sensitive popover
- UI is **extensible**—new plugins or panels can register via manifest
- Designed for both desktop and VR/AR (gesture, gaze, speech input)
- All actions logged for undo, replay, AI learning, and debugging

---

## Example User Flow

1. **Open Dev Suite UI**:  
   Press hotkey or say “Open Dev Suite.”

2. **Inject an event**:  
   Type or say: “Trigger solar flare at year 2100.”
   - Drag on timeline to year 2100, drop event, edit its parameters.

3. **Live code a logic tweak**:  
   Open Code Console, paste JavaScript to add new portal rule.
   - Preview changes instantly in-world.

4. **Visualize & Debug**:  
   Toggle overlays to see mesh stats, physics, causality graph.

5. **Get AI suggestions**:  
   Click “Suggest” in AI Sidebar, see auto-generated world events and style changes.

6. **Export state or animation**:  
   Click Export, choose format (GLTF, 4D, replay video, etc.)

---

## Extending the Dev Suite UI

- **Register a new tool or panel:**  
  Create plugin with manifest (see PLUGIN_EXTENSION_TEMPLATE.md) and register UI hooks.

- **VR/AR/Voice:**  
  Add gesture and speech support via the plugin API.

- **Agent Collaboration:**  
  Both human and AI agents can use and extend the UI; support “handoff” (AI continues a human-initiated flow, or vice versa).

---

## Best Practices

- Maintain accessibility for all user types (devs, artists, AI agents, testers)
- Every tool should have help/hints, ideally AI-powered
- Always log every action and change for versioning, debugging, and rollback

---

**See Also:**  
- [ENGINE_FABRIC_BLUEPRINT.md](./ENGINE_FABRIC_BLUEPRINT.md)  
- [FABRIC_OBJECT_TIMELINE.md](./FABRIC_OBJECT_TIMELINE.md)  
- [PLUGIN_EXTENSION_TEMPLATE.md](./PLUGIN_EXTENSION_TEMPLATE.md)  