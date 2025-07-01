# Plugin & Extension Boilerplate

## Overview
Plugins and extensions are the backbone of the Fabric Engine's modularity. This template guides developers and AI integrators in adding new systems—physics, AI, renderer, UI panels, asset pipelines, world logic—without changing core code.

## Requirements
- Hot-swappable at runtime (reloadable, no engine restart needed)
- Self-registering via a standard API contract
- Clearly document all public methods, events, and input/output types

## Plugin Contract

- **Entry Point:**  
  `register(pluginAPI)` — main function exported by every plugin
- **Lifecycle Events:**  
  - `onInit`
  - `onEvent`
  - `onExport`
  - `onTimelineChange`
- **Methods/Services:**  
  - Must declare any services provided or required
  - Must provide a `manifest` (name, version, dependencies, description)
- **Schema:**  
  Prefer TypeScript, JSON Schema, or JSDoc for data/interface types

## Example Skeleton (TypeScript)

```typescript
export function register(pluginAPI) {
  pluginAPI.on('init', () => {
    // Initialization code here
  });
  pluginAPI.on('event', (event) => {
    // Respond to world or user events
  });
  pluginAPI.provide('myCustomFeature', (params) => {
    // Main feature logic
    return { result: 'success' };
  });
}

export const manifest = {
  name: "MyAwesomePlugin",
  version: "1.0.0",
  dependencies: ["physics", "timeline"],
  description: "Adds procedural physics with AI-based feedback."
};
Best Practices
Unit tests: All plugins should include a test suite for CI.

Live reload: Plugins should detect code changes and reload state gracefully.

Documentation: Every new service, event, and method should be documented in /docs/plugins/.

Examples
Physics engine add-on

Style transfer module (AI-powered)

Asset importer/exporter

Custom UI or HUD panels

World logic or event generators

pgsql
Copy
Edit

---

## 6. `docs/INWORLD_DEV_SUITE_UI.md`

```markdown
# In-World Dev Suite UI Guide

## Mission
To make the developer and AI toolkit fully accessible in-world, via a modern, minimalist, extensible user interface. All engine features—prompt input, timeline, causality, code console, export, and event editing—are available live, to both players and AI.

## Core UI Patterns

- **Prompt Input**
  - Accept text, code, images, or audio as direct world-altering commands
  - Always available via hotkey and overlay
- **Timeline Scrubber**
  - Drag to jump/branch in time
  - Event browser for injection, editing, or replay
- **Event Injection**
  - Drag & drop events, click-to-inject (right-click or palette)
  - Visualize causal links and ripple effects
- **Code/Logic Console**
  - Live JS/Python/GLSL eval (sandboxed, with AI assist)
  - Scriptable triggers and UI macros
- **Visual Debug Overlay**
  - Toggle overlays for mesh, state, causality, perf metrics
  - Inspect objects, events, timelines in real time
- **AI Suggestions Sidebar**
  - LLM-generated prompts, corrections, and “what if?” ideas
  - Click to auto-fill prompt box or preview changes

## Design Guidelines

- Minimalist, transparent overlays that never block gameplay or worldbuilding
- Customizable hotkeys for all tools
- “Paintable” material and reality layers (see Fabric Layers UI)
- Fast context switching: jump between prompt, code, timeline, and export with a single key
- Every tool is AI-usable (support speech input, code injection, etc.)

## Example Flow

1. Open Dev Suite overlay (hotkey or AI trigger)
2. Type or speak: “Add asteroid impact at year 3, paint rust over 10 years”
3. Scrub timeline to year 3, see the event branch
4. Inject or edit shader code (GLSL/JS) in live console to morph the sky
5. Review, export, or revert—all from in-world UI

## Extending the UI

- Add custom panels via plugin boilerplate (`PLUGIN_EXTENSION_TEMPLATE.md`)
- Register new overlays or toolbars
- Support for VR/AR controls and gestures
- Persistent layouts and favorites

## Dev & AI Collaboration

- Both human and AI can use the UI—design for agent handoff and co-creation
- Always log actions for replay, undo, or learning

---

**See also:**  
- [ENGINE_FABRIC_BLUEPRINT.md](./ENGINE_FABRIC_BLUEPRINT.md)  
- [FABRIC_OBJECT_TIMELINE.md](./FABRIC_OBJECT_TIMELINE.md)  
- [SPRINT_EXPANSION_GUIDE.md](./SPRINT_EXPANSION_GUIDE.md)  