# Fabric Object & Timeline Schema

## FabricObject (TypeScript/JSON)
```json
{
  "id": "uuid",
  "name": "string",
  "geometry": { /* mesh, 4D, etc. */ },
  "material": { /* style, ai-mutation, etc. */ },
  "timeline": [
    {
      "timestamp": "float",
      "state": { /* position, morph, logic, etc. */ },
      "events": [ /* event ids */ ]
    }
  ],
  "causal_links": [
    {
      "event_id": "uuid",
      "effect_on": "FabricObject.id"
    }
  ]
}
Event (Schema)
json
Copy
Edit
{
  "id": "uuid",
  "type": "string",
  "parameters": { },
  "timestamp": "float",
  "targets": [ "FabricObject.id" ]
}
Timeline/Branch Methods
find_nearest(timestamp)

apply_event(event, timestamp)

branch(from_timestamp)

merge(otherTimeline)

export(state | animation | full history)

Example Pseudocode: Event Propagation
python
Copy
Edit
def apply_event(fabric_object, event, timestamp):
    state = fabric_object.timeline.find_nearest(timestamp)
    new_state = event.apply_to(state)
    fabric_object.timeline.append({
      "timestamp": timestamp, "state": new_state, "events": [event.id]
    })
    for link in fabric_object.causal_links:
        if link.event_id == event.id:
            apply_event(link.effect_on, event, timestamp)
yaml
Copy
Edit

---

## 5. `docs/PLUGIN_EXTENSION_TEMPLATE.md`

```markdown
# Plugin & Extension Boilerplate

## Overview
- Plug-ins extend any core module: physics, AI, renderer, UI, logic, asset.
- All plug-ins are hot-swappable and support runtime registration.

## Plugin Contract
- `register(pluginAPI)` — required entry point
- Events: `onInit`, `onEvent`, `onExport`, `onTimelineChange`
- Input/output schema (TypeScript/JSON)

## Example Skeleton (TypeScript)
```typescript
export function register(pluginAPI) {
  pluginAPI.on('init', () => { /*...*/ });
  pluginAPI.on('event', (event) => { /*...*/ });
  pluginAPI.provide('feature', (params) => { /*...*/ });
}
Testing & Hot-Reload
All plug-ins should include unit tests and live reload triggers.

yaml
Copy
Edit

---

## 6. `docs/INWORLD_DEV_SUITE_UI.md`

```markdown
# In-World Dev Suite UI Guide

## Core UI Patterns
- Prompt input (text/code/image/audio)
- Timeline scrubber, event/branch browser
- Event injection (drag/drop, click-to-inject)
- Code/logic console (live in-world eval)
- Visual debug overlay (fabric, state, causality)

## Design Guidelines
- Minimalist, transparent overlays
- Hotkey support for all tools
- “Paintable” material/fabric layers
- AI suggestions sidebar

## Example Flow
1. Player (or AI) opens Dev Suite overlay
2. Types: “Add asteroid impact at year 3, paint rust over 10 years”
3. Scrubs timeline, views branching
4. Injects GLSL shader via code console for dynamic effect