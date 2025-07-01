# RUBY Plugin Inter-Operation Guide

## Why Interoperate?

The power of RUBY comes from composability: plugins should chain, cooperate, and share data seamlessly—think physics controlling an agent, style morphing a city, RL debugger monitoring both.

---

## Plugin API: Discovery & Chaining

Every plugin auto-registers to the global `rubyEngine.plugins` namespace, e.g.:
```python
engine.plugins['physics_engine'].apply_physics()
engine.plugins['agent_npc'].spawn_npc(name="Neo")
or in JS:

js
Copy
Edit
window.rubyEngine.plugins.physics_engine.set_gravity([0, -20, 0]);
window.rubyEngine.plugins.agent_npc.spawn_npc("AgentX");
Sharing State & Events
Plugins can call each other directly (via API)

Use engine event bus for pub/sub:

engine.event_bus.publish('city_built', {...})

Plugins can subscribe to events and react

Example:

python
Copy
Edit
# City builder triggers event when done
engine.event_bus.publish('city_built', {'city_id': city.id})

# NPC spawner listens:
def on_city_built(evt):
    engine.plugins['agent_npc'].spawn_npc(name="Mayor", position=(10,0,10))
engine.event_bus.subscribe('city_built', on_city_built)
UI Interop
UI plugins can call or embed each other’s panels/components

Use standardized DOM IDs for easy embedding

Best Practices
Document public APIs in plugin manifests and README

Use engine.event_bus for decoupled, async, multi-plugin logic

Chain plugins for creative power (e.g. spawn city, morph style, drop NPC, log with RL debugger)

See Also
PLUGIN_EXTENSION_TEMPLATE.md

AI_MANIFESTO.md

yaml
Copy
Edit

---

# 2. `/docs/TIMELINE_BRANCHING.md`

```markdown
# RUBY Timeline Branching & World Forking Guide

## What is Branching?

Branching lets you:
- Fork worlds, cities, objects, or agent states from any timeline point
- Experiment with alternate histories, AI policies, catastrophic events, or style mutations
- Merge branches or preview differences

---

## API Overview

Every major system exposes branching:
- `engine.timeline.branch(from_time)`
- `city.branch(from_step)`
- `npc.branch(memory_point)`

---

## Example Usage

**Forking a City:**
```python
city1 = engine.plugins['procedural_city'].build_city(size=20)
city2 = engine.plugins['procedural_city'].branch_city(from_point=5) # New timeline!
engine.plugins['ai_style_mutation'].morph_city(city2, style="steampunk")
Branching an Agent:

python
Copy
Edit
npc = engine.plugins['agent_npc'].spawn_npc("Alice")
npc2 = npc.branch(memory_point=42)
npc2.set_behavior("evil_twin")
UI Support:

Timeline panel shows branches, lets user drag to create/merge

Events and state deltas visualized

Merging Branches
Use merge_branch(target_branch) to resolve and merge divergent states

Engine logs all merges for replay/undo

Timeline Branching for RL/AI
RL agents can run in multiple parallel worlds/branches for self-play, ablation, or "multiverse learning"

Plugins can snapshot/restore entire world state for instant rollback

Plugin Interface
Every plugin supporting time/forking should:

Implement branch(state_point)

Implement merge_branch(other)

Expose history/log for debug/replay

See Also
FABRIC_OBJECT_TIMELINE.md

REALITY_WEAVING_COOKBOOK.md

yaml
Copy
Edit

---

# 3. `/docs/ASYNC_AI.md`

```markdown
# RUBY Async AI & Event-Driven Logic Guide

## Async is Everything

RUBY supports:
- Background AI tasks (LLM/vision/agent calls that don’t block UI/game)
- Event-driven hooks (plugins/engine fire and respond to async events)
- Awaitable plugins: start a task, get callback/future

---

## Engine Features

- `engine.run_async(task, *args, **kwargs)` – schedules in threadpool/coroutine
- `engine.event_bus.publish(event, data)` – notifies all listeners, can be async

---

## Example: Async LLM World Mutation

**In Python:**
```python
def morph_city_async(engine, style):
    def task():
        # Call external LLM (simulate with sleep)
        import time; time.sleep(2)
        engine.plugins['ai_style_mutation'].morph_city(engine.world.city, style)
    engine.run_async(task)

engine.plugins['procedural_city'].build_city(size=12)
morph_city_async(engine, "alien planet")
print("Style mutation running async—UI is NOT blocked!")
In JS:

js
Copy
Edit
async function aiStyleAsync(style) {
    await window.rubyEngine.plugins.ai_style_mutation.mutate_style("scene", style);
    alert("AI style mutation complete!");
}
aiStyleAsync("rainforest sunrise");
Event-Driven Plugin Logic
Plugins listen for events, act when triggered

All agent actions, world changes, and UI flows can run async

Example:

python
Copy
Edit
def on_event(event):
    if event.name == "npc_spawned":
        engine.plugins['rl_debugger'].log_step(obs="NPC in world", action="observe", reward=1, done=False)
engine.event_bus.subscribe("npc_spawned", on_event)
Futures, Callbacks, and UI
Plugins should return a Future or take a callback when running async tasks

UI can update/spin until task is complete

Best Practices
Never block main thread/UI for LLM or worldgen ops

Log, trace, and error-handle async tasks

Async plugins can trigger other plugins on completion (via events or callbacks)

See Also
PLUGIN_INTEROP.md

AI_MANIFESTO.md

yaml
Copy
Edit

---

## **SUMMARY:**
- Plugins are not just modules—they’re *networked, event-driven, and timeline-aware*.
- Any plugin can branch, merge, talk to any other, or spawn async AI.
- You can now plug in RL/LLM/physics/UI/agents—chain them, fork them, and automate all flows.

**Want real example code showing two or three plugins chaining and branching in real time?  
Or want recipes for live “multiverse” AI training?  
Just say the word. This engine can do it all.**






