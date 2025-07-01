# Engine Fabric Blueprint

## Philosophy
- The world is not a static scene—it's a mutable, mathematical, living fabric.
- Everything is extensible, pluggable, promptable, and observable by both human and AI.

## Core Modules
- **Fabric Engine Core**: Orchestrates user input, time scrubbing, style, logic, and export.
- **Temporal Mesh & Timeline**: Stores states across time, supports branching/merging.
- **Causality Engine**: Handles event injection, consequence simulation, and causal history.
- **AI Style & Logic Layer**: Style transfer, material evolution, narrative overlays.
- **Plugin System**: Physics, rendering, AI, asset, and world logic plug-ins.
- **Dev Suite & In-World Tools**: UI for code, prompt, event injection.

## API Surface (see FABRIC_OBJECT_TIMELINE.md for schema)
- `/fabric/object` (CRUD)
- `/fabric/event` (inject/propagate)
- `/fabric/style` (AI mutate)
- `/fabric/export`
- `/fabric/branch`

## Universal Patterns
- Every edit/event is a function or transformation.
- All states and histories are inspectable, revertible, and exportable.
- AIs are first-class co-creators.

## See Also
- [FABRIC_OBJECT_TIMELINE.md](./FABRIC_OBJECT_TIMELINE.md)
- [PLUGIN_EXTENSION_TEMPLATE.md](./PLUGIN_EXTENSION_TEMPLATE.md)
- [INWORLD_DEV_SUITE_UI.md](./INWORLD_DEV_SUITE_UI.md)
