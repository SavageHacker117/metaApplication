
# AI Collab v6.2

**Hybrid Python↔JS AI Agent Collaboration with Modular Event Protocol**

---

## Quickstart

### Backend (Python)
1. `cd backend`
2. `pip install -r requirements.txt`
3. `python server_v6_2.py`

### Frontend (Browser)
1. Open `frontend_v6_2/index.html` in your browser (or run `python -m http.server` inside `frontend_v6_2/`)
2. Use the buttons to send/receive AI-to-AI and UI events.

---

## Migration & Extensibility

- v6.2 is compatible with v6.1 (legacy config/code can be copied as needed)
- To add new agent types or event types, update `/shared/protocol.md` and backend/frontend handlers

## Features

- Modular event-driven protocol
- Multiple agents (Manus, ChatGBT, ...others)
- Easy to add new AI events (see protocol)
- WebSocket-based, low latency

---
