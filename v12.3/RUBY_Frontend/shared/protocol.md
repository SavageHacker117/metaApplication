
# Shared Event Protocol v6.2

All AI-to-AI and UI messages use JSON over WebSocket.

## Envelope
{
  "type": "visualization:request" | "visualization:response" | "code:generate" | "code:response" | ...,
  "from": "ChatGBT" | "Manus" | ...,
  "to": "Manus" | "ChatGBT" | ...,
  "payload": { ... }
}

- "from" and "to" are agent names (can be extended).
- "payload" is structured per event type.
- New event types can be added for richer AI collaboration.
