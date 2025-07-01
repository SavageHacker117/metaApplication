class EmergencySim:
    def __init__(self, engine):
        self.engine = engine
        self.disasters = []

    def trigger_event(self, type, location, magnitude=1.0):
        event = {"type": type, "location": location, "magnitude": magnitude}
        self.disasters.append(event)
        print(f"[RUBY][Emergency] Disaster triggered: {event}")

    def branch_disaster(self, event_idx):
        print(f"[RUBY][Emergency] Branch created at disaster {event_idx}")
        # Placeholder for state copy/branch

    def rollback_state(self, to_event):
        print(f"[RUBY][Emergency] Rollback to event {to_event}")
        # Placeholder for rollback

def register(pluginAPI):
    e = EmergencySim(pluginAPI.engine)
    pluginAPI.provide("trigger_event", e.trigger_event)
    pluginAPI.provide("branch_disaster", e.branch_disaster)
    pluginAPI.provide("rollback_state", e.rollback_state)
    print("[RUBY] emergency_response plugin registered.")
