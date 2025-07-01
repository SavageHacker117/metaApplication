class Pulse:
    def __init__(self, engine):
        self.engine = engine
        self.metrics = {"health":100, "nerf_particles":0, "agents":0, "latency":1.2}

    def get_pulse(self):
        # Example: gather live data from engine
        return self.metrics

    def subscribe_metric(self, metric, callback):
        print(f"[RUBY][Pulse] Subscribed to {metric}")
        # Placeholder: would trigger callback on metric change

def register(pluginAPI):
    p = Pulse(pluginAPI.engine)
    pluginAPI.provide("get_pulse", p.get_pulse)
    pluginAPI.provide("subscribe_metric", p.subscribe_metric)
    print("[RUBY] world_pulse_monitor plugin registered.")
