class Constitution:
    def __init__(self, engine):
        self.engine = engine
        self.laws = []  # List of law dicts

    def propose_law(self, title, text):
        law = {"title": title, "text": text, "votes": 0, "enforced": False}
        self.laws.append(law)
        print(f"[RUBY][Constitution] Law proposed: {title}")

    def vote_law(self, law_idx, up=True):
        if law_idx < len(self.laws):
            self.laws[law_idx]["votes"] += 1 if up else -1
            print(f"[RUBY][Constitution] Vote {'up' if up else 'down'} on {self.laws[law_idx]['title']}")

    def enforce_law(self, law_idx):
        if law_idx < len(self.laws) and not self.laws[law_idx]["enforced"]:
            self.laws[law_idx]["enforced"] = True
            print(f"[RUBY][Constitution] Law ENFORCED: {self.laws[law_idx]['title']}")

def register(pluginAPI):
    c = Constitution(pluginAPI.engine)
    pluginAPI.provide("propose_law", c.propose_law)
    pluginAPI.provide("vote_law", c.vote_law)
    pluginAPI.provide("enforce_law", c.enforce_law)
    print("[RUBY] city_constitution plugin registered.")
