import os
import importlib.util
import json
import traceback

class PluginBus:
    def __init__(self):
        self.plugins = {}

    def register(self, name, plugin):
        self.plugins[name] = plugin
        print(f"[RUBY][PLUGIN] Registered: {name}")

    def get(self, name):
        return self.plugins.get(name, None)

def load_plugins(plugin_dir="plugins"):
    """
    Loads all plugins in plugin_dir with a valid plugin.json manifest.
    Returns a PluginBus object with loaded plugin APIs.
    """
    plugin_bus = PluginBus()
    plugin_root = os.path.abspath(plugin_dir)

    for folder in os.listdir(plugin_root):
        folder_path = os.path.join(plugin_root, folder)
        if not os.path.isdir(folder_path):
            continue

        manifest_path = os.path.join(folder_path, "plugin.json")
        if not os.path.isfile(manifest_path):
            print(f"[RUBY][PLUGIN] Skipping {folder}: No plugin.json found.")
            continue

        # Parse manifest
        try:
            with open(manifest_path, "r") as mf:
                manifest = json.load(mf)
        except Exception as e:
            print(f"[RUBY][PLUGIN] ERROR parsing manifest in {folder}: {e}")
            continue

        # Load plugin code
        entry = manifest.get("entry")
        if not entry or not entry.endswith(".py"):
            print(f"[RUBY][PLUGIN] Skipping {folder}: No valid Python entry specified.")
            continue

        entry_path = os.path.join(folder_path, entry)
        if not os.path.isfile(entry_path):
            print(f"[RUBY][PLUGIN] Entry file not found: {entry_path}")
            continue

        try:
            spec = importlib.util.spec_from_file_location(f"{folder}", entry_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            # Try to register the plugin
            if hasattr(mod, "register"):
                mod.register(plugin_bus)
                print(f"[RUBY][PLUGIN] Loaded {folder} ({entry}) successfully.")
            else:
                print(f"[RUBY][PLUGIN] WARNING: {entry} in {folder} has no register() function.")
            plugin_bus.register(folder, mod)
        except Exception as e:
            print(f"[RUBY][PLUGIN] FAILED to load {folder}: {e}\n{traceback.format_exc()}")
            continue

    print(f"[RUBY][PLUGIN] Plugin loading complete: {len(plugin_bus.plugins)} loaded.\n")
    return plugin_bus

# USAGE EXAMPLE:
# plugin_bus = load_plugins("RUBY/plugins")
# plugin_bus.plugins['city_constitution'].propose_law(...)
# plugin_bus.get('nft_creator_market').mint_nft(...)
