
import http.server
import socketserver
import os
import json

# Load configuration to find the plugin directory and port
with open("config.json", "r") as f:
    config = json.load(f)

PLUGIN_DIR = config["plugins"]["web_plugin_directory"]
PORT = config["plugins"]["available_plugins"]["tower_defense_game"]["config"]["port"]

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PLUGIN_DIR, **kwargs)

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving at port {PORT}")
    httpd.serve_forever()


