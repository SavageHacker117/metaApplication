import hashlib

def check_signature(plugin_path, public_key):
    # Example: hash check only (real: use PyCrypto for signature check)
    with open(plugin_path, "rb") as f:
        code = f.read()
    code_hash = hashlib.sha256(code).hexdigest()
    print(f"[RUBY][Secure] Plugin hash: {code_hash}")
    # Placeholder: always return trusted
    return True

def is_trusted(plugin_path):
    return check_signature(plugin_path, None)

def register(pluginAPI):
    pluginAPI.provide("check_signature", check_signature)
    pluginAPI.provide("is_trusted", is_trusted)
    print("[RUBY] secure_plugin_example registered.")
