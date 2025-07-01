import asyncio
import websockets
import json
import datetime
import os
from flask import Flask, send_file, jsonify, request
from flask_cors import CORS
import threading
from model_3d_generator import Model3DGenerator

# Flask app for file serving
app = Flask(__name__)
CORS(app)

# Initialize 3D model generator
model_generator = Model3DGenerator()

clients = set()

def log_event(event):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] EVENT: {json.dumps(event)}")

@app.route('/models/<model_id>/<filename>')
def serve_model(model_id, filename):
    """Serve generated 3D model files"""
    file_path = os.path.join(model_generator.models_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return jsonify({"error": "File not found"}), 404

@app.route('/api/models')
def list_models():
    """List all generated models"""
    models = model_generator.list_models()
    return jsonify(models)

@app.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a specific model"""
    result = model_generator.delete_model(model_id)
    return jsonify(result)

@app.route('/api/generate-3d', methods=['POST'])
def generate_3d_api():
    """HTTP API endpoint for 3D model generation"""
    data = request.get_json()
    prompt = data.get('prompt', 'A simple 3D object')
    format_type = data.get('format', 'glb')
    quality = data.get('quality', 'medium')
    
    result = model_generator.generate_model(prompt, format_type, quality)
    
    if result['status'] == 'success':
        return jsonify({
            "status": "success",
            "model_id": result['model_id'],
            "preview_url": f"/models/{result['model_id']}/{result['model_id']}_preview.jpg",
            "download_url": f"/models/{result['model_id']}/{result['filename']}",
            "thumbnail_url": f"/models/{result['model_id']}/{result['model_id']}_thumb.jpg",
            "prompt": prompt,
            "format": format_type
        })
    else:
        return jsonify(result), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "server": "Manus AI v6.2 Enhanced",
        "timestamp": datetime.datetime.now().isoformat(),
        "models_count": len(model_generator.list_models())
    })

async def handler(websocket):
    clients.add(websocket)
    try:
        async for msg in websocket:
            data = json.loads(msg)
            log_event(data)

            # Handle 3D model generation request
            if data.get('type') == '3d:generate':
                prompt = data['payload'].get('prompt', 'A simple 3D object')
                format_type = data['payload'].get('format', 'glb')
                quality = data['payload'].get('quality', 'medium')
                
                log_event(f"Generating 3D model: {prompt}")
                
                # Generate the 3D model
                result = model_generator.generate_model(prompt, format_type, quality)
                
                if result['status'] == 'success':
                    response = {
                        "type": "3d:response",
                        "from": "Manus",
                        "to": data['from'],
                        "payload": {
                            "model_id": result['model_id'],
                            "preview_url": f"/models/{result['model_id']}/{result['model_id']}_preview.jpg",
                            "download_url": f"/models/{result['model_id']}/{result['filename']}",
                            "thumbnail_url": f"/models/{result['model_id']}/{result['model_id']}_thumb.jpg",
                            "status": "success",
                            "prompt": prompt,
                            "format": format_type,
                            "created_at": result['created_at']
                        }
                    }
                    log_event(f"3D model generated successfully: {result['model_id']}")
                else:
                    response = {
                        "type": "3d:response",
                        "from": "Manus",
                        "to": data['from'],
                        "payload": {
                            "status": "error",
                            "error": result['error']
                        }
                    }
                    log_event(f"3D model generation failed: {result['error']}")
                
                await websocket.send(json.dumps(response))

            # Handle 3D model list request
            elif data.get('type') == '3d:list':
                models = model_generator.list_models()
                response = {
                    "type": "3d:list_response",
                    "from": "Manus",
                    "to": data['from'],
                    "payload": {
                        "models": models,
                        "count": len(models)
                    }
                }
                await websocket.send(json.dumps(response))

            # Handle 3D model delete request
            elif data.get('type') == '3d:delete':
                model_id = data['payload'].get('model_id')
                if model_id:
                    result = model_generator.delete_model(model_id)
                    response = {
                        "type": "3d:delete_response",
                        "from": "Manus",
                        "to": data['from'],
                        "payload": result
                    }
                    await websocket.send(json.dumps(response))

            # Handle visualization request (existing functionality)
            elif data.get('type') == 'visualization:request':
                await websocket.send(json.dumps({
                    "type": "visualization:response",
                    "from": "Manus",
                    "to": data['from'],
                    "payload": {
                        "what": data['payload']['what'],
                        "result": "Visualization ready (Manus v6.2 Python backend)!"
                    }
                }))
            
            # Handle code generation request (existing functionality)
            elif data.get('type') == 'code:generate':
                code = 'console.log("Hello from Manus v6.2 AI!");'
                await websocket.send(json.dumps({
                    "type": "code:response",
                    "from": "Manus",
                    "to": data['from'],
                    "payload": {"code": code}
                }))
            
    finally:
        clients.remove(websocket)

def run_flask():
    """Run Flask server in a separate thread"""
    app.run(host="0.0.0.0", port=5000, debug=False)

async def main():
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Start WebSocket server
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("=" * 60)
        print("🚀 Manus AI v6.2 Enhanced Server Started!")
        print("=" * 60)
        print("📡 WebSocket server: ws://localhost:8765")
        print("🌐 HTTP server: http://localhost:5000")
        print("🎯 3D Model API: http://localhost:5000/api/generate-3d")
        print("📊 Health check: http://localhost:5000/api/health")
        print("📁 Models list: http://localhost:5000/api/models")
        print("=" * 60)
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())

