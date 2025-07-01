#!/bin/bash

# RL-LLM NeRF Integration Setup Script
# This script adds CUDA-enabled NeRF functionality to the Darkmatter RL-LLM trainer
# Author: Manus AI
# Version: 1.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(pwd)"
NERF_DIR="$PROJECT_ROOT/nerf_integration"
PYTHON_ENV="$PROJECT_ROOT/venv"
CUDA_VERSION="12.0"

echo -e "${BLUE}🚀 RL-LLM NeRF Integration Setup${NC}"
echo -e "${BLUE}=================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the project root directory."
    exit 1
fi

print_status "Found package.json - proceeding with setup"

# Check CUDA availability
check_cuda() {
    echo -e "${BLUE}🔍 Checking CUDA availability...${NC}"
    if command -v nvcc &> /dev/null; then
        CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        print_status "CUDA $CUDA_VER detected"
        return 0
    else
        print_warning "CUDA not detected. Installing CPU-only version."
        return 1
    fi
}

# Create directory structure
create_directories() {
    echo -e "${BLUE}📁 Creating directory structure...${NC}"
    
    mkdir -p "$NERF_DIR"/{systems,cuda/{kernels,utils},components,rl_integration,config,assets/{shaders,models}}
    mkdir -p "$PROJECT_ROOT/src/nerf"
    mkdir -p "$PROJECT_ROOT/backend"
    
    print_status "Directory structure created"
}

# Install Python dependencies
setup_python_env() {
    echo -e "${BLUE}🐍 Setting up Python environment...${NC}"
    
    if [ ! -d "$PYTHON_ENV" ]; then
        python3 -m venv "$PYTHON_ENV"
        print_status "Python virtual environment created"
    fi
    
    source "$PYTHON_ENV/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install CUDA-enabled PyTorch if CUDA is available
    if check_cuda; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other dependencies
    pip install numpy opencv-python flask flask-socketio flask-cors
    pip install matplotlib plotly scipy scikit-image
    pip install trimesh open3d
    
    print_status "Python dependencies installed"
}

# Install Node.js dependencies
setup_node_deps() {
    echo -e "${BLUE}📦 Installing Node.js dependencies...${NC}"
    
    # Add new dependencies to package.json
    npm install --save socket.io-client @types/three
    npm install --save-dev @types/socket.io-client
    
    print_status "Node.js dependencies installed"
}

# Create NeRF System files
create_nerf_systems() {
    echo -e "${BLUE}🧠 Creating NeRF system files...${NC}"
    
    # NeRFSystem.ts
    cat > "$NERF_DIR/systems/NeRFSystem.ts" << 'EOF'
import * as THREE from 'three';
import { io, Socket } from 'socket.io-client';

export interface NeRFConfig {
  resolution: number;
  maxSteps: number;
  stepSize: number;
  densityThreshold: number;
  learningRate: number;
}

export class NeRFSystem {
  private scene: THREE.Group | null = null;
  private socket: Socket | null = null;
  private isTraining: boolean = false;
  private config: NeRFConfig;
  private sceneData: any[] = [];
  private renderTarget: THREE.WebGLRenderTarget;
  private volumetricMaterial: THREE.ShaderMaterial;

  constructor(config: Partial<NeRFConfig> = {}) {
    this.config = {
      resolution: 256,
      maxSteps: 128,
      stepSize: 0.01,
      densityThreshold: 0.01,
      learningRate: 0.001,
      ...config
    };

    this.initializeSocket();
    this.createRenderTarget();
    this.createVolumetricMaterial();
  }

  private initializeSocket(): void {
    this.socket = io('http://localhost:5000');
    
    this.socket.on('connect', () => {
      console.log('🔗 Connected to NeRF backend');
    });

    this.socket.on('training_progress', (data) => {
      this.handleTrainingProgress(data);
    });

    this.socket.on('nerf_update', (data) => {
      this.updateNeRFRepresentation(data);
    });
  }

  private createRenderTarget(): void {
    this.renderTarget = new THREE.WebGLRenderTarget(
      this.config.resolution,
      this.config.resolution,
      {
        format: THREE.RGBAFormat,
        type: THREE.FloatType,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter
      }
    );
  }

  private createVolumetricMaterial(): void {
    this.volumetricMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uResolution: { value: new THREE.Vector2(this.config.resolution, this.config.resolution) },
        uMaxSteps: { value: this.config.maxSteps },
        uStepSize: { value: this.config.stepSize },
        uDensityThreshold: { value: this.config.densityThreshold },
        uVolumeTexture: { value: null },
        uCameraPosition: { value: new THREE.Vector3() },
        uCameraDirection: { value: new THREE.Vector3() }
      },
      vertexShader: this.getVertexShader(),
      fragmentShader: this.getFragmentShader(),
      transparent: true,
      side: THREE.DoubleSide
    });
  }

  setScene(scene: THREE.Group): void {
    this.scene = scene;
  }

  startTraining(): void {
    if (!this.socket || this.isTraining) return;

    this.isTraining = true;
    this.socket.emit('start_nerf_training', {
      config: this.config,
      sceneData: this.sceneData
    });

    console.log('🎯 NeRF training started');
  }

  stopTraining(): void {
    if (!this.socket || !this.isTraining) return;

    this.isTraining = false;
    this.socket.emit('stop_nerf_training');

    console.log('⏹️ NeRF training stopped');
  }

  captureScene(camera: THREE.Camera): void {
    if (!this.scene) return;

    const sceneCapture = {
      timestamp: Date.now(),
      cameraPosition: camera.position.toArray(),
      cameraRotation: camera.rotation.toArray(),
      objects: this.extractSceneObjects()
    };

    this.sceneData.push(sceneCapture);

    // Send to backend for processing
    if (this.socket) {
      this.socket.emit('scene_capture', sceneCapture);
    }
  }

  private extractSceneObjects(): any[] {
    if (!this.scene) return [];

    const objects: any[] = [];
    
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh) {
        objects.push({
          position: object.position.toArray(),
          rotation: object.rotation.toArray(),
          scale: object.scale.toArray(),
          geometry: object.geometry.type,
          material: this.extractMaterialData(object.material)
        });
      }
    });

    return objects;
  }

  private extractMaterialData(material: any): any {
    if (material instanceof THREE.MeshBasicMaterial) {
      return {
        type: 'basic',
        color: material.color.getHex(),
        opacity: material.opacity
      };
    }
    // Add more material types as needed
    return { type: 'unknown' };
  }

  private handleTrainingProgress(data: any): void {
    console.log(`📈 NeRF Training Progress: ${data.progress}% - Loss: ${data.loss}`);
    
    // Emit event for UI updates
    window.dispatchEvent(new CustomEvent('nerf-training-progress', { detail: data }));
  }

  private updateNeRFRepresentation(data: any): void {
    // Update the volumetric representation with new NeRF data
    if (data.volumeTexture) {
      this.volumetricMaterial.uniforms.uVolumeTexture.value = data.volumeTexture;
    }
  }

  update(deltaTime: number, camera: THREE.Camera): void {
    if (!this.scene) return;

    // Update uniforms
    this.volumetricMaterial.uniforms.uTime.value += deltaTime;
    this.volumetricMaterial.uniforms.uCameraPosition.value.copy(camera.position);
    
    const direction = new THREE.Vector3();
    camera.getWorldDirection(direction);
    this.volumetricMaterial.uniforms.uCameraDirection.value.copy(direction);

    // Capture scene periodically during training
    if (this.isTraining && Math.random() < 0.01) { // 1% chance per frame
      this.captureScene(camera);
    }
  }

  private getVertexShader(): string {
    return `
      varying vec2 vUv;
      varying vec3 vPosition;
      
      void main() {
        vUv = uv;
        vPosition = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `;
  }

  private getFragmentShader(): string {
    return `
      uniform float uTime;
      uniform vec2 uResolution;
      uniform int uMaxSteps;
      uniform float uStepSize;
      uniform float uDensityThreshold;
      uniform sampler3D uVolumeTexture;
      uniform vec3 uCameraPosition;
      uniform vec3 uCameraDirection;
      
      varying vec2 vUv;
      varying vec3 vPosition;
      
      vec4 rayMarch(vec3 rayOrigin, vec3 rayDirection) {
        vec4 color = vec4(0.0);
        float depth = 0.0;
        
        for (int i = 0; i < uMaxSteps; i++) {
          vec3 samplePos = rayOrigin + rayDirection * depth;
          
          // Sample volume texture
          vec4 sample = texture(uVolumeTexture, samplePos * 0.5 + 0.5);
          
          if (sample.a > uDensityThreshold) {
            // Accumulate color and opacity
            color.rgb += sample.rgb * sample.a * (1.0 - color.a);
            color.a += sample.a * (1.0 - color.a);
            
            if (color.a > 0.95) break; // Early termination
          }
          
          depth += uStepSize;
          if (depth > 10.0) break; // Max distance
        }
        
        return color;
      }
      
      void main() {
        vec3 rayDirection = normalize(vPosition - uCameraPosition);
        vec4 color = rayMarch(uCameraPosition, rayDirection);
        
        gl_FragColor = color;
      }
    `;
  }

  getTrainingStats(): any {
    return {
      isTraining: this.isTraining,
      scenesCaptured: this.sceneData.length,
      config: this.config
    };
  }

  dispose(): void {
    if (this.socket) {
      this.socket.disconnect();
    }
    
    if (this.renderTarget) {
      this.renderTarget.dispose();
    }
    
    if (this.volumetricMaterial) {
      this.volumetricMaterial.dispose();
    }
  }
}
EOF

    print_status "NeRF system files created"
}

# Create CUDA backend
create_cuda_backend() {
    echo -e "${BLUE}⚡ Creating CUDA backend...${NC}"
    
    # CUDAManager.py
    cat > "$NERF_DIR/cuda/CUDAManager.py" << 'EOF'
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class CUDAManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_pool = {}
        self.logger = logging.getLogger(__name__)
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA initialized on {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.logger.warning("CUDA not available, using CPU")
    
    def allocate_memory(self, name: str, shape: Tuple, dtype=torch.float32) -> torch.Tensor:
        """Allocate GPU memory with caching"""
        key = f"{name}_{shape}_{dtype}"
        
        if key not in self.memory_pool:
            tensor = torch.zeros(shape, dtype=dtype, device=self.device)
            self.memory_pool[key] = tensor
            self.logger.debug(f"Allocated {key}: {tensor.numel() * tensor.element_size() / 1e6:.1f} MB")
        
        return self.memory_pool[key]
    
    def free_memory(self, name: str = None):
        """Free GPU memory"""
        if name:
            keys_to_remove = [k for k in self.memory_pool.keys() if k.startswith(name)]
            for key in keys_to_remove:
                del self.memory_pool[key]
        else:
            self.memory_pool.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict:
        """Get current memory usage statistics"""
        if not torch.cuda.is_available():
            return {"device": "cpu", "allocated": 0, "cached": 0}
        
        return {
            "device": "cuda",
            "allocated": torch.cuda.memory_allocated() / 1e9,
            "cached": torch.cuda.memory_reserved() / 1e9,
            "max_allocated": torch.cuda.max_memory_allocated() / 1e9
        }
    
    def optimize_memory(self):
        """Optimize memory usage"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

class NeRFNetwork(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=8, output_dim=4):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for i in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            # Skip connection at middle layer
            if i == num_layers // 2 - 1:
                layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
                layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.network = nn.Sequential(*layers)
        self.skip_layer = num_layers // 2
    
    def forward(self, x):
        input_x = x
        
        for i, layer in enumerate(self.network):
            if i == self.skip_layer * 2:  # Account for ReLU layers
                x = torch.cat([x, input_x], dim=-1)
            x = layer(x)
        
        return x

class NeRFTrainer:
    def __init__(self, cuda_manager: CUDAManager, config: Dict):
        self.cuda_manager = cuda_manager
        self.config = config
        self.device = cuda_manager.device
        
        # Initialize network
        self.network = NeRFNetwork(
            input_dim=config.get('input_dim', 3),
            hidden_dim=config.get('hidden_dim', 256),
            num_layers=config.get('num_layers', 8),
            output_dim=config.get('output_dim', 4)
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.get('learning_rate', 0.001)
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        self.training_step = 0
        self.logger = logging.getLogger(__name__)
    
    def train_step(self, rays_o: torch.Tensor, rays_d: torch.Tensor, target_colors: torch.Tensor) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Sample points along rays
        t_vals = torch.linspace(0, 1, self.config.get('num_samples', 64), device=self.device)
        points = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., None]
        
        # Flatten for network input
        points_flat = points.reshape(-1, 3)
        
        # Forward pass
        outputs = self.network(points_flat)
        outputs = outputs.reshape(*points.shape[:-1], 4)  # Reshape back
        
        # Volume rendering
        colors, weights = self.volume_render(outputs, t_vals)
        
        # Compute loss
        loss = self.criterion(colors, target_colors)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        self.training_step += 1
        
        return loss.item()
    
    def volume_render(self, outputs: torch.Tensor, t_vals: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Volume rendering equation"""
        rgb = outputs[..., :3]
        density = outputs[..., 3]
        
        # Compute distances between samples
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        
        # Compute alpha values
        alpha = 1.0 - torch.exp(-density * dists)
        
        # Compute transmittance
        transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
        transmittance = torch.cat([torch.ones_like(transmittance[..., :1]), transmittance[..., :-1]], dim=-1)
        
        # Compute weights
        weights = alpha * transmittance
        
        # Compute final color
        colors = torch.sum(weights[..., None] * rgb, dim=-2)
        
        return colors, weights
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
EOF

    # Flask backend server
    cat > "$PROJECT_ROOT/backend/nerf_server.py" << 'EOF'
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import sys
import os
import threading
import time
import logging

# Add the nerf_integration directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nerf_integration'))

from cuda.CUDAManager import CUDAManager, NeRFTrainer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'nerf_secret_key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
cuda_manager = None
nerf_trainer = None
training_thread = None
is_training = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "cuda_available": cuda_manager.device.type == 'cuda'})

@app.route('/memory_stats')
def memory_stats():
    return jsonify(cuda_manager.get_memory_stats())

@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    emit('status', {'connected': True, 'cuda_available': cuda_manager.device.type == 'cuda'})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('start_nerf_training')
def handle_start_training(data):
    global nerf_trainer, training_thread, is_training
    
    if is_training:
        emit('error', {'message': 'Training already in progress'})
        return
    
    config = data.get('config', {})
    scene_data = data.get('sceneData', [])
    
    logger.info(f"Starting NeRF training with {len(scene_data)} scene captures")
    
    # Initialize trainer
    nerf_trainer = NeRFTrainer(cuda_manager, config)
    
    # Start training in separate thread
    is_training = True
    training_thread = threading.Thread(target=training_loop, args=(scene_data,))
    training_thread.start()
    
    emit('training_started', {'message': 'NeRF training started'})

@socketio.on('stop_nerf_training')
def handle_stop_training():
    global is_training
    is_training = False
    logger.info("Stopping NeRF training")
    emit('training_stopped', {'message': 'NeRF training stopped'})

@socketio.on('scene_capture')
def handle_scene_capture(data):
    logger.info(f"Received scene capture at {data.get('timestamp')}")
    # Process scene capture data here
    emit('scene_processed', {'status': 'processed'})

def training_loop(scene_data):
    global is_training, nerf_trainer
    
    step = 0
    while is_training and nerf_trainer:
        try:
            # Simulate training step (replace with actual training logic)
            loss = 0.1 * (1.0 - step / 1000.0)  # Decreasing loss
            
            # Emit progress
            socketio.emit('training_progress', {
                'step': step,
                'loss': loss,
                'progress': min(100, step / 10)  # 100% at step 1000
            })
            
            step += 1
            time.sleep(0.1)  # Simulate training time
            
            if step >= 1000:  # Stop after 1000 steps
                break
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            socketio.emit('training_error', {'error': str(e)})
            break
    
    is_training = False
    socketio.emit('training_completed', {'message': 'Training completed'})

if __name__ == '__main__':
    # Initialize CUDA manager
    cuda_manager = CUDAManager()
    
    logger.info("Starting NeRF backend server...")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
EOF

    print_status "CUDA backend created"
}

# Create React components
create_react_components() {
    echo -e "${BLUE}⚛️ Creating React components...${NC}"
    
    # NeRF Dashboard component
    cat > "$PROJECT_ROOT/src/nerf/NeRFDashboard.tsx" << 'EOF'
import React, { useState, useEffect } from 'react';
import { io, Socket } from 'socket.io-client';

interface TrainingStats {
  step: number;
  loss: number;
  progress: number;
}

interface MemoryStats {
  device: string;
  allocated: number;
  cached: number;
  max_allocated?: number;
}

export const NeRFDashboard: React.FC = () => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStats, setTrainingStats] = useState<TrainingStats>({ step: 0, loss: 0, progress: 0 });
  const [memoryStats, setMemoryStats] = useState<MemoryStats>({ device: 'cpu', allocated: 0, cached: 0 });

  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('connect', () => {
      setIsConnected(true);
      console.log('Connected to NeRF backend');
    });

    newSocket.on('disconnect', () => {
      setIsConnected(false);
      console.log('Disconnected from NeRF backend');
    });

    newSocket.on('training_progress', (data: TrainingStats) => {
      setTrainingStats(data);
    });

    newSocket.on('training_started', () => {
      setIsTraining(true);
    });

    newSocket.on('training_stopped', () => {
      setIsTraining(false);
    });

    newSocket.on('training_completed', () => {
      setIsTraining(false);
    });

    // Fetch memory stats periodically
    const memoryInterval = setInterval(() => {
      fetch('http://localhost:5000/memory_stats')
        .then(res => res.json())
        .then(setMemoryStats)
        .catch(console.error);
    }, 2000);

    return () => {
      newSocket.close();
      clearInterval(memoryInterval);
    };
  }, []);

  const startTraining = () => {
    if (socket && !isTraining) {
      socket.emit('start_nerf_training', {
        config: {
          resolution: 256,
          maxSteps: 128,
          stepSize: 0.01,
          densityThreshold: 0.01,
          learningRate: 0.001
        },
        sceneData: []
      });
    }
  };

  const stopTraining = () => {
    if (socket && isTraining) {
      socket.emit('stop_nerf_training');
    }
  };

  return (
    <div className="nerf-dashboard p-6 bg-gray-900 text-white rounded-lg">
      <h2 className="text-2xl font-bold mb-4">NeRF Training Dashboard</h2>
      
      {/* Connection Status */}
      <div className="mb-4">
        <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm ${
          isConnected ? 'bg-green-600' : 'bg-red-600'
        }`}>
          <div className={`w-2 h-2 rounded-full mr-2 ${
            isConnected ? 'bg-green-300' : 'bg-red-300'
          }`}></div>
          {isConnected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      {/* Memory Stats */}
      <div className="mb-6 p-4 bg-gray-800 rounded">
        <h3 className="text-lg font-semibold mb-2">Memory Usage</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-400">Device</p>
            <p className="text-lg font-mono">{memoryStats.device.toUpperCase()}</p>
          </div>
          <div>
            <p className="text-sm text-gray-400">Allocated</p>
            <p className="text-lg font-mono">{memoryStats.allocated.toFixed(2)} GB</p>
          </div>
          {memoryStats.device === 'cuda' && (
            <>
              <div>
                <p className="text-sm text-gray-400">Cached</p>
                <p className="text-lg font-mono">{memoryStats.cached.toFixed(2)} GB</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Max Allocated</p>
                <p className="text-lg font-mono">{(memoryStats.max_allocated || 0).toFixed(2)} GB</p>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Training Controls */}
      <div className="mb-6">
        <div className="flex gap-4">
          <button
            onClick={startTraining}
            disabled={!isConnected || isTraining}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded"
          >
            Start Training
          </button>
          <button
            onClick={stopTraining}
            disabled={!isConnected || !isTraining}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded"
          >
            Stop Training
          </button>
        </div>
      </div>

      {/* Training Progress */}
      {isTraining && (
        <div className="mb-6 p-4 bg-gray-800 rounded">
          <h3 className="text-lg font-semibold mb-2">Training Progress</h3>
          <div className="mb-2">
            <div className="flex justify-between text-sm">
              <span>Progress</span>
              <span>{trainingStats.progress.toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${trainingStats.progress}%` }}
              ></div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-gray-400">Step</p>
              <p className="font-mono">{trainingStats.step}</p>
            </div>
            <div>
              <p className="text-gray-400">Loss</p>
              <p className="font-mono">{trainingStats.loss.toFixed(6)}</p>
            </div>
          </div>
        </div>
      )}

      {/* NeRF Visualization */}
      <div className="p-4 bg-gray-800 rounded">
        <h3 className="text-lg font-semibold mb-2">NeRF Visualization</h3>
        <div className="w-full h-64 bg-gray-700 rounded flex items-center justify-center">
          <p className="text-gray-400">3D NeRF render will appear here</p>
        </div>
      </div>
    </div>
  );
};
EOF

    print_status "React components created"
}

# Update GameEngine integration
update_game_engine() {
    echo -e "${BLUE}🎮 Updating GameEngine integration...${NC}"
    
    # Create integration file
    cat > "$PROJECT_ROOT/src/nerf/GameEngineNeRFIntegration.ts" << 'EOF'
import { NeRFSystem } from '../../nerf_integration/systems/NeRFSystem';

// Extend the existing GameEngine class with NeRF functionality
export class GameEngineNeRFExtension {
  private nerfSystem: NeRFSystem;
  private originalGameEngine: any;

  constructor(gameEngine: any) {
    this.originalGameEngine = gameEngine;
    this.nerfSystem = new NeRFSystem({
      resolution: 256,
      maxSteps: 128,
      stepSize: 0.01,
      densityThreshold: 0.01,
      learningRate: 0.001
    });
  }

  initialize(): void {
    // Set the scene for NeRF system
    if (this.originalGameEngine.scene) {
      this.nerfSystem.setScene(this.originalGameEngine.scene);
    }

    // Start NeRF training
    this.nerfSystem.startTraining();

    console.log('🧠 NeRF system integrated with GameEngine');
  }

  update(deltaTime: number, camera: any): void {
    // Update NeRF system
    this.nerfSystem.update(deltaTime, camera);

    // Capture scene periodically for training
    if (Math.random() < 0.005) { // 0.5% chance per frame
      this.nerfSystem.captureScene(camera);
    }
  }

  getNeRFStats(): any {
    return this.nerfSystem.getTrainingStats();
  }

  dispose(): void {
    this.nerfSystem.dispose();
  }
}

// Helper function to integrate NeRF with existing GameEngine
export function integrateNeRFWithGameEngine(gameEngine: any): GameEngineNeRFExtension {
  const nerfExtension = new GameEngineNeRFExtension(gameEngine);
  
  // Store original update method
  const originalUpdate = gameEngine.update.bind(gameEngine);
  
  // Override update method to include NeRF updates
  gameEngine.update = function(deltaTime: number, camera?: any) {
    // Call original update
    originalUpdate(deltaTime);
    
    // Update NeRF system if camera is available
    if (camera) {
      nerfExtension.update(deltaTime, camera);
    }
  };

  // Add NeRF methods to GameEngine
  gameEngine.nerfExtension = nerfExtension;
  gameEngine.getNeRFStats = () => nerfExtension.getNeRFStats();
  gameEngine.initializeNeRF = () => nerfExtension.initialize();

  return nerfExtension;
}
EOF

    print_status "GameEngine integration updated"
}

# Create configuration files
create_config_files() {
    echo -e "${BLUE}⚙️ Creating configuration files...${NC}"
    
    # NeRF configuration
    cat > "$NERF_DIR/config/nerf_config.json" << 'EOF'
{
  "model": {
    "input_dim": 3,
    "hidden_dim": 256,
    "num_layers": 8,
    "output_dim": 4,
    "skip_connections": [4]
  },
  "training": {
    "learning_rate": 0.001,
    "batch_size": 1024,
    "num_epochs": 1000,
    "num_samples": 64,
    "near": 0.1,
    "far": 10.0
  },
  "rendering": {
    "resolution": 256,
    "max_steps": 128,
    "step_size": 0.01,
    "density_threshold": 0.01
  },
  "optimization": {
    "use_cuda": true,
    "mixed_precision": true,
    "gradient_clipping": 1.0
  }
}
EOF

    # CUDA configuration
    cat > "$NERF_DIR/config/cuda_config.json" << 'EOF'
{
  "device": "auto",
  "memory_management": {
    "pool_size": "auto",
    "cache_size": 1024,
    "garbage_collection": true
  },
  "optimization": {
    "use_tensorrt": false,
    "use_amp": true,
    "compile_mode": "default"
  },
  "kernels": {
    "ray_marching": {
      "block_size": 256,
      "grid_size": "auto"
    },
    "volume_rendering": {
      "block_size": 512,
      "grid_size": "auto"
    }
  }
}
EOF

    # Training configuration
    cat > "$NERF_DIR/config/training_config.json" << 'EOF'
{
  "rl_integration": {
    "reward_weight": 0.1,
    "scene_capture_frequency": 0.01,
    "training_frequency": 10,
    "evaluation_frequency": 100
  },
  "data": {
    "max_scenes": 10000,
    "scene_buffer_size": 1000,
    "augmentation": {
      "rotation": true,
      "translation": true,
      "scaling": false
    }
  },
  "logging": {
    "log_level": "INFO",
    "save_frequency": 100,
    "checkpoint_frequency": 500
  }
}
EOF

    print_status "Configuration files created"
}

# Create startup script
create_startup_script() {
    echo -e "${BLUE}🚀 Creating startup script...${NC}"
    
    cat > "$PROJECT_ROOT/start_nerf_rl_llm.sh" << 'EOF'
#!/bin/bash

# RL-LLM NeRF Integration Startup Script

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="$PROJECT_ROOT/venv"

echo "🚀 Starting RL-LLM NeRF Integration..."

# Activate Python environment
source "$PYTHON_ENV/bin/activate"

# Start the NeRF backend server
echo "🐍 Starting NeRF backend server..."
cd "$PROJECT_ROOT/backend"
python nerf_server.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start the frontend development server
echo "⚛️ Starting frontend development server..."
cd "$PROJECT_ROOT"
npm run dev &
FRONTEND_PID=$!

echo "✅ RL-LLM NeRF Integration started!"
echo "📊 NeRF Dashboard: http://localhost:5173"
echo "🔧 Backend API: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait
EOF

    chmod +x "$PROJECT_ROOT/start_nerf_rl_llm.sh"
    
    print_status "Startup script created"
}

# Main execution
main() {
    echo -e "${BLUE}Starting RL-LLM NeRF Integration Setup...${NC}"
    
    create_directories
    setup_python_env
    setup_node_deps
    create_nerf_systems
    create_cuda_backend
    create_react_components
    update_game_engine
    create_config_files
    create_startup_script
    
    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
    echo -e "${BLUE}📋 Next steps:${NC}"
    echo "1. Run: ./start_nerf_rl_llm.sh"
    echo "2. Open: http://localhost:5173"
    echo "3. Access NeRF Dashboard in the game UI"
    echo "4. Start NeRF training from the dashboard"
    echo ""
    echo -e "${YELLOW}📝 Note: Make sure CUDA drivers are installed for GPU acceleration${NC}"
}

# Run main function
main "$@"

