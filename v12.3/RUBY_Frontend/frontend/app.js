// Enhanced AI Collab v6.2 with 3D Model Generation
const logDiv = document.getElementById('log');
const statusDiv = document.getElementById('status');
const statusIndicator = document.getElementById('statusIndicator');
const modelsList = document.getElementById('modelsList');
const modelCount = document.getElementById('modelCount');
const viewerModal = document.getElementById('viewerModal');
const threejsContainer = document.getElementById('threejs-container');

// WebSocket connection
let ws = null;
let isConnected = false;

// 3D Viewer variables
let scene, camera, renderer, controls;
let currentModel = null;

// Models data
let modelsData = [];

// Initialize the application
function init() {
  connectWebSocket();
  initThreeJS();
  refreshModels();
}

function log(msg, type = 'info') {
  const timestamp = new Date().toLocaleTimeString();
  const typeIcon = {
    'info': '<i class="fas fa-info-circle" style="color: #00d4ff;"></i>',
    'success': '<i class="fas fa-check-circle" style="color: #10b981;"></i>',
    'error': '<i class="fas fa-exclamation-circle" style="color: #ef4444;"></i>',
    'warning': '<i class="fas fa-exclamation-triangle" style="color: #f59e0b;"></i>'
  };
  
  logDiv.innerHTML += `<div style="margin-bottom: 8px;">
    <span style="color: #6b7280;">[${timestamp}]</span> 
    ${typeIcon[type]} ${msg}
  </div>`;
  logDiv.scrollTop = logDiv.scrollHeight;
}

function updateStatus(message, connected = false) {
  statusDiv.textContent = message;
  statusIndicator.className = `status-indicator ${connected ? 'connected' : ''}`;
  isConnected = connected;
}

function connectWebSocket() {
  try {
    ws = new WebSocket("ws://localhost:8765");
    
    ws.onopen = () => {
      updateStatus("WebSocket connected! Agent: ChatGBT", true);
      log("Connected to Manus AI v6.2 Enhanced backend", 'success');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    };

    ws.onclose = () => {
      updateStatus("WebSocket disconnected", false);
      log("Connection lost. Attempting to reconnect...", 'warning');
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
      updateStatus("WebSocket error", false);
      log("WebSocket error occurred", 'error');
    };
  } catch (error) {
    updateStatus("Failed to connect", false);
    log("Failed to establish WebSocket connection", 'error');
  }
}

function handleWebSocketMessage(data) {
  log(`<b>From ${data.from}:</b> ${data.type}`, 'info');
  
  switch (data.type) {
    case '3d:response':
      handle3DResponse(data.payload);
      break;
    case '3d:list_response':
      handleModelsListResponse(data.payload);
      break;
    case '3d:delete_response':
      handleDeleteResponse(data.payload);
      break;
    case 'visualization:response':
      log(`Visualization: ${JSON.stringify(data.payload)}`, 'success');
      break;
    case 'code:response':
      log(`Generated code: <code style="background: rgba(0,0,0,0.3); padding: 2px 4px; border-radius: 3px;">${data.payload.code}</code>`, 'success');
      break;
    default:
      log(`Received: ${JSON.stringify(data.payload)}`, 'info');
  }
}

function handle3DResponse(payload) {
  if (payload.status === 'success') {
    log(`3D model generated successfully! Model ID: ${payload.model_id}`, 'success');
    
    // Add the new model to the list
    const newModel = {
      model_id: payload.model_id,
      filename: payload.model_id + '.' + payload.format,
      format: payload.format,
      download_url: payload.download_url,
      preview_url: payload.preview_url,
      thumbnail_url: payload.thumbnail_url,
      prompt: payload.prompt,
      created_at: payload.created_at
    };
    
    modelsData.unshift(newModel);
    updateModelsList();
    
    // Show preview in chat
    showModelPreview(newModel);
  } else {
    log(`3D generation failed: ${payload.error}`, 'error');
  }
  
  // Re-enable the generate button
  const generateBtn = document.getElementById('generateBtn');
  generateBtn.disabled = false;
  generateBtn.innerHTML = '<i class="fas fa-cube"></i> Generate 3D Model';
}

function showModelPreview(model) {
  const previewHtml = `
    <div style="background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); border-radius: 8px; padding: 15px; margin: 10px 0;">
      <div style="display: flex; align-items: center; gap: 15px;">
        <img src="http://localhost:5000${model.preview_url}" 
             style="width: 80px; height: 80px; object-fit: cover; border-radius: 6px; cursor: pointer;"
             onclick="openViewer('${model.model_id}')"
             onerror="this.style.display='none'">
        <div style="flex: 1;">
          <div style="color: #00d4ff; font-weight: bold; margin-bottom: 5px;">New 3D Model Generated</div>
          <div style="color: #a0a9b8; font-size: 0.9rem; margin-bottom: 8px;">${model.prompt}</div>
          <div style="display: flex; gap: 10px;">
            <button onclick="openViewer('${model.model_id}')" 
                    style="background: rgba(0,212,255,0.2); border: 1px solid #00d4ff; color: #00d4ff; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.8rem;">
              <i class="fas fa-eye"></i> View
            </button>
            <button onclick="downloadModel('${model.model_id}', '${model.filename}')" 
                    style="background: rgba(16,185,129,0.2); border: 1px solid #10b981; color: #10b981; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.8rem;">
              <i class="fas fa-download"></i> Download
            </button>
          </div>
        </div>
      </div>
    </div>
  `;
  
  logDiv.innerHTML += previewHtml;
  logDiv.scrollTop = logDiv.scrollHeight;
}

function handleModelsListResponse(payload) {
  modelsData = payload.models || [];
  updateModelsList();
  log(`Loaded ${modelsData.length} models`, 'success');
}

function handleDeleteResponse(payload) {
  if (payload.status === 'success') {
    log(`Model deleted successfully (${payload.deleted_files} files removed)`, 'success');
    refreshModels();
  } else {
    log(`Delete failed: ${payload.error}`, 'error');
  }
}

function updateModelsList() {
  modelCount.textContent = modelsData.length;
  
  if (modelsData.length === 0) {
    modelsList.innerHTML = `
      <div style="text-align: center; padding: 40px; color: #6b7280;">
        <i class="fas fa-cube" style="font-size: 3rem; margin-bottom: 15px; opacity: 0.3;"></i>
        <p>No 3D models generated yet</p>
        <p style="font-size: 0.9rem; margin-top: 5px;">Create your first model using the generator!</p>
      </div>
    `;
    return;
  }
  
  const modelsHtml = modelsData.map(model => `
    <div class="model-item">
      <div class="model-preview" onclick="openViewer('${model.model_id}')">
        <img src="http://localhost:5000${model.thumbnail_url || model.preview_url}" 
             alt="Model preview" 
             onerror="this.parentElement.innerHTML='<i class=\\"fas fa-cube\\" style=\\"font-size: 2rem; color: #6b7280;\\"></i>'">
      </div>
      <div class="model-info">
        <div style="font-weight: bold; color: #e1e5e9; margin-bottom: 5px;">
          ${model.prompt ? (model.prompt.length > 40 ? model.prompt.substring(0, 40) + '...' : model.prompt) : 'Untitled Model'}
        </div>
        <div>Format: ${model.format.toUpperCase()}</div>
        <div>ID: ${model.model_id.substring(0, 8)}...</div>
      </div>
      <div class="model-actions">
        <button onclick="openViewer('${model.model_id}')" title="View in 3D">
          <i class="fas fa-eye"></i>
        </button>
        <button onclick="downloadModel('${model.model_id}', '${model.filename}')" 
                class="secondary-btn" title="Download">
          <i class="fas fa-download"></i>
        </button>
        <button onclick="deleteModel('${model.model_id}')" 
                class="danger-btn" title="Delete">
          <i class="fas fa-trash"></i>
        </button>
      </div>
    </div>
  `).join('');
  
  modelsList.innerHTML = modelsHtml;
}

// 3D Model Generation
function generate3DModel() {
  if (!isConnected) {
    log("Not connected to server", 'error');
    return;
  }
  
  const prompt = document.getElementById('promptInput').value.trim();
  if (!prompt) {
    log("Please enter a description for your 3D model", 'warning');
    return;
  }
  
  const format = document.getElementById('formatSelect').value;
  const quality = document.getElementById('qualitySelect').value;
  
  const generateBtn = document.getElementById('generateBtn');
  generateBtn.disabled = true;
  generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
  
  const message = {
    type: "3d:generate",
    from: "ChatGBT",
    to: "Manus",
    payload: {
      prompt: prompt,
      format: format,
      quality: quality
    }
  };
  
  ws.send(JSON.stringify(message));
  log(`Generating 3D model: "${prompt}" (${format.toUpperCase()}, ${quality} quality)`, 'info');
}

// Model Management
function refreshModels() {
  if (!isConnected) {
    setTimeout(refreshModels, 1000);
    return;
  }
  
  const message = {
    type: "3d:list",
    from: "ChatGBT",
    to: "Manus",
    payload: {}
  };
  
  ws.send(JSON.stringify(message));
}

function downloadModel(modelId, filename) {
  const url = `http://localhost:5000/models/${modelId}/${filename}`;
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  log(`Downloading: ${filename}`, 'success');
}

function deleteModel(modelId) {
  if (!confirm('Are you sure you want to delete this model?')) {
    return;
  }
  
  const message = {
    type: "3d:delete",
    from: "ChatGBT",
    to: "Manus",
    payload: {
      model_id: modelId
    }
  };
  
  ws.send(JSON.stringify(message));
  log(`Deleting model: ${modelId}`, 'info');
}

function clearAllModels() {
  if (!confirm('Are you sure you want to delete ALL models? This action cannot be undone.')) {
    return;
  }
  
  modelsData.forEach(model => {
    const message = {
      type: "3d:delete",
      from: "ChatGBT",
      to: "Manus",
      payload: {
        model_id: model.model_id
      }
    };
    ws.send(JSON.stringify(message));
  });
  
  log('Clearing all models...', 'warning');
}

// Three.js 3D Viewer
function initThreeJS() {
  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1419);
  
  // Camera
  camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000);
  camera.position.set(0, 0, 5);
  
  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(400, 400);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  
  // Lighting
  const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
  scene.add(ambientLight);
  
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
  directionalLight.position.set(10, 10, 5);
  directionalLight.castShadow = true;
  scene.add(directionalLight);
  
  const pointLight = new THREE.PointLight(0x00d4ff, 0.5);
  pointLight.position.set(-10, -10, -5);
  scene.add(pointLight);
}

function openViewer(modelId) {
  const model = modelsData.find(m => m.model_id === modelId);
  if (!model) {
    log('Model not found', 'error');
    return;
  }
  
  document.getElementById('viewerTitle').innerHTML = 
    `<i class="fas fa-eye"></i> 3D Model Viewer - ${model.prompt || 'Untitled'}`;
  
  viewerModal.classList.add('active');
  
  // Resize renderer to fit container
  const container = threejsContainer;
  const rect = container.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  
  // Clear container and add renderer
  container.innerHTML = '';
  container.appendChild(renderer.domElement);
  
  // Initialize controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  
  // Load and display the model
  loadModel(model);
  
  // Start render loop
  animate();
}

function loadModel(model) {
  // Clear previous model
  if (currentModel) {
    scene.remove(currentModel);
  }
  
  // For now, we'll create a placeholder 3D object since we're using placeholder files
  // In a real implementation, you would load the actual GLB/OBJ file
  createPlaceholder3DObject(model);
}

function createPlaceholder3DObject(model) {
  // Create a group to hold the model
  currentModel = new THREE.Group();
  
  // Create a stylized cube representing the 3D model
  const geometry = new THREE.BoxGeometry(2, 2, 2);
  const material = new THREE.MeshPhongMaterial({ 
    color: 0x00d4ff,
    transparent: true,
    opacity: 0.8
  });
  const cube = new THREE.Mesh(geometry, material);
  cube.castShadow = true;
  cube.receiveShadow = true;
  
  // Add wireframe
  const wireframe = new THREE.WireframeGeometry(geometry);
  const line = new THREE.LineSegments(wireframe, new THREE.LineBasicMaterial({ color: 0xffffff, opacity: 0.3, transparent: true }));
  
  currentModel.add(cube);
  currentModel.add(line);
  
  // Add some floating particles for effect
  const particlesGeometry = new THREE.BufferGeometry();
  const particlesCount = 100;
  const posArray = new Float32Array(particlesCount * 3);
  
  for (let i = 0; i < particlesCount * 3; i++) {
    posArray[i] = (Math.random() - 0.5) * 10;
  }
  
  particlesGeometry.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
  const particlesMaterial = new THREE.PointsMaterial({
    size: 0.02,
    color: 0x00d4ff,
    transparent: true,
    opacity: 0.6
  });
  
  const particlesMesh = new THREE.Points(particlesGeometry, particlesMaterial);
  currentModel.add(particlesMesh);
  
  scene.add(currentModel);
  
  // Add rotation animation
  currentModel.userData.animate = true;
}

function animate() {
  if (!viewerModal.classList.contains('active')) {
    return;
  }
  
  requestAnimationFrame(animate);
  
  // Rotate the model
  if (currentModel && currentModel.userData.animate) {
    currentModel.rotation.y += 0.005;
    currentModel.children[2].rotation.x += 0.002; // Rotate particles differently
  }
  
  controls.update();
  renderer.render(scene, camera);
}

function closeViewer() {
  viewerModal.classList.remove('active');
  if (currentModel) {
    scene.remove(currentModel);
    currentModel = null;
  }
}

// Legacy functions
function requestVisualization() {
  if (!isConnected) {
    log("Not connected to server", 'error');
    return;
  }
  
  ws.send(JSON.stringify({
    type: "visualization:request",
    from: "ChatGBT",
    to: "Manus",
    payload: { what: "reward_curve", params: { episodes: [10,11,12] } }
  }));
  log("Requested <i>reward_curve</i> visualization from Manus.", 'info');
}

function generateCode() {
  if (!isConnected) {
    log("Not connected to server", 'error');
    return;
  }
  
  ws.send(JSON.stringify({
    type: "code:generate",
    from: "ChatGBT",
    to: "Manus",
    payload: { prompt: "Print hello world" }
  }));
  log("Asked Manus for code generation.", 'info');
}

// Handle window resize for 3D viewer
window.addEventListener('resize', () => {
  if (viewerModal.classList.contains('active') && renderer) {
    const container = threejsContainer;
    const rect = container.getBoundingClientRect();
    renderer.setSize(rect.width, rect.height);
    camera.aspect = rect.width / rect.height;
    camera.updateProjectionMatrix();
  }
});

// Close viewer with Escape key
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && viewerModal.classList.contains('active')) {
    closeViewer();
  }
});

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', init);

