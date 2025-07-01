
import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';

let scene, camera, renderer;

function init() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Load scene configuration from Python backend
    fetch('../scene_config.json')
        .then(response => response.json())
        .then(config => {
            console.log('Loaded scene config:', config);
            buildScene(config);
            animate();
        })
        .catch(error => console.error('Error loading scene config:', error));

    window.addEventListener('resize', onWindowResize, false);
}

function buildScene(config) {
    // Set camera position and lookAt
    if (config.camera) {
        camera.position.set(config.camera.position[0], config.camera.position[1], config.camera.position[2]);
        camera.lookAt(new THREE.Vector3(config.camera.lookAt[0], config.camera.lookAt[1], config.camera.lookAt[2]));
    }

    // Add lights
    if (config.lights) {
        config.lights.forEach(lightConfig => {
            let light;
            if (lightConfig.type === 'AmbientLight') {
                light = new THREE.AmbientLight(lightConfig.color);
            } else if (lightConfig.type === 'DirectionalLight') {
                light = new THREE.DirectionalLight(lightConfig.color, lightConfig.intensity);
                if (lightConfig.position) {
                    light.position.set(lightConfig.position[0], lightConfig.position[1], lightConfig.position[2]);
                }
            }
            if (light) scene.add(light);
        });
    }

    // Add objects
    if (config.objects) {
        config.objects.forEach(objConfig => {
            let geometry;
            let material;

            // Create geometry
            if (objConfig.type === 'PlaneGeometry') {
                geometry = new THREE.PlaneGeometry(objConfig.width, objConfig.height);
            } else if (objConfig.type === 'BoxGeometry') {
                geometry = new THREE.BoxGeometry(objConfig.width, objConfig.height, objConfig.depth);
            } else if (objConfig.type === 'CylinderGeometry') {
                geometry = new THREE.CylinderGeometry(objConfig.radiusTop, objConfig.radiusBottom, objConfig.height, objConfig.radialSegments);
            } else if (objConfig.type === 'SphereGeometry') {
                geometry = new THREE.SphereGeometry(objConfig.radius, objConfig.widthSegments, objConfig.heightSegments);
            }

            // Create material
            if (objConfig.material) {
                if (objConfig.material.type === 'MeshPhongMaterial') {
                    material = new THREE.MeshPhongMaterial({ color: objConfig.material.color });
                } else if (objConfig.material.type === 'MeshBasicMaterial') {
                    material = new THREE.MeshBasicMaterial({ color: objConfig.material.color });
                }
            }

            if (geometry && material) {
                const mesh = new THREE.Mesh(geometry, material);
                if (objConfig.position) {
                    mesh.position.set(objConfig.position[0], objConfig.position[1], objConfig.position[2]);
                }
                if (objConfig.rotation) {
                    mesh.rotation.set(objConfig.rotation[0], objConfig.rotation[1], objConfig.rotation[2]);
                }
                scene.add(mesh);
            }
        });
    }
}

function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

init();


