import * as THREE from 'three';

export class ShaderManager {
    constructor() {
        this.shaders = new Map(); // Stores { shaderName: { vertexShader: string, fragmentShader: string } }
    }

    async loadShader(name, vertexPath, fragmentPath) {
        const [vertexShader, fragmentShader] = await Promise.all([
            fetch(vertexPath).then(res => res.text()),
            fetch(fragmentPath).then(res => res.text())
        ]);
        this.shaders.set(name, { vertexShader, fragmentShader });
    }

    createShaderMaterial(name, uniforms = {}) {
        const shader = this.shaders.get(name);
        if (!shader) {
            console.error(`Shader ${name} not loaded.`);
            return null;
        }
        return new THREE.ShaderMaterial({
            uniforms: {
                ...uniforms,
                time: { value: 0.0 } // Example dynamic uniform
            },
            vertexShader: shader.vertexShader,
            fragmentShader: shader.fragmentShader,
            // ... other material properties like blending, depthTest
        });
    }
}


