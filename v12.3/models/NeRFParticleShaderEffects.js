import * as THREE from 'three';

export class NeRFParticleShaderEffects {

    /**
     * Returns a ShaderMaterial for a glowing particle effect.
     * This shader will make particles glow and pulsate.
     * @param {THREE.Color} glowColor - The color of the glow.
     * @param {number} glowIntensity - The intensity of the glow.
     * @returns {THREE.ShaderMaterial}
     */
    static createGlowingParticleMaterial(glowColor = new THREE.Color(0x00ffff), glowIntensity = 1.0) {
        const vertexShader = `
            uniform float time;
            attribute float size;
            attribute vec3 customColor;
            varying vec3 vColor;

            void main() {
                vColor = customColor;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (100.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `;

        const fragmentShader = `
            uniform vec3 glowColor;
            uniform float glowIntensity;
            uniform float time;
            varying vec3 vColor;

            void main() {
                float dist = length(gl_PointCoord - vec2(0.5));
                float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
                
                // Pulsating effect
                float pulse = sin(time * 5.0) * 0.2 + 0.8;
                alpha *= pulse;

                gl_FragColor = vec4(glowColor * glowIntensity * alpha, alpha);
            }
        `;

        return new THREE.ShaderMaterial({
            uniforms: {
                glowColor: { value: glowColor },
                glowIntensity: { value: glowIntensity },
                time: { value: 0.0 } // Will be updated by NeRFRenderer's animate loop
            },
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            depthWrite: false
        });
    }

    /**
     * Returns a ShaderMaterial for a fireball effect.
     * @returns {THREE.ShaderMaterial}
     */
    static createFireballMaterial() {
        const vertexShader = `
            varying vec2 vUv;
            void main() {
                vUv = uv;
                gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
            }
        `;

        const fragmentShader = `
            uniform float time;
            uniform float opacity;
            varying vec2 vUv;

            void main() {
                float strength = 1.0 - length(vUv - vec2(0.5)) * 2.0; // Center is brightest
                strength = pow(strength, 2.0);

                // Simple noise for fire effect
                float noise = fract(sin(dot(vUv.xy, vec2(12.9898, 78.233))) * 43758.5453);
                noise = mix(noise, 1.0, 0.5 + sin(time * 5.0) * 0.5); // Animate noise

                vec3 color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 0.0, 0.0), strength);
                color = mix(color, vec3(1.0, 1.0, 0.0), noise * 0.5);

                gl_FragColor = vec4(color, strength * opacity);
            }
        `;

        return new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0.0 },
                opacity: { value: 1.0 }
            },
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            blending: THREE.AdditiveBlending,
            side: THREE.DoubleSide
        });
    }

    /**
     * Returns a ShaderMaterial for a smoke plume effect.
     * @returns {THREE.ShaderMaterial}
     */
    static createSmokeMaterial() {
        const vertexShader = `
            uniform float time;
            attribute float size;
            attribute vec3 customColor;
            varying vec3 vColor;

            void main() {
                vColor = customColor;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (100.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `;

        const fragmentShader = `
            uniform float time;
            uniform float opacity;
            varying vec3 vColor;

            void main() {
                float dist = length(gl_PointCoord - vec2(0.5));
                float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
                
                // Simple smoke dissipation
                alpha *= (1.0 - opacity);

                gl_FragColor = vec4(vColor, alpha);
            }
        `;

        return new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0.0 },
                opacity: { value: 1.0 }
            },
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            blending: THREE.NormalBlending,
            depthWrite: false
        });
    }

    /**
     * Returns a ShaderMaterial for debris particles.
     * @returns {THREE.ShaderMaterial}
     */
    static createDebrisMaterial() {
        const vertexShader = `
            uniform float time;
            attribute float size;
            attribute vec3 customColor;
            varying vec3 vColor;

            void main() {
                vColor = customColor;
                vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                gl_PointSize = size * (100.0 / -mvPosition.z);
                gl_Position = projectionMatrix * mvPosition;
            }
        `;

        const fragmentShader = `
            uniform float time;
            uniform float opacity;
            varying vec3 vColor;

            void main() {
                float dist = length(gl_PointCoord - vec2(0.5));
                float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
                
                gl_FragColor = vec4(vColor, alpha * opacity);
            }
        `;

        return new THREE.ShaderMaterial({
            uniforms: {
                time: { value: 0.0 },
                opacity: { value: 1.0 }
            },
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            blending: THREE.NormalBlending,
            depthWrite: false
        });
    }

    /**
     * Applies a given shader material to a point cloud object.
     * This assumes the point cloud is a THREE.Points object.
     * @param {THREE.Points} pointCloud - The Three.js Points object.
     * @param {THREE.ShaderMaterial} shaderMaterial - The custom shader material to apply.
     */
    static applyShaderToPointCloud(pointCloud, shaderMaterial) {
        if (pointCloud.isPoints) {
            pointCloud.material = shaderMaterial;
            // Ensure attributes are set up if the shader expects them (e.g., 'size', 'customColor')
            // This might require modifying the point cloud's geometry or creating a new one
            // For simplicity, this example assumes basic attributes are handled or not strictly required by the shader.
        } else {
            console.warn("Provided object is not a THREE.Points instance. Cannot apply particle shader.");
        }
    }
}


