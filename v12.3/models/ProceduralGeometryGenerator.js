import * as THREE from 'three';

export class ProceduralGeometryGenerator {
    static createWavePlane(width, height, segmentsX, segmentsY, frequency, amplitude) {
        const geometry = new THREE.PlaneGeometry(width, height, segmentsX, segmentsY);
        const positionAttribute = geometry.getAttribute('position');

        for (let i = 0; i < positionAttribute.count; i++) {
            const x = positionAttribute.getX(i);
            const y = positionAttribute.getY(i);
            // Apply a sine wave based on x coordinate
            positionAttribute.setZ(i, Math.sin(x * frequency) * amplitude);
        }
        geometry.attributes.position.needsUpdate = true;
        geometry.computeVertexNormals();
        return geometry;
    }

    static createSpiral(radius, turns, segments, height) {
        const points = [];
        for (let i = 0; i <= segments; i++) {
            const angle = (i / segments) * Math.PI * 2 * turns;
            const x = radius * Math.cos(angle);
            const y = radius * Math.sin(angle);
            const z = (i / segments) * height;
            points.push(new THREE.Vector3(x, y, z));
        }
        return new THREE.BufferGeometry().setFromPoints(points);
    }

    // Add more procedural geometry functions as needed
}


