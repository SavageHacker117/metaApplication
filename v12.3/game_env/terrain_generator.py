
import numpy as np
from typing import Dict, Any, Tuple, List

class TerrainGenerator:
    """
    Generates 3D terrain data, including heightmaps and material distribution,
    to simulate diverse ecosystems (e.g., water, land, mountains).
    """
    def __init__(self, grid_size: Tuple[int, int] = (100, 100), scale: float = 0.1, octaves: int = 6, persistence: float = 0.5, lacunarity: float = 2.0):
        self.grid_size = grid_size
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity

    def generate_heightmap(self) -> np.ndarray:
        """
        Generates a heightmap using Perlin noise or similar algorithms.
        """
        heightmap = np.zeros(self.grid_size)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Simple Perlin-like noise (conceptual)
                # In a real implementation, use a proper noise library (e.g., noise, opensimplex)
                val = 0
                freq = self.scale
                amp = 1
                for _ in range(self.octaves):
                    # Placeholder for noise function
                    # val += noise.pnoise2(i * freq, j * freq, octaves=1, persistence=1, lacunarity=1) * amp
                    val += np.sin(i * freq) * np.cos(j * freq) * amp # Very simple placeholder
                    freq *= self.lacunarity
                    amp *= self.persistence
                heightmap[i][j] = val
        
        # Normalize heightmap to a reasonable range (e.g., 0 to 10)
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min()) * 10
        return heightmap

    def generate_biome_map(self, heightmap: np.ndarray) -> List[List[str]]:
        """
        Generates a biome map (e.g., 'water', 'land', 'mountain') based on height.
        """
        biome_map = []
        for row in heightmap:
            biome_row = []
            for height in row:
                if height < 2:
                    biome_row.append("water")
                elif height < 6:
                    biome_row.append("land")
                else:
                    biome_row.append("mountain")
            biome_map.append(biome_row)
        return biome_map

    def generate_full_terrain(self) -> Dict[str, Any]:
        """
        Generates a complete terrain definition including heightmap and biome map.
        """
        heightmap = self.generate_heightmap()
        biome_map = self.generate_biome_map(heightmap)
        return {"heightmap": heightmap.tolist(), "biome_map": biome_map}

# Example Usage:
# if __name__ == "__main__":
#     terrain_gen = TerrainGenerator(grid_size=(50, 50))
#     terrain_data = terrain_gen.generate_full_terrain()
#     print("Generated Heightmap (first 5x5):")
#     print(np.array(terrain_data["heightmap"][:5, :5]))
#     print("Generated Biome Map (first 5x5):")
#     for row in terrain_data["biome_map"][:5]:
#         print(row[:5])


