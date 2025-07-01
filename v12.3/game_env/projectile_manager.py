
from typing import Dict, Any, Tuple, List
from game_env.physics_engine_interface import PhysicsEngineInterface

class ProjectileManager:
    """
    Manages the lifecycle and physics of projectiles (missiles, rockets) in the game.
    Interacts with the physics engine to simulate their movement and collisions.
    """
    def __init__(self, physics_engine: PhysicsEngineInterface):
        self.physics_engine = physics_engine
        self.projectiles: Dict[str, Dict[str, Any]] = {}
        self.next_projectile_id = 0

    def create_projectile(self, start_pos: Tuple[float, float, float], target_pos: Tuple[float, float, float], 
                          speed: float, projectile_type: str, damage: float) -> str:
        """
        Creates a new projectile and adds it to the physics simulation.
        """
        proj_id = f"projectile_{self.next_projectile_id}"
        self.next_projectile_id += 1

        mass = 1.0 # Default mass
        collider_shape = "sphere"
        dimensions = (0.5,) # Radius

        self.physics_engine.add_rigid_body(proj_id, start_pos, mass, collider_shape, dimensions)

        # Calculate initial velocity vector towards target
        direction = (np.array(target_pos) - np.array(start_pos))
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        velocity_vector = direction * speed
        
        # Apply initial impulse to set velocity (conceptual)
        # In a real engine, you might set linear velocity directly
        self.physics_engine.apply_force(proj_id, tuple(velocity_vector * mass)) # Force = mass * acceleration (velocity change)

        self.projectiles[proj_id] = {
            "id": proj_id,
            "type": projectile_type,
            "damage": damage,
            "target_pos": target_pos,
            "start_pos": start_pos,
            "current_pos": start_pos,
            "active": True
        }
        print(f"Created projectile {proj_id} of type {projectile_type} from {start_pos} to {target_pos}")
        return proj_id

    def update_projectiles(self, delta_time: float) -> List[Dict[str, Any]]:
        """
        Updates all active projectiles based on physics simulation.
        Returns a list of projectiles that have hit their target or expired.
        """
        hit_projectiles = []
        updated_physics_states = self.physics_engine.update_simulation(delta_time)

        for proj_id, proj_info in list(self.projectiles.items()): # Iterate over a copy
            if not proj_info["active"]: continue

            if proj_id in updated_physics_states:
                new_pos = updated_physics_states[proj_id]["position"]
                proj_info["current_pos"] = new_pos

                # Simple check if projectile has passed its target (conceptual)
                # More complex logic for homing missiles, etc., would go here
                if np.linalg.norm(np.array(new_pos) - np.array(proj_info["target_pos"])) < 1.0: # Close enough to target
                    hit_projectiles.append(proj_info)
                    proj_info["active"] = False
                    self.physics_engine.remove_rigid_body(proj_id)
            else:
                # If physics engine removed it (e.g., fell off world), deactivate
                proj_info["active"] = False
                self.physics_engine.remove_rigid_body(proj_id)

        # Clean up inactive projectiles
        self.projectiles = {pid: pinfo for pid, pinfo in self.projectiles.items() if pinfo["active"]}
        return hit_projectiles

    def get_active_projectiles_info(self) -> List[Dict[str, Any]]:
        """
        Returns information about all currently active projectiles.
        """
        return [p for p in self.projectiles.values() if p["active"]]

import numpy as np


