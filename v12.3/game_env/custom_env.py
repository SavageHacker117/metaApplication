"""
Custom Tower Defense Environment Implementation

This module implements a specific tower defense environment with procedural
generation and NeRF integration capabilities.
"""

import gym
import numpy as np
from typing import Dict, Any, Tuple, List
import random
import math
from .base_env import BaseTowerDefenseEnv
import logging

logger = logging.getLogger(__name__)


class ProceduralTowerDefenseEnv(BaseTowerDefenseEnv):
    """
    Custom tower defense environment with procedural generation.
    
    Features:
    - Procedurally generated maps
    - Dynamic enemy spawning
    - Multiple tower types
    - Resource management
    - Wave-based progression
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the custom environment."""
        # Set default config values
        default_config = {
            'map_size': (12, 12),
            'max_towers': 20,
            'tower_types': 3,
            'enemy_types': 4,
            'wave_size_multiplier': 1.2,
            'initial_wave_size': 5,
            'max_waves': 20,
            'tower_cost_base': 10,
            'enemy_reward_base': 5
        }
        default_config.update(config)
        
        super().__init__(default_config)
        
        # Environment-specific attributes
        self.map_size = self.config['map_size']
        self.max_towers = self.config['max_towers']
        self.tower_types = self.config['tower_types']
        self.enemy_types = self.config['enemy_types']
        
        # Wave management
        self.current_wave = 0
        self.max_waves = self.config['max_waves']
        self.wave_size = self.config['initial_wave_size']
        self.enemies_remaining_in_wave = 0
        
        # Tower definitions
        self.tower_definitions = self._create_tower_definitions()
        
        # Path for enemies (simple straight path for now)
        self.enemy_path = self._generate_enemy_path()
        
        logger.info(f"Initialized ProceduralTowerDefenseEnv with map size {self.map_size}")
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space: map state + game info
        map_obs_size = self.map_size[0] * self.map_size[1]
        game_info_size = 10  # wave, resources, lives, etc.
        obs_size = map_obs_size + game_info_size
        
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(obs_size,), dtype=np.float32
        )
        
        # Action space: [action_type, x, y, tower_type]
        # action_type: 0=no_op, 1=build_tower, 2=upgrade_tower, 3=sell_tower
        self.action_space = gym.spaces.MultiDiscrete([
            4,  # action_type
            self.map_size[0],  # x coordinate
            self.map_size[1],  # y coordinate
            self.tower_types   # tower_type
        ])
    
    def _create_tower_definitions(self) -> List[Dict[str, Any]]:
        """Create tower type definitions."""
        return [
            {
                'name': 'Basic Tower',
                'cost': 10,
                'damage': 5,
                'range': 2,
                'fire_rate': 1.0,
                'upgrade_cost': 15
            },
            {
                'name': 'Sniper Tower',
                'cost': 25,
                'damage': 15,
                'range': 4,
                'fire_rate': 0.5,
                'upgrade_cost': 30
            },
            {
                'name': 'Splash Tower',
                'cost': 20,
                'damage': 8,
                'range': 2,
                'fire_rate': 0.8,
                'upgrade_cost': 25,
                'splash_radius': 1
            }
        ]
    
    def _generate_enemy_path(self) -> List[Tuple[int, int]]:
        """Generate a path for enemies to follow."""
        path = []
        # Simple path from left to right
        y = self.map_size[1] // 2
        for x in range(self.map_size[0]):
            path.append((x, y))
        return path
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Map state (flattened)
        map_state = np.zeros(self.map_size, dtype=np.float32)
        
        # Mark towers
        for tower in self.towers:
            x, y = tower['position']
            map_state[x, y] = tower['type'] + 1  # +1 to avoid 0
        
        # Mark enemies
        for enemy in self.enemies:
            x, y = enemy['position']
            if 0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]:
                map_state[x, y] += 10  # Different encoding for enemies
        
        # Mark path
        for x, y in self.enemy_path:
            if map_state[x, y] == 0:  # Only mark empty path cells
                map_state[x, y] = 0.5
        
        map_obs = map_state.flatten()
        
        # Game info
        game_info = np.array([
            self.current_wave / self.max_waves,
            self.resources / 100.0,  # Normalize
            self.lives / 20.0,       # Normalize
            len(self.towers) / self.max_towers,
            len(self.enemies) / 20.0,  # Normalize
            self.enemies_remaining_in_wave / 20.0,
            self.episode_step / self.max_episode_steps,
            self.episode_reward / 100.0,  # Normalize
            float(self.current_wave > 0),  # Wave active flag
            float(len(self.enemies) > 0)   # Enemies present flag
        ], dtype=np.float32)
        
        return np.concatenate([map_obs, game_info])
    
    def _calculate_reward(self, action: Any, prev_state: Dict) -> float:
        """Calculate reward for the current step."""
        reward = 0.0
        
        # Reward for defeating enemies
        enemies_defeated = prev_state.get('enemies_count', 0) - len(self.enemies)
        reward += enemies_defeated * self.config['enemy_reward_base']
        
        # Penalty for losing lives
        lives_lost = prev_state.get('lives', self.lives) - self.lives
        reward -= lives_lost * 10
        
        # Small reward for surviving each step
        reward += 0.1
        
        # Bonus for completing waves
        if self.enemies_remaining_in_wave == 0 and len(self.enemies) == 0:
            reward += 20 * (self.current_wave + 1)
        
        # Penalty for invalid actions
        if action[0] == 1:  # Build tower action
            tower_cost = self.tower_definitions[action[3]]['cost']
            if self.resources < tower_cost:
                reward -= 5  # Penalty for trying to build without resources
        
        return reward
    
    def _is_terminal(self) -> bool:
        """Check if episode should terminate."""
        # Terminal conditions
        if self.lives <= 0:
            return True
        if self.episode_step >= self.max_episode_steps:
            return True
        if self.current_wave >= self.max_waves and len(self.enemies) == 0:
            return True
        
        return False
    
    def _execute_action(self, action: Any):
        """Execute the given action."""
        action_type, x, y, tower_type = action
        
        if action_type == 1:  # Build tower
            self._build_tower(x, y, tower_type)
        elif action_type == 2:  # Upgrade tower
            self._upgrade_tower(x, y)
        elif action_type == 3:  # Sell tower
            self._sell_tower(x, y)
        # action_type == 0 is no-op
    
    def _build_tower(self, x: int, y: int, tower_type: int):
        """Build a tower at the specified position."""
        if not (0 <= x < self.map_size[0] and 0 <= y < self.map_size[1]):
            return  # Invalid position
        
        if tower_type >= len(self.tower_definitions):
            return  # Invalid tower type
        
        # Check if position is available
        if (x, y) in self.enemy_path:
            return  # Cannot build on path
        
        for tower in self.towers:
            if tower['position'] == (x, y):
                return  # Position occupied
        
        tower_def = self.tower_definitions[tower_type]
        if self.resources >= tower_def['cost']:
            self.resources -= tower_def['cost']
            self.towers.append({
                'position': (x, y),
                'type': tower_type,
                'level': 1,
                'last_fire_time': 0,
                **tower_def
            })
            self.episode_metrics['towers_built'] += 1
            self.episode_metrics['resources_spent'] += tower_def['cost']
    
    def _upgrade_tower(self, x: int, y: int):
        """Upgrade tower at the specified position."""
        for tower in self.towers:
            if tower['position'] == (x, y):
                if self.resources >= tower['upgrade_cost']:
                    self.resources -= tower['upgrade_cost']
                    tower['level'] += 1
                    tower['damage'] = int(tower['damage'] * 1.5)
                    tower['upgrade_cost'] = int(tower['upgrade_cost'] * 1.3)
                    self.episode_metrics['resources_spent'] += tower['upgrade_cost']
                break
    
    def _sell_tower(self, x: int, y: int):
        """Sell tower at the specified position."""
        for i, tower in enumerate(self.towers):
            if tower['position'] == (x, y):
                sell_value = tower['cost'] // 2
                self.resources += sell_value
                del self.towers[i]
                break
    
    def _update_game_state(self):
        """Update the game state after action execution."""
        super()._update_game_state()
        
        # Update enemies
        self._update_enemies()
        
        # Tower attacks
        self._process_tower_attacks()
        
        # Spawn new enemies if needed
        self._spawn_enemies()
        
        # Update wave progression
        self._update_wave_progression()
    
    def _update_enemies(self):
        """Update enemy positions and states."""
        for enemy in self.enemies[:]:  # Copy list to allow modification
            # Move enemy along path
            current_pos = enemy['position']
            path_index = enemy.get('path_index', 0)
            
            if path_index < len(self.enemy_path) - 1:
                enemy['path_index'] = path_index + 1
                enemy['position'] = self.enemy_path[path_index + 1]
            else:
                # Enemy reached the end
                self.lives -= 1
                self.enemies.remove(enemy)
                self.episode_metrics['damage_taken'] += 1
    
    def _process_tower_attacks(self):
        """Process tower attacks on enemies."""
        current_time = self.episode_step
        
        for tower in self.towers:
            if current_time - tower['last_fire_time'] >= (1.0 / tower['fire_rate']):
                # Find enemies in range
                tower_x, tower_y = tower['position']
                targets = []
                
                for enemy in self.enemies:
                    enemy_x, enemy_y = enemy['position']
                    distance = math.sqrt((tower_x - enemy_x)**2 + (tower_y - enemy_y)**2)
                    if distance <= tower['range']:
                        targets.append(enemy)
                
                if targets:
                    # Attack first target
                    target = targets[0]
                    target['health'] -= tower['damage']
                    tower['last_fire_time'] = current_time
                    self.episode_metrics['damage_dealt'] += tower['damage']
                    
                    # Remove dead enemies
                    if target['health'] <= 0:
                        self.enemies.remove(target)
                        self.episode_metrics['enemies_defeated'] += 1
                        self.enemies_remaining_in_wave -= 1
    
    def _spawn_enemies(self):
        """Spawn new enemies for the current wave."""
        if len(self.enemies) == 0 and self.enemies_remaining_in_wave == 0:
            # Start new wave
            if self.current_wave < self.max_waves:
                self.current_wave += 1
                self.enemies_remaining_in_wave = int(self.wave_size)
                self.wave_size *= self.config['wave_size_multiplier']
        
        # Spawn enemies gradually
        if (self.enemies_remaining_in_wave > 0 and 
            len(self.enemies) < 5 and  # Limit concurrent enemies
            self.episode_step % 10 == 0):  # Spawn every 10 steps
            
            enemy_type = random.randint(0, self.enemy_types - 1)
            health = 10 + enemy_type * 5 + self.current_wave * 2
            
            self.enemies.append({
                'position': self.enemy_path[0],
                'path_index': 0,
                'type': enemy_type,
                'health': health,
                'max_health': health,
                'speed': 1.0
            })
    
    def _update_wave_progression(self):
        """Update wave progression logic."""
        # Wave progression is handled in _spawn_enemies
        pass
    
    def _initialize_game_state(self) -> Dict[str, Any]:
        """Initialize the game state."""
        return {
            'wave_number': 0,
            'time_step': 0,
            'map_state': np.zeros(self.map_size),
            'tower_positions': [],
            'enemy_positions': [],
            'enemies_count': 0,
            'lives': self.lives
        }

