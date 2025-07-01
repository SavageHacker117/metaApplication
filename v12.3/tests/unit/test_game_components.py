

import unittest
import numpy as np
from game_env.tower_defense_env import TowerDefenseEnv, Enemy
from game_env.game_generator import GameGenerator
from llm_integration.llm_api_interface import LLMAPIInterface

class TestTowerDefenseEnv(unittest.TestCase):

    def setUp(self):
        self.env = TowerDefenseEnv(grid_size=(5,5), initial_cash=100, initial_lives=10)

    def test_initial_state(self):
        state = self.env.get_state()
        self.assertEqual(state["cash"], 100)
        self.assertEqual(state["lives"], 10)
        self.assertFalse(state["game_over"])
        self.assertEqual(state["wave_number"], 0)
        self.assertEqual(len(state["towers"]), 0)
        self.assertEqual(len(state["enemies"]), 0)
        self.assertEqual(np.array(state["grid"]).shape, (5,5))

    def test_place_tower_success(self):
        initial_cash = self.env.cash
        success = self.env.place_tower(1, 1, "basic")
        self.assertTrue(success)
        self.assertEqual(self.env.grid[1, 1], 2) # 2 represents a tower
        self.assertIn((1, 1), self.env.towers)
        self.assertLess(self.env.cash, initial_cash)

    def test_place_tower_invalid_coords(self):
        success = self.env.place_tower(10, 10, "basic") # Out of bounds
        self.assertFalse(success)

    def test_place_tower_occupied_spot(self):
        self.env.place_tower(1, 1, "basic")
        success = self.env.place_tower(1, 1, "basic") # Already occupied
        self.assertFalse(success)

    def test_place_tower_not_enough_cash(self):
        self.env.cash = 10 # Set cash too low
        success = self.env.place_tower(2, 2, "basic")
        self.assertFalse(success)

    def test_spawn_wave(self):
        self.env.spawn_wave(num_enemies=3)
        self.assertEqual(self.env.wave_number, 1)
        self.assertEqual(len(self.env.enemies), 3)

    def test_enemy_movement(self):
        path = [(0,0), (0,1), (0,2)]
        enemy = Enemy(path=path, speed=1)
        self.assertEqual(enemy.position, (0,0))
        enemy.move()
        self.assertEqual(enemy.position, (0,1))
        enemy.move()
        self.assertEqual(enemy.position, (0,2))
        enemy.move()
        self.assertEqual(enemy.position, (0,2)) # Should stay at end of path

    def test_game_over_lives(self):
        self.env.lives = 1
        self.env.enemies.append(Enemy(path=[(0,0),(0,1)], health=1, speed=1)) # Add an enemy to trigger update
        self.env.update_game_state() # Simulate enemy reaching end
        self.assertTrue(self.env.game_over)

class MockLLMAPIInterface(LLMAPIInterface):
    def __init__(self, *args, **kwargs):
        super().__init__(api_key="dummy", model_name="dummy")

    def generate_response(self, prompt, max_tokens, temperature):
        # Simulate a valid JSON response for game config
        return '''
        {
            "grid_size": [8, 8],
            "initial_cash": 200,
            "initial_lives": 15,
            "path": [[0,0], [1,0], [2,0], [2,1], [2,2], [3,2], [4,2], [4,3], [4,4], [5,4], [6,4], [6,5], [6,6], [7,6], [7,7]],
            "spawn_point": [0,0],
            "exit_point": [7,7],
            "tower_types": {
                "laser": {"cost": 100, "range": 4, "damage": 20}
            },
            "enemy_waves": [
                {"num_enemies": 7, "enemy_health": 60}
            ]
        }
        '''

class TestGameGenerator(unittest.TestCase):

    def test_generate_game_config(self):
        mock_llm = MockLLMAPIInterface()
        generator = GameGenerator(mock_llm)
        config = generator.generate_game_config("easy")
        
        self.assertIsNotNone(config)
        self.assertIn("grid_size", config)
        self.assertEqual(config["grid_size"], [8, 8])
        self.assertIn("path", config)
        self.assertGreater(len(config["path"]), 0)

if __name__ == "__main__":
    unittest.main()


