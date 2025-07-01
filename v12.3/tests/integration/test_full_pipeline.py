
import unittest
import os
import json
from unittest.mock import patch, MagicMock

# Assuming these modules exist and are importable from the project root
from training.rl_llm_trainer import RL_LLMTrainer
from core.config.config_schema import MainConfig

class TestFullPipelineIntegration(unittest.TestCase):

    def setUp(self):
        # Setup a dummy config for testing
        self.test_config_data = {
            "llm": {"model_name": "test-llm", "api_key": "dummy_key", "temperature": 0.5, "max_tokens": 50},
            "rl": {"algorithm": "PPO", "learning_rate": 0.0001, "gamma": 0.99, "num_episodes": 2},
            "system": {"environment_name": "test_env", "log_level": "DEBUG", "output_dir": "./test_outputs", "debug_mode": True}
        }
        self.config = MainConfig(**self.test_config_data)

        # Mock external dependencies to isolate the test
        self.mock_llm_interface = MagicMock()
        self.mock_llm_interface.generate_response.return_value = "This is a simulated LLM response."

        self.mock_reward_model = MagicMock()
        self.mock_reward_model.get_reward.return_value = 0.8

        self.mock_text_environment = MagicMock()
        self.mock_text_environment.reset.return_value = "Initial state."
        self.mock_text_environment.step.side_effect = [
            ("Next state 1", 0.5, False, {}), # First step
            ("<END_OF_DIALOGUE>", 1.0, True, {}) # Second step, ends episode
        ]
        self.mock_text_environment.get_history.return_value = [{"role": "user", "content": "Initial state."}]

    @patch("training.rl_llm_trainer.LLMAPIInterface")
    @patch("training.rl_llm_trainer.RewardModel")
    @patch("training.rl_llm_trainer.TextEnvironment")
    def test_trainer_pipeline(self, MockTextEnvironment, MockRewardModel, MockLLMAPIInterface):
        MockLLMAPIInterface.return_value = self.mock_llm_interface
        MockRewardModel.return_value = self.mock_reward_model
        MockTextEnvironment.return_value = self.mock_text_environment

        trainer = RL_LLMTrainer(self.config)
        trainer.train()

        # Assertions to check if components were called as expected
        self.assertEqual(self.mock_text_environment.reset.call_count, self.config.rl.num_episodes)
        # Each episode has 2 steps in our mock setup
        self.assertEqual(self.mock_text_environment.step.call_count, self.config.rl.num_episodes * 2)
        self.assertEqual(self.mock_llm_interface.generate_response.call_count, self.config.rl.num_episodes * 2)
        self.assertEqual(self.mock_reward_model.get_reward.call_count, self.config.rl.num_episodes * 2)

        # Check if output directory was created (if applicable, depends on logging setup)
        self.assertTrue(os.path.exists(self.config.system.output_dir))

    def tearDown(self):
        # Clean up test outputs
        if os.path.exists(self.config.system.output_dir):
            import shutil
            shutil.rmtree(self.config.system.output_dir)

if __name__ == "__main__":
    unittest.main()


