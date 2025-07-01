
import unittest
import os
import sys

# Add the core directory to the Python path to import training_loop
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "core")))
from training_loop import run_training_loop

class TestCoreComponents(unittest.TestCase):

    def setUp(self):
        self.test_output_dir = "./test_output"
        os.makedirs(self.test_output_dir, exist_ok=True)

    def tearDown(self):
        # Clean up generated files and directories
        if os.path.exists(self.test_output_dir):
            for f in os.listdir(self.test_output_dir):
                os.remove(os.path.join(self.test_output_dir, f))
            os.rmdir(self.test_output_dir)

    def test_training_loop_runs(self):
        # Test if the training loop can be called without errors
        try:
            run_training_loop("dummy_config.yaml", self.test_output_dir)
            self.assertTrue(True) # If no exception, test passes
        except Exception as e:
            self.fail(f"run_training_loop raised an exception: {e}")

    def test_training_loop_creates_output(self):
        # Test if the training loop creates expected output files
        run_training_loop("dummy_config.yaml", self.test_output_dir)
        # Check for the existence of at least one model file
        model_files = [f for f in os.listdir(self.test_output_dir) if f.startswith("model_episode_")]
        self.assertGreater(len(model_files), 0, "No model files were created.")

if __name__ == '__main__':
    unittest.main()


