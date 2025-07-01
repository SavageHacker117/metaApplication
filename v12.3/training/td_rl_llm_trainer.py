
import torch
from typing import Dict, Any

from ..game_env.tower_defense_env import TowerDefenseEnv
from ..game_env.game_generator import GameGenerator
from ..game_env.game_state_processor import GameStateProcessor
from ..game_env.visualizer import GameVisualizer
from ..rl_agents.td_agent import TDAgent
from ..llm_integration.llm_api_interface import LLMAPIInterface
from ..llm_integration.llm_game_action_parser import LLMGameActionParser
from ..evaluation.reward_model import RewardModel # Re-using for general reward signal
from ..utils.experiment_tracker import ExperimentTracker
from ..core.config.config_schema import MainConfig

class TDRL_LLMTrainer:
    """
    The specialized trainer for the Tower Defense RL-LLM system.
    Orchestrates game generation, RL agent training, LLM interaction, and visualization.
    """
    def __init__(self, config: MainConfig):
        self.config = config
        self.llm_api = LLMAPIInterface(api_key=config.llm.api_key, model_name=config.llm.model_name)
        self.game_generator = GameGenerator(self.llm_api)
        self.llm_action_parser = LLMGameActionParser()
        self.reward_model = RewardModel() # For general reward feedback
        self.experiment_tracker = ExperimentTracker()

        # Initialize environment and related components after game config generation
        self.env = None
        self.state_processor = None
        self.visualizer = None
        self.agent = None

    def _initialize_game_components(self, game_config: Dict[str, Any]):
        """
        Initializes the game environment, state processor, visualizer, and RL agent
        based on the generated game configuration.
        """
        self.env = TowerDefenseEnv(grid_size=tuple(game_config["grid_size"]),
                                   initial_cash=game_config["initial_cash"],
                                   initial_lives=game_config["initial_lives"])
        self.state_processor = GameStateProcessor(grid_size=tuple(game_config["grid_size"]))
        self.visualizer = GameVisualizer(grid_size=tuple(game_config["grid_size"]))

        # Define action space for the RL agent based on possible game actions
        # This is a simplified representation. A real system would have a more nuanced action space.
        # Actions: 0: place_tower_basic, 1: place_tower_archer, 2: place_tower_cannon, 3: start_wave, 4: no_op
        action_dim = len(game_config["tower_types"]) + 2 # +1 for start_wave, +1 for no_op
        state_dim = self.state_processor.get_state_dimension()
        self.agent = TDAgent(state_dim=state_dim, action_dim=action_dim,
                             lr=self.config.rl.learning_rate, gamma=self.config.rl.gamma)

    def train(self, num_training_episodes: int):
        """
        Runs the training loop for the TD RL-LLM system.
        Each episode involves generating a game, playing it, and training the agent.
        """
        self.experiment_tracker.start_experiment("td_rl_llm_training", self.config.dict())
        
        for episode_num in range(1, num_training_episodes + 1):
            print(f"\n--- Starting Episode {episode_num} ---")
            
            # 1. Generate a new game environment configuration using LLM
            game_config = self.game_generator.generate_game_config(complexity_level="medium")
            self._initialize_game_components(game_config)
            
            current_game_state = self.env.reset()
            done = False
            episode_total_reward = 0
            game_turn = 0

            while not done and game_turn < 50: # Limit turns per game to prevent infinite loops
                processed_state = self.state_processor.process_state(current_game_state)
                rl_action_idx = self.agent.select_action(processed_state)

                # Map RL action index to a concrete game action for the LLM
                # This mapping needs to be consistent with the action_dim definition
                game_action_type = "no_op"
                if rl_action_idx == 0: game_action_type = "place_tower_basic"
                elif rl_action_idx == 1: game_action_type = "place_tower_archer"
                elif rl_action_idx == 2: game_action_type = "place_tower_cannon"
                elif rl_action_idx == 3: game_action_type = "start_wave"
                
                # LLM generates a natural language command based on the RL action and current state
                llm_prompt = f"Given the current game state:\n{self.visualizer.render(current_game_state)}\n\nAnd the RL agent suggests to {game_action_type}. Generate a precise command for this action, including coordinates if placing a tower (e.g., 'place basic tower at (X,Y)'). If starting a wave, just say 'start wave'. If no_op, say 'do nothing'."
                llm_response_text = self.llm_api.generate_response(llm_prompt, max_tokens=100, temperature=0.7)
                
                # Parse LLM's natural language response into a structured game action
                parsed_game_action = self.llm_action_parser.parse_action(llm_response_text)

                if parsed_game_action:
                    next_game_state, reward, done, info = self.env.step(parsed_game_action)
                else:
                    # Penalize for unparseable LLM output
                    reward = -10.0
                    next_game_state = current_game_state # State doesn't change
                    done = False
                    info = {"error": "LLM output unparseable"}

                # Reward for RL agent: combine game reward with LLM quality reward
                # (e.g., using the reward_model to assess LLM response quality)
                llm_quality_reward = self.reward_model.get_reward(llm_prompt, llm_response_text) - 0.5 # Center around 0
                total_step_reward = reward + llm_quality_reward

                self.agent.rewards.append(total_step_reward)
                self.agent.is_terminals.append(done)
                self.agent.values.append(self.agent.actor_critic(torch.FloatTensor(processed_state))[1]) # Store value for GAE

                episode_total_reward += total_step_reward
                current_game_state = next_game_state
                game_turn += 1

                if done:
                    print(f"Game Over. Final Lives: {current_game_state['lives']}")
                    break
            
            self.agent.update()
            self.experiment_tracker.log_metric("episode_reward", episode_total_reward, step=episode_num)
            self.experiment_tracker.log_metric("final_lives", current_game_state["lives"], step=episode_num)
            print(f"Episode {episode_num} finished. Total Reward: {episode_total_reward:.2f}")

        self.experiment_tracker.end_experiment({"total_episodes": num_training_episodes, "final_avg_reward": self.experiment_tracker.metrics["episode_reward"][-1]})
        print("TD RL-LLM training finished.")

# Example usage:
# if __name__ == "__main__":
#     from pydantic import ValidationError
#     import os

#     # Dummy config for demonstration
#     example_config_data = {
#         "llm": {"model_name": "gpt-3.5-turbo", "api_key": os.environ.get("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")},
#         "rl": {"algorithm": "PPO", "learning_rate": 0.0001, "gamma": 0.99, "num_episodes": 5},
#         "system": {"environment_name": "tower_defense", "log_level": "INFO", "output_dir": "./td_outputs", "debug_mode": False}
#     }

#     try:
#         validated_config = MainConfig(**example_config_data)
#         trainer = TDRL_LLMTrainer(validated_config)
#         trainer.train(num_training_episodes=2)
#     except ValidationError as e:
#         print(f"Configuration validation error: {e}")
#     except ValueError as e:
#         print(f"Error: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")


