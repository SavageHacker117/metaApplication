
import os
import torch
from pydantic import ValidationError

from go_env.minigo_go_env import MinigoGoEnv
from rl_agent.go_policy_value_net import GoPolicyValueNet
from rl_agent.alpha_go_trainer import AlphaGoTrainer
from llm_integration.llm_api_interface import LLMAPIInterface
from llm_integration.go_llm_analyzer import GoLLMAnalyzer
from llm_integration.go_llm_formatter import GoLLMFormatter
from core.config.go_rl_llm_config import GoRL_LLMMainConfig
from utils.config_loader import load_config

def run_go_rl_llm_system(config_path: str):
    print("\n--- Starting Go RL-LLM System ---")

    # 1. Load Configuration
    try:
        config = load_config(config_path, GoRL_LLMMainConfig)
        print(f"Loaded configuration for board size: {config.game_specific.board_size}")
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return

    # 2. Initialize LLM Interface and Analyzer
    llm_api = LLMAPIInterface(api_key=config.llm.api_key, model_name=config.llm.model_name)
    go_llm_analyzer = GoLLMAnalyzer(llm_api, board_size=config.game_specific.board_size)
    go_llm_formatter = GoLLMFormatter(board_size=config.game_specific.board_size)

    # 3. Initialize RL Agent Components
    policy_value_net = GoPolicyValueNet(board_size=config.game_specific.board_size)
    if config.rl.load_model_path and os.path.exists(config.rl.load_model_path):
        policy_value_net.load_state_dict(torch.load(config.rl.load_model_path))
        print(f"Loaded pre-trained model from {config.rl.load_model_path}")
    
    # 4. Initialize AlphaGo Trainer
    trainer = AlphaGoTrainer(
        policy_value_net=policy_value_net,
        board_size=config.game_specific.board_size,
        learning_rate=config.rl.learning_rate,
        l2_const=config.rl.l2_reg,
        n_playout=config.rl.mcts_n_playout,
        buffer_size=config.rl.buffer_size,
        batch_size=config.rl.batch_size,
        epochs=config.rl.epochs,
        check_freq=config.rl.save_freq,
        cuda=config.system.use_cuda
    )

    # 5. Run Training (Self-Play)
    print("\n--- Starting RL Agent Training (Self-Play) ---")
    trainer.run(num_iterations=config.rl.num_iterations)

    # 6. Post-Training Analysis with LLM (Example)
    print("\n--- Performing LLM-based Game Analysis ---")
    # Simulate a game for analysis
    env = MinigoGoEnv(board_size=config.game_specific.board_size)
    env.reset()
    moves_history = []
    
    # Play a few moves (e.g., from a trained agent or random)
    # For demonstration, let's make a few dummy moves
    dummy_moves = [
        (2,2), (3,3), (2,3), (3,2), (None, None), (None, None) # Example moves
    ]
    current_board_state = env.board.copy()
    current_player = env.current_player

    for i, move in enumerate(dummy_moves):
        if not env.is_game_over():
            current_board_state, reward, done, info = env.step(move)
            moves_history.append(move)
            if i == 2: # Analyze a specific move
                explanation = go_llm_analyzer.explain_move(env.board.tolist(), env.current_player, move)
                print(f"\nLLM Explanation for move {move}:\n{explanation}")
            if done: break

    if moves_history:
        sgf_history = go_llm_formatter.format_to_sgf(moves_history, config.game_specific.board_size)
        strategy_summary = go_llm_analyzer.summarize_strategy(sgf_history)
        print(f"\nLLM Summary of Game Strategy:\n{strategy_summary}")

        weakness_analysis = go_llm_analyzer.spot_weaknesses(sgf_history)
        print(f"\nLLM Weakness Analysis:\n{weakness_analysis}")

    print("\n--- Go RL-LLM System Finished ---")

if __name__ == "__main__":
    # Default configuration file path
    default_config_path = "./config/go_rl_llm_config.json"
    
    # Ensure the config directory exists
    os.makedirs(os.path.dirname(default_config_path), exist_ok=True)

    # Create a dummy config file if it doesn't exist for first run
    if not os.path.exists(default_config_path):
        print(f"Creating a default configuration file at {default_config_path}")
        dummy_config_content = GoRL_LLMMainConfig().json(indent=4)
        with open(default_config_path, "w") as f:
            f.write(dummy_config_content)

    run_go_rl_llm_system(default_config_path)


