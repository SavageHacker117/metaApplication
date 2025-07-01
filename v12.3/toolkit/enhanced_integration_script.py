
import argparse
import os
import time
import sys

# Add the core directory to the Python path to import training_loop
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "core"))
from training_loop import run_training_loop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_production.yaml", help="Configuration file")
    parser.add_argument("--v8", action="store_true", help="V8 flag")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    args = parser.parse_args()

    print(f"Running enhanced integration script with config: {args.config}, V8: {args.v8}, output_dir: {args.output_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    log_file_path = os.path.join(args.output_dir, 'run.log')

    # Redirect stdout and stderr to the log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with open(log_file_path, 'w') as f:
        sys.stdout = f
        sys.stderr = f
        print(f"Simulation started at {time.ctime()}\n")
        print("Calling RL training loop...\n")
        
        # Call the actual training loop
        run_training_loop(args.config, args.output_dir)

        print("Simulation finished.\n")
    
    # Restore stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    print(f"Demo complete! Check {args.output_dir} for results, logs, and artifacts.")

if __name__ == '__main__':
    main()


