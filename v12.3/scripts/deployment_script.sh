#!/bin/bash

# This script automates the deployment of the RL-LLM system.

# Exit immediately if a command exits with a non-zero status.
set -e

# Configuration variables
ENV_NAME="rl_llm_env"
PYTHON_VERSION="python3.11"
REQUIREMENTS_FILE="requirements.txt"
MAIN_APP_FILE="main.py"

echo "Starting RL-LLM deployment process..."

# 1. Create and activate virtual environment
echo "Creating virtual environment: $ENV_NAME"
$PYTHON_VERSION -m venv $ENV_NAME
source $ENV_NAME/bin/activate

# 2. Install dependencies
echo "Installing Python dependencies from $REQUIREMENTS_FILE"
pip install --no-cache-dir -r $REQUIREMENTS_FILE

# 3. Run any necessary migrations or setup (placeholder)
echo "Running pre-deployment setup (if any)..."
# python your_migration_script.py

# 4. Start the main application (example: a training run or API server)
echo "Starting the main RL-LLM application..."
# For training:
python $MAIN_APP_FILE

# For a web service, you might use gunicorn or uvicorn:
# uvicorn your_app:app --host 0.0.0.0 --port 8000

echo "Deployment process completed. The RL-LLM application is running."

# Deactivate virtual environment (optional, if you want to keep it active for debugging)
# deactivate