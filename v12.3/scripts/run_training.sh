
#!/bin/bash
# Run the RL-LLM training session

CONFIG=${1:-../../config/training.json}
OUTDIR=../../output_training_$(date +"%Y%m%d_%H%M%S")

mkdir -p $OUTDIR

python3 ../RUBY/toolkit/enhanced_integration_script.py --config $CONFIG --v8 --output_dir $OUTDIR

echo "Training complete! Check $OUTDIR for results, logs, and artifacts."


