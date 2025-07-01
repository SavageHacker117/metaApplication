
#!/bin/bash
# Run a V5 demo session and save all artifacts

CONFIG=${1:-config_production.yaml}
OUTDIR=output_$(date +"%Y%m%d_%H%M%S")
mkdir -p $OUTDIR

python enhanced_integration_script.py --config $CONFIG --v8 --output_dir $OUTDIR | tee $OUTDIR/run.log
echo "Demo complete! Check $OUTDIR for results, logs, and artifacts."


