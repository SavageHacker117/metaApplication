
#!/bin/bash
# Intentionally "stress" the V5 system: kill renderer, remove assets, fill GPU, etc.
echo "Running CUDA OOM test..."
# CUDA OOM test requires PyTorch to be installed. If not installed, this will fail gracefully.
# CUDA_VISIBLE_DEVICES=0 python -c "import torch; x = torch.randn(100000,10000,10000, device=\'cuda\');"
# (Uncomment the line above and install PyTorch if you want to test CUDA OOM)
echo "Simulating asset loss..."
rm -f assets/nerf_asset_critical.glb
echo "Simulating renderer crash..."
pkill -f threejs_renderer_enhanced.py

# V5 Specific Stress Tests (Suggestions):
# 1. Corrupt a V5 configuration file
# echo "Corrupting V8 training configuration..."
# echo "malformed_json" > /home/ubuntu/RL-LLM-dev-tool/v8.2/config/training.json

# 2. Simulate high I/O load by creating many small files
# echo "Simulating high I/O load..."
# for i in $(seq 1 1000); do echo "dummy content" > /tmp/dummy_file_$i.txt; done

echo "Run your pipeline again and check error handling/logs!"



