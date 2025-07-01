# RL-LLM NeRF Integration - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Setup
```bash
# Navigate to your RL-LLM project directory
cd /path/to/your/project

# Copy and run the setup script
cp /path/to/setup_nerf_rl_llm.sh .
chmod +x setup_nerf_rl_llm.sh
./setup_nerf_rl_llm.sh
```

### Step 2: Launch
```bash
# Start the integrated system
./start_nerf_rl_llm.sh
```

### Step 3: Access
- **Game with NeRF**: http://localhost:5173
- **Backend API**: http://localhost:5000

## 🎯 What You Get

✅ **CUDA-accelerated NeRF training** integrated with your game  
✅ **Real-time 3D scene understanding** for enhanced AI training  
✅ **Web dashboard** for monitoring NeRF performance  
✅ **Seamless integration** with existing RL-LLM systems  
✅ **Automatic scene capture** during gameplay  
✅ **Enhanced reward system** using spatial understanding  

## 🔧 Key Features

- **Neural Radiance Fields (NeRF)** for 3D scene representation
- **CUDA acceleration** for fast training (CPU fallback available)
- **WebGL volumetric rendering** for real-time visualization
- **Socket.IO communication** for live updates
- **Configurable training parameters** via JSON files
- **Memory management** with intelligent caching

## 📊 Dashboard Features

The integrated NeRF Dashboard provides:
- Real-time training progress monitoring
- GPU/CPU memory usage tracking
- Interactive 3D NeRF visualization
- Performance metrics and statistics
- Training controls (start/stop/configure)

## ⚙️ Quick Configuration

Edit `nerf_integration/config/nerf_config.json` to adjust:
- **Resolution**: Higher = better quality, slower training
- **Learning rate**: Lower = more stable, slower convergence
- **Batch size**: Higher = faster training, more memory usage

## 🐛 Troubleshooting

**CUDA not detected?**
- Install NVIDIA drivers and CUDA Toolkit
- System automatically falls back to CPU

**Port conflicts?**
```bash
sudo lsof -ti:5000 | xargs kill -9
sudo lsof -ti:5173 | xargs kill -9
```

**Memory issues?**
- Reduce batch_size in nerf_config.json
- Lower resolution settings
- Enable garbage collection

## 📚 Next Steps

1. **Explore the Dashboard**: Monitor training progress and performance
2. **Customize Configuration**: Adjust settings for your specific needs
3. **Integrate with RL**: Use NeRF data for enhanced reward calculations
4. **Read Full Documentation**: See `RL_LLM_NeRF_Integration_Documentation.md`

---

*Ready to enhance your RL-LLM with advanced 3D understanding? Start with the setup script above!*

