
#!/usr/bin/env python3
"""
Setup Script for RL Training System Version 8

This script handles the installation and setup of the Version 8
training system, including dependency installation, environment setup,
and initial configuration.

Usage:
    python setup.py --install
    python setup.py --configure
    python setup.py --test
    python setup.py --all

Author: Manus AI Team
Version: 8.0.0
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import platform
import shutil
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class V8Setup:
    """
    Setup manager for Version 8.
    
    Handles installation, configuration, and initial setup of the
    RL training system.
    """
    
    def __init__(self):
        self.version = "8.0.0"
        self.project_root = Path(__file__).parent
        self.system_info = self._get_system_info()
        
        logger.info(f"V8 Setup initialized for version {self.version}")
        logger.info(f"System: {self.system_info['platform']} {self.system_info['python_version']}")
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'python_executable': sys.executable
        }
    
    def check_requirements(self) -> bool:
        """Check system requirements."""
        
        logger.info("Checking system requirements...")
        
        requirements_met = True
        
        # Check Python version
        python_version_parts = platform.python_version().split(".")
        python_version = tuple(int(p) for p in python_version_parts if p.isdigit())
        if python_version < (3, 8):
            logger.error(f"Python 3.8+ required, found {platform.python_version()}")
            requirements_met = False
        else:
            logger.info(f"✓ Python version: {platform.python_version()}")
        
        # Check available disk space
        try:
            disk_usage = shutil.disk_usage(self.project_root)
            free_gb = disk_usage.free / (1024**3)
            if free_gb < 5:  # 5GB minimum
                logger.error(f"Insufficient disk space: {free_gb:.1f}GB available, 5GB required")
                requirements_met = False
            else:
                logger.info(f"✓ Disk space: {free_gb:.1f}GB available")
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
        
        # Check for CUDA (optional)
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✓ NVIDIA GPU detected")
            else:
                logger.info("ℹ No NVIDIA GPU detected (CPU training will be used)")
        except FileNotFoundError:
            logger.info("ℹ nvidia-smi not found (CPU training will be used)")
        
        return requirements_met
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies."""
        
        logger.info("Installing Python dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        if not requirements_file.exists():
            logger.error(f"Requirements file not found: {requirements_file}")
            return False
        
        try:
            # Upgrade pip first
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--upgrade", "pip"
            ], check=True)
            
            # Install requirements
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            
            logger.info("✓ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            return False
    
    def setup_directories(self) -> bool:
        """Setup necessary directories."""
        
        logger.info("Setting up directories...")
        
        directories = [
            "outputs",
            "models",
            "logs",
            "visualizations",
            "feedback",
            "checkpoints",
            "exports",
            "assets/nerf",
            "error_reports"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✓ Created directory: {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create directories: {e}")
            return False
    
    def create_default_config(self) -> bool:
        """Create default configuration files."""
        
        logger.info("Creating default configuration...")
        
        config_dir = self.project_root / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Check if config already exists
        config_file = config_dir / "training.json"
        if config_file.exists():
            logger.info("✓ Configuration file already exists")
            return True
        
        # Default configuration is already created in config/training.json
        # Just verify it exists
        if config_file.exists():
            logger.info("✓ Default configuration verified")
            return True
        else:
            logger.error("Default configuration file not found")
            return False
    
    def setup_environment_variables(self) -> bool:
        """Setup environment variables."""
        
        logger.info("Setting up environment variables...")
        
        env_vars = {
            'RL_TRAINING_VERSION': '8.0.0',
            'RL_CONFIG_PATH': str(self.project_root / 'config' / 'training.json'),
            'RL_OUTPUT_DIR': str(self.project_root / 'outputs'),
            'PYTHONPATH': str(self.project_root)
        }
        
        # Create environment setup script
        if self.system_info['platform'] == 'Windows':
            env_script = self.project_root / "setup_env.bat"
            with open(env_script, 'w') as f:
                f.write("@echo off\n")
                f.write("echo Setting up Version 8 environment...\n")
                for key, value in env_vars.items():
                    f.write(f"set {key}={value}\n")
                f.write("echo Environment setup complete!\n")
        else:
            env_script = self.project_root / "setup_env.sh"
            with open(env_script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("echo 'Setting up Version 8 environment!'\n")
                for key, value in env_vars.items():
                    f.write(f"export {key}=\"{value}\"\n")
                f.write("echo 'Environment setup complete!'\n")
            
            # Make script executable
            os.chmod(env_script, 0o755)
        
        logger.info(f"✓ Environment setup script created: {env_script}")
        return True
    
    def verify_installation(self) -> bool:
        """Verify the installation."""
        
        logger.info("Verifying installation...")
        
        try:
            # Test imports
            sys.path.insert(0, str(self.project_root))
            
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            
            # Test core modules
            from core.training_loop import EnhancedTrainingLoop
            from core.nerf_integration import NeRFIntegrationManager
            from main import RLTrainingSystemV8
            
            logger.info("✓ Core modules imported successfully")
            
            # Test CUDA availability
            if torch.cuda.is_available():
                logger.info(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("ℹ CUDA not available, using CPU")
            
            # Test configuration loading
            config_path = self.project_root / "config" / "training.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                    if config.get("version") == self.version:
                        logger.info("✓ Configuration file valid")
                    else:
                        logger.warning("Configuration version mismatch")
            
            return True
            
        except Exception as e:
            logger.error(f"Installation verification failed: {e}")
            return False
    
    def run_quick_test(self) -> bool:
        """Run quick functionality test."""
        
        logger.info("Running quick functionality test...")
        
        try:
            # Run the integration test
            test_script = self.project_root / "test_integration.py"
            if test_script.exists():
                result = subprocess.run([
                    sys.executable, str(test_script), "--quick-test"
                ], capture_output=True, text=True, cwd=self.project_root)
                
                if result.returncode == 0:
                    logger.info("✓ Quick test passed")
                    return True
                else:
                    logger.error(f"Quick test failed: {result.stderr}")
                    return False
            else:
                logger.warning("Test script not found, skipping tests")
                return True
                
        except Exception as e:
            logger.error(f"Quick test failed: {e}")
            return False
    
    def install(self) -> bool:
        """Run complete installation."""
        
        logger.info("Starting Version 8 installation...")
        
        steps = [
            ("Checking requirements", self.check_requirements),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up directories", self.setup_directories),
            ("Creating configuration", self.create_default_config),
            ("Setting up environment", self.setup_environment_variables),
            ("Verifying installation", self.verify_installation)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Installation failed at step: {step_name}")
                return False
        
        logger.info("✓ Installation completed successfully!")
        logger.info("Run 'python main.py --help' to get started")
        
        return True
    
    def configure(self) -> bool:
        """Run configuration setup."""
        
        logger.info("Configuring Version 8...")
        
        # Interactive configuration could be added here
        # For now, just verify the default configuration
        
        config_path = self.project_root / "config" / "training.json"
        if config_path.exists():
            logger.info("✓ Configuration file exists")
            
            # Optionally, could add interactive configuration here
            logger.info("Using default configuration. Edit config/training.json to customize.")
            return True
        else:
            logger.error("Configuration file not found")
            return False
    
    def test(self) -> bool:
        """Run test suite."""
        
        logger.info("Running test suite...")
        return self.run_quick_test()
    
    def all(self) -> bool:
        """Run complete setup (install + configure + test)."""
        
        logger.info("Running complete setup...")
        
        if not self.install():
            return False
        
        if not self.configure():
            return False
        
        if not self.test():
            logger.warning("Tests failed, but installation may still be functional")
        
        logger.info("Complete setup finished!")
        return True

def main():
    """Main entry point for setup script."""
    parser = argparse.ArgumentParser(
        description='Setup script for RL Training System Version 8')
    parser.add_argument('--install', action='store_true', help='Install dependencies and setup')
    parser.add_argument('--configure', action='store_true', help='Configure the system')
    parser.add_argument('--test', action='store_true', help='Run test suite')
    parser.add_argument('--all', action='store_true', help='Run complete setup')
    parser.add_argument('--check', action='store_true', help='Check requirements only')

    args = parser.parse_args()

    setup = V8Setup()

    try:
        success = False

        if args.install:
            success = setup.install()
        elif args.configure:
            success = setup.configure()
        elif args.test:
            success = setup.test()
        elif args.all:
            success = setup.all()
        elif args.check:
            success = setup.check_requirements()
        else:
            # Default to complete setup
            success = setup.all()

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


