#!/usr/bin/env python3
"""
Setup script for Weights & Biases integration
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed successfully!")

def setup_wandb():
    """Setup wandb login"""
    print("\nSetting up Weights & Biases...")
    print("Please run the following command to login to wandb:")
    print("wandb login")
    print("\nOr if you don't have an account, visit: https://wandb.ai/")
    print("After creating an account, run: wandb login")

def main():
    print("🚀 Setting up Weights & Biases for Mask Regularization Project")
    print("=" * 60)
    
    install_requirements()
    setup_wandb()
    
    print("\n🎉 Setup complete!")
    print("\nTo run your experiments with wandb tracking:")
    print("python mask_main.py --dataset MNIST --model_type MLP --lambda_reg 0.01")

if __name__ == "__main__":
    main()
