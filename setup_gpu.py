"""
GPU-Optimized Setup Script for CAPMeme
Sets up the environment for training with 10,000 images on GPU
"""

import os
import sys
import subprocess
import platform
import torch


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_gpu_setup():
    """Check GPU setup and CUDA availability"""
    print("Checking GPU setup...")
    
    if not torch.cuda.is_available():
        print("✗ CUDA is not available. GPU training will not be possible.")
        print("Please install CUDA-compatible PyTorch.")
        return False
    
    print(f"✓ CUDA is available")
    print(f"  - CUDA Version: {torch.version.cuda}")
    print(f"  - GPU Count: {torch.cuda.device_count()}")
    print(f"  - Current GPU: {torch.cuda.current_device()}")
    print(f"  - GPU Name: {torch.cuda.get_device_name()}")
    print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True


def install_gpu_dependencies():
    """Install GPU-optimized dependencies"""
    print("\nInstalling GPU-optimized dependencies...")
    
    # Install PyTorch with CUDA support
    print("\nInstalling PyTorch with CUDA support...")
    if platform.system() == "Windows":
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    
    if not run_command(torch_command, "Installing PyTorch with CUDA"):
        print("Warning: PyTorch CUDA installation failed. Trying CPU-only version...")
        if not run_command("pip install torch torchvision torchaudio", 
                          "Installing PyTorch (CPU-only)"):
            return False
    
    # Install other dependencies
    dependencies = [
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "pillow>=9.5.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "tensorboard>=2.13.0",
        "nltk>=3.8.0",
        "spacy>=3.6.0",
        "networkx>=3.1.0",
        "rdflib>=6.3.0",
        "pyyaml>=6.0",
        "requests>=2.28.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"Warning: Failed to install {dep}")
    
    return True


def setup_directories():
    """Create necessary directories for 10K image training"""
    print("\nSetting up directories for 10K image training...")
    
    directories = [
        'data/images',
        'data/conceptnet_cache',
        'outputs/checkpoints',
        'outputs/logs',
        'outputs/results',
        'models/saved',
        'configs',
        'notebooks'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    return True


def create_gpu_config():
    """Create GPU-optimized configuration"""
    print("\nCreating GPU-optimized configuration...")
    
    config_content = """# CAPMeme Configuration - GPU Optimized for 10K Images
model:
  name: "CAPMeme"
  vision_encoder: "clip"
  text_encoder: "bert"
  embedding_dim: 512
  hidden_dim: 256
  num_classes: 4
  
# Dataset configuration - Optimized for 10K images
data:
  dataset_name: "meme_dataset_10k"
  image_size: 224
  max_text_length: 128
  batch_size: 64  # Optimized for GPU
  num_workers: 8  # Optimized for data loading
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  
# Training configuration - Optimized for large dataset
training:
  epochs: 100
  learning_rate: 2e-4
  weight_decay: 1e-5
  warmup_steps: 2000
  gradient_clip_norm: 1.0
  save_every: 10
  eval_every: 2
  
# Contrastive Affective Pretraining
cap:
  temperature: 0.07
  contrastive_weight: 0.5
  affective_weight: 0.3
  knowledge_weight: 0.2
  
# Knowledge Graph
knowledge_graph:
  use_conceptnet: true
  use_comet: true
  kg_embedding_dim: 200
  max_relations: 10
  
# Evaluation
evaluation:
  metrics: ["accuracy", "f1", "auroc"]
  save_predictions: true
  
# Logging
logging:
  use_wandb: true
  project_name: "capmeme-10k-gpu"
  log_every: 50
"""
    
    try:
        with open('configs/gpu_config.yaml', 'w') as f:
            f.write(config_content)
        print("✓ Created GPU-optimized configuration")
        return True
    except Exception as e:
        print(f"✗ Failed to create configuration: {e}")
        return False


def test_gpu_setup():
    """Test GPU setup"""
    print("\nTesting GPU setup...")
    
    try:
        import torch
        if torch.cuda.is_available():
            # Test GPU memory allocation
            device = torch.device('cuda')
            test_tensor = torch.randn(1000, 1000).to(device)
            result = torch.matmul(test_tensor, test_tensor)
            print("✓ GPU tensor operations working")
            
            # Test mixed precision
            with torch.cuda.amp.autocast():
                result = torch.matmul(test_tensor, test_tensor)
            print("✓ Mixed precision training available")
            
            # Clear GPU memory
            del test_tensor, result
            torch.cuda.empty_cache()
            print("✓ GPU memory management working")
            
            return True
        else:
            print("✗ CUDA not available for testing")
            return False
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        return False


def main():
    """Main setup function for GPU training"""
    print("CAPMeme GPU Setup Script")
    print("=" * 60)
    print("Setting up environment for training with 10,000 images on GPU")
    print("=" * 60)
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.8 or higher.")
        return False
    
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    
    # Install dependencies
    if not install_gpu_dependencies():
        print("✗ Dependency installation failed")
        return False
    
    # Setup directories
    if not setup_directories():
        print("✗ Directory setup failed")
        return False
    
    # Create GPU config
    if not create_gpu_config():
        print("✗ Configuration creation failed")
        return False
    
    # Test GPU setup
    if not test_gpu_setup():
        print("✗ GPU setup test failed")
        return False
    
    print("\n" + "=" * 60)
    print("✓ CAPMeme GPU setup completed successfully!")
    print("\nNext steps:")
    print("1. Organize your 10,000 images:")
    print("   python organize_images.py --source_dir /path/to/your/images --target_dir data/images")
    print("\n2. Start GPU training:")
    print("   python train_gpu.py --config configs/gpu_config.yaml")
    print("\n3. Monitor training:")
    print("   - Check Weights & Biases dashboard")
    print("   - Monitor GPU usage with nvidia-smi")
    print("\n4. For inference:")
    print("   python inference.py --checkpoint outputs/best_model.pt --image data/images/meme_00001.jpg --text \"Your text here\"")
    
    print("\nGPU Training Tips:")
    print("- Use batch size 64-128 for optimal GPU utilization")
    print("- Monitor GPU memory usage during training")
    print("- Enable mixed precision for faster training")
    print("- Use multiple workers for data loading")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
