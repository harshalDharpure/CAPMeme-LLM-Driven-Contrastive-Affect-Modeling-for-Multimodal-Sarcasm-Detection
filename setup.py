"""
Setup script for CAPMeme
Installs dependencies and sets up the project environment
"""

import os
import sys
import subprocess
import platform


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


def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"✗ Python {version.major}.{version.minor} is not supported.")
        print("Please use Python 3.8 or higher.")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    # Check if pip is available
    if not run_command("pip --version", "Checking pip availability"):
        print("✗ pip is not available. Please install pip first.")
        return False
    
    # Install PyTorch (with CUDA support if available)
    print("\nInstalling PyTorch...")
    if platform.system() == "Windows":
        torch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else:
        torch_command = "pip install torch torchvision torchaudio"
    
    if not run_command(torch_command, "Installing PyTorch"):
        print("Warning: PyTorch installation failed. Trying CPU-only version...")
        if not run_command("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu", 
                          "Installing PyTorch (CPU-only)"):
            return False
    
    # Install other dependencies
    dependencies = [
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "pillow>=9.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "wandb>=0.15.0",
        "opencv-python>=4.8.0",
        "sentence-transformers>=2.2.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "rouge-score>=0.1.2",
        "nltk>=3.8.0",
        "spacy>=3.6.0",
        "networkx>=3.1.0",
        "rdflib>=6.3.0",
        "pyyaml>=6.0"
    ]
    
    for dep in dependencies:
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"Warning: Failed to install {dep}")
    
    return True


def setup_directories():
    """Create necessary directories"""
    print("\nSetting up directories...")
    
    directories = [
        'data/images',
        'data/conceptnet_cache',
        'outputs/checkpoints',
        'outputs/logs',
        'outputs/results',
        'models/saved'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        except Exception as e:
            print(f"✗ Failed to create directory {directory}: {e}")
            return False
    
    return True


def download_sample_data():
    """Create sample data for testing"""
    print("\nCreating sample data...")
    
    try:
        from data.dataset import MemeDataset
        
        # This will create sample data automatically
        dataset = MemeDataset(
            data_path='data/meme_dataset.json',
            image_dir='data/images',
            image_size=224,
            max_text_length=128
        )
        
        print(f"✓ Sample dataset created with {len(dataset)} samples")
        return True
    except Exception as e:
        print(f"✗ Failed to create sample data: {e}")
        return False


def test_installation():
    """Test the installation"""
    print("\nTesting installation...")
    
    try:
        result = subprocess.run([sys.executable, "test_installation.py"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ Installation test passed")
            return True
        else:
            print("✗ Installation test failed")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Failed to run installation test: {e}")
        return False


def main():
    """Main setup function"""
    print("CAPMeme Setup Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("✗ Dependency installation failed")
        return False
    
    # Setup directories
    if not setup_directories():
        print("✗ Directory setup failed")
        return False
    
    # Create sample data
    if not download_sample_data():
        print("✗ Sample data creation failed")
        return False
    
    # Test installation
    if not test_installation():
        print("✗ Installation test failed")
        return False
    
    print("\n" + "=" * 50)
    print("✓ CAPMeme setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python train.py' to start training")
    print("2. Run 'python inference.py --checkpoint outputs/best_model.pt --image data/images/meme1.jpg --text \"Your text here\"' for inference")
    print("3. Open notebooks/capmeme_demo.ipynb for interactive demo")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
