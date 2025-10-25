"""
Test script for CAPMeme
Verifies that all components can be imported and basic functionality works
"""

import sys
import os
import torch
import yaml

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from models.capmeme import CAPMeme
        print("✓ CAPMeme model imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import CAPMeme: {e}")
        return False
    
    try:
        from data.dataset import MemeDataset
        print("✓ Dataset module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import dataset: {e}")
        return False
    
    try:
        from evaluation.metrics import CAPMemeEvaluator
        print("✓ Evaluation module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import evaluation: {e}")
        return False
    
    try:
        from utils.config import load_config
        print("✓ Utils module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    return True


def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        from utils.config import load_config
        config = load_config('configs/default.yaml')
        print("✓ Configuration loaded successfully")
        print(f"  - Vision encoder: {config['model']['vision_encoder']}")
        print(f"  - Text encoder: {config['model']['text_encoder']}")
        print(f"  - Embedding dim: {config['model']['embedding_dim']}")
        return True
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        return False


def test_model_initialization():
    """Test model initialization"""
    print("\nTesting model initialization...")
    
    try:
        from models.capmeme import CAPMeme
        from utils.config import load_config
        
        config = load_config('configs/default.yaml')
        model = CAPMeme(config)
        
        print("✓ Model initialized successfully")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False


def test_dataset_creation():
    """Test dataset creation"""
    print("\nTesting dataset creation...")
    
    try:
        from data.dataset import MemeDataset
        
        dataset = MemeDataset(
            data_path='data/meme_dataset.json',
            image_dir='data/images',
            image_size=224,
            max_text_length=128
        )
        
        print("✓ Dataset created successfully")
        print(f"  - Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  - Sample keys: {list(sample.keys())}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False


def test_device_setup():
    """Test device setup"""
    print("\nTesting device setup...")
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Device setup successful: {device}")
        
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU count: {torch.cuda.device_count()}")
            print(f"  - Current GPU: {torch.cuda.current_device()}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to setup device: {e}")
        return False


def main():
    """Run all tests"""
    print("CAPMeme Installation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config_loading,
        test_model_initialization,
        test_dataset_creation,
        test_device_setup
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! CAPMeme is ready to use.")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
